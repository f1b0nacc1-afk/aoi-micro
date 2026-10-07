from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .align import BoardFrame, ecc_refine, local_patch, parallax_px, rectify
from .cad import Component

REGION_MARGIN_MM = 0.25
BODY_MARGIN_MM = 0.1
FLATTEN_SIGMA_MM = 6.0


@dataclass
class PartGeom:
    center: tuple
    ccw: float
    region: tuple
    body: tuple
    pad: int
    patch: tuple
    search: int


def _odd(v: float) -> int:
    return int(round(v)) | 1


def part_geometry(c: Component, frame: BoardFrame, principal: tuple, cfg: dict) -> PartGeom:
    s = frame.s
    p = frame.to_px(c.x_mm, c.y_mm)
    dx, dy = parallax_px(p, principal, c.package.height, cfg["optics"]["wd_mm"])
    region = (_odd((c.package.total[0] + 2 * REGION_MARGIN_MM) * s),
              _odd((c.package.total[1] + 2 * REGION_MARGIN_MM) * s))
    body = (_odd((c.package.body[0] + 2 * BODY_MARGIN_MM) * s),
            _odd((c.package.body[1] + 2 * BODY_MARGIN_MM) * s))
    search = int(math.ceil(cfg["inspect"]["search_mm"] * s))
    slack = int(math.ceil(math.sin(math.radians(cfg["inspect"]["rot_search_deg"])) * max(region)))
    pad = search + slack
    return PartGeom(center=(p[0] + dx, p[1] + dy), ccw=frame.visual_ccw(c.rot_deg),
                    region=region, body=body, pad=pad, patch=(region[0] + 2 * pad, region[1] + 2 * pad),
                    search=search)


def center_crop(img: np.ndarray, size: tuple) -> np.ndarray:
    h, w = img.shape[:2]
    x, y = (w - size[0]) // 2, (h - size[1]) // 2
    return img[y:y + size[1], x:x + size[0]]


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or a.shape != b.shape:
        return 0.0
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    d = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return float(np.dot(a, b)) / d if d > 1e-6 else 0.0


def _rot(img: np.ndarray, deg: float) -> np.ndarray:
    if deg == 0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def match_part(patch: np.ndarray, body_tmpl: np.ndarray, region: tuple, angles: list,
               fine_step: float = 0.0, max_shift: int | None = None) -> dict:
    """Locate the body template in the patch over de-rotation angles (coarse list, then +-fine_step
    around the best); return the registered inspection region centred on the found body.
    dtheta = part rotation vs golden, CCW."""
    best = {"ncc": -1.0}
    bh, bw = body_tmpl.shape[:2]

    def trial(a):
        nonlocal best
        res = cv2.matchTemplate(_rot(patch, a), body_tmpl, cv2.TM_CCOEFF_NORMED)
        ox = oy = 0
        if max_shift is not None:
            cy0, cx0 = (res.shape[0] - 1) // 2, (res.shape[1] - 1) // 2
            oy, ox = max(0, cy0 - max_shift), max(0, cx0 - max_shift)
            res = res[oy:cy0 + max_shift + 1, ox:cx0 + max_shift + 1]
        _, mx, _, loc = cv2.minMaxLoc(res)
        loc = (loc[0] + ox, loc[1] + oy)
        if mx > best["ncc"]:
            best = {"ncc": float(mx), "loc": loc, "angle": a, "dtheta": -a}

    for a in angles:
        trial(a)
    if fine_step:
        a0 = best["angle"]
        for a in (a0 - fine_step, a0 + fine_step):
            trial(a)
    ph, pw = patch.shape[:2]
    cx, cy = best["loc"][0] + bw / 2, best["loc"][1] + bh / 2
    x0, y0 = int(round(cx - region[0] / 2)), int(round(cy - region[1] / 2))
    x0, y0 = min(max(0, x0), pw - region[0]), min(max(0, y0), ph - region[1])
    best["registered"] = _rot(patch, best["angle"])[y0:y0 + region[1], x0:x0 + region[0]]
    best["dx_px"], best["dy_px"] = cx - pw / 2, cy - ph / 2
    return best


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def offset_mad(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean |a - b| over mask after removing the median brightness offset (batch shade change)."""
    if not mask.any():
        return 0.0
    a, b = a.astype(np.float32)[mask], b.astype(np.float32)[mask]
    return float(np.mean(np.abs((a - np.median(a)) - (b - np.median(b)))))


def zones(c: Component, shape: tuple, s: float) -> dict:
    """Boolean masks in region coordinates (region centred on the body):
    body - package body; interior - marking / batch-dependent colour (median test only);
    frame - body minus interior (terminals, body edges); mark - polarity feature."""
    h, w = shape
    pkg = c.package

    def rect(cx, cy, rw, rh):
        m = np.zeros(shape, bool)
        x0, x1 = int(round(w / 2 + (cx - rw / 2) * s)), int(round(w / 2 + (cx + rw / 2) * s))
        y0, y1 = int(round(h / 2 + (cy - rh / 2) * s)), int(round(h / 2 + (cy + rh / 2) * s))
        m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
        return m

    L, W = pkg.body
    body = rect(0, 0, L, W)
    iw, ih = pkg.interior or (max(L - 0.6, 0.4 * L), max(W - 0.6, 0.4 * W))
    interior = rect(0, 0, iw, ih)
    mark = np.zeros(shape, bool)
    if pkg.mark:
        mx, my, mw, mh = pkg.mark
        for k in range(4 if is_square(pkg) else 2):
            px, py = mx, my
            for _ in range(k * (1 if is_square(pkg) else 2)):
                px, py = py, -px
            mark |= rect(px, py, mw + 0.2, mh + 0.2)
    interior &= ~mark
    TL, TW = pkg.total
    joint = rect(0, 0, TL, TW) & ~body
    return {"body": body, "interior": interior, "frame": body & ~interior & ~mark, "mark": mark,
            "joint": joint}


def is_square(pkg) -> bool:
    return pkg.mark is not None and abs(pkg.body[0] - pkg.body[1]) < 0.05 and abs(pkg.mark[2] - pkg.mark[3]) < 0.05


def angle_list(cfg: dict, train: bool = False) -> tuple:
    """Coarse angle list and the fine refinement step."""
    m, st = cfg["inspect"]["rot_search_deg"], cfg["inspect"]["rot_step_deg"]
    if train:
        m = min(m, 4)
    coarse = 2 * st
    n = int(m // coarse)
    return [0.0] + [sg * k * coarse for k in range(1, n + 1) for sg in (1, -1)], st


def mark_rects(c: Component, s: float, shape: tuple):
    """Mark rect and its 180° counterpart in region pixel coordinates."""
    r = mark_rects_k(c, s, shape)
    return r[0], r[2]


def mark_rects_k(c: Component, s: float, shape: tuple) -> dict:
    """Mark rect after k CCW quarter turns of the part, k = 0..3 (region pixel coordinates)."""
    th, tw = shape
    mx, my, mw, mh = c.package.mark
    out = {}
    for k in range(4):
        px, py, rw, rh = mx, my, mw, mh
        for _ in range(k):
            px, py, rw, rh = py, -px, rh, rw
        w, h = max(3, int(rw * s)), max(3, int(rh * s))
        x, y = int(round(tw / 2 + px * s - w / 2)), int(round(th / 2 + py * s - h / 2))
        out[k] = (max(0, x), max(0, y), w, h)
    return out


def crop(img, r):
    x, y, w, h = r
    return img[y:y + h, x:x + w]


def flatten_to(board: np.ndarray, ref: np.ndarray, sigma_px: float) -> np.ndarray:
    """Match low-frequency illumination of board to ref; keeps local contrast untouched."""
    k = 16
    small = lambda im: cv2.resize(im, None, fx=1 / k, fy=1 / k, interpolation=cv2.INTER_AREA).astype(np.float32)
    sb = cv2.GaussianBlur(small(board), (0, 0), sigma_px / k)
    sr = cv2.GaussianBlur(small(ref), (0, 0), sigma_px / k)
    gain = np.clip(sr / np.maximum(sb, 1.0), 0.5, 2.0)
    gain = cv2.resize(gain, (board.shape[1], board.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(board.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def prepare_board(raw: np.ndarray, cfg: dict, frame: BoardFrame, ref: np.ndarray | None = None):
    board, info = rectify(raw, cfg, frame)
    if ref is not None:
        board, warp, cc = ecc_refine(board, ref, cfg["inspect"]["ecc_scale"])
        board = flatten_to(board, ref, FLATTEN_SIGMA_MM * frame.s)
        info["ecc_cc"] = cc
        info["ecc_warp"] = warp.tolist()
    return board, info


class Welford:
    def __init__(self):
        self.n, self.mean, self.m2 = 0, None, None

    def add(self, x: np.ndarray):
        x = x.astype(np.float64)
        self.n += 1
        if self.mean is None:
            self.mean, self.m2 = x.copy(), np.zeros_like(x)
            return
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def std(self):
        return np.sqrt(self.m2 / max(1, self.n - 1))


def component_mask(comps: list, frame: BoardFrame, principal: tuple, cfg: dict,
                   margin_mm: float = REGION_MARGIN_MM) -> np.ndarray:
    mask = np.zeros(frame.size[::-1], np.uint8)
    for c in comps:
        if c.package is None:
            continue
        g = part_geometry(c, frame, principal, cfg)
        p0 = frame.to_px(c.x_mm, c.y_mm)
        w = (c.package.total[0] + 2 * margin_mm) * frame.s
        h = (c.package.total[1] + 2 * margin_mm) * frame.s
        for ctr in (p0, g.center):
            box = cv2.boxPoints(((ctr[0], ctr[1]), (w, h), -g.ccw))
            cv2.fillPoly(mask, [np.int32(np.round(box))], 255)
    for f in cfg["board"]["fiducials"]:
        x, y = frame.to_px(f["x_mm"], f["y_mm"])
        cv2.circle(mask, (int(x), int(y)), int(1.8 * frame.s), 255, -1)
    return mask


def _mean_u8(boards: list) -> np.ndarray:
    acc = np.zeros(boards[0].shape, np.float32)
    for b in boards:
        acc += b
    return np.clip(np.round(acc / len(boards)), 0, 255).astype(np.uint8)


def train(images: list, comps: list, cfg: dict, out_dir: Path, log=print) -> dict:
    frame = BoardFrame(cfg)
    ic = cfg["inspect"]
    boards, principals = [], []
    for i, raw in enumerate(images):
        b, info = prepare_board(raw, cfg, frame)
        boards.append(b)
        principals.append(info["principal_px"])
        log(f"  golden {i + 1}/{len(images)}: fiducial residual {info['fid_residual_mm']}")
    for _ in range(2):
        ref = _mean_u8(boards)
        boards = [flatten_to(ecc_refine(b, ref, ic["ecc_scale"])[0], ref, FLATTEN_SIGMA_MM * frame.s)
                  for b in boards]
    principal = tuple(np.median(np.array(principals), axis=0).tolist())

    acc = Welford()
    for b in boards:
        acc.add(b)
    mean_board = np.clip(np.round(acc.mean), 0, 255).astype(np.uint8)
    arrays = {"mean_board": mean_board, "std_board": acc.std.astype(np.float16)}
    parts = {}
    angles, fine = angle_list(cfg, train=True)
    for c in comps:
        if c.package is None:
            log(f"  {c.ref}: no package model for '{c.footprint}', skipped")
            continue
        g = part_geometry(c, frame, principal, cfg)
        region = center_crop(local_patch(mean_board, g.center, g.ccw, g.patch), g.region)
        for _ in range(2):
            body = center_crop(region, g.body)
            loc, regs, scores = Welford(), [], []
            for b in boards:
                m = match_part(local_patch(b, g.center, g.ccw, g.patch), body, g.region, angles, fine, g.search)
                loc.add(m["registered"])
                regs.append(m["registered"])
                scores.append(m["ncc"])
            region = np.clip(np.round(loc.mean), 0, 255).astype(np.uint8)
        z = zones(c, region.shape, frame.s)
        mads = [offset_mad(r, region, z["frame"]) for r in regs]
        meds = [float(np.median(r[z["interior"]])) if z["interior"].any() else 0.0 for r in regs]
        mu, sd = float(np.mean(scores)), float(np.std(scores))
        thr = float(np.clip(mu - ic["ncc_sigma_k"] * max(sd, 0.01), ic["ncc_floor"], ic["ncc_present"]))
        mad_mu, mad_sd = float(np.mean(mads)), float(np.std(mads))
        mad_thr = mad_mu + max(ic["mad_sigma_k"] * mad_sd, ic["mad_min_delta"])
        arrays[f"{c.ref}__mean"] = region
        arrays[f"{c.ref}__std"] = loc.std.astype(np.float16)
        parts[c.ref] = {"ncc_mu": mu, "ncc_sd": sd, "ncc_thr": thr, "mad_mu": mad_mu, "mad_sd": mad_sd,
                        "mad_thr": mad_thr, "interior_med": float(np.median(meds)),
                        "interior_sd": float(np.std(meds))}
        log(f"  {c.ref:6s} {c.package.name:14s} ncc {mu:.3f}±{sd:.3f} thr {thr:.3f} | "
            f"frame mad {mad_mu:.1f}±{mad_sd:.1f} thr {mad_thr:.1f} | interior {np.median(meds):.0f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "model.npz", **arrays)
    cv2.imwrite(str(out_dir / "mean_board.png"), mean_board)
    meta = {"n_golden": len(images), "principal_px": principal, "parts": parts,
            "px_per_mm": frame.s, "board_size_px": frame.size}
    (out_dir / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_model(golden_dir: Path):
    meta = json.loads((golden_dir / "model.json").read_text(encoding="utf-8"))
    arrays = dict(np.load(golden_dir / "model.npz"))
    return meta, arrays
