from __future__ import annotations

import math

import cv2
import numpy as np


class BoardFrame:
    """Canonical board image: fixed px/mm, origin top-left, y down."""

    def __init__(self, cfg: dict):
        b = cfg["board"]
        self.s = float(b["px_per_mm"])
        self.w_mm = float(b["width_mm"])
        self.h_mm = float(b["height_mm"])
        self.y_up = bool(b["cad_y_up"])
        self.rot_ccw = bool(b["cad_rot_ccw"])
        self.size = (int(round(self.w_mm * self.s)), int(round(self.h_mm * self.s)))

    def to_px(self, x_mm: float, y_mm: float) -> tuple:
        y = self.h_mm - y_mm if self.y_up else y_mm
        return x_mm * self.s, y * self.s

    def visual_ccw(self, rot_deg: float) -> float:
        """Rotation seen in the canonical image, degrees counter-clockwise."""
        ccw = rot_deg if self.rot_ccw else -rot_deg
        return ccw if self.y_up else -ccw


def _fiducial_template(diam_px: float) -> np.ndarray:
    r = diam_px / 2
    half = int(math.ceil(r * 2))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    d = np.hypot(xx, yy)
    return np.clip(r + 0.5 - d, 0, 1).astype(np.float32)


def find_fiducial(img: np.ndarray, approx: tuple | None, diam_px: float, search_px: float):
    """Bright round fiducial on a darker clearance: NCC with a disk-in-ring template, then subpixel
    centroid of the locally thresholded disk. Returns (x, y) or None."""
    h, w = img.shape[:2]
    if approx is None:
        return _find_fiducial_otsu(img, None, diam_px, search_px)
    x0, y0 = int(max(0, approx[0] - search_px)), int(max(0, approx[1] - search_px))
    x1, y1 = int(min(w, approx[0] + search_px)), int(min(h, approx[1] + search_px))
    roi = img[y0:y1, x0:x1].astype(np.float32)
    tmpl = _fiducial_template(diam_px)
    if roi.shape[0] <= tmpl.shape[0] or roi.shape[1] <= tmpl.shape[1]:
        return None
    res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < 0.6:
        return None
    th, tw = tmpl.shape
    cx, cy = loc[0] + tw / 2, loc[1] + th / 2
    r = diam_px / 2
    k = int(math.ceil(r * 1.6))
    xa, ya = int(max(0, cx - k)), int(max(0, cy - k))
    win = roi[ya:int(cy + k) + 1, xa:int(cx + k) + 1]
    yy, xx = np.mgrid[ya:ya + win.shape[0], xa:xa + win.shape[1]].astype(np.float32)
    inner = np.hypot(xx - cx, yy - cy) <= r * 0.6
    ring = np.hypot(xx - cx, yy - cy) >= r * 1.35
    if not inner.any() or not ring.any():
        return None
    t = (np.median(win[inner]) + np.median(win[ring])) / 2
    m = (win > t).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
    lbl = lab[int(cy) - ya, int(cx) - xa] if 0 <= int(cy) - ya < lab.shape[0] and 0 <= int(cx) - xa < lab.shape[1] else 0
    if lbl == 0:
        return None
    area = stats[lbl, cv2.CC_STAT_AREA]
    if not (0.5 * math.pi * r * r < area < 1.8 * math.pi * r * r):
        return None
    wts = np.where(lab == lbl, win - t, 0).astype(np.float64)
    sx = float((wts * xx).sum() / wts.sum())
    sy = float((wts * yy).sum() / wts.sum())
    return sx + x0, sy + y0


def _find_fiducial_otsu(img: np.ndarray, approx: tuple | None, diam_px: float, search_px: float):
    h, w = img.shape[:2]
    if approx is None:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0, y0 = int(max(0, approx[0] - search_px)), int(max(0, approx[1] - search_px))
        x1, y1 = int(min(w, approx[0] + search_px)), int(min(h, approx[1] + search_px))
    roi = cv2.GaussianBlur(img[y0:y1, x0:x1], (0, 0), max(1.0, diam_px / 12))
    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    area_nom = math.pi * (diam_px / 2) ** 2
    best, best_cost = None, math.inf
    for c in cnts:
        a = cv2.contourArea(c)
        if not (0.5 * area_nom < a < 1.8 * area_nom):
            continue
        per = cv2.arcLength(c, True)
        circularity = 4 * math.pi * a / per ** 2 if per else 0
        if circularity < 0.8:
            continue
        m = cv2.moments(c)
        cx, cy = m["m10"] / m["m00"] + x0, m["m01"] / m["m00"] + y0
        cost = -circularity if approx is None else math.hypot(cx - approx[0], cy - approx[1])
        if cost < best_cost:
            best, best_cost = (cx, cy), cost
    return best


def find_all_fiducials(img: np.ndarray, diam_px: float) -> list:
    """All bright round blobs of roughly the fiducial size, for first-time setup."""
    blur = cv2.GaussianBlur(img, (0, 0), max(1.0, diam_px / 12))
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                               int(diam_px * 4) | 1, -20)
    cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    area_nom = math.pi * (diam_px / 2) ** 2
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        per = cv2.arcLength(c, True)
        if 0.5 * area_nom < a < 1.8 * area_nom and per and 4 * math.pi * a / per ** 2 > 0.8:
            m = cv2.moments(c)
            out.append((m["m10"] / m["m00"], m["m01"] / m["m00"], 4 * math.pi * a / per ** 2))
    return out


def transform_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2 points: similarity, 3: affine, >=4: homography. Returns 3x3 src->dst."""
    src, dst = np.float32(src), np.float32(dst)
    n = len(src)
    if n < 2:
        raise ValueError("need at least 2 fiducials")
    if n == 2:
        (a, b), (c, d) = src
        (e, f), (g, h) = dst
        vs, vd = complex(c - a, d - b), complex(g - e, h - f)
        z = vd / vs
        t = complex(e, f) - z * complex(a, b)
        return np.array([[z.real, -z.imag, t.real], [z.imag, z.real, t.imag], [0, 0, 1]], np.float64)
    if n == 3:
        return np.vstack([cv2.getAffineTransform(src, dst), [0, 0, 1]]).astype(np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    return H


def rectify(raw: np.ndarray, cfg: dict, frame: BoardFrame):
    """Find fiducials in the raw frame and warp it into the canonical board frame."""
    b = cfg["board"]
    diam_px = b["fiducial_diameter_mm"] * b["raw_px_per_mm"]
    search_px = b["fiducial_search_mm"] * b["raw_px_per_mm"]
    src, dst, found = [], [], []
    for f in b["fiducials"]:
        c = find_fiducial(raw, f.get("approx_px"), diam_px, search_px)
        found.append(c)
        if c is not None:
            src.append(c)
            dst.append(frame.to_px(f["x_mm"], f["y_mm"]))
    if len(src) < 2:
        raise RuntimeError(f"fiducials found {len(src)}/{len(b['fiducials'])}: {found}")
    H = transform_from_points(np.array(src), np.array(dst))
    board = cv2.warpPerspective(raw, H, frame.size, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
    h, w = raw.shape[:2]
    pp = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2]]]), H)[0, 0]
    residual = None
    if len(src) >= 4:
        # Exact fits leave no residual; an affine least-squares check flags a bad fiducial or board bow.
        A = np.hstack([np.float64(src), np.ones((len(src), 1))])
        X, *_ = np.linalg.lstsq(A, np.float64(dst), rcond=None)
        residual = float(np.max(np.linalg.norm(A @ X - np.float64(dst), axis=1)) / frame.s)
    return board, {"H": H.tolist(), "fiducials_px": found, "principal_px": pp.tolist(),
                   "fid_residual_mm": residual}


def ecc_refine(board: np.ndarray, ref: np.ndarray, scale: float) -> tuple:
    """Affine ECC of board onto ref at reduced scale; returns (aligned board, 2x3 warp, cc)."""
    small = lambda im: cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    a, b = small(ref).astype(np.float32), small(board).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5),
                                        None, 5)
    except cv2.error:
        return board, np.eye(2, 3, dtype=np.float32), 0.0
    warp[:, 2] /= scale
    out = cv2.warpAffine(board, warp, (board.shape[1], board.shape[0]),
                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return out, warp, float(cc)


def parallax_px(p: tuple, principal: tuple, height_mm: float, wd_mm: float) -> tuple:
    """Top of a part of given height projects radially away from the optical axis."""
    k = height_mm / max(wd_mm - height_mm, 1e-6)
    return (p[0] - principal[0]) * k, (p[1] - principal[1]) * k


def local_patch(img: np.ndarray, center: tuple, ccw_deg: float, size_px: tuple,
                interp=cv2.INTER_LINEAR) -> np.ndarray:
    """Crop size_px (w, h) around center, de-rotated so the part is at CAD rotation 0."""
    w, h = size_px
    M = cv2.getRotationMatrix2D(center, -ccw_deg, 1.0)
    M[0, 2] += w / 2 - center[0]
    M[1, 2] += h / 2 - center[1]
    return cv2.warpAffine(img, M, (int(w), int(h)), flags=interp, borderMode=cv2.BORDER_REPLICATE)
