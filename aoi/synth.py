"""Synthetic mono board shots for a dry run of the pipeline without the stand."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .cad import load_packages

BOARD_W, BOARD_H = 90.0, 60.0
RENDER_PX_MM = 22.0
RAW_SIZE = (2400, 1600)
WD_MM = 300.0
FIDUCIALS = [(4.0, 4.0), (86.0, 4.0), (86.0, 56.0), (4.0, 56.0)]

PARTS = [
    ("R1", "10k", "R_0805_2012Metric", 15, 45, 0), ("R2", "10k", "R_0805_2012Metric", 15, 40, 0),
    ("R3", "4k7", "R_0805_2012Metric", 15, 35, 90), ("R4", "1k", "R_0603_1608Metric", 22, 45, 0),
    ("R5", "1k", "R_0603_1608Metric", 22, 40, 180), ("C1", "100n", "C_0603_1608Metric", 22, 35, 90),
    ("C2", "1u", "C_0805_2012Metric", 30, 45, 0), ("C3", "10u", "C_1206_3216Metric", 30, 38, 90),
    ("C4", "100n", "C_0402_1005Metric", 30, 32, 0), ("D1", "SS14", "D_SOD-123", 42, 48, 0),
    ("D2", "1N4148W", "D_SOD-123", 42, 42, 180), ("Q1", "BSS138", "SOT-23", 42, 34, 0),
    ("U1", "LM358", "SOIC-8_3.9x4.9mm_P1.27mm", 56, 42, 0),
    ("U2", "STM8S003", "TQFP-32_7x7mm_P0.8mm", 72, 40, 0),
    ("C5", "47u", "CP_Elec_5x5.8", 58, 22, 90), ("R6", "0R", "R_1206_3216Metric", 72, 22, 0),
]
PKG_OF = {"R_0805": "0805", "R_0603": "0603", "C_0603": "0603", "C_0805": "0805", "C_1206": "1206",
          "C_0402": "0402", "R_1206": "1206", "D_SOD": "SOD-123", "SOT-23": "SOT-23",
          "SOIC-8": "SOIC-8", "TQFP-32": "TQFP-32", "CP_Elec": "CP_Elec_5x5.8"}


class Canvas:
    def __init__(self, s: float, rng):
        self.s = s
        w, h = int(BOARD_W * s), int(BOARD_H * s)
        base = rng.normal(62, 4, (h // 8 + 1, w // 8 + 1)).astype(np.float32)
        self.img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
        self.img += rng.normal(0, 2, self.img.shape).astype(np.float32)

    def px(self, x, y):
        return x * self.s, (BOARD_H - y) * self.s

    def rect(self, cx, cy, ccw, dx, dy, w, h, val):
        """Rectangle at local offset (dx, dy) [image y down] of a part centred at board (cx, cy)."""
        px, py = self.px(cx, cy)
        a = math.radians(ccw)
        ox = px + (dx * math.cos(a) + dy * math.sin(a)) * self.s
        oy = py + (-dx * math.sin(a) + dy * math.cos(a)) * self.s
        box = cv2.boxPoints(((ox, oy), (w * self.s, h * self.s), -ccw))
        cv2.fillPoly(self.img, [np.int32(np.round(box * 16))], float(val), cv2.LINE_8, 4)

    def circle(self, cx, cy, ccw, dx, dy, r, val):
        px, py = self.px(cx, cy)
        a = math.radians(ccw)
        ox = px + (dx * math.cos(a) + dy * math.sin(a)) * self.s
        oy = py + (-dx * math.sin(a) + dy * math.cos(a)) * self.s
        cv2.circle(self.img, (int(ox * 16), int(oy * 16)), int(r * self.s * 16), float(val), -1,
                   cv2.LINE_8, 4)


def draw_pads(cv, pkg, x, y, rot):
    L, W = pkg.total
    if pkg.name in ("0402", "0603", "0805", "1206"):
        for sx in (-1, 1):
            cv.rect(x, y, rot, sx * L / 2, 0, L * 0.35, W * 1.15, 175)
    elif pkg.name == "SOD-123":
        for sx in (-1, 1):
            cv.rect(x, y, rot, sx * 1.65, 0, 0.9, 1.2, 175)
    elif pkg.name == "SOT-23":
        for dx, dy in ((-0.95, 1.1), (0.95, 1.1), (0.0, -1.1)):
            cv.rect(x, y, rot, dx, dy, 0.7, 0.9, 175)
    elif pkg.name == "SOIC-8":
        for i in range(4):
            for sy in (-1, 1):
                cv.rect(x, y, rot, -1.905 + 1.27 * i, sy * 2.7, 0.6, 1.5, 175)
    elif pkg.name == "TQFP-32":
        for i in range(8):
            t = -2.8 + 0.8 * i
            for sy in (-1, 1):
                cv.rect(x, y, rot, t, sy * 4.2, 0.45, 1.3, 175)
                cv.rect(x, y, rot, sy * 4.2, t, 1.3, 0.45, 175)
    elif pkg.name == "CP_Elec_5x5.8":
        for sx in (-1, 1):
            cv.rect(x, y, rot, sx * 2.2, 0, 1.6, 1.4, 175)


def draw_part(cv, pkg, x, y, rot):
    n = pkg.name
    if n in ("0402", "0603", "0805", "1206"):
        L, W = pkg.body
        cv.rect(x, y, rot, 0, 0, L, W, 150 if L < 1.2 else 35)
        for sx in (-1, 1):
            cv.rect(x, y, rot, sx * (L / 2 - L * 0.1), 0, L * 0.2, W, 215)
    elif n == "SOD-123":
        for sx in (-1, 1):
            cv.rect(x, y, rot, sx * 1.6, 0, 0.7, 0.6, 215)
        cv.rect(x, y, rot, 0, 0, 2.7, 1.6, 40)
        cv.rect(x, y, rot, -1.0, 0, 0.4, 1.6, 185)
    elif n == "SOT-23":
        for dx, dy in ((-0.95, 0.9), (0.95, 0.9), (0.0, -0.9)):
            cv.rect(x, y, rot, dx, dy, 0.4, 0.6, 215)
        cv.rect(x, y, rot, 0, 0, 2.9, 1.3, 38)
    elif n == "SOIC-8":
        for i in range(4):
            for sy in (-1, 1):
                cv.rect(x, y, rot, -1.905 + 1.27 * i, sy * 2.5, 0.42, 1.1, 215)
        cv.rect(x, y, rot, 0, 0, 4.9, 3.9, 35)
        cv.circle(x, y, rot, -1.8, 1.2, 0.3, 95)
    elif n == "TQFP-32":
        for i in range(8):
            t = -2.8 + 0.8 * i
            for sy in (-1, 1):
                cv.rect(x, y, rot, t, sy * 4.0, 0.35, 1.0, 215)
                cv.rect(x, y, rot, sy * 4.0, t, 1.0, 0.35, 215)
        cv.rect(x, y, rot, 0, 0, 7.0, 7.0, 35)
        cv.circle(x, y, rot, -2.6, 2.6, 0.4, 95)
    elif n == "CP_Elec_5x5.8":
        cv.rect(x, y, rot, 0, 0, 5.3, 5.3, 120)
        cv.circle(x, y, rot, 0, 0, 2.5, 165)
        cv.rect(x, y, rot, -1.9, 0, 0.9, 3.4, 70)
        cv.circle(x, y, rot, 0, 0, 0.6, 140)


def render(rng, defects: dict | None = None):
    lib = load_packages(None)
    defects = defects or {}
    cv = Canvas(RENDER_PX_MM, rng)
    for k in range(14):
        y0 = 8 + k * 3.5
        cv.rect(45 + (k % 3), y0, 0, 0, 0, 70, 0.25, 88)
    for fx, fy in FIDUCIALS:
        cv.circle(fx, fy, 0, 0, 0, 1.5, 40)
        cv.circle(fx, fy, 0, 0, 0, 0.5, 225)
    for ref, _val, fp, x, y, rot in PARTS:
        pkg = lib[next(v for k, v in PKG_OF.items() if fp.startswith(k))]
        draw_pads(cv, pkg, x, y, rot)
        d = defects.get(ref, {})
        if d.get("missing"):
            continue
        jx, jy, jr = rng.normal(0, 0.03), rng.normal(0, 0.03), rng.normal(0, 0.4)
        k = pkg.height / (WD_MM - pkg.height)
        px, py = (x - BOARD_W / 2) * k, (y - BOARD_H / 2) * k
        draw_part(cv, pkg, x + px + jx + d.get("dx", 0), y + py + jy + d.get("dy", 0),
                  rot + jr + d.get("drot", 0))
    for kind, x, y, w, h in defects.get("_extra", []):
        cv.rect(x, y, 0, 0, 0, w, h, 215 if kind == "bright" else 20)

    img = cv.img
    gain = rng.uniform(0.92, 1.08)
    gy, gx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
    img = img * gain * (1 + 0.06 * (gx / img.shape[1] - 0.5) + 0.04 * (gy / img.shape[0] - 0.5))

    raw_s = 22.0 / RENDER_PX_MM
    ang = math.radians(rng.uniform(-1.2, 1.2))
    cx, cy = RAW_SIZE[0] / 2 + rng.uniform(-30, 30), RAW_SIZE[1] / 2 + rng.uniform(-30, 30)
    A = np.array([[raw_s * math.cos(ang), -raw_s * math.sin(ang), 0],
                  [raw_s * math.sin(ang), raw_s * math.cos(ang), 0]], np.float64)
    c0 = A[:, :2] @ np.array([img.shape[1] / 2, img.shape[0] / 2])
    A[:, 2] = [cx - c0[0], cy - c0[1]]
    raw = cv2.warpAffine(img, A, RAW_SIZE, flags=cv2.INTER_LINEAR, borderValue=15)
    raw = cv2.GaussianBlur(raw, (0, 0), 0.8)
    raw += rng.normal(0, 2.5, raw.shape).astype(np.float32)
    return np.clip(raw, 0, 255).astype(np.uint8), A


BAD = {
    "R3": {"missing": True},
    "C2": {"dx": 0.55},
    "D1": {"drot": 180},
    "U1": {"drot": 180},
    "Q1": {"drot": 15},
    "_extra": [("bright", 70.4, 44.0, 0.6, 0.4),
               ("bright", 23.6, 40.8, 0.35, 0.35),
               ("dark", 10, 20, 3.0, 0.2)],
}


def make_demo(out: Path, n_golden: int = 20, seed: int = 7):
    rng = np.random.default_rng(seed)
    (out / "golden").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    A0 = None
    for i in range(n_golden):
        raw, A = render(rng)
        A0 = A0 if A0 is not None else A
        cv2.imwrite(str(out / "golden" / f"golden_{i:02d}.png"), raw)
    for i in range(5):
        cv2.imwrite(str(out / "test" / f"good_{i}.png"), render(rng)[0])
    cv2.imwrite(str(out / "test" / "bad.png"), render(rng, BAD)[0])

    with (out / "centroid.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Ref", "Val", "Package", "PosX", "PosY", "Rot", "Side"])
        for ref, val, fp, x, y, rot in PARTS:
            w.writerow([ref, val, fp, x, y, rot, "top"])
    with (out / "bom.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Designator", "MPN"])
        w.writerow(["U1", "LM358DR"])
        w.writerow(["U2", "STM8S003F3P6"])
        w.writerow(["Q1", "BSS138"])
        w.writerow(["D1, D2", "SS14"])

    fids = []
    for fx, fy in FIDUCIALS:
        p = A0 @ np.array([fx * RENDER_PX_MM, (BOARD_H - fy) * RENDER_PX_MM, 1.0])
        fids.append({"x_mm": fx, "y_mm": fy, "approx_px": [round(p[0]), round(p[1])]})
    cfg = {
        "camera": {"backend": "file"},
        "optics": {"focal_mm": 12.0, "wd_mm": WD_MM},
        "board": {"width_mm": BOARD_W, "height_mm": BOARD_H, "cad_y_up": True, "cad_rot_ccw": True,
                  "px_per_mm": 22.0, "raw_px_per_mm": 22.0, "fiducials": fids,
                  "fiducial_diameter_mm": 1.0, "fiducial_search_mm": 3.0},
        "files": {"centroid": "centroid.csv", "bom": "bom.csv"},
        "inspect": {"edge_tol_mm": 0.05, "joint_frac": 0.1},
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
