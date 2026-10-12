"""Physically-motivated mono rendering of an assembled board as seen by IMX183 + 12 mm at f/4.

Units: board mm, y up; rotation = visual CCW (matches cad_y_up/cad_rot_ccw = true).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..cad import load_packages
from .geometry import CHIPS, geometry

S = 43.0            # px/mm on the board, ~23 µm/px at WD 100 mm from the lens front
WD = 127.0          # board -> principal plane, mm
FOCUS_Z = 0.5       # focused height above the board, mm
APERTURE = 3.0      # 12 mm / f4
MARGIN = 6.0        # fixture visible around the board, mm
FONT = cv2.FONT_HERSHEY_SIMPLEX
LIB = load_packages(None)

BATCH = {
    "A": {"r_body": 30, "ic_body": 34, "mark": 78, "r_mark": 0, "date": "2231", "ejector": True,
          "c_shift": 0, "tant": 150, "led": 186, "ferrite": 95},
    "B": {"r_body": 40, "ic_body": 41, "mark": 94, "r_mark": 1, "date": "2318", "ejector": False,
          "c_shift": 18, "tant": 138, "led": 196, "ferrite": 108},
}


def _rot(dx, dy, deg):
    a = math.radians(deg)
    return dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)


class Canvas:
    def __init__(self, W, H, rng):
        self.W, self.H, self.rng = W, H, rng
        self.img = np.zeros((int(round(H * S)), int(round(W * S))), np.float32)

    def P(self, x, y):
        return x * S, (self.H - y) * S

    def L(self, cx, cy, ccw, dx, dy):
        px, py = self.P(cx, cy)
        rx, ry = _rot(dx, dy, ccw)
        return px + rx * S, py + ry * S

    def _blend(self, pts, val, alpha=1.0, noise=0.0, grad=None, thick=None):
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        pad = 2 + (int(thick) if thick else 0)
        x0, y0 = np.floor(pts.min(0)).astype(int) - pad
        x1, y1 = np.ceil(pts.max(0)).astype(int) + pad
        X0, Y0 = max(0, x0), max(0, y0)
        X1, Y1 = min(self.img.shape[1], x1), min(self.img.shape[0], y1)
        if X1 <= X0 or Y1 <= Y0:
            return None
        m = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
        q = np.int32(np.round((pts - [X0, Y0]) * 16))
        if thick:
            cv2.polylines(m, [q], False, 255, max(1, int(round(thick))), cv2.LINE_AA, 4)
        else:
            cv2.fillPoly(m, [q], 255, cv2.LINE_AA, 4)
        a = m.astype(np.float32) * (alpha / 255.0)
        roi = self.img[Y0:Y1, X0:X1]
        v = np.full(roi.shape, val, np.float32) if grad is None else grad(X0, Y0, roi.shape)
        if noise:
            v = v + self.rng.normal(0, noise, roi.shape).astype(np.float32)
        roi[:] = roi * (1 - a) + v * a
        return X0, Y0, m

    def rect(self, cx, cy, ccw, dx, dy, w, h, val, **kw):
        ox, oy = self.L(cx, cy, ccw, dx, dy)
        return self._blend(cv2.boxPoints(((ox, oy), (w * S, h * S), -ccw)), val, **kw)

    def rect_grad(self, cx, cy, ccw, dx, dy, w, h, v0, v1, axis, **kw):
        """Linear ramp v0 -> v1 along the local axis (+x or +y, sign included in axis)."""
        ox, oy = self.L(cx, cy, ccw, dx, dy)
        ux, uy = _rot(*axis, ccw)
        span = (w if axis[0] else h) * S

        def grad(X0, Y0, shape):
            yy, xx = np.mgrid[Y0:Y0 + shape[0], X0:X0 + shape[1]].astype(np.float32)
            t = np.clip(((xx - ox) * ux + (yy - oy) * uy) / span + 0.5, 0, 1)
            return v0 + (v1 - v0) * t
        return self._blend(cv2.boxPoints(((ox, oy), (w * S, h * S), -ccw)), 0, grad=grad, **kw)

    def circle(self, cx, cy, ccw, dx, dy, r, val, n=48, **kw):
        ox, oy = self.L(cx, cy, ccw, dx, dy)
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return self._blend(np.c_[ox + r * S * np.cos(t), oy + r * S * np.sin(t)], val, **kw)

    def segment(self, cx, cy, ccw, dx, dy, r, x_cut, val, **kw):
        """Part of a circle (local centre dx,dy, radius r) with local x <= x_cut."""
        a = math.acos(max(-1, min(1, x_cut / r)))
        t = np.linspace(a, 2 * np.pi - a, 40)
        loc = [(dx + r * math.cos(v), dy + r * math.sin(v)) for v in t]
        return self._blend([self.L(cx, cy, ccw, lx, ly) for lx, ly in loc], val, **kw)

    def poly_local(self, cx, cy, ccw, pts, val, **kw):
        return self._blend([self.L(cx, cy, ccw, lx, ly) for lx, ly in pts], val, **kw)

    def line_board(self, pts_mm, width_mm, val, **kw):
        return self._blend([self.P(x, y) for x, y in pts_mm], val, thick=width_mm * S, **kw)

    def text(self, cx, cy, ccw, dx, dy, txt, h_mm, val, alpha=1.0):
        th = h_mm * S
        (tw0, th0), _ = cv2.getTextSize(txt, FONT, 1.0, 1)
        sc = th / th0
        thick = max(1, int(round(th / 7)))
        tw = int(tw0 * sc) + 4 * thick
        side = int(math.hypot(tw, th * 1.8)) + 6
        patch = np.zeros((side, side), np.uint8)
        org = (int((side - tw0 * sc) / 2), int((side + th) / 2))
        cv2.putText(patch, txt, org, FONT, sc, 255, thick, cv2.LINE_AA)
        M = cv2.getRotationMatrix2D((side / 2, side / 2), ccw, 1.0)
        patch = cv2.warpAffine(patch, M, (side, side), flags=cv2.INTER_LINEAR)
        ox, oy = self.L(cx, cy, ccw, dx, dy)
        x0, y0 = int(round(ox - side / 2)), int(round(oy - side / 2))
        X0, Y0 = max(0, x0), max(0, y0)
        X1, Y1 = min(self.img.shape[1], x0 + side), min(self.img.shape[0], y0 + side)
        if X1 <= X0 or Y1 <= Y0:
            return
        a = patch[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0].astype(np.float32) * (alpha / 255.0)
        roi = self.img[Y0:Y1, X0:X1]
        roi[:] = roi * (1 - a) + val * a

    def defocus(self, cx, cy, half_mm, sigma_px):
        if sigma_px < 0.3:
            return
        px, py = self.P(cx, cy)
        r = int(half_mm * S + 4 * sigma_px)
        x0, y0 = max(0, int(px) - r), max(0, int(py) - r)
        x1, y1 = min(self.img.shape[1], int(px) + r), min(self.img.shape[0], int(py) + r)
        roi = self.img[y0:y1, x0:x1]
        roi[:] = cv2.GaussianBlur(roi, (0, 0), sigma_px)


def _lowfreq(shape, rng, cell, sigma):
    small = rng.normal(0, sigma, (shape[0] // cell + 2, shape[1] // cell + 2)).astype(np.float32)
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def pad_value(design):
    return 186.0 if design["finish"] == "HASL" else 206.0


def render_board(design: dict) -> np.ndarray:
    """Bare board after reflow minus parts: mask, copper under mask, pads, vias, silk, fiducials."""
    rng = np.random.default_rng(design["seed"])
    cv = Canvas(design["W"], design["H"], rng)
    m = design["mask_level"]
    cu = m + 26
    cv.img[:] = m
    cv.img += _lowfreq(cv.img.shape, rng, 90, 2.5) + rng.normal(0, 1.2, cv.img.shape).astype(np.float32)
    for p in design["pours"]:
        cv._blend([cv.P(p["x"], p["y"]), cv.P(p["x"] + p["w"], p["y"]), cv.P(p["x"] + p["w"], p["y"] + p["h"]),
                   cv.P(p["x"], p["y"] + p["h"])], cu)
    for t in design["traces"]:
        cv.line_board(t["pts"], t["w"], cu)
    pv = pad_value(design)
    hasl = design["finish"] == "HASL"
    for part in design["parts"]:
        g = geometry(LIB[part["pkg"]])
        for (dx, dy, w, h) in g["pads"]:
            cv.rect(part["x"], part["y"], part["rot"], dx, dy, w + 0.6, h + 0.6, m)
    for v in design["vias"]:
        cv.circle(v["x"], v["y"], 0, 0, 0, 0.6, m)
        if v["tented"]:
            cv.circle(v["x"], v["y"], 0, 0, 0, 0.35, cu + 8)
        else:
            cv.circle(v["x"], v["y"], 0, 0, 0, 0.35, pv)
            cv.circle(v["x"], v["y"], 0, 0, 0, 0.15, 14)
    for part in design["parts"]:
        g = geometry(LIB[part["pkg"]])
        for (dx, dy, w, h) in g["pads"]:
            cv.rect(part["x"], part["y"], part["rot"], dx, dy, w, h, pv, noise=7 if hasl else 2.5)
    for f in design["fiducials"]:
        cv.circle(f["x_mm"], f["y_mm"], 0, 0, 0, 1.5, 96, noise=3)
        cv.circle(f["x_mm"], f["y_mm"], 0, 0, 0, 0.5, pv + 10, noise=2)
    silk = 196
    for part in design["parts"]:
        pkg = LIB[part["pkg"]]
        T, TW = pkg.total
        ox, oy = T / 2 + 0.35, TW / 2 + 0.35
        corners = [(-ox, -oy), (ox, -oy), (ox, oy), (-ox, oy), (-ox, -oy)]
        pts = []
        for lx, ly in corners:
            rx, ry = _rot(lx, ly, part["rot"])
            pts.append((part["x"] + rx, part["y"] - ry))
        cv.line_board(pts, 0.15, silk)
        if pkg.orient != "none":
            mx = pkg.mark[0] if pkg.mark else -T / 2
            sx = -1 if mx <= 0 else 1
            sy = 1 if (pkg.mark and pkg.mark[1] > 0) else 0
            cv.circle(part["x"], part["y"], part["rot"], sx * (ox + 0.45), sy * oy, 0.2, silk)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cv.text(min(xs) + 0.1, max(ys) + 0.65, 0, 0.6, 0, part["ref"], 0.7, silk)
    return cv.img


def _fillet(cv, x, y, rot, lead, pad, pv, rng, wet=True):
    ldx, ldy, lw, lh = lead
    pdx, pdy, pw, ph = pad
    if abs(pdx) >= abs(pdy):
        s = 1 if pdx > 0 else -1
        a, b = ldx + s * lw / 2, pdx + s * pw / 2
        if (b - a) * s <= 0.02:
            return
        c, ln, wd = (a + b) / 2, abs(b - a), min(pw, lh * 1.15)
        if wet:
            cv.rect_grad(x, y, rot, c, pdy, ln, wd, 135, pv + 4, (s, 0), noise=4)
        else:
            cv.rect(x, y, rot, a + s * 0.03, pdy, 0.06, wd, 40)
        glint_xy = [(c + rng.uniform(-ln / 3, ln / 3), pdy + rng.uniform(-wd / 3, wd / 3))]
    else:
        s = 1 if pdy > 0 else -1
        a, b = ldy + s * lh / 2, pdy + s * ph / 2
        if (b - a) * s <= 0.02:
            return
        c, ln, wd = (a + b) / 2, abs(b - a), min(pw, lw * 1.15)
        if wet:
            cv.rect_grad(x, y, rot, pdx, c, wd, ln, 135, pv + 4, (0, s), noise=4)
        else:
            cv.rect(x, y, rot, pdx, a + s * 0.03, wd, 0.06, 40)
        glint_xy = [(pdx + rng.uniform(-wd / 3, wd / 3), c + rng.uniform(-ln / 3, ln / 3))]
    if wet:
        for gx, gy in glint_xy[: int(rng.integers(0, 2))]:
            cv.circle(x, y, rot, gx, gy, rng.uniform(0.025, 0.05), 240)


def _ic_marking(cv, x, y, rot, part, look, body_l, body_w, ov):
    name = ov.get("marking") or part["marks"][0]
    lines = [name, look["date"] + ("A" if look["r_mark"] else "")]
    h1 = min(0.75, body_w * 0.17, body_l / max(4, len(name)) * 1.3)
    cv.text(x, y, rot, 0, -h1 * 0.7, lines[0], h1, look["mark"], alpha=0.9)
    cv.text(x, y, rot, 0, h1 * 0.8, lines[1], h1 * 0.75, look["mark"], alpha=0.9)


def draw_part(cv: Canvas, part: dict, look: dict, ov: dict, rng, design: dict):
    pkg = LIB[part["pkg"]]
    n = pkg.name
    g = geometry(pkg)
    pv = pad_value(design)
    x = part["x"] + rng.normal(0, 0.035) + ov.get("dx", 0)
    y = part["y"] + rng.normal(0, 0.035) + ov.get("dy", 0)
    rot = part["rot"] + rng.normal(0, 0.5) + ov.get("drot", 0)
    leads_top = n in CHIPS or n in ("Tantalum_Case-A", "Crystal_SMD_3225", "L_4x4")
    L, W = pkg.body
    kind = ov.get("as_kind", part["kind"])

    if ov.get("missing"):
        for (dx, dy, w, h) in g["pads"]:
            cv.rect(part["x"], part["y"], part["rot"], dx, dy, w * 0.8, h * 0.8, pv + 12, noise=6)
            cv.circle(part["x"], part["y"], part["rot"], dx, dy, min(w, h) * 0.15, 238)
        return
    if ov.get("tombstone"):
        (d0, d1) = g["pads"][0], g["pads"][1]
        cv.rect(x, y, rot, d0[0], d0[1], d0[2] * 0.8, d0[3] * 0.8, pv + 12, noise=6)
        k = 1.0 / (WD - L)
        sx, sy = (x - design["W"] / 2) * k * L, (y - design["H"] / 2) * k * L
        cv.rect(x, y, rot, d1[0] - L * 0.05, d1[1], pkg.height + 0.05, W, 70)
        cv.rect(x + sx, y + sy, rot, d1[0] - L * 0.1, d1[1], pkg.height, W, 205, noise=6)
        cv.defocus(x + sx, y + sy, W, 0.41 * abs(L - FOCUS_Z))
        return

    wet = not ov.get("no_solder")
    if not wet:
        for (dx, dy, w, h) in g["pads"]:
            cv.rect(x, y, rot, dx, dy, w * 0.9, h * 0.9, 218, noise=1.5)
    if g["leads"] and len(g["leads"]) == len(g["pads"]):
        for lead, pad in zip(g["leads"], g["pads"]):
            _fillet(cv, x, y, rot, lead, pad, pv, rng, wet)
    elif not g["leads"]:
        for (dx, dy, w, h) in g["pads"]:
            if wet:
                cv.rect(x, y, rot, dx, dy, w, h, pv - 25, noise=6)
    if not leads_top:
        for (dx, dy, w, h) in g["leads"]:
            if n.startswith(("SOIC", "TSSOP", "TQFP", "LQFP", "SOT")):
                ax = (0, 1 if dy > 0 else -1) if abs(dy) > abs(dx) else (1 if dx > 0 else -1, 0)
                cv.rect_grad(x, y, rot, dx, dy, w, h, 120, 212, ax, noise=5)
            else:
                cv.rect(x, y, rot, dx, dy, w, h, 196, noise=5)

    h = pkg.height
    if ov.get("billboard"):
        W = h
    k = h / (WD - h)
    tx, ty = x + (x - design["W"] / 2) * k, y + (y - design["H"] / 2) * k

    if kind == "R" and n in CHIPS:
        body = look["r_body"]
    elif kind == "C" and n in CHIPS:
        body = float(np.clip(design["c_body"] + look["c_shift"], 80, 170))
    elif n in ("SOIC-8", "SOIC-16", "TSSOP-20", "TQFP-32", "LQFP-48", "QFN-32", "SOT-23", "SOT-223",
               "SOD-123", "SMA"):
        body = look["ic_body"]
    elif n == "Tantalum_Case-A":
        body = look["tant"]
    elif n == "LED_0805":
        body = look["led"]
    elif n == "L_4x4":
        body = look["ferrite"]
    elif n == "Crystal_SMD_3225":
        body = 120
    else:
        body = 200
    if h >= 0.8 and not n.startswith("CP_"):
        cv.rect(x, y, rot, 0, 0, L, W, body * 0.55 + 10)

    if n.startswith("CP_"):
        T = L
        cv.poly_local(x, y, rot, [(-T / 2, -T / 2), (T / 2 - 0.9, -T / 2), (T / 2, -T / 2 + 0.9),
                                  (T / 2, T / 2 - 0.9), (T / 2 - 0.9, T / 2), (-T / 2, T / 2)], 200, noise=3)
        r = T / 2 * 0.95
        cv.circle(x, y, rot, 0, 0, r, 120)
        cv.circle(tx, ty, rot, 0, 0, r, 172, noise=3)
        cv.circle(tx, ty, rot, 0, 0, r * 0.82, 182, noise=2)
        mx = pkg.mark[0]
        cv.segment(tx, ty, rot, 0, 0, r, mx + pkg.mark[2] / 2, 62, noise=3)
        for ang in (0, 90):
            cv.rect(tx, ty, rot, 0, 0, r * 1.1 if ang == 0 else 0.1, 0.1 if ang == 0 else r * 1.1, 110)
        cv.defocus(tx, ty, r, 0.41 * (h - FOCUS_Z))
        return

    cv.rect(tx, ty, rot, 0, 0, L, W, body, noise=2.5)
    if kind == "R" and n in CHIPS:
        if L >= 1.6:
            cv.text(tx, ty, rot, 0, 0, part["marks"][look["r_mark"]] if "marks" in part else "103",
                    W * 0.42, 150, alpha=0.9)
        tw = 0.2 * L
        for s in (-1, 1):
            cv.rect(tx, ty, rot, s * (L / 2 - tw / 2), 0, tw, W, 202, noise=7)
    elif n in CHIPS:
        tw = 0.22 * L
        if n == "LED_0805":
            cv.rect(tx, ty, rot, 0, 0, L * 0.5, W * 0.7, body + 18, noise=2)
            cv.rect(tx, ty, rot, pkg.mark[0], 0, 0.15, W * 0.75, 95)
        for s in (-1, 1):
            cv.rect(tx, ty, rot, s * (L / 2 - tw / 2), 0, tw, W, 196, noise=6)
    elif n == "Tantalum_Case-A":
        cv.rect(tx, ty, rot, pkg.mark[0], 0, 0.45, W, 82)
        for (dx, dy, w, hh) in g["leads"]:
            cv.rect(x, y, rot, dx, dy, w, hh, 190, noise=5)
    elif n == "Crystal_SMD_3225":
        cv.rect_grad(tx, ty, rot, 0, 0, L * 0.9, W * 0.88, 182, 204, (1, 0), noise=2)
    elif n == "L_4x4":
        cv.circle(tx, ty, rot, 0, 0, 1.6, body + 12, noise=4)
        cv.text(tx, ty, rot, 0, 0, part["marks"][0], 0.8, 165, alpha=0.9)
        for (dx, dy, w, hh) in g["leads"]:
            cv.rect(x, y, rot, dx, dy, w, hh, 185, noise=5)
    elif n in ("SOD-123", "SMA"):
        if not ov.get("no_band"):
            cv.rect(tx, ty, rot, pkg.mark[0], 0, pkg.mark[2], W, 150 if n == "SOD-123" else 160)
        if n == "SMA":
            cv.text(tx, ty, rot, 0.3, 0, "SS34", 0.55, look["mark"], alpha=0.8)
    elif n in ("SOT-23", "SOT-223"):
        cv.text(tx, ty, rot, 0, 0 if n == "SOT-23" else -0.4, ov.get("marking") or part["marks"][0],
                0.5 if n == "SOT-23" else 0.8, look["mark"], alpha=0.9)
    else:
        mx, my = pkg.mark[0], pkg.mark[1]
        r = 0.28 if n in ("SOIC-8", "SOIC-16") else 0.22 if n == "TSSOP-20" else 0.35
        if n == "QFN-32":
            cv.circle(tx, ty, rot, mx, my, 0.22, 92)
        else:
            cv.circle(tx, ty, rot, mx, my, r, body + 22)
            cv.circle(tx, ty, rot, mx, my, r * 0.75, body - 8)
        if n in ("TQFP-32", "LQFP-48") and look["ejector"]:
            for ex, ey in ((2.2, -2.2), (2.2, 2.2)):
                cv.circle(tx, ty, rot, ex, ey, 0.9, body + 5)
        if n in ("SOIC-8", "SOIC-16"):
            cv.rect(tx, ty, rot, 0, W / 2 - 0.25, L * 0.96, 0.12, body + 10)
        _ic_marking(cv, tx, ty, rot, part, look, L, W, ov)
    if ov.get("damaged"):
        if n in CHIPS:
            s = 1 if rng.random() < 0.5 else -1
            cv.poly_local(tx, ty, rot, [(s * L / 2, -W / 2), (s * (L / 2 - L * 0.35), -W / 2),
                                        (s * L / 2, W * 0.15)], 176, noise=6)
        else:
            cv.line_board([(tx - L * 0.3, ty + W * 0.35), (tx, ty - W * 0.05), (tx + L * 0.25, ty - W * 0.3)],
                          0.08, 140)
    if h - FOCUS_Z > 1.0:
        cv.defocus(tx, ty, max(L, W) / 2, 0.41 * (h - FOCUS_Z))


def draw_extras(cv: Canvas, part: dict, ov: dict, design: dict, rng):
    pkg = LIB[part["pkg"]]
    g = geometry(pkg)
    x, y, rot = part["x"], part["y"], part["rot"]
    if "bridge" in ov:
        i, j = ov["bridge"]
        (ax, ay, aw, ah), (bx, by, bw, bh) = g["pads"][i], g["pads"][j]
        cx, cy = (ax + bx) / 2, (ay + by) / 2
        if abs(ax - bx) > abs(ay - by):
            cv.rect(x, y, rot, cx, cy, abs(ax - bx) + aw * 0.6, ah * 0.45, 214, noise=6)
        else:
            cv.rect(x, y, rot, cx, cy, aw * 0.45, abs(ay - by) + ah * 0.6, 214, noise=6)
    if "excess" in ov:
        dx, dy, w, h = g["pads"][ov["excess"]]
        cv.circle(x, y, rot, dx, dy, max(w, h) * 0.62 + 0.1, 120)
        cv.circle(x, y, rot, dx, dy, max(w, h) * 0.55 + 0.1, 226, noise=8)
        cv.circle(x, y, rot, dx - 0.1, dy - 0.1, 0.08, 248)
    if "ball" in ov:
        bx, by = ov["ball"]
        r = ov.get("ball_r", 0.15)
        cv.circle(bx, by, 0, 0, 0, r + 0.03, 110)
        cv.circle(bx, by, 0, 0, 0, r, 228)
        cv.circle(bx, by, 0, 0, 0, r * 0.35, 250)


def draw_foreign(cv: Canvas, f: dict, rng):
    if f["kind"] == "hair":
        cv.line_board(f["pts"], f["w"], f["val"])
    else:
        cv._blend([cv.P(px, py) for px, py in f["pts"]], f["val"], noise=6)


def render_instance(design: dict, board: np.ndarray, batch: str, ovs: dict, foreign: list,
                    rng: np.random.Generator, pose=True, info: dict | None = None) -> np.ndarray:
    look = BATCH[batch]
    cv = Canvas(design["W"], design["H"], rng)
    cv.img = board + _lowfreq(board.shape, rng, 120, 1.5)
    parts = sorted(design["parts"], key=lambda p: LIB[p["pkg"]].height)
    for part in parts:
        draw_part(cv, part, look, ovs.get(part["ref"], {}), rng, design)
    for part in design["parts"]:
        if part["ref"] in ovs:
            draw_extras(cv, part, ovs[part["ref"]], design, rng)
    for f in foreign:
        draw_foreign(cv, f, rng)
    return camera(cv.img, design, rng, pose, info)


def frame_px(design, x_mm, y_mm):
    return (x_mm + MARGIN) * S, (design["H"] - y_mm + MARGIN) * S


def camera(board: np.ndarray, design: dict, rng, pose=True, info: dict | None = None) -> np.ndarray:
    """info, if given, receives "H": board canvas px -> frame px homography."""
    fw, fh = int((design["W"] + 2 * MARGIN) * S), int((design["H"] + 2 * MARGIN) * S)
    th = math.radians(rng.uniform(-0.8, 0.8)) if pose else 0.0
    tx = rng.uniform(-1.2, 1.2) * S if pose else 0.0
    ty = rng.uniform(-1.2, 1.2) * S if pose else 0.0
    bc = np.array([board.shape[1] / 2, board.shape[0] / 2])
    fc = np.array([fw / 2 + tx, fh / 2 + ty])
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    A = np.eye(3)
    A[:2, :2] = R
    A[:2, 2] = fc - R @ bc
    P = np.eye(3)
    if pose:
        P[2, 0], P[2, 1] = rng.normal(0, 6e-7), rng.normal(0, 6e-7)
    C = np.array([[1, 0, -fw / 2], [0, 1, -fh / 2], [0, 0, 1]])
    Hm = np.linalg.inv(C) @ P @ C @ A
    if info is not None:
        info["H"] = Hm
    img = cv2.warpPerspective(board, Hm, (fw, fh), flags=cv2.INTER_LINEAR, borderValue=17.0)
    yy, xx = np.mgrid[0:fh, 0:fw].astype(np.float32)
    rr = np.hypot(xx - fw / 2, yy - fh / 2) / math.hypot(fw / 2, fh / 2)
    gain = rng.uniform(0.95, 1.05) * (1 - 0.12 * rr ** 2) * (1 + rng.uniform(-0.03, 0.03) * (xx / fw - 0.5))
    sheen = 9.0 * np.exp(-((rr - 0.62) / 0.12) ** 2)
    img = img * gain + sheen
    img = cv2.GaussianBlur(img, (0, 0), 0.55)
    img = np.clip(img, 0, 255)
    img = img + rng.normal(0, 1, img.shape).astype(np.float32) * np.sqrt(img / 58.8 + 0.25)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)
