"""Procedural board designs: placement, routing under mask, pours, vias, silkscreen, fiducials."""
from __future__ import annotations

import math

import numpy as np

from ..cad import load_packages
from .geometry import geometry

FOOTPRINTS = {
    "0402": ("C_0402_1005Metric", "R_0402_1005Metric"), "0603": ("C_0603_1608Metric", "R_0603_1608Metric"),
    "0805": ("C_0805_2012Metric", "R_0805_2012Metric"), "1206": ("C_1206_3216Metric", "R_1206_3216Metric"),
    "LED_0805": ("LED_0805_2012Metric",), "Tantalum_Case-A": ("CP_Tantalum_Case-A_EIA-3216-18",),
    "SOD-123": ("D_SOD-123",), "SMA": ("D_SMA",), "SOT-23": ("SOT-23",), "SOT-223": ("SOT-223-3_TabPin2",),
    "SOIC-8": ("SOIC-8_3.9x4.9mm_P1.27mm",), "SOIC-16": ("SOIC-16_3.9x9.9mm_P1.27mm",),
    "TSSOP-20": ("TSSOP-20_4.4x6.5mm_P0.65mm",), "TQFP-32": ("TQFP-32_7x7mm_P0.8mm",),
    "LQFP-48": ("LQFP-48_7x7mm_P0.5mm",), "QFN-32": ("QFN-32-1EP_5x5mm_P0.5mm",),
    "CP_Elec_5x5.8": ("CP_Elec_5x5.8",), "CP_Elec_6.3x7.7": ("CP_Elec_6.3x7.7",),
    "Crystal_SMD_3225": ("Crystal_SMD_3225-4Pin_3.2x2.5mm",), "L_4x4": ("L_4x4mm_Shielded",),
}
WEIGHTS = {"0402": 6, "0603": 10, "0805": 10, "1206": 4, "LED_0805": 2, "Tantalum_Case-A": 2,
           "SOD-123": 3, "SMA": 1, "SOT-23": 4, "SOT-223": 1, "SOIC-8": 2, "SOIC-16": 1, "TSSOP-20": 1,
           "TQFP-32": 1, "LQFP-48": 1, "QFN-32": 1, "CP_Elec_5x5.8": 1, "CP_Elec_6.3x7.7": 1,
           "Crystal_SMD_3225": 1, "L_4x4": 1}
MUST = ["LQFP-48", "QFN-32", "SOIC-8", "TSSOP-20", "CP_Elec_5x5.8", "SOT-23", "SOD-123", "Tantalum_Case-A"]
IC_NAMES = {"SOIC-8": ["LM358", "NE555", "TL072", "24C02"], "SOIC-16": ["74HC595", "CD4051", "MAX232"],
            "TSSOP-20": ["STM32F030", "74LVC245", "PCA9685"], "TQFP-32": ["ATMEGA328P", "STM8S105"],
            "LQFP-48": ["STM32F103", "GD32F303", "RP2040"], "QFN-32": ["ESP32-C3", "CH340", "nRF52810"],
            "SOT-23": ["J3Y", "A7", "W1", "S4"], "SOT-223": ["AMS1117", "LM1117", "LD1117"]}
R_CODES = [("10k", "103", "1002"), ("4k7", "472", "4701"), ("1k", "102", "1001"), ("220", "221", "2200"),
           ("100k", "104", "1003"), ("0R", "000", "0")]
C_VALUES = ["100n", "1u", "10u", "4u7", "22p", "10n"]


def _rot(dx, dy, deg):
    a = math.radians(deg)
    return dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)


def _bbox(pkg, rot, clear):
    T, TW = pkg.total
    pts = [_rot(sx * T / 2, sy * TW / 2, rot) for sx in (-1, 1) for sy in (-1, 1)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (max(xs) - min(xs)) / 2 + clear, (max(ys) - min(ys)) / 2 + clear


def make_design(rng: np.random.Generator, name: str) -> dict:
    lib = load_packages(None)
    W, H = [(70, 50), (80, 55), (64, 46), (76, 52)][int(rng.integers(4))]
    grid_res = 0.25
    occ = np.zeros((int(H / grid_res) + 1, int(W / grid_res) + 1), bool)

    corners = [(3.5, 3.5), (W - 3.5, 3.5), (W - 3.5, H - 3.5), (3.5, H - 3.5)]
    if rng.random() < 0.4:
        corners.pop(int(rng.integers(4)))
    fiducials = [{"x_mm": x, "y_mm": y} for x, y in corners]

    def block(x0, y0, x1, y1):
        occ[max(0, int(y0 / grid_res)):int(y1 / grid_res) + 1, max(0, int(x0 / grid_res)):int(x1 / grid_res) + 1] = True

    for x, y in corners:
        block(x - 4, y - 4, x + 4, y + 4)
    edge = 2.5
    block(0, 0, W, edge); block(0, H - edge, W, H); block(0, 0, edge, H); block(W - edge, 0, W, H)

    def try_place(pkg, rot, tries=400):
        hx, hy = _bbox(pkg, rot, 1.1)
        for _ in range(tries):
            x, y = rng.uniform(hx, W - hx), rng.uniform(hy, H - hy)
            y0, y1 = int((y - hy) / grid_res), int((y + hy) / grid_res) + 1
            x0, x1 = int((x - hx) / grid_res), int((x + hx) / grid_res) + 1
            if not occ[y0:y1, x0:x1].any():
                occ[y0:y1, x0:x1] = True
                return round(x, 2), round(y, 2)
        return None

    counters, parts = {}, []
    names = list(WEIGHTS)
    p = np.array([WEIGHTS[k] for k in names], float)
    p /= p.sum()
    queue = MUST + list(rng.choice(names, size=90, p=p))
    for pk in queue:
        pkg = lib[pk]
        rot = float(rng.choice([0, 90, 180, 270]))
        if pk in ("0603", "0805") and rng.random() < 0.06:
            rot = 45.0
        pos = try_place(pkg, rot)
        if pos is None:
            continue
        if pk in ("0402", "0603", "0805", "1206"):
            kind = "R" if rng.random() < 0.45 else "C"
        elif pk in ("LED_0805", "SOD-123", "SMA"):
            kind = "D"
        elif pk in ("SOT-23",):
            kind = "Q"
        elif pk.startswith("CP_") or pk.startswith("Tant"):
            kind = "C"
        elif pk == "Crystal_SMD_3225":
            kind = "Y"
        elif pk == "L_4x4":
            kind = "L"
        else:
            kind = "U"
        counters[kind] = counters.get(kind, 0) + 1
        ref = f"{kind}{counters[kind]}"
        fps = FOOTPRINTS[pk]
        fp = fps[1] if (kind == "R" and len(fps) > 1) else fps[0]
        part = {"ref": ref, "pkg": pk, "footprint": fp, "x": pos[0], "y": pos[1], "rot": rot, "kind": kind}
        if kind == "R" and pk in ("0402", "0603", "0805", "1206"):
            v = R_CODES[int(rng.integers(len(R_CODES)))]
            part.update(value=v[0], marks=[v[1], v[2]])
        elif kind == "C" and pk in ("0402", "0603", "0805", "1206"):
            part.update(value=str(rng.choice(C_VALUES)))
        elif pk in IC_NAMES:
            nm = str(rng.choice(IC_NAMES[pk]))
            part.update(value=nm, marks=[nm, nm])
        elif pk.startswith("CP_"):
            part.update(value=str(rng.choice(["47u", "100u", "220u"])))
        elif pk == "L_4x4":
            part.update(value="4u7", marks=["4R7", "4R7"])
        else:
            part.update(value=pk)
        parts.append(part)

    traces, vias = [], []
    for part in parts:
        g = geometry(lib[part["pkg"]])
        for (dx, dy, w, h) in g["pads"]:
            if rng.random() < 0.25:
                continue
            ax = (math.copysign(1, dx), 0) if abs(dx) >= abs(dy) else (0, math.copysign(1, dy))
            if dx == 0 and dy == 0:
                continue
            l1 = rng.uniform(0.8, 2.5)
            sx, sy = dx + ax[0] * (w / 2), dy + ax[1] * (h / 2)
            ex, ey = sx + ax[0] * l1, sy + ax[1] * l1
            l2 = rng.uniform(-4, 4)
            fx, fy = (ex, ey + l2) if ax[0] else (ex + l2, ey)
            pts = []
            for lx, ly in ((dx, dy), (ex, ey), (fx, fy)):
                rx, ry = _rot(lx, ly, part["rot"])
                pts.append((part["x"] + rx, part["y"] - ry))
            width = float(rng.choice([0.25, 0.3, 0.4]))
            traces.append({"pts": pts, "w": width})
            vx, vy = pts[-1]
            if 3 < vx < W - 3 and 3 < vy < H - 3:
                vias.append({"x": vx, "y": vy, "tented": bool(rng.random() < 0.5)})
    pours = []
    for _ in range(int(rng.integers(2, 5))):
        pw, ph = rng.uniform(8, 25), rng.uniform(6, 18)
        pours.append({"x": rng.uniform(3, W - 3 - pw), "y": rng.uniform(3, H - 3 - ph), "w": pw, "h": ph})

    return {"name": name, "W": W, "H": H, "finish": str(rng.choice(["HASL", "ENIG"])),
            "mask_level": float(rng.uniform(48, 62)), "c_body": float(rng.uniform(105, 140)), "fiducials": fiducials, "parts": parts,
            "traces": traces, "vias": vias, "pours": pours, "seed": int(rng.integers(1 << 30))}
