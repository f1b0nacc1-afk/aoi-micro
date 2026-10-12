"""Second opinion from the component detector for CAD positions inspected by the classic pipeline."""
from __future__ import annotations

import re

import cv2
import numpy as np

from ..bench.render import S as TRAIN_PX_PER_MM
from .classes import BENCH_PKG, ID, NAMES
from .detect import detect_tiled

REF_CLASS = {"R": "resistor", "RN": "resistor", "C": "capacitor", "L": "inductor", "FB": "inductor",
             "D": "diode", "LED": "led", "Q": "transistor", "U": "ic", "IC": "ic", "Y": "crystal", "X": "crystal"}


def expected_class(c) -> str | None:
    if c.package is not None and c.package.name in BENCH_PKG:
        return BENCH_PKG[c.package.name]
    fp = c.footprint.lower()
    if fp.startswith("cp_elec"):
        return "electrolytic"
    if fp.startswith("led"):
        return "led"
    m = re.match(r"[A-Za-z]+", c.ref)
    return REF_CLASS.get(m.group(0).upper()) if m else None


def to_train_scale(img: np.ndarray, px_per_mm: float) -> tuple:
    k = TRAIN_PX_PER_MM / px_per_mm
    if abs(k - 1) < 0.03:
        return img, 1.0
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR), k


def _inside(pt, center, size, ccw) -> bool:
    a = np.radians(ccw)
    dx, dy = pt[0] - center[0], pt[1] - center[1]
    u = dx * np.cos(a) - dy * np.sin(a)
    v = dx * np.sin(a) + dy * np.cos(a)
    return abs(u) <= size[0] / 2 and abs(v) <= size[1] / 2


def nn_check(model, board: np.ndarray, results: list, comps: list, px_per_mm: float, conf: float = 0.25,
             veto: bool = False) -> np.ndarray:
    """Adds r["nn"], r["nn_expected"], r["nn_flags"] to each inspected part; returns detections in board px.
    veto: flags also become defects and fail the part."""
    img, k = to_train_scale(board, px_per_mm)
    det = detect_tiled(model, img, conf=conf)
    det[:, :4] /= k
    centers = np.c_[(det[:, 0] + det[:, 2]) / 2, (det[:, 1] + det[:, 3]) / 2]
    by_ref = {c.ref: c for c in comps}
    for r in results:
        g = r.get("_geom")
        if g is None:
            continue
        hits = [i for i, p in enumerate(centers) if _inside(p, g.center, g.region, g.ccw)]
        best = max(hits, key=lambda i: det[i, 4], default=None)
        exp = expected_class(by_ref[r["ref"]])
        r["nn_expected"] = exp
        r["nn"] = None if best is None else f"{NAMES[int(det[best, 5])]} {det[best, 4]:.2f}"
        flags = []
        if best is None or int(det[best, 5]) == ID["footprint"]:
            flags.append("NN_EMPTY")
        elif exp and int(det[best, 5]) not in (ID[exp], ID["other"]):
            flags.append("NN_CLASS")
        r["nn_flags"] = flags
        if veto and flags:
            r["defects"] += [f for f in flags if f not in r["defects"]]
            r["status"] = "FAIL"
    return det
