"""Defect injection with ground truth. Each defect: type, ref (or None), CAD location, expected codes."""
from __future__ import annotations

import math

import numpy as np

from ..cad import load_packages
from .geometry import CHIPS, ICS, SQUARE, geometry

LIB = load_packages(None)
ANY = []

# type: (applicability, expected codes; [] = any FAIL counts, ocr = needs OCR, not scored in recall)
TYPES = {
    "missing": (lambda p, k: True, ["MISSING", "MISMATCH"]),
    "shift": (lambda p, k: True, ["SHIFT"]),
    "rotate": (lambda p, k: not p.name.startswith("CP_"), ["ROTATED"]),
    "polarity": (lambda p, k: p.orient != "none", ["POLARITY"]),
    "rot90": (lambda p, k: p.name in SQUARE and p.orient != "none", ["POLARITY", "ROTATED"]),
    "wrong": (lambda p, k: (p.name in ("0603", "0805", "1206") and k in "RC") or p.name == "SOD-123",
              ["MISMATCH", "MISSING", "POLARITY"]),
    "tombstone": (lambda p, k: p.name in ("0402", "0603", "0805", "1206"), ANY),
    "billboard": (lambda p, k: p.name in ("0603", "0805", "1206"), ANY),
    "damaged": (lambda p, k: p.name in ("0805", "1206") or p.name in ICS[:6], ["MISMATCH", "ANOMALY"]),
    "no_solder": (lambda p, k: p.name in CHIPS or p.name in ("SOIC-8", "SOIC-16", "SOT-223"), ["ANOMALY"]),
    "excess": (lambda p, k: p.name in CHIPS or p.name in ICS, ["ANOMALY"]),
    "bridge": (lambda p, k: p.name in ("SOIC-8", "SOIC-16", "TSSOP-20", "TQFP-32", "LQFP-48", "QFN-32",
                                       "SOT-223"), ["ANOMALY"]),
    "ball": (lambda p, k: True, ["ANOMALY"]),
    "foreign": (None, ["ANOMALY"]),
    "marking": (lambda p, k: p.name in ICS and p.name != "SOT-23", ANY),
}
OCR_ONLY = {"marking"}
FAKE_MARKS = ["LM393", "NE556", "74HC14", "STM32F031", "ATTINY85", "CH9102", "AMS1084"]


def _local_to_cad(part, dx, dy):
    a = math.radians(part["rot"])
    rx, ry = dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)
    return part["x"] + rx, part["y"] - ry


def _free_spot(design, rng, clear=1.8):
    for _ in range(500):
        x, y = rng.uniform(5, design["W"] - 5), rng.uniform(5, design["H"] - 5)
        ok = True
        for p in design["parts"]:
            T, TW = LIB[p["pkg"]].total
            if math.hypot(x - p["x"], y - p["y"]) < max(T, TW) / 2 + clear:
                ok = False
                break
        if ok and all(math.hypot(x - f["x_mm"], y - f["y_mm"]) > 4 for f in design["fiducials"]):
            return x, y
    return None


def inject(design: dict, types: list, rng: np.random.Generator):
    """Returns (overrides per ref, foreign objects, ground truth list)."""
    ovs, foreign, gt, used = {}, [], [], set()
    for t in types:
        if t == "foreign":
            spot = _free_spot(design, rng)
            if spot is None:
                continue
            x, y = spot
            if rng.random() < 0.55:
                pts, a = [(x, y)], rng.uniform(0, 2 * math.pi)
                for _ in range(int(rng.integers(6, 12))):
                    a += rng.normal(0, 0.35)
                    pts.append((pts[-1][0] + 0.4 * math.cos(a), pts[-1][1] + 0.4 * math.sin(a)))
                f = {"kind": "hair", "pts": pts, "w": float(rng.uniform(0.05, 0.09)),
                     "val": float(rng.choice([24, 190]))}
                cx, cy = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
            else:
                r = rng.uniform(0.18, 0.4)
                ang = np.sort(rng.uniform(0, 2 * np.pi, 9))
                pts = [(x + r * rng.uniform(0.6, 1.2) * math.cos(v), y + r * rng.uniform(0.6, 1.2) * math.sin(v))
                       for v in ang]
                f = {"kind": "debris", "pts": pts,
                     "val": float(rng.uniform(18, 30) if rng.random() < 0.5 else rng.uniform(150, 200))}
                cx, cy = x, y
            foreign.append(f)
            gt.append({"type": t, "ref": None, "x": round(float(cx), 2), "y": round(float(cy), 2),
                       "expect": TYPES[t][1]})
            continue
        app, expect = TYPES[t]
        cands = [p for p in design["parts"] if p["ref"] not in used and app(LIB[p["pkg"]], p["kind"])]
        if not cands:
            continue
        p = cands[int(rng.integers(len(cands)))]
        pkg = LIB[p["pkg"]]
        g = geometry(pkg)
        ov = {}
        loc = (p["x"], p["y"])
        if t == "missing":
            ov["missing"] = True
        elif t == "shift":
            tol = min(0.35 * min(pkg.body), 0.5)
            mag = tol * rng.uniform(1.6, 2.4)
            ang = math.radians(p["rot"] + float(rng.choice([0, 90, 180, 270])))
            ov["dx"], ov["dy"] = mag * math.cos(ang), mag * math.sin(ang)
        elif t == "rotate":
            ov["drot"] = float(rng.choice([-1, 1]) * rng.uniform(9, 17))
        elif t == "polarity":
            ov["drot"] = 180.0
        elif t == "rot90":
            ov["drot"] = float(rng.choice([90, -90]))
        elif t == "wrong":
            if pkg.name == "SOD-123":
                ov["no_band"] = True
            else:
                ov["as_kind"] = "C" if p["kind"] == "R" else "R"
        elif t == "tombstone":
            ov["tombstone"] = True
        elif t == "billboard":
            ov["billboard"] = True
        elif t == "damaged":
            ov["damaged"] = True
        elif t == "no_solder":
            ov["no_solder"] = True
        elif t == "excess":
            i = int(rng.integers(len(g["pads"])))
            ov["excess"] = i
            loc = _local_to_cad(p, *g["pads"][i][:2])
        elif t == "bridge":
            row = g["rows"][int(rng.integers(len(g["rows"])))]
            k = int(rng.integers(len(row) - 1))
            ov["bridge"] = (row[k], row[k + 1])
            a, b = g["pads"][row[k]], g["pads"][row[k + 1]]
            loc = _local_to_cad(p, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        elif t == "ball":
            dx, dy, w, h = g["pads"][int(rng.integers(len(g["pads"])))]
            if abs(dx) >= abs(dy):
                off = (math.copysign(w / 2 + rng.uniform(0.35, 0.6), dx or 1), rng.uniform(-h / 2, h / 2))
            else:
                off = (rng.uniform(-w / 2, w / 2), math.copysign(h / 2 + rng.uniform(0.35, 0.6), dy or 1))
            loc = _local_to_cad(p, dx + off[0], dy + off[1])
            ov["ball"] = loc
            ov["ball_r"] = float(rng.uniform(0.1, 0.18))
        elif t == "marking":
            ov["marking"] = str(rng.choice([m for m in FAKE_MARKS if m != p.get("value")]))
        used.add(p["ref"])
        ovs.setdefault(p["ref"], {}).update(ov)
        gt.append({"type": t, "ref": p["ref"], "pkg": pkg.name, "x": round(float(loc[0]), 2),
                   "y": round(float(loc[1]), 2), "expect": expect})
    return ovs, foreign, gt
