from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Body = package body, total = body + leads + land pattern (inspection window), mm; local frame at CAD rotation 0: x = length, y = width (image y down).
# mark: [cx, cy, w, h] of the polarity feature in that local frame.
# orient: "none" (symmetric), "mark" (mark vs 180°, and vs 90/270° for square bodies), "template" (whole body 0° vs 180°).
# interior: [w, h] centred zone with marking/body colour that may change between component batches;
# compared only by median brightness. Default: body minus 0.3 mm per side.
DEFAULT_PACKAGES = {
    "0402": {"body": [1.0, 0.5], "total": [1.6, 0.6], "height": 0.35, "orient": "none", "interior": [0.5, 0.4]},
    "0603": {"body": [1.6, 0.8], "total": [2.5, 0.95], "height": 0.45, "orient": "none", "interior": [0.9, 0.65]},
    "0805": {"body": [2.0, 1.25], "total": [2.9, 1.35], "height": 0.5, "orient": "none", "interior": [1.15, 1.05]},
    "1206": {"body": [3.2, 1.6], "total": [4.2, 1.8], "height": 0.55, "orient": "none", "interior": [1.9, 1.4]},
    "SOD-123": {"body": [2.7, 1.6], "total": [4.2, 1.6], "height": 1.1, "orient": "mark",
                "mark": [-1.0, 0.0, 0.5, 1.4]},
    "SOT-23": {"body": [2.9, 1.3], "total": [2.9, 3.1], "height": 1.0, "orient": "template"},
    "SOIC-8": {"body": [4.9, 3.9], "total": [4.9, 7.0], "height": 1.5, "orient": "mark",
               "mark": [-1.8, 1.2, 0.8, 0.8]},
    "TQFP-32": {"body": [7.0, 7.0], "total": [9.7, 9.7], "height": 1.2, "orient": "mark",
                "mark": [-2.6, 2.6, 1.0, 1.0]},
    "CP_Elec_5x5.8": {"body": [5.3, 5.3], "total": [6.0, 5.3], "height": 5.8, "orient": "mark",
                      "mark": [-1.9, 0.0, 1.0, 3.0]},
    "CP_Elec_6.3x7.7": {"body": [6.6, 6.6], "total": [7.4, 6.6], "height": 7.7, "orient": "mark",
                        "mark": [-2.4, 0.0, 1.2, 3.6]},
    "SMA": {"body": [4.3, 2.6], "total": [6.5, 2.6], "height": 2.3, "orient": "mark",
            "mark": [-1.7, 0.0, 0.5, 2.4]},
    "SOT-223": {"body": [6.5, 3.5], "total": [6.5, 7.8], "height": 1.7, "orient": "template"},
    "SOIC-16": {"body": [9.9, 3.9], "total": [9.9, 7.0], "height": 1.5, "orient": "mark",
                "mark": [-4.4, 1.2, 0.8, 0.8]},
    "TSSOP-20": {"body": [6.5, 4.4], "total": [6.5, 7.1], "height": 1.1, "orient": "mark",
                 "mark": [-2.8, 1.5, 0.6, 0.6]},
    "LQFP-48": {"body": [7.0, 7.0], "total": [9.7, 9.7], "height": 1.4, "orient": "mark",
                "mark": [-2.7, 2.7, 0.8, 0.8]},
    "QFN-32": {"body": [5.0, 5.0], "total": [5.7, 5.7], "height": 0.9, "orient": "mark",
               "mark": [-2.0, 2.0, 0.6, 0.6]},
    "Tantalum_Case-A": {"body": [3.2, 1.6], "total": [4.6, 1.6], "height": 1.6, "orient": "mark",
                        "mark": [-1.2, 0.0, 0.5, 1.4], "interior": [1.8, 1.4]},
    "LED_0805": {"body": [2.0, 1.25], "total": [2.9, 1.35], "height": 0.8, "orient": "mark",
                 "mark": [-0.7, 0.0, 0.3, 1.0], "interior": [1.15, 1.05]},
    "Crystal_SMD_3225": {"body": [3.2, 2.5], "total": [3.6, 2.75], "height": 0.8, "orient": "none"},
    "L_4x4": {"body": [4.0, 4.0], "total": [4.6, 4.0], "height": 2.0, "orient": "none"},
}


@dataclass
class Package:
    name: str
    body: tuple
    total: tuple
    height: float
    orient: str = "none"
    mark: tuple | None = None
    interior: tuple | None = None


@dataclass
class Component:
    ref: str
    value: str
    footprint: str
    x_mm: float
    y_mm: float
    rot_deg: float
    side: str = "top"
    mpn: str = ""
    package: Package | None = None
    extra: dict = field(default_factory=dict)


def load_packages(path: Path | None) -> dict:
    lib = dict(DEFAULT_PACKAGES)
    if path:
        lib.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return {k: Package(name=k, body=tuple(v["body"]), total=tuple(v.get("total", v["body"])),
                       height=float(v.get("height", 1.0)), orient=v.get("orient", "none"),
                       mark=tuple(v["mark"]) if v.get("mark") else None,
                       interior=tuple(v["interior"]) if v.get("interior") else None)
            for k, v in lib.items()}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def match_package(footprint: str, lib: dict) -> Package | None:
    f = _norm(footprint)
    best = None
    for name, pkg in lib.items():
        n = _norm(name)
        if n and n in f and (best is None or len(n) > len(_norm(best.name))):
            best = pkg
    return best


_ALIASES = {
    "ref": ["ref", "designator", "refdes", "reference", "part"],
    "value": ["val", "value", "comment"],
    "footprint": ["package", "footprint", "pattern"],
    "x": ["posx", "midx", "centerx", "x", "locationx"],
    "y": ["posy", "midy", "centery", "y", "locationy"],
    "rot": ["rot", "rotation", "angle"],
    "side": ["side", "layer", "tb"],
}


def _col(header: list, key: str) -> int | None:
    norm = [_norm(h) for h in header]
    for alias in _ALIASES[key]:
        if alias in norm:
            return norm.index(alias)
    return None


def _num(s: str) -> float:
    return float(re.sub(r"[^0-9eE+\-.]", "", s.replace(",", ".")))


def load_centroid(path: Path, lib: dict, bom: dict | None = None) -> list:
    """KiCad .pos/.csv, Altium Pick&Place, or any CSV with ref/x/y/rot columns (mm)."""
    rows = [r for r in csv.reader(Path(path).read_text(encoding="utf-8-sig").splitlines())
            if r and not r[0].lstrip().startswith("#")]
    header, data = rows[0], rows[1:]
    idx = {k: _col(header, k) for k in _ALIASES}
    for need in ("ref", "x", "y"):
        if idx[need] is None:
            raise ValueError(f"centroid: no column for '{need}' in {header}")
    comps = []
    for r in data:
        g = lambda k, d="": r[idx[k]].strip() if idx[k] is not None and idx[k] < len(r) else d
        side = g("side", "top").lower()
        if side.startswith("b"):
            continue
        fp = g("footprint")
        c = Component(ref=g("ref"), value=g("value"), footprint=fp, x_mm=_num(g("x")),
                      y_mm=_num(g("y")), rot_deg=_num(g("rot", "0")) if g("rot") else 0.0,
                      side="top", package=match_package(fp, lib))
        if bom and c.ref in bom:
            c.mpn = bom[c.ref]
        comps.append(c)
    return comps


def load_bom(path: Path | None) -> dict:
    """CSV with a designator column (comma/space-separated refs allowed) and an MPN column."""
    if not path:
        return {}
    rows = list(csv.reader(Path(path).read_text(encoding="utf-8-sig").splitlines()))
    norm = [_norm(h) for h in rows[0]]
    ri = next(i for i, h in enumerate(norm) if h in ("ref", "designator", "refdes", "reference"))
    mi = next((i for i, h in enumerate(norm) if h in ("mpn", "partnumber", "manufacturerpartnumber")), None)
    out = {}
    for r in rows[1:]:
        if mi is None or len(r) <= max(ri, mi):
            continue
        for ref in re.split(r"[,\s;]+", r[ri].strip()):
            if ref:
                out[ref] = r[mi].strip()
    return out
