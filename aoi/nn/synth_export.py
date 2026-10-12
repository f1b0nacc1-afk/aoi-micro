"""Labelled tiles from the synthetic bench: same camera, scale and mono rendering as the real setup.

    python -m aoi.nn.synth_export --out data/yolo/synth --designs 24 --val-designs 6
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from ..bench.defects import TYPES, inject
from ..bench.geometry import geometry
from ..bench.layout import make_design
from ..bench.render import LIB, MARGIN, S, WD, render_board, render_instance
from .classes import ID, NAMES, bench_class

KEEP_FRAC = 0.4


def _corners(cx, cy, rot, dx, dy, w, h, H_mm):
    """Board-canvas px corners of a local rect (same convention as render.Canvas.L)."""
    a = math.radians(rot)
    out = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        lx, ly = dx + sx * w / 2, dy + sy * h / 2
        rx, ry = lx * math.cos(a) + ly * math.sin(a), -lx * math.sin(a) + ly * math.cos(a)
        out.append((cx * S + rx * S, (H_mm - cy) * S + ry * S))
    return out


def part_box(design: dict, part: dict, ov: dict, Hm: np.ndarray) -> tuple | None:
    """(class id, x0, y0, x1, y1) in frame px for one placed (or missing) part."""
    pkg = LIB[part["pkg"]]
    g = geometry(pkg)
    H_mm = design["H"]
    if ov.get("missing"):
        cid, pts = ID["footprint"], []
        for dx, dy, w, h in g["pads"]:
            pts += _corners(part["x"], part["y"], part["rot"], dx, dy, w, h, H_mm)
    else:
        cid = bench_class(pkg.name, ov.get("as_kind", part["kind"]))
        x, y = part["x"] + ov.get("dx", 0), part["y"] + ov.get("dy", 0)
        rot = part["rot"] + ov.get("drot", 0)
        L, W = pkg.body
        h = pkg.height
        if ov.get("billboard"):
            W = h
        k = h / (WD - h)
        tx, ty = x + (x - design["W"] / 2) * k, y + (y - design["H"] / 2) * k
        pts = _corners(tx, ty, rot, 0, 0, L, W, H_mm)
        for dx, dy, w, hh in g["leads"]:
            pts += _corners(x, y, rot, dx, dy, w, hh, H_mm)
        if ov.get("tombstone"):
            for dx, dy, w, hh in g["pads"]:
                pts += _corners(x, y, rot, dx, dy, w, hh, H_mm)
    p = cv2.perspectiveTransform(np.float64(pts).reshape(-1, 1, 2), Hm).reshape(-1, 2)
    return cid, *p.min(0), *p.max(0)


def tiles(img: np.ndarray, boxes: list, size: int, stride: int):
    h, w = img.shape
    ys = list(range(0, max(1, h - size) + 1, stride)) + ([h - size] if h > size else [])
    xs = list(range(0, max(1, w - size) + 1, stride)) + ([w - size] if w > size else [])
    for y0 in sorted(set(max(0, v) for v in ys)):
        for x0 in sorted(set(max(0, v) for v in xs)):
            t = img[y0:y0 + size, x0:x0 + size]
            th, tw = t.shape
            rows = []
            for cid, bx0, by0, bx1, by1 in boxes:
                cx0, cy0 = max(bx0 - x0, 0), max(by0 - y0, 0)
                cx1, cy1 = min(bx1 - x0, tw), min(by1 - y0, th)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                if (cx1 - cx0) * (cy1 - cy0) < KEEP_FRAC * (bx1 - bx0) * (by1 - by0):
                    continue
                rows.append((cid, (cx0 + cx1) / 2 / tw, (cy0 + cy1) / 2 / th, (cx1 - cx0) / tw, (cy1 - cy0) / th))
            yield (x0, y0), t, rows


def export(out: Path, split: str, seeds: list, per_design: int, size: int, stride: int, stats: Counter,
           n_missing: int = 0, log=print):
    idir, ldir = out / "images" / split, out / "labels" / split
    idir.mkdir(parents=True, exist_ok=True)
    ldir.mkdir(parents=True, exist_ok=True)
    types = list(TYPES)
    n = 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        design = make_design(rng, f"s{seed}")
        board = render_board(design)
        for i in range(per_design):
            batch = "AB"[i % 2]
            n_def = 0 if i < 2 else int(rng.integers(4, 10))
            todo = list(rng.choice(types, n_def)) + ["missing"] * (n_missing if n_def else 0)
            ovs, foreign, _ = inject(design, todo, rng) if todo else ({}, [], [])
            info = {}
            img = render_instance(design, board, batch, ovs, foreign, rng, info=info)
            boxes = [part_box(design, p, ovs.get(p["ref"], {}), info["H"]) for p in design["parts"]]
            for (x0, y0), t, rows in tiles(img, boxes, size, stride):
                stem = f"{design['name']}_{i}_{x0}_{y0}"
                cv2.imwrite(str(idir / f"{stem}.png"), t)
                (ldir / f"{stem}.txt").write_text(
                    "\n".join(f"{c} {a:.6f} {b:.6f} {w:.6f} {h:.6f}" for c, a, b, w, h in rows))
                for r in rows:
                    stats[NAMES[r[0]]] += 1
                n += 1
        log(f"  [{split}] {design['name']}: {len(design['parts'])} parts, {design['finish']}")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/yolo/synth")
    ap.add_argument("--designs", type=int, default=24)
    ap.add_argument("--val-designs", type=int, default=6)
    ap.add_argument("--per-design", type=int, default=4, help="2 good (batch A/B) + defective boards")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=800)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--missing", type=int, default=0, help="extra missing parts per defective board")
    a = ap.parse_args(argv)
    out, stats = Path(a.out), Counter()
    tr = [a.seed + i for i in range(a.designs)]
    va = [a.seed + 10_000 + i for i in range(a.val_designs)]
    n_tr = export(out, "train", tr, a.per_design, a.tile, a.stride, stats, a.missing)
    n_va = export(out, "val", va, a.per_design, a.tile, a.stride, stats, a.missing)
    print(f"tiles: train {n_tr}, val {n_va}; boxes per class: {dict(stats.most_common())}")


if __name__ == "__main__":
    main()
