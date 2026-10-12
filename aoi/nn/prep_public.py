"""Convert public PCB datasets (Roboflow YOLO or COCO export) to the unified classes, mono images.

    python -m aoi.nn.prep_public data/raw/circuit-elements data/raw/rf100-vl/smd-components --out data/yolo/public
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import yaml

from .classes import NAMES, map_public

SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "test"}
EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _names(src: Path) -> list:
    y = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
    n = y["names"]
    return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)


def _box(vals: list) -> tuple | None:
    """YOLO box, or polygon (segmentation export) -> enclosing box."""
    if len(vals) == 4:
        return tuple(vals)
    if len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    return None


def _yolo_items(src: Path, sdir: str):
    """(image path, [(source class name, cx, cy, w, h normalised)]) from a Roboflow YOLO export."""
    names = _names(src)
    idir = src / sdir / "images"
    for img_path in sorted(p for p in idir.iterdir() if p.suffix.lower() in EXTS):
        lbl = src / sdir / "labels" / (img_path.stem + ".txt")
        boxes = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                v = line.split()
                b = _box([float(x) for x in v[1:]]) if v else None
                if b:
                    boxes.append((names[int(v[0])], *b))
        yield img_path, boxes


def _coco_items(src: Path, sdir: str):
    d = json.loads((src / sdir / "_annotations.coco.json").read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    anns = {}
    for a in d["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    for im in d["images"]:
        W, H = im["width"], im["height"]
        boxes = [(cats[a["category_id"]], (a["bbox"][0] + a["bbox"][2] / 2) / W,
                  (a["bbox"][1] + a["bbox"][3] / 2) / H, a["bbox"][2] / W, a["bbox"][3] / H)
                 for a in anns.get(im["id"], [])]
        yield src / sdir / im["file_name"], boxes


def convert(src: Path, out: Path, max_side: int, stats: Counter) -> int:
    tag = src.name
    n_img = 0
    for sdir, split in SPLITS.items():
        if (src / sdir / "images").is_dir():
            items = _yolo_items(src, sdir)
        elif (src / sdir / "_annotations.coco.json").exists():
            items = _coco_items(src, sdir)
        else:
            continue
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img_path, boxes in items:
            rows = []
            for name, cx, cy, w, h in boxes:
                cid = map_public(name)
                if cid is None or w <= 0 or h <= 0:
                    continue
                rows.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                stats[NAMES[cid]] += 1
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            k = max_side / max(img.shape)
            if k < 1:
                img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            stem = f"{tag}__{img_path.stem}"
            cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(rows))
            n_img += 1
    return n_img


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--out", default="data/yolo/public")
    ap.add_argument("--max-side", type=int, default=1600)
    a = ap.parse_args(argv)
    out, stats = Path(a.out), Counter()
    for s in a.sources:
        print(f"{s}: {convert(Path(s), out, a.max_side, stats)} images")
    print("boxes per class:", dict(stats.most_common()))


if __name__ == "__main__":
    main()
