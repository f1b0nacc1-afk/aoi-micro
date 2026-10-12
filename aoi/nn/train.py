"""Train the component detector on prepared YOLO datasets (unified classes).

    python -m aoi.nn.train --data data/yolo/public data/yolo/synth --name det_v0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .classes import NAMES


def write_data_yaml(roots: list, out: Path, val_roots: list | None = None) -> Path:
    def split(rs, s):
        return [str((Path(r) / "images" / s).resolve()) for r in rs if (Path(r) / "images" / s).is_dir()]
    cfg = {"train": split(roots, "train"), "val": split(val_roots or roots, "val"),
           "names": dict(enumerate(NAMES))}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="dataset roots with images/{train,val}")
    ap.add_argument("--val", nargs="*", help="roots used for validation (default: same as --data)")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=6, help="8 at 1024 px overflows 10 GB VRAM into shared RAM")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--name", default="det")
    ap.add_argument("--project", default="runs/nn")
    a = ap.parse_args(argv)
    from ultralytics import YOLO
    project = Path(a.project).resolve()  # relative paths go under the global ultralytics runs_dir
    data = write_data_yaml(a.data, project / f"{a.name}_data.yaml", a.val)
    YOLO(a.model).train(data=str(data), imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, project=str(project),
                        name=a.name, exist_ok=True, workers=a.workers, hsv_h=0.0, hsv_s=0.0, degrees=10.0,
                        flipud=0.5, fliplr=0.5, mosaic=1.0, close_mosaic=10, patience=20, plots=True)


if __name__ == "__main__":
    main()
