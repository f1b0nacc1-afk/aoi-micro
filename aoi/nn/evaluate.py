"""Detector evaluation: per-source validation mAP and full-frame bench test against CAD.

    python -m aoi.nn.evaluate runs/nn/det_v0/weights/best.pt --sources data/yolo/public data/yolo/synth
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..bench.defects import inject
from ..bench.layout import make_design
from ..bench.render import render_board, render_instance
from .classes import ID, NAMES, bench_class
from .detect import detect_tiled
from .synth_export import part_box
from .train import write_data_yaml

BENCH_DEFECTS = ["missing", "missing", "shift", "rotate", "polarity", "wrong", "tombstone", "billboard"]


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def val_per_source(model, sources: list, imgsz: int, work: Path) -> dict:
    out = {}
    for s in sources:
        y = write_data_yaml([s], work / f"val_{Path(s).name}.yaml")
        m = model.val(data=str(y), imgsz=imgsz, batch=6, workers=2, verbose=False, plots=False,
                      project=str(work), name=f"val_{Path(s).name}", exist_ok=True)
        per = {NAMES[int(c)]: round(float(m.box.maps[int(c)]), 3) for c in m.box.ap_class_index}
        ap50 = {NAMES[int(c)]: round(float(v), 3) for c, v in zip(m.box.ap_class_index, m.box.ap50)}
        out[Path(s).name] = {"mAP50": round(float(m.box.map50), 3), "mAP50-95": round(float(m.box.map), 3),
                             "ap50": ap50, "ap50_95": per}
    return out


def bench_test(model, n_designs: int, seed: int, conf: float, iou_thr: float, log=print) -> dict:
    """Full synthetic frames (unseen designs): placed parts found with the right class, missing parts
    reported as empty footprint / no component, false boxes."""
    per_cls = defaultdict(lambda: {"n": 0, "found": 0, "right_class": 0})
    defects = defaultdict(lambda: {"n": 0, "flagged": 0})
    n_false, n_det, times = 0, 0, []
    for k in range(n_designs):
        rng = np.random.default_rng(seed + k)
        design = make_design(rng, f"e{seed + k}")
        board = render_board(design)
        ovs, foreign, gt = inject(design, BENCH_DEFECTS, rng)
        info = {}
        img = render_instance(design, board, "AB"[k % 2], ovs, foreign, rng, info=info)
        t = time.time()
        det = detect_tiled(model, img, conf=conf)
        times.append(time.time() - t)
        n_det += len(det)
        used = set()
        gt_type = {g["ref"]: g["type"] for g in gt if g["ref"]}
        for p in design["parts"]:
            ov = ovs.get(p["ref"], {})
            cid, *box = part_box(design, p, ov, info["H"])
            ious = [(iou(box, d[:4]), j) for j, d in enumerate(det)]
            best = max(ious, default=(0, -1))
            match = det[best[1]] if best[0] >= iou_thr else None
            if match is not None:
                used.add(best[1])
            if ov.get("missing"):
                comp_here = [d for d in det if iou(box, d[:4]) >= 0.3 and int(d[5]) != ID["footprint"]]
                d = defects["missing"]
                d["n"] += 1
                d["flagged"] += int(not comp_here)
                c = per_cls["footprint"]
                c["n"] += 1
                c["found"] += int(match is not None)
                c["right_class"] += int(match is not None and int(match[5]) == cid)
                continue
            c = per_cls[NAMES[cid]]
            c["n"] += 1
            c["found"] += int(match is not None)
            c["right_class"] += int(match is not None and int(match[5]) == cid)
            if p["ref"] in gt_type and gt_type[p["ref"]] == "wrong":
                d = defects["wrong"]
                d["n"] += 1
                d["flagged"] += int(match is not None and int(match[5]) != bench_class(p["pkg"], p["kind"]))
        n_false += len(det) - len(used)
        log(f"  {design['name']}: {len(design['parts'])} parts, {len(det)} boxes, {times[-1]:.2f} s")
    return {"per_class": {k: {**v, "recall": round(v["found"] / max(v["n"], 1), 3),
                              "class_acc": round(v["right_class"] / max(v["n"], 1), 3)} for k, v in per_cls.items()},
            "defects": {k: {**v, "rate": round(v["flagged"] / max(v["n"], 1), 3)} for k, v in defects.items()},
            "false_boxes": n_false, "boxes": n_det, "frame_s": round(float(np.mean(times)), 2)}


def report(val: dict, bench: dict, weights: str) -> str:
    L = [f"# Детектор компонентов: {weights}", "", "## Валидация по источникам", "",
         "| источник | mAP50 | mAP50-95 |", "|---|---|---|"]
    L += [f"| {k} | {v['mAP50']} | {v['mAP50-95']} |" for k, v in val.items()]
    L += ["", "AP50 по классам:", "", "| класс | " + " | ".join(val) + " |", "|---|" + "---|" * len(val)]
    for n in NAMES:
        L.append(f"| {n} | " + " | ".join(str(v["ap50"].get(n, "—")) for v in val.values()) + " |")
    L += ["", "## Стенд: целые кадры новых дизайнов", "",
          f"Время детекции кадра (тайлы 1024): {bench['frame_s']} с; рамок {bench['boxes']}, "
          f"лишних (без CAD-позиции) {bench['false_boxes']}.", "",
          "| класс | позиций | найдено | верный класс |", "|---|---|---|---|"]
    for k, v in sorted(bench["per_class"].items(), key=lambda kv: -kv[1]["n"]):
        L.append(f"| {k} | {v['n']} | {100 * v['recall']:.1f}% | {100 * v['class_acc']:.1f}% |")
    L += ["", "| дефект | n | замечен детектором |", "|---|---|---|"]
    for k, v in bench["defects"].items():
        L.append(f"| {k} | {v['n']} | {100 * v['rate']:.1f}% |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--sources", nargs="*", default=["data/yolo/public", "data/yolo/synth"])
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--designs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=90_000)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args(argv)
    from ultralytics import YOLO
    model = YOLO(a.weights)
    work = Path(a.weights).resolve().parent.parent / "eval"
    work.mkdir(parents=True, exist_ok=True)
    val = val_per_source(model, a.sources, a.imgsz, work)
    bench = bench_test(model, a.designs, a.seed, a.conf, a.iou)
    text = report(val, bench, a.weights)
    (work / "eval.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"-> {work / 'eval.md'}")


if __name__ == "__main__":
    main()
