"""Bench runner: designs -> golden -> train -> good/alt-batch/defective boards -> inspect -> score -> reports."""
from __future__ import annotations

import csv
import html
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..align import BoardFrame
from ..cad import load_bom, load_centroid, load_packages
from ..config import load_config, resolve
from ..inspect import inspect_board
from ..model import load_model, train
from ..report import write_report
from .defects import OCR_ONLY, TYPES, inject
from .layout import make_design
from .render import S, frame_px, render_board, render_instance

LIB = load_packages(None)


def _write_design(design: dict, ddir: Path, inspect_over: dict) -> Path:
    ddir.mkdir(parents=True, exist_ok=True)
    with (ddir / "centroid.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Ref", "Val", "Package", "PosX", "PosY", "Rot", "Side"])
        for p in design["parts"]:
            w.writerow([p["ref"], p.get("value", ""), p["footprint"], p["x"], p["y"], p["rot"], "top"])
    fids = [{**f, "approx_px": [round(v) for v in frame_px(design, f["x_mm"], f["y_mm"])]}
            for f in design["fiducials"]]
    cfg = {"camera": {"backend": "file"}, "optics": {"focal_mm": 12.0, "wd_mm": 127.0},
           "board": {"width_mm": design["W"], "height_mm": design["H"], "cad_y_up": True, "cad_rot_ccw": True,
                     "px_per_mm": S, "raw_px_per_mm": S, "fiducials": fids, "fiducial_diameter_mm": 1.0,
                     "fiducial_search_mm": 4.0},
           "files": {"centroid": "centroid.csv"}, "inspect": inspect_over}
    (ddir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (ddir / "design.json").write_text(json.dumps({k: v for k, v in design.items()
                                                  if k not in ("traces", "vias", "pours")}, indent=1,
                                                 default=str), encoding="utf-8")
    return ddir / "config.json"


def _near(ax, ay, bx, by, r):
    return math.hypot(ax - bx, ay - by) <= r


def score_board(design: dict, results: list, extra: list, gt: list) -> dict:
    parts = {r["ref"]: r for r in results}
    geo = {p["ref"]: (p["x"], p["y"], max(LIB[p["pkg"]].total) / 2) for p in design["parts"]}
    gt_refs = {g["ref"] for g in gt if g["ref"]}
    explained, used_extra, scored = set(), set(), []
    for g in gt:
        det, codes = False, set()
        r = parts.get(g["ref"]) if g["ref"] else None
        if r and r["status"] == "FAIL":
            det = True
            codes |= set(r["defects"])
        for i, e in enumerate(extra):
            if _near(e["x_mm"], e["y_mm"], g["x"], g["y"], 1.2 + max(e["w_mm"], e["h_mm"]) / 2):
                used_extra.add(i)
                if g["type"] in ("ball", "foreign", "excess", "bridge"):
                    det = True
                    codes.add("ANOMALY")
        if g["type"] in ("ball", "foreign"):
            for ref, (x, y, rad) in geo.items():
                pr = parts.get(ref, {})
                if _near(x, y, g["x"], g["y"], rad + 0.3) and pr.get("status") == "FAIL" \
                        and "ANOMALY" in pr.get("defects", []):
                    det = True
                    codes.add("ANOMALY")
                    explained.add(ref)
        cls = det and (not g["expect"] or bool(codes & set(g["expect"])))
        scored.append({**g, "detected": det, "classified": cls, "codes": sorted(codes)})
    false_parts = [r for r in results if r["status"] == "FAIL" and r["ref"] not in gt_refs
                   and r["ref"] not in explained]
    false_extra = []
    for i, e in enumerate(extra):
        if i in used_extra:
            continue
        if any(_near(e["x_mm"], e["y_mm"], *geo[ref][:2], geo[ref][2] + 1.5) for ref in gt_refs if ref in geo):
            continue
        false_extra.append(e)
    return {"gt": scored, "false_parts": [{"ref": r["ref"], "package": r["package"], "defects": r["defects"]}
                                          for r in false_parts],
            "false_board": false_extra, "n_parts": sum(r["status"] != "SKIP" for r in results)}


def _balanced_types(rng, n):
    types = [t for t in TYPES]
    out = []
    while len(out) < n:
        rng.shuffle(types)
        out += types
    return out[:n]


def run_bench(out_root: Path, tag: str, n_designs=4, n_golden=20, n_good=2, n_alt=2, n_bad=4, per_bad=6,
              seed=2026, inspect_over: dict | None = None, log=print) -> Path:
    run_dir = out_root / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}"
    run_dir.mkdir(parents=True)
    inspect_over = inspect_over or {}
    trng = np.random.default_rng(seed)
    type_queue = _balanced_types(trng, n_designs * n_bad * per_bad)
    boards, timings = [], {"train_s": [], "inspect_s": [], "render_s": []}
    for di in range(n_designs):
        design = make_design(np.random.default_rng(seed * 100 + di), f"d{di}")
        ddir = run_dir / design["name"]
        cfg = load_config(_write_design(design, ddir, inspect_over))
        comps = load_centroid(resolve(cfg, cfg["files"]["centroid"]), LIB, {})
        board = render_board(design)
        irng = np.random.default_rng(seed * 1000 + di)
        t = time.time()
        goldens = [render_instance(design, board, "A", {}, [], irng) for _ in range(n_golden)]
        timings["render_s"].append((time.time() - t) / n_golden)
        cv2.imwrite(str(ddir / "sample_golden.jpg"),
                    cv2.resize(goldens[0], None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA))
        t = time.time()
        train(goldens, comps, cfg, ddir / "_model", log=lambda *_: None)
        timings["train_s"].append(time.time() - t)
        meta, arrays = load_model(ddir / "_model")
        del goldens
        frame = BoardFrame(cfg)
        tests = [(f"good_A{i}", "A", []) for i in range(n_good)] + \
                [(f"good_B{i}", "B", []) for i in range(n_alt)] + \
                [(f"bad{i}", "AB"[i % 2], [type_queue.pop() for _ in range(per_bad)]) for i in range(n_bad)]
        log(f"[{design['name']}] {design['W']}x{design['H']} mm, {design['finish']}, {len(design['parts'])} parts, "
            f"{len(design['fiducials'])} fiducials, train {timings['train_s'][-1]:.1f} s")
        for name, batch, types in tests:
            ovs, foreign, gt = inject(design, types, irng) if types else ({}, [], [])
            raw = render_instance(design, board, batch, ovs, foreign, irng)
            t = time.time()
            bimg, info, results, extra = inspect_board(raw, comps, meta, arrays, cfg)
            timings["inspect_s"].append(time.time() - t)
            rdir = ddir / name
            write_report(rdir, bimg, info, results, extra, frame, f"{design['name']}/{name}", max_side=1800,
                         crop_refs={g["ref"] for g in gt if g["ref"]})
            sc = score_board(design, [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
                             extra, gt)
            (rdir / "ground_truth.json").write_text(json.dumps(sc, indent=1, default=str), encoding="utf-8")
            boards.append({"design": design["name"], "name": name, "batch": batch,
                           "kind": "bad" if types else f"good_{batch}", **sc,
                           "verdict": "FAIL" if any(r["status"] == "FAIL" for r in results) or extra else "PASS"})
            miss = [g["type"] + ":" + str(g["ref"]) for g in sc["gt"] if not g["detected"]]
            log(f"   {name:8s} batch {batch}  verdict {boards[-1]['verdict']}  false parts "
                f"{len(sc['false_parts'])}  false board {len(sc['false_board'])}" + (f"  MISSED {miss}" if miss else ""))
        shutil.rmtree(ddir / "_model")
    summary = aggregate(boards, timings)
    summary.update(tag=tag, run=run_dir.name, seed=seed, inspect_over=inspect_over,
                   config=dict(n_designs=n_designs, n_golden=n_golden, n_good=n_good, n_alt=n_alt, n_bad=n_bad,
                               per_bad=per_bad))
    (run_dir / "summary.json").write_text(json.dumps({"summary": summary, "boards": boards}, indent=1,
                                                     default=str), encoding="utf-8")
    write_index(run_dir, summary, boards)
    append_log(out_root.parent / "RESULTS.md", summary)
    return run_dir


def aggregate(boards: list, timings: dict) -> dict:
    per_type = defaultdict(lambda: {"n": 0, "detected": 0, "classified": 0})
    per_pkg_false = Counter()
    fc = defaultdict(lambda: {"boards": 0, "parts": 0, "false_parts": 0, "false_board": 0, "fail_verdict": 0})
    for b in boards:
        for g in b["gt"]:
            t = per_type[g["type"]]
            t["n"] += 1
            t["detected"] += g["detected"]
            t["classified"] += g["classified"]
        k = fc[b["kind"]]
        k["boards"] += 1
        k["parts"] += b["n_parts"]
        k["false_parts"] += len(b["false_parts"])
        k["false_board"] += len(b["false_board"])
        k["fail_verdict"] += b["verdict"] == "FAIL"
        for fp in b["false_parts"]:
            per_pkg_false[fp["package"]] += 1
    scored = {t: v for t, v in per_type.items() if t not in OCR_ONLY}
    n = sum(v["n"] for v in scored.values())
    det = sum(v["detected"] for v in scored.values())
    cls = sum(v["classified"] for v in scored.values())
    bad = [b for b in boards if b["kind"] == "bad"]
    good = [b for b in boards if b["kind"] != "bad"]
    return {
        "recall": round(det / max(n, 1), 4), "classification": round(cls / max(n, 1), 4), "n_defects": n,
        "per_type": {t: {**v, "recall": round(v["detected"] / max(v["n"], 1), 3),
                         "class_acc": round(v["classified"] / max(v["n"], 1), 3)} for t, v in sorted(per_type.items())},
        "false_calls": {k: {**v, "false_call_rate": round(v["false_parts"] / max(v["parts"], 1), 4)}
                        for k, v in sorted(fc.items())},
        "false_by_package": dict(per_pkg_false.most_common()),
        "good_board_fail_rate": round(sum(b["verdict"] == "FAIL" for b in good) / max(len(good), 1), 3),
        "bad_board_escape_rate": round(sum(b["verdict"] == "PASS" for b in bad) / max(len(bad), 1), 3),
        "timing": {k: round(float(np.mean(v)), 2) for k, v in timings.items() if v},
    }


def _pct(v):
    return f"{100 * v:.1f}%"


def write_index(run_dir: Path, s: dict, boards: list):
    rows_t = "".join(f"<tr><td>{t}{' (OCR)' if t in OCR_ONLY else ''}</td><td>{v['n']}</td><td>{_pct(v['recall'])}</td>"
                     f"<td>{_pct(v['class_acc'])}</td></tr>" for t, v in s["per_type"].items())
    rows_f = "".join(f"<tr><td>{k}</td><td>{v['boards']}</td><td>{v['parts']}</td><td>{v['false_parts']}</td>"
                     f"<td>{_pct(v['false_call_rate'])}</td><td>{v['false_board']}</td><td>{v['fail_verdict']}</td></tr>"
                     for k, v in s["false_calls"].items())
    rows_b, misses, falses = [], [], []
    for b in boards:
        link = f"{b['design']}/{b['name']}/report.html"
        nd = sum(g["detected"] for g in b["gt"])
        rows_b.append(f"<tr><td><a href='{link}'>{b['design']}/{b['name']}</a></td><td>{b['batch']}</td>"
                      f"<td>{b['verdict']}</td><td>{nd}/{len(b['gt'])}</td><td>{len(b['false_parts'])}</td>"
                      f"<td>{len(b['false_board'])}</td></tr>")
        for g in b["gt"]:
            if not g["detected"] or not g["classified"]:
                img = (f"<img src='{b['design']}/{b['name']}/crops/{g['ref']}.png' height=90>" if g["ref"] else "")
                misses.append(f"<tr><td><a href='{link}'>{b['design']}/{b['name']}</a></td><td>{g['type']}</td>"
                              f"<td>{g['ref'] or ''} {g.get('pkg', '')}</td><td>{'нет' if not g['detected'] else 'класс'}"
                              f"</td><td>{' '.join(g['codes'])}</td><td>{img}</td></tr>")
        for fp in b["false_parts"]:
            falses.append(f"<tr><td><a href='{link}'>{b['design']}/{b['name']}</a></td><td>{fp['ref']}</td>"
                          f"<td>{fp['package']}</td><td>{' '.join(fp['defects'])}</td>"
                          f"<td><img src='{b['design']}/{b['name']}/crops/{fp['ref']}.png' height=90></td></tr>")
        for e in b["false_board"]:
            falses.append(f"<tr><td><a href='{link}'>{b['design']}/{b['name']}</a></td><td>плата</td>"
                          f"<td>({e['x_mm']}, {e['y_mm']})</td><td>ANOMALY {e['area_mm2']} мм²</td><td></td></tr>")
    doc = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><title>AOI bench {s['run']}</title>
<style>body{{font:14px -apple-system,Segoe UI,sans-serif;margin:24px;background:#f4f1ea;color:#1c1915}}
table{{border-collapse:collapse;background:#fff;margin:8px 0 20px}}td,th{{border-bottom:1px solid #ddd;padding:5px 10px;text-align:left}}
.k{{display:inline-block;background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px 16px;margin:4px}}
.k b{{display:block;font-size:22px}}</style>
<h1>AOI bench: {html.escape(s['tag'])}</h1><p>{s['run']} · seed {s['seed']} · {s['config']} · overrides {s['inspect_over']}</p>
<div class=k><b>{_pct(s['recall'])}</b>обнаружение ({s['n_defects']} дефектов, без OCR)</div>
<div class=k><b>{_pct(s['classification'])}</b>верный класс</div>
<div class=k><b>{_pct(s['good_board_fail_rate'])}</b>хорошие платы с FAIL</div>
<div class=k><b>{_pct(s['bad_board_escape_rate'])}</b>брак прошёл как PASS</div>
<div class=k><b>{s['timing'].get('inspect_s')} с</b>проверка платы</div>
<h2>По типам дефектов</h2><table><tr><th>тип</th><th>n</th><th>обнаружено</th><th>класс верный</th></tr>{rows_t}</table>
<h2>Ложные срабатывания</h2><table><tr><th>платы</th><th>шт</th><th>позиций</th><th>ложных позиций</th><th>доля</th>
<th>ложных вне деталей</th><th>FAIL-вердиктов</th></tr>{rows_f}</table>
<p>Ложные по корпусам: {s['false_by_package']}</p>
<h2>Пропуски и ошибки класса</h2><table><tr><th>плата</th><th>тип</th><th>позиция</th><th>что</th><th>коды</th><th>кроп</th></tr>{''.join(misses)}</table>
<h2>Ложные срабатывания</h2><table><tr><th>плата</th><th>позиция</th><th>корпус</th><th>коды</th><th>кроп</th></tr>{''.join(falses)}</table>
<h2>Платы</h2><table><tr><th>плата</th><th>партия</th><th>вердикт</th><th>найдено</th><th>ложн. поз.</th><th>ложн. плата</th></tr>{''.join(rows_b)}</table>
</html>"""
    (run_dir / "index.html").write_text(doc, encoding="utf-8")


def append_log(path: Path, s: dict):
    fc = s["false_calls"]
    line = lambda k: (f"{fc[k]['false_parts']}/{fc[k]['parts']} ({_pct(fc[k]['false_call_rate'])}), "
                      f"вне деталей {fc[k]['false_board']}") if k in fc else "—"
    weak = ", ".join(f"{t} {_pct(v['recall'])}/{_pct(v['class_acc'])}" for t, v in s["per_type"].items()
                     if (v["recall"] < 1 or v["class_acc"] < 1) and t not in OCR_ONLY)
    block = (f"\n## {s['run']}\n\n"
             f"- Обнаружение {_pct(s['recall'])}, верный класс {_pct(s['classification'])} ({s['n_defects']} дефектов)\n"
             f"- Хорошие платы с FAIL {_pct(s['good_board_fail_rate'])}, брак как PASS {_pct(s['bad_board_escape_rate'])}\n"
             f"- Ложные позиции: партия A {line('good_A')}; партия B {line('good_B')}; на бракованных {line('bad')}\n"
             f"- Слабые типы (обнаружение/класс): {weak or 'нет'}\n"
             f"- OCR-only (маркировка): {_pct(s['per_type'].get('marking', {}).get('recall', 0))} обнаружено\n"
             f"- Время: {s['timing']}; overrides: {s['inspect_over'] or '—'}\n"
             f"- Отчёт: `bench/runs/{s['run']}/index.html`\n")
    if not path.exists():
        path.write_text("# Журнал прогонов AOI-стенда\n\nКаждый прогон: синтетические платы, golden партии A, "
                        "проверка хороших плат партий A и B и бракованных плат с размеченными дефектами.\n",
                        encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(block)
