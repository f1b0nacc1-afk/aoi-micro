from __future__ import annotations

import csv
import html
import json
import time
from pathlib import Path

import cv2
import numpy as np

COLORS = {"PASS": (60, 170, 60), "FAIL": (40, 40, 220), "SKIP": (160, 160, 160)}
CSV_FIELDS = ["ref", "status", "defects", "value", "package", "mpn", "ncc", "ncc_thr", "ncc_180", "mad", "mad_thr", "mad_180",
              "mark_ncc", "mark_ncc_180", "dx_mm", "dy_mm", "dtheta_deg", "anomaly_px"]


def _crop_panel(r: dict, scale_to: int = 160) -> np.ndarray:
    im = r["_img"]
    live = cv2.cvtColor(im["live"], cv2.COLOR_GRAY2BGR)
    gold = cv2.cvtColor(im["golden"], cv2.COLOR_GRAY2BGR)
    over = live.copy()
    over[im["amask"] > 0] = (0, 0, 255)
    for x, y, w, h, _ in im["blobs"]:
        cv2.rectangle(over, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 255), 1)
    if im["marks"]:
        (x, y, w, h), (ox, oy, ow, oh) = im["marks"]
        cv2.rectangle(gold, (x, y), (x + w, y + h), (0, 200, 255), 1)
        cv2.rectangle(gold, (ox, oy), (ox + ow, oy + oh), (255, 160, 0), 1)
    panel = np.hstack([live, np.full((live.shape[0], 4, 3), 255, np.uint8), gold,
                       np.full((live.shape[0], 4, 3), 255, np.uint8), over])
    k = scale_to / panel.shape[0]
    if k > 1:
        panel = cv2.resize(panel, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
    return panel


def write_report(out: Path, board: np.ndarray, info: dict, results: list, extra: list,
                 frame, image_name: str, max_side: int = 2400, crop_refs: set | None = None) -> dict:
    """crop_refs: if set, crops are written only for FAIL parts and these refs."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "crops").mkdir(exist_ok=True)
    k = min(1.0, max_side / max(board.shape))
    ov = cv2.cvtColor(cv2.resize(board, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    for r in results:
        if "_geom" not in r:
            continue
        g = r["_geom"]
        box = cv2.boxPoints(((g.center[0] * k, g.center[1] * k), (g.region[0] * k, g.region[1] * k), -g.ccw))
        cv2.polylines(ov, [np.int32(np.round(box))], True, COLORS[r["status"]], 2 if r["status"] == "FAIL" else 1)
        if r["status"] == "FAIL":
            cv2.putText(ov, r["ref"], (int(g.center[0] * k) + 6, int(g.center[1] * k) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["FAIL"], 2)
        if crop_refs is None or r["status"] == "FAIL" or r["ref"] in crop_refs:
            cv2.imwrite(str(out / "crops" / f"{r['ref']}.png"), _crop_panel(r))
            r["_crop"] = True
    for e in extra:
        x, y, w, h = [v * k for v in e["_box_px"]]
        cv2.rectangle(ov, (int(x) - 4, int(y) - 4), (int(x + w) + 4, int(y + h) + 4), (0, 140, 255), 2)
    cv2.imwrite(str(out / "overview.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 90])

    cropped = {r["ref"] for r in results if r.get("_crop")}
    clean = [{k2: v for k2, v in r.items() if not k2.startswith("_")} for r in results]
    extra_clean = [{k2: v for k2, v in e.items() if not k2.startswith("_")} for e in extra]
    n_fail = sum(r["status"] == "FAIL" for r in clean) + len(extra_clean)
    summary = {"image": image_name, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
               "verdict": "FAIL" if n_fail else "PASS", "n_parts": len(clean),
               "n_fail_parts": sum(r["status"] == "FAIL" for r in clean),
               "n_board_anomalies": len(extra_clean),
               "fid_residual_mm": info.get("fid_residual_mm"), "ecc_cc": info.get("ecc_cc")}
    (out / "result.json").write_text(json.dumps({"summary": summary, "parts": clean,
                                                 "board_anomalies": extra_clean}, indent=2,
                                                ensure_ascii=False), encoding="utf-8")
    with (out / "result.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in clean:
            w.writerow({**r, "defects": " ".join(r["defects"])})

    order = sorted(clean, key=lambda r: (r["status"] != "FAIL", r["ref"]))
    rows = []
    for r in order:
        img = f'<img src="crops/{html.escape(r["ref"])}.png">' if r["ref"] in cropped else ""
        metrics = " · ".join(f"{m} {r[m]}" for m in ("ncc", "ncc_thr", "ncc_180", "mad", "mad_thr", "mad_180", "mark_ncc", "mark_ncc_180",
                                                       "dx_mm", "dy_mm", "dtheta_deg", "anomaly_px") if m in r)
        rows.append(f'<tr class="{r["status"].lower()}"><td><b>{html.escape(r["ref"])}</b><br>'
                    f'<small>{html.escape(str(r["value"]))} · {html.escape(r["package"])}'
                    f'{" · " + html.escape(r["mpn"]) if r.get("mpn") else ""}</small></td>'
                    f'<td>{r["status"]}<br><small>{" ".join(r["defects"])}</small></td>'
                    f'<td>{img}<br><small>{metrics}</small></td></tr>')
    extra_rows = "".join(f"<li>x {e['x_mm']} мм, y {e['y_mm']} мм (кадр платы), {e['w_mm']}×{e['h_mm']} мм,"
                         f" {e['area_mm2']} мм²</li>" for e in extra_clean)
    doc = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><title>AOI {html.escape(image_name)}</title>
<style>body{{font:14px -apple-system,Segoe UI,sans-serif;margin:24px;color:#1c1915;background:#f4f1ea}}
table{{border-collapse:collapse;width:100%;background:#fff}}td{{border-bottom:1px solid #ddd;padding:6px 10px;vertical-align:top}}
tr.fail td:nth-child(2){{color:#b01e1e;font-weight:600}}tr.pass td:nth-child(2){{color:#1f5c4a}}
img{{max-width:100%;image-rendering:pixelated}}.v{{font-size:28px;font-weight:700}}
.FAIL{{color:#b01e1e}}.PASS{{color:#1f5c4a}}small{{color:#666}}</style>
<h1>AOI: {html.escape(image_name)}</h1>
<p class="v {summary['verdict']}">{summary['verdict']}</p>
<p>{summary['time']} · позиций {summary['n_parts']} · FAIL {summary['n_fail_parts']} ·
аномалий вне деталей {summary['n_board_anomalies']} · невязка фидуциалов {summary['fid_residual_mm']} мм ·
ECC {summary['ecc_cc'] and round(summary['ecc_cc'], 4)}</p>
<img src="overview.jpg" style="max-width:1200px">
{"<h2>Аномалии вне деталей</h2><ul>" + extra_rows + "</ul>" if extra_rows else ""}
<h2>Позиции</h2><p><small>Кроп: живая плата (выровнена) | эталон (жёлтое — метка полярности,
синее — место метки при 180°) | карта отклонений σ.<br>
MISSING — корпус не найден; MISMATCH — на месте не то, что в эталоне (голые площадки, другая деталь,
повреждение); POLARITY — развёрнута на 180° / метка не там; SHIFT / ROTATED — вне допуска;
ANOMALY — локальное отклонение от golden (мост, шарик, лишний припой, посторонний предмет).</small></p>
<table>{"".join(rows)}</table></html>"""
    (out / "report.html").write_text(doc, encoding="utf-8")
    return summary
