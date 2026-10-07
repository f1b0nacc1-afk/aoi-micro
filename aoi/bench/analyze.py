"""Per-run error breakdown: misclassified defects and false calls with their decision metrics.

    python -m aoi.bench.analyze bench/runs/<run>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

KEYS = ("package", "defects", "ncc", "ncc_thr", "mad", "mad_thr", "interior", "interior_ref",
        "dx_mm", "dy_mm", "dtheta_deg", "anomaly_px", "joint", "mark_ncc", "ncc_180")


def analyze(run: Path, out=print):
    s = json.loads((run / "summary.json").read_text())["summary"]
    out(f"recall {s['recall']:.3f}  class {s['classification']:.3f}  good FAIL {s['good_board_fail_rate']:.2f}")
    for k, v in s["false_calls"].items():
        out(f"  {k:7s} false parts {v['false_parts']:3d}/{v['parts']}  board {v['false_board']}")
    codes = Counter()
    for f in sorted(run.glob("d*/*/ground_truth.json")):
        b = f.parent
        gt = json.loads(f.read_text())
        parts = {p["ref"]: p for p in json.loads((b / "result.json").read_text())["parts"]}
        tag = f"{b.parent.name}/{b.name}"
        for fp in gt["false_parts"]:
            p = parts[fp["ref"]]
            codes.update(p["defects"])
            out(f"FALSE {tag:9s} {fp['ref']:5s} " + json.dumps({k: p.get(k) for k in KEYS if k in p}))
        for e in gt["false_board"]:
            out(f"FBOARD {tag:9s} ({e['x_mm']}, {e['y_mm']}) {e['area_mm2']} mm2")
        for g in gt["gt"]:
            if not g.get("classified"):
                p = parts.get(g.get("ref"), {})
                out(f"MISS  {tag:9s} {g['type']:9s} {str(g.get('ref')):5s} got {g.get('codes')} "
                    + json.dumps({k: p.get(k) for k in KEYS if k in p and k != 'defects'}))
    out(f"false-call codes: {dict(codes)}")


if __name__ == "__main__":
    analyze(Path(sys.argv[1]))
