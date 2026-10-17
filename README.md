# aoi-micro

**Ivan Iliukhin** — Applied AI / Computer Vision / Python

CAD-driven AOI prototype for a single top mono camera (IMX183 + Computar 12 mm; Basler acA5472-17um / Hikrobot MV-CE200-10UM backends). Classical OpenCV pipeline: fiducials, CAD windows, golden variation model, defect checks, PASS/FAIL reports. Optional YOLO is an experiment only — **not** claimed as line inspection or production deployment.

**Pitch:** industrial 2D AOI without deep learning when geometry, lighting, and a golden batch are under control — the same variation-model idea used in production tools (e.g. MVTec HALCON-style), implemented in OpenCV.

Camera notes live under `basler-aca5472-17um-report/` and `hikrobot-mv-ce200-10um-report/`.

## Pipeline

1. **Align** — Fiducials → board frame (fixed px/mm). Two fiducials → similarity; three → affine; ≥4 → homography. ECC refine vs mean golden; low-frequency illumination normalize.
2. **CAD windows** — Centroid file (KiCad `.pos`, Altium Pick&Place, CSV) + package library → one window per placement. Window center shifted for package-height **parallax**: `r·h/(WD−h)`.
3. **Golden model** — From ~20–30 known-good boards: per-position mean/σ, NCC and photometric MAD distributions. Thresholds from statistics, not hand-tuned constants.
4. **Inspect** — NCC template search (±0.8 mm, ±20°) for shift/rotation; body zones (inner median tone vs border MAD); polarity mark search; shift vs tolerance; solder/foreign via golden envelope (σ) and lead-zone `joint_frac`; board exterior variation model for debris/bridges/opens.
5. **Report** — PASS/FAIL per position; live | golden | deviation crops; CSV/JSON + HTML.

Presence needs **NCC and border MAD and inner tone** — NCC alone is contrast-invariant and can match empty pads. See [docs/INTERVIEW.md](docs/INTERVIEW.md).

## Synthetic bench metrics

Label: **synthetic bench** (procedural PCBs). Do not treat as line KPIs.

Source: `bench/RESULTS.md` (v6, 2026-10-07) — 4 designs, 89 labeled defects, ~2000 inspect positions; holdout = seed 7 (unseen designs).

| Metric | v0 baseline | v6 | v6 holdout |
|---|---|---|---|
| Defect detection | 76.4% | **93.3%** | **93.3%** |
| Correct defect class | 59.6% | **87.6%** | **84.3%** |
| Bad board as PASS | 0% | 0% | 0% |
| False positions (good A) | 0.8% | 0.6% | 1.4% |
| False positions (good B, new batch) | 61.1% | 0.8% | 1.2% |

Weak spots on the same bench (from README / RESULTS): chip chips on bright leads, same-package IC swap (needs OCR), SOT-23 class confusion, excess solder often labeled MISMATCH/MISSING though detected; false positions ~1% vs a line goal of &lt;0.2%.

Optional NN experiment (`aoi/nn/`, `nn_results/det_v0/`): synth mAP50 0.93; public photos mAP50 0.38 — second opinion / research only.

## Quick start (no hardware)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m aoi optics --wd 100 150 200
.venv/bin/python -m aoi synth --out demo
.venv/bin/python -m aoi train   -c demo/config.json --images demo/golden --golden demo/golden_model
.venv/bin/python -m aoi inspect -c demo/config.json --images demo/test --golden demo/golden_model --out demo/reports
.venv/bin/python -m unittest tests.test_pipeline
```

Windows: use `.venv\Scripts\` instead of `.venv/bin/`.

### Synthetic bench

```bash
.venv/bin/python -m aoi bench --tag my_change                 # ~2.5 min: 4 designs, 89 defects
.venv/bin/python -m aoi bench --tag holdout --seed 7
.venv/bin/python -m aoi.bench.analyze bench/runs/<run>
```

Each run writes `bench/runs/<time>_<tag>/index.html` and appends a row to `bench/RESULTS.md`.

### On a stand (summary)

1. Fixed exposure, locked focus/iris (f/4); ring/dome light — coax fails on a wide FOV with steep angles.
2. Copy `config.example.json` → `config.json`; set `camera.backend` to `basler` (`pypylon`) or `hikrobot` (MVS SDK + `mvs_import_path`).
3. `python -m aoi capture` → `fiducials` → fill `approx_px`; check CAD frames on first golden (`overview.jpg`).
4. Capture 20–30 good boards → `train` → `python -m aoi run -c config.json`.

ChArUco calibration: `calib-target` → capture → `calibrate` → `camera.npz` / `camera.json` (see original flow in git history / comments in CLI). Synthetic calib check: `tests/test_calib.py`.

## Limitations

From existing project boundaries:

- Camera backends follow pylon/MVS APIs but are **not verified on hardware**.
- `edge_tol_mm` must sit above placement scatter (~2–3σ) and below half the lead pitch; at ~23 µm/px (WD 100) a 0.8 mm-pitch bridge is catchable; at ~45 µm/px it is not.
- **Not covered:** fillet volume, lifted lead, under-package BGA/QFN, color, OCR (next stage).
- Bench weaknesses: chips on bright leads (needs oblique light), same-footprint IC swap (OCR only), SOT-23 / excess class labels; ~1% false positions vs &lt;0.2% line target.

## What I would do next

- Oblique / multi-angle lighting for top-face chips; OCR for marking / wrong-IC.
- Drive false positions toward &lt;0.2% (more golden, batch-B σ adaptation).
- Fine-tune or replace the optional detector on real camera frames (public mAP is weak on tiny R/C).
- Hardware bring-up of Basler/Hikrobot backends; bridge CAD/golden ideas into warehouse CV ([sklad-digital-twin](../sklad-digital-twin)) — shared optics/camera discipline, different domain.

## Docs

- [Interview notes](docs/INTERVIEW.md) — classical CV vs YOLO, NCC/MAD, polarity, parallax, bench, portfolio bridge.
- Suggested commit message (manual): `docs: English README and interview notes for AOI portfolio`
