# Interview notes — aoi-micro

Short talking points for Applied AI / CV / Python interviews. Owner: **Ivan Iliukhin**.

This repo is a **CAD-driven classical AOI prototype**, not a deployed production line and not “NN inspection as the product.”

---

## Classical CV vs YOLO for AOI

| | Classical (this repo) | YOLO-style detector |
|---|---|---|
| Prior | CAD placement + package library + golden batch | Learned appearance from labeled boxes |
| Output | Per-refdes PASS/FAIL with measurable shift/polarity/solder cues | Class + box; weak on “is the solder joint OK?” without more heads |
| Batch change | Explicit zone model (inner vs border) | Needs retrain / domain adaptation |
| Empty pad | Photometric MAD + tone break NCC ambiguity | Needs “footprint / empty” class and enough empties in data |
| When it wins | Controlled optics, known BOM, golden samples available | Dense clutter, no CAD, open-world part types |

**Positioning:** use classical variation models when the problem is metrology + defect taxonomy under fixed lighting; use detectors as a **second opinion** (empty seat / wrong family) or when CAD is missing. The optional `aoi/nn/` path is an experiment (synth mAP50 strong, public photos weak) — not claimed as line inspection.

---

## Why NCC alone is insufficient

Normalized cross-correlation is **contrast-invariant**. Bare pads with similar gap geometry can score like a present chip. Presence in this pipeline requires:

1. NCC match in the search window, **and**
2. Border **MAD** after brightness-shift compensation (leads / edges), **and**
3. Inner-zone median tone within golden tolerance (body, not marking detail).

Zones exist so a new component batch (different hue, marking, date code) does not flood **MISMATCH** on the body while leads still look wrong when the part is missing or shifted.

---

## MAD on borders (leads / edges)

- **Inner zone:** marking and batch tone → median with absolute/relative tolerance (ignore ink/date-code variation).
- **Border / frame:** leads and package outline → mean absolute difference vs golden after local brightness compensation.

That split is what dropped false positions on synthetic **batch B** from ~61% (v0) to ~1% (v6) in `bench/RESULTS.md` — still labeled **synthetic bench**.

---

## Polarity

Golden polarity mark (pin 1, diode band, electrolytic “−”) is searched in 4 orientations for square packages or 2 for polarized two-sided parts, near the registered and nominal pose. Fallback: whole template flipped. Wrong orientation → polarity / rot90-style fail rather than a vague mismatch.

---

## Golden model and batch variation

Train on 20–30 known-good boards of batch A: per-pixel mean/σ in package coordinates, plus distributions of NCC and body MAD → thresholds from those stats.

Batch B (different body shades, markings, no ejector marks on the bench) stresses the model. Without zones, good B boards look “bad”; with zones, false rate collapses on the same synthetic protocol. Remaining line-style goal called out in docs: false positions &lt;0.2% (bench still ~0.6–1.4%).

---

## Parallax and working distance

Tall parts off the optical axis shift in the image: window center offset ≈ `r·h/(WD−h)` (`optics` / `parallax_px`). Working distance and px/mm must match the locked lens setup; wrong WD misplaces CAD windows and looks like systematic shift. Coaxial light fails on wide FOV with steep chief-ray angles — ring/dome preferred (project boundary notes).

---

## Bench evaluation (honest framing)

`aoi bench` builds procedural boards (mask, traces, packages, parallax, defocus, vignette, pose, sensor noise) with labeled defect types. Metrics in README / `bench/RESULTS.md` are **synthetic bench** only:

- Detection / class accuracy, bad-as-PASS, false positions on good A/B, holdout seed for overfitting check.
- Analyze misses and false alarms with crops (`aoi.bench.analyze`).

Do not quote these as factory OEE or customer SLA.

---

## Bridge to sklad-digital-twin

Same owner stack: industrial camera discipline, calibration mindset, Python CV services.

| aoi-micro | sklad-digital-twin |
|---|---|
| CAD + golden → PCB PASS/FAIL | Camera → warehouse cells / object state |
| Metrology windows, mm tolerances | Spatial binding to layout / history |
| Classical AOI first | System + detection product scope |

Talking point: “AOI taught me fiducials, variation models, and why photometry beats pure template scores; the warehouse twin reuses camera/backend habits in a logistics digital twin.”

---

## 30-second recruiter checklist

1. **One line:** CAD-driven classical AOI (OpenCV), optional YOLO experiment only.
2. **Pipeline:** align → CAD windows → golden model → inspect → report.
3. **Insight:** NCC alone fails on empty pads; border MAD + zones handle new batches.
4. **Evidence:** synthetic bench v6 ~93% detection / ~88% class; 0% bad-as-PASS; cite `bench/RESULTS.md`.
5. **Honesty:** not production-deployed; camera backends unproven on iron; no fillet/BGA/OCR yet.
