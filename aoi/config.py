from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULTS = {
    "camera": {
        "backend": "file",
        "exposure_us": 3000,
        "gain_db": 0.0,
        "average_frames": 1,
        "timeout_ms": 3000,
        "mvs_import_path": None,
    },
    "optics": {
        "focal_mm": 12.0,
        "pixel_um": 2.4,
        "sensor_px": [5472, 3648],
        "wd_mm": 100.0,
        "undistort_npz": None,
    },
    "board": {
        "width_mm": 100.0,
        "height_mm": 80.0,
        "cad_y_up": True,
        "cad_rot_ccw": True,
        "px_per_mm": 40.0,
        "fiducials": [],
        "fiducial_diameter_mm": 1.0,
        "fiducial_search_mm": 4.0,
        "raw_px_per_mm": 43.4,
    },
    "files": {
        "centroid": "centroid.csv",
        "bom": None,
        "packages": None,
    },
    "inspect": {
        "search_mm": 0.8,
        "rot_search_deg": 20,
        "rot_step_deg": 2,
        "shift_tol_frac": 0.35,
        "rot_tol_deg": 6,
        "ncc_floor": 0.35,
        "ncc_present": 0.6,
        "ncc_sigma_k": 6.0,
        "mad_sigma_k": 6.0,
        "mad_min_delta": 6.0,
        "polarity_margin": 0.1,
        "mark_min": 0.4,
        "mark_search_mm": 0.12,
        "int_dev_min": 25.0,
        "int_text_margin": 35.0,
        "int_open_mm": 0.18,
        "joint_k": 2.5,
        "joint_tol_mm": 0.1,
        "joint_frac": 0.05,
        "shift_tol_max_mm": 0.5,
        "body_tol_abs": 14.0,
        "body_tol_rel": 0.15,
        "body_edge_tol_px": 1,
        "var_k": 5.0,
        "var_sigma_floor": 6.0,
        "edge_tol_mm": 0.1,
        "anomaly_min_area_mm2": 0.03,
        "board_var_k": 4.0,
        "board_sigma_floor": 3.0,
        "board_anomaly_min_area_mm2": 0.02,
        "board_edge_mm": 0.6,
        "ecc_scale": 0.25,
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = _merge(DEFAULTS, json.loads(path.read_text(encoding="utf-8")))
    cfg["_dir"] = str(path.parent.resolve())
    return cfg


def resolve(cfg: dict, rel: str | None) -> Path | None:
    if not rel:
        return None
    p = Path(rel)
    return p if p.is_absolute() else Path(cfg["_dir"]) / p
