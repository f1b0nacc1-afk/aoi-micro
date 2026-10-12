"""Camera calibration with a ChArUco target: intrinsics + distortion (undistort_npz), WD and px/mm.

The target is printed at 100 % scale; measure the real square size with a caliper and pass it
as square_mm. 15-25 views across the whole field, small tilts (the 12 mm lens at f/4 has only a
few mm of depth of field at WD 100-150 mm), the last view flat on the inspection plane.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

DICT = cv2.aruco.DICT_5X5_250


def make_board(squares=(12, 8), square_mm=8.0, marker_mm=6.0):
    return cv2.aruco.CharucoBoard(tuple(squares), float(square_mm), float(marker_mm),
                                  cv2.aruco.getPredefinedDictionary(DICT))


def target_image(board, square_mm: float, dpi: int = 600, margin_mm: float = 5.0) -> np.ndarray:
    sx, sy = board.getChessboardSize()
    px_mm = dpi / 25.4
    w, h = int(round(sx * square_mm * px_mm)), int(round(sy * square_mm * px_mm))
    m = int(round(margin_mm * px_mm))
    return board.generateImage((w + 2 * m, h + 2 * m), marginSize=m, borderBits=1)


def detect(img: np.ndarray, board, min_corners: int = 8):
    """ChArUco corners -> (object points mm, image points px) or None."""
    det = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = det.detectBoard(img)
    if ids is None or len(ids) < min_corners:
        return None
    obj, imgp = board.matchImagePoints(corners, ids)
    return obj.reshape(-1, 3).astype(np.float32), imgp.reshape(-1, 2).astype(np.float32)


def calibrate(images: list, board, k3: bool = False, log=print) -> dict:
    obj_pts, img_pts, names = [], [], []
    size = None
    for name, img in images:
        size = img.shape[1], img.shape[0]
        d = detect(img, board)
        if d is None:
            log(f"  {name}: target not found")
            continue
        obj_pts.append(d[0])
        img_pts.append(d[1])
        names.append(name)
        log(f"  {name}: {len(d[0])} corners")
    if len(obj_pts) < 4:
        raise RuntimeError(f"need >= 4 usable views, got {len(obj_pts)}")
    flags = 0 if k3 else cv2.CALIB_FIX_K3
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None, flags=flags)
    views = []
    for n, o, i, r, t in zip(names, obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, D)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i) ** 2, axis=1))))
        R, _ = cv2.Rodrigues(r)
        tilt = float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), 0, 1))))
        n_cam = R[:, 2]
        dist = float(abs(n_cam @ t.ravel()))
        views.append({"name": n, "rms_px": round(err, 3), "tilt_deg": round(tilt, 2),
                      "plane_dist_mm": round(dist, 2), "corners": len(o)})
    flat = min(views, key=lambda v: v["tilt_deg"])
    f_px = float((K[0, 0] + K[1, 1]) / 2)
    return {"K": K, "D": D, "rms_px": float(rms), "image_size": size, "views": views,
            "flat_view": flat["name"], "wd_mm": flat["plane_dist_mm"],
            "raw_px_per_mm": f_px / flat["plane_dist_mm"], "f_px": f_px}


def save(res: dict, out: Path, pixel_um: float | None = None) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, K=res["K"], D=res["D"], image_size=np.array(res["image_size"]), rms=res["rms_px"])
    rep = {k: v for k, v in res.items() if k not in ("K", "D")}
    rep["K"] = res["K"].round(3).tolist()
    rep["D"] = res["D"].ravel().round(6).tolist()
    if pixel_um:
        rep["focal_mm_est"] = round(res["f_px"] * pixel_um / 1000, 3)
    rep["config_hint"] = {"optics": {"undistort_npz": out.name, "wd_mm": round(res["wd_mm"], 1)},
                          "board": {"raw_px_per_mm": round(res["raw_px_per_mm"], 2)}}
    out.with_suffix(".json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return rep
