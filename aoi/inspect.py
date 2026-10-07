from __future__ import annotations

import cv2
import numpy as np

from .align import BoardFrame, local_patch
from .model import (angle_list, center_crop, component_mask, crop, is_square, mark_rects, mark_rects_k,
                    match_part, ncc, offset_mad, part_geometry, prepare_board, zones)


def _blobs(z: np.ndarray, k: float, min_area_px: float, valid: np.ndarray | None = None,
           open_px: int = 3):
    m = (z > k).astype(np.uint8)
    if valid is not None:
        m &= valid
    if open_px > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = [(int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]),
            int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, n)
           if stats[i, cv2.CC_STAT_AREA] >= min_area_px]
    return out, m


def envelope_z(live, mean, std, tol_px: float, floor: float) -> np.ndarray:
    """Deviation outside the golden min/max envelope over a tol_px neighbourhood: pads and body
    edges may move independently by up to tol_px without firing."""
    r = max(0, int(round(tol_px)))
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    if r:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        hi, lo = cv2.dilate(mean, k), cv2.erode(mean, k)
    else:
        hi = lo = mean
    live = live.astype(np.float32)
    return np.maximum(np.maximum(live - hi, lo - live), 0) / (std + floor)


def mark_scores(live, region0, rects, ks, r_px):
    """Best NCC of the golden mark, turned by k quarter turns, near its k-th position."""
    ref = crop(region0, rects[0]).astype(np.float32)
    out = {}
    for k in ks:
        t = np.ascontiguousarray(np.rot90(ref, k))
        x, y, w, h = rects[k]
        win = live[max(0, y - r_px):y + h + r_px, max(0, x - r_px):x + w + r_px].astype(np.float32)
        if win.shape[0] < t.shape[0] or win.shape[1] < t.shape[1] or t.std() < 1:
            out[k] = 0.0
            continue
        out[k] = float(cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED).max())
    return out


def interior_blobs(live, golden, interior, ic, min_area_px, open_px):
    """Local defects on the body top (chips, cracks, scratches) that the median test cannot see.
    Marking text changes between batches, so thin strokes are tolerated unless far brighter or
    darker than the golden text contrast."""
    if not interior.any():
        return [], np.zeros(live.shape, np.uint8)
    g = golden.astype(np.float32)
    lv = live.astype(np.float32) - (np.median(live[interior]) - np.median(golden[interior]))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dev = np.maximum(np.maximum(lv - cv2.dilate(g, k), cv2.erode(g, k) - lv), 0)
    dev[~interior] = 0
    gi = g[interior]
    text = float(np.percentile(gi, 99.5) - np.median(gi))
    hi = max(2 * ic["int_dev_min"], text + ic["int_text_margin"])
    thick = cv2.morphologyEx((dev > ic["int_dev_min"]).astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((open_px, open_px), np.uint8))
    m = thick | (dev > hi).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    blobs = [tuple(int(v) for v in st[i, :5]) for i in range(1, n) if st[i, 4] >= min_area_px]
    return blobs, m


def inspect_part(c, board, meta, arrays, frame, cfg, angles):
    angles, fine = angles
    ic = cfg["inspect"]
    s = frame.s
    pkg = c.package
    pm = meta["parts"][c.ref]
    g = part_geometry(c, frame, tuple(meta["principal_px"]), cfg)
    region0 = arrays[f"{c.ref}__mean"]
    std0 = arrays[f"{c.ref}__std"].astype(np.float32)
    zn = zones(c, region0.shape, s)
    patch = local_patch(board, g.center, g.ccw, g.patch)

    def evaluate(region_t, wide=False):
        m = match_part(patch, center_crop(region_t, g.body), g.region, angles, fine,
                       None if wide else g.search)
        m["mad"] = offset_mad(m["registered"], region_t, zn["frame"])
        m["present"] = m["ncc"] >= pm["ncc_thr"] and m["mad"] <= pm["mad_thr"]
        return m

    flip = lambda a: np.ascontiguousarray(a[::-1, ::-1])
    shift_tol = min(ic["shift_tol_frac"] * min(pkg.body), ic["shift_tol_max_mm"])
    is_shifted = lambda m: max(abs(m["dx_px"]), abs(m["dy_px"])) / s > shift_tol
    m0 = evaluate(region0)
    use, region_t, std_t = m0, region0, std0
    r = {"ref": c.ref, "value": c.value, "package": pkg.name, "mpn": c.mpn,
         "ncc": round(m0["ncc"], 3), "ncc_thr": round(pm["ncc_thr"], 3),
         "mad": round(m0["mad"], 1), "mad_thr": round(pm["mad_thr"], 1), "defects": []}

    if pkg.orient == "mark" and pkg.mark:
        rects = mark_rects_k(c, s, region0.shape)
        ks = (0, 1, 2, 3) if is_square(pkg) else (0, 2)
        r_px = int(round(ic["mark_search_mm"] * s))
        sc = mark_scores(m0["registered"], region0, rects, ks, r_px)
        nominal = center_crop(patch, region0.shape[::-1])
        sc = {k: max(v, mark_scores(nominal, region0, rects, (k,), r_px)[k]) for k, v in sc.items()}
        r["mark_ncc"] = {str(k * 90): round(v, 3) for k, v in sc.items()}
        kb = max(sc, key=sc.get)
        if kb and sc[kb] >= ic["mark_min"] and sc[kb] - sc[0] > ic["polarity_margin"]:
            r["defects"].append("POLARITY")
            r["orientation_deg"] = kb * 90
        elif sc[0] < ic["mark_min"] or is_shifted(m0) or not m0["present"]:
            m180 = evaluate(flip(region0))
            r["ncc_180"] = round(m180["ncc"], 3)
            if m180["ncc"] >= pm["ncc_thr"] and m180["ncc"] - m0["ncc"] > ic["polarity_margin"]:
                use, region_t, std_t = m180, flip(region0), flip(std0)
                r["defects"].append("POLARITY")
                r["orientation_deg"] = 180
    elif pkg.orient == "template":
        m180 = evaluate(flip(region0))
        r["ncc_180"], r["mad_180"] = round(m180["ncc"], 3), round(m180["mad"], 1)
        if m180["present"] and (not m0["present"] or m180["ncc"] - m0["ncc"] > ic["polarity_margin"]):
            use, region_t, std_t = m180, flip(region0), flip(std0)
            r["defects"].append("POLARITY")

    polarity = "POLARITY" in r["defects"]
    if not polarity and not m0["present"]:
        mw = evaluate(region0, wide=True)
        if mw["ncc"] > m0["ncc"] + 0.05:
            use = mw
            r["wide_search"] = True

    live = use["registered"]
    if zn["interior"].any():
        med = float(np.median(live[zn["interior"]]))
        tol = ic["body_tol_abs"] + ic["body_tol_rel"] * pm["interior_med"]
        r["interior"], r["interior_ref"] = round(med, 1), round(pm["interior_med"], 1)
        interior_ok = abs(med - pm["interior_med"]) <= tol
    else:
        interior_ok = True

    floor = ic["var_sigma_floor"]
    min_px = ic["anomaly_min_area_mm2"] * s * s
    z_body = envelope_z(live, region_t, std_t, ic["body_edge_tol_px"], floor)
    z_out = envelope_z(live, region_t, std_t, ic["edge_tol_mm"] * s, floor)
    z = np.where(zn["body"], z_body, z_out)
    z[zn["interior"] | zn["mark"]] = 0
    blobs, amask = _blobs(z, ic["var_k"], min_px, open_px=1)
    iblobs, imask = interior_blobs(live, region_t, zn["interior"], ic, min_px,
                                   max(3, int(round(ic["int_open_mm"] * s))))
    blobs, amask = blobs + iblobs, amask | imask
    r["anomaly_px"] = int(sum(b[4] for b in blobs))
    joint = zn["joint"]
    if joint.any():
        tol = max(ic["edge_tol_mm"], ic["joint_tol_mm"]) * s
        zj = z_out if tol == ic["edge_tol_mm"] * s else envelope_z(live, region_t, std_t, tol, floor)
        r["joint"] = round(float((zj[joint] > ic["joint_k"]).mean()), 3)
    else:
        r["joint"] = 0.0
    local = bool(blobs) or r["joint"] > ic["joint_frac"]

    r["dx_mm"], r["dy_mm"] = round(use["dx_px"] / s, 3), round(use["dy_px"] / s, 3)
    r["dtheta_deg"] = use["dtheta"]
    if not polarity:
        shifted = is_shifted(use)
        rotated = abs(use["dtheta"]) > ic["rot_tol_deg"]
        located = use["ncc"] >= pm["ncc_thr"]
        if not interior_ok:
            r["defects"].append("MISMATCH")
        elif located and (shifted or rotated):
            r["defects"] += ["SHIFT"] * shifted + ["ROTATED"] * rotated
        elif not use["present"]:
            moderate = located and use["mad"] <= 2 * pm["mad_thr"]
            r["defects"].append("ANOMALY" if moderate and local else "MISSING")
    if local and not r["defects"]:
        r["defects"].append("ANOMALY")
    r["status"] = "FAIL" if r["defects"] else "PASS"
    r["_img"] = {"live": live, "golden": region0, "amask": amask, "blobs": blobs,
                 "marks": mark_rects(c, s, region0.shape) if pkg.mark else None}
    r["_geom"] = g
    return r


def board_anomalies(board, arrays, comps, meta, frame, cfg, scale=1.0):
    """Variation model outside component footprints: foreign objects, bare-board shorts/opens."""
    ic = cfg["inspect"]
    mask = component_mask(comps, frame, tuple(meta["principal_px"]), cfg)
    sm = lambda im: cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mean = sm(arrays["mean_board"]).astype(np.float32)
    std = sm(arrays["std_board"].astype(np.float32))
    z = np.abs(sm(board).astype(np.float32) - mean) / (std + ic["board_sigma_floor"])
    valid = ((sm(mask) == 0) & (sm(board) > 0)).astype(np.uint8)
    ss = frame.s * scale
    e = int(round(ic["board_edge_mm"] * ss))
    if e:
        valid[:e], valid[-e:], valid[:, :e], valid[:, -e:] = 0, 0, 0, 0
    blobs, _ = _blobs(z, ic["board_var_k"], ic["board_anomaly_min_area_mm2"] * ss * ss, valid, open_px=1)
    out = []
    for x, y, w, h, a in blobs:
        cx, cy = (x + w / 2) / ss, (y + h / 2) / ss
        out.append({"x_mm": round(cx, 2), "y_mm": round(frame.h_mm - cy if frame.y_up else cy, 2),
                    "w_mm": round(w / ss, 2), "h_mm": round(h / ss, 2), "area_mm2": round(a / ss ** 2, 3),
                    "_box_px": (x / scale, y / scale, w / scale, h / scale)})
    return out


def inspect_board(raw, comps, meta, arrays, cfg):
    frame = BoardFrame(cfg)
    board, info = prepare_board(raw, cfg, frame, arrays["mean_board"])
    angles = angle_list(cfg)
    results = []
    for c in comps:
        if c.package is None or c.ref not in meta["parts"]:
            results.append({"ref": c.ref, "value": c.value, "package": c.footprint, "mpn": c.mpn,
                            "status": "SKIP", "defects": ["NO_MODEL"]})
            continue
        results.append(inspect_part(c, board, meta, arrays, frame, cfg, angles))
    extra = board_anomalies(board, arrays, comps, meta, frame, cfg)
    gross = {"MISSING", "SHIFT", "ROTATED", "POLARITY", "MISMATCH"}
    owners = [(r, r["_geom"]) for r in results if r.get("status") == "FAIL" and gross & set(r["defects"])]
    kept = []
    for e in extra:
        x, y, w, h = e["_box_px"]
        cx, cy = x + w / 2, y + h / 2
        owner = next((r for r, g in owners
                      if np.hypot(cx - g.center[0], cy - g.center[1]) <= max(g.region) / 2 + g.pad), None)
        if owner is None:
            kept.append(e)
        else:
            owner["board_spill_mm2"] = round(owner.get("board_spill_mm2", 0) + e["area_mm2"], 3)
    return board, info, results, kept
