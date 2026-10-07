"""Leads and land patterns per package, local frame at rotation 0 (mm, y down)."""
from __future__ import annotations

CHIPS = ("0402", "0603", "0805", "1206", "LED_0805")
SQUARE = ("TQFP-32", "LQFP-48", "QFN-32", "L_4x4")
ICS = ("SOIC-8", "SOIC-16", "TSSOP-20", "TQFP-32", "LQFP-48", "QFN-32", "SOT-23", "SOT-223")


def _chip_pads(L, T, TW):
    inner, outer = 0.35 * L, T / 2
    c, ln = (inner + outer) / 2, outer - inner
    return [(-c, 0, ln, TW), (c, 0, ln, TW)]


def _rows_dual(n_side, pitch, lead_y, lead_wh, pad_y, pad_wh):
    leads, pads, rows = [], [], []
    for sy in (1, -1):
        row = []
        for i in range(n_side):
            x = (i - (n_side - 1) / 2) * pitch
            row.append(len(leads))
            leads.append((x, sy * lead_y, *lead_wh))
            pads.append((x, sy * pad_y, *pad_wh))
        rows.append(row)
    return leads, pads, rows


def _rows_quad(n_side, pitch, lead_c, lead_wh, pad_c, pad_wh):
    leads, pads, rows = [], [], []
    for side in range(4):
        row = []
        for i in range(n_side):
            t = (i - (n_side - 1) / 2) * pitch
            if side in (0, 1):
                s = 1 if side == 0 else -1
                leads.append((t, s * lead_c, *lead_wh))
                pads.append((t, s * pad_c, *pad_wh))
            else:
                s = 1 if side == 2 else -1
                leads.append((s * lead_c, t, lead_wh[1], lead_wh[0]))
                pads.append((s * pad_c, t, pad_wh[1], pad_wh[0]))
            row.append(len(leads) - 1)
        rows.append(row)
    return leads, pads, rows


def geometry(pkg) -> dict:
    """leads: visible metal (terminal caps / gull-wing feet); pads: copper lands; rows: lead indices
    of neighbouring pins (for bridges)."""
    n = pkg.name
    L, W = pkg.body
    T, TW = pkg.total
    g = {"leads": [], "pads": [], "rows": []}
    if n in CHIPS:
        g["pads"] = _chip_pads(L, T, TW)
        tw = 0.2 * L
        g["leads"] = [(-L / 2 + tw / 2, 0, tw, W), (L / 2 - tw / 2, 0, tw, W)]
    elif n == "Tantalum_Case-A":
        g["pads"] = [(-1.6, 0, 1.4, 1.4), (1.6, 0, 1.4, 1.4)]
        g["leads"] = [(-1.45, 0, 0.3, 1.2), (1.45, 0, 0.3, 1.2)]
    elif n == "SOD-123":
        g["pads"] = [(-1.65, 0, 0.9, 1.2), (1.65, 0, 0.9, 1.2)]
        g["leads"] = [(-1.55, 0, 0.4, 0.6), (1.55, 0, 0.4, 0.6)]
    elif n == "SMA":
        g["pads"] = [(-2.4, 0, 1.7, 1.7), (2.4, 0, 1.7, 1.7)]
        g["leads"] = [(-2.35, 0, 0.5, 1.4), (2.35, 0, 0.5, 1.4)]
    elif n == "Crystal_SMD_3225":
        g["pads"] = [(sx * 1.1, sy * 0.8, 1.4, 1.15) for sx in (-1, 1) for sy in (-1, 1)]
    elif n == "L_4x4":
        g["pads"] = [(-1.6, 0, 1.4, 3.6), (1.6, 0, 1.4, 3.6)]
        g["leads"] = [(-1.9, 0, 0.2, 3.0), (1.9, 0, 0.2, 3.0)]
    elif n == "SOT-23":
        g["leads"] = [(-0.95, 0.93, 0.4, 0.55), (0.95, 0.93, 0.4, 0.55), (0, -0.93, 0.4, 0.55)]
        g["pads"] = [(-0.95, 1.1, 0.6, 0.9), (0.95, 1.1, 0.6, 0.9), (0, -1.1, 0.6, 0.9)]
        g["rows"] = [[0, 1]]
    elif n == "SOT-223":
        g["leads"] = [(-2.3, 2.2, 0.7, 0.9), (0, 2.2, 0.7, 0.9), (2.3, 2.2, 0.7, 0.9), (0, -2.2, 3.0, 0.9)]
        g["pads"] = [(-2.3, 3.15, 1.0, 1.5), (0, 3.15, 1.0, 1.5), (2.3, 3.15, 1.0, 1.5), (0, -3.15, 3.3, 1.5)]
        g["rows"] = [[0, 1, 2]]
    elif n in ("SOIC-8", "SOIC-16"):
        g["leads"], g["pads"], g["rows"] = _rows_dual(4 if n == "SOIC-8" else 8, 1.27, 2.45, (0.42, 1.0),
                                                      2.75, (0.6, 1.5))
    elif n == "TSSOP-20":
        g["leads"], g["pads"], g["rows"] = _rows_dual(10, 0.65, 2.65, (0.3, 0.9), 2.85, (0.4, 1.4))
    elif n == "TQFP-32":
        g["leads"], g["pads"], g["rows"] = _rows_quad(8, 0.8, 4.0, (0.35, 1.0), 4.2, (0.45, 1.3))
    elif n == "LQFP-48":
        g["leads"], g["pads"], g["rows"] = _rows_quad(12, 0.5, 4.0, (0.22, 1.0), 4.2, (0.28, 1.3))
    elif n == "QFN-32":
        g["leads"], g["pads"], g["rows"] = _rows_quad(8, 0.5, 2.6, (0.22, 0.2), 2.45, (0.25, 0.8))
    elif n == "CP_Elec_5x5.8":
        g["pads"] = [(-2.2, 0, 1.6, 1.4), (2.2, 0, 1.6, 1.4)]
        g["leads"] = [(-2.75, 0, 0.5, 0.6), (2.75, 0, 0.5, 0.6)]
    elif n == "CP_Elec_6.3x7.7":
        g["pads"] = [(-2.7, 0, 2.0, 1.6), (2.7, 0, 2.0, 1.6)]
        g["leads"] = [(-3.4, 0, 0.6, 0.7), (3.4, 0, 0.6, 0.7)]
    else:
        raise KeyError(n)
    return g
