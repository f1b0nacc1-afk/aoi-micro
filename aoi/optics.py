"""Field, scale, depth of field and parallax for the IMX183 + fixed-focal setup."""
from __future__ import annotations

import math


def optics_table(focal_mm: float, pixel_um: float, sensor_px: tuple, wds: list, fstops: list,
                 part_height_mm: float = 2.0, pupil_offset_mm: float = 27.0) -> str:
    """wds are measured from the lens front; pupil_offset_mm moves them to the principal plane
    (27 mm reproduces the Computar V1226-MPZ catalogue field at MOD 100 mm)."""
    sw, sh = sensor_px[0] * pixel_um / 1000, sensor_px[1] * pixel_um / 1000
    coc = 2 * pixel_um / 1000
    lines = [f"f={focal_mm} мм, пиксель {pixel_um} мкм, CoC {coc * 1000:.1f} мкм, WD от передней линзы "
             f"(+{pupil_offset_mm:.0f} мм до главной плоскости), параллакс для детали h={part_height_mm} мм у края поля",
             f"{'WD':>6} {'поле, мм':>14} {'мкм/px':>7} {'0402 px':>9} {'шаг0.4 px':>9} "
             + " ".join(f"{'ГРИП f/' + str(n):>10}" for n in fstops) + f" {'параллакс':>10}"]
    for wd_front in wds:
        wd = wd_front + pupil_offset_mm
        m = focal_mm / (wd - focal_mm)
        fw, fh = sw / m, sh / m
        um = fw / sensor_px[0] * 1000
        dofs = []
        for n in fstops:
            dofs.append(2 * n * coc * (1 + m) / (m * m))
        r_edge = math.hypot(fw, fh) / 2
        par = r_edge * part_height_mm / (wd - part_height_mm)
        lines.append(f"{wd_front:>6.0f} {fw:>6.0f} × {fh:<5.0f} {um:>7.1f} "
                     f"{1000 / um:>4.0f}×{500 / um:<4.0f} {400 / um:>9.1f} "
                     + " ".join(f"{d:>7.1f} мм" for d in dofs) + f" {par:>7.2f} мм")
    return "\n".join(lines)
