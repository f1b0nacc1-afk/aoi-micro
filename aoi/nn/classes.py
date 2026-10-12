"""Unified component classes shared by public datasets, the synthetic bench and the detector."""
from __future__ import annotations

import re

NAMES = ["resistor", "capacitor", "electrolytic", "ic", "transistor", "diode", "led", "inductor",
         "crystal", "footprint", "other"]
ID = {n: i for i, n in enumerate(NAMES)}

# Bench package -> class; chip R/C is resolved by part kind.
BENCH_PKG = {
    "LED_0805": "led", "Tantalum_Case-A": "capacitor", "SOD-123": "diode", "SMA": "diode",
    "SOT-23": "transistor", "SOT-223": "ic", "SOIC-8": "ic", "SOIC-16": "ic", "TSSOP-20": "ic",
    "TQFP-32": "ic", "LQFP-48": "ic", "QFN-32": "ic", "CP_Elec_5x5.8": "electrolytic",
    "CP_Elec_6.3x7.7": "electrolytic", "Crystal_SMD_3225": "crystal", "L_4x4": "inductor",
}


def bench_class(pkg: str, kind: str) -> int:
    if pkg in BENCH_PKG:
        return ID[BENCH_PKG[pkg]]
    return ID["resistor" if kind == "R" else "capacitor"]


def _n(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def map_public(name: str) -> int | None:
    """Source class name (circuit-elements, smd-components, ...) -> unified id; None drops the box."""
    n = _n(name)
    if "footprint" in n or n == "pads":
        return ID["footprint"]
    if "electrolytic" in n:
        return ID["electrolytic"]
    if n.startswith(("resist", "ressist")) or n in ("rp", "jumper"):
        return ID["resistor"]
    if n.startswith("capacitor"):
        return ID["capacitor"]
    if n.startswith("ic"):
        return ID["ic"]
    if n.startswith("transistor"):
        return ID["transistor"]
    if "diode" in n:
        return ID["diode"]
    if n.startswith("led") or n == "indicator":
        return ID["led"]
    if n in ("inductor", "coil", "ferritebead"):
        return ID["inductor"]
    if n in ("clock", "crystal"):
        return ID["crystal"]
    if n in ("unknownunlabeled",):
        return None
    return ID["other"]
