from __future__ import annotations

from settings import RISK_CFG


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def compute_sl_pips(atr_pips: float) -> int:
    if RISK_CFG.get("sl_mode") == "atr_aggressive":
        sl = atr_pips * RISK_CFG["sl_atr_mult"]
        limited = clamp(sl, RISK_CFG["sl_min_pips"], RISK_CFG["sl_max_pips"])
        return int(round(limited))
    return int(RISK_CFG["sl_min_pips"])
