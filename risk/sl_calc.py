from __future__ import annotations

from settings import RISK_CFG


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_sl_pips(atr_pips: float) -> int:
    if RISK_CFG["sl_mode"] == "atr_aggressive":
        sl = atr_pips * RISK_CFG["sl_atr_mult"]
        return int(round(clamp(sl, RISK_CFG["sl_min_pips"], RISK_CFG["sl_max_pips"])) or 0)
    return int(RISK_CFG["sl_min_pips"])
