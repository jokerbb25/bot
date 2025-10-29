from settings import RISK_CFG


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def compute_sl_pips(atr_pips: float) -> int:
    if RISK_CFG["sl_mode"] == "atr_aggressive":
        sl_value = atr_pips * RISK_CFG["sl_atr_mult"]
        return int(round(clamp(sl_value, RISK_CFG["sl_min_pips"], RISK_CFG["sl_max_pips"])))
    return int(RISK_CFG["sl_min_pips"])
