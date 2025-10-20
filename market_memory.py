# =========================================================
# MODULE: market_memory.py
# Memory of market performance per symbol
# =========================================================
from collections import defaultdict
from typing import DefaultDict, Dict

import numpy as np


class MarketMemory:
    def __init__(self) -> None:
        self.stats: DefaultDict[str, Dict[str, int]] = defaultdict(lambda: {"wins": 0, "losses": 0})

    def update(self, symbol: str, result: str) -> None:
        bucket = self.stats[symbol]
        if (result or "").strip().upper() == "WIN":
            bucket["wins"] += 1
        else:
            bucket["losses"] += 1

    def bias(self, symbol: str) -> float:
        bucket = self.stats[symbol]
        total = bucket["wins"] + bucket["losses"]
        if total == 0:
            return 1.0
        ratio = bucket["wins"] / total
        bias = 0.8 + (ratio - 0.5) * 0.4
        return float(np.clip(bias, 0.8, 1.2))
