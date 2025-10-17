# =========================================================
# MODULE: ai_calibrator.py
# Adaptive Bayesian Confidence Calibration
# =========================================================
from collections import defaultdict, deque
from typing import Deque, DefaultDict, Tuple

import numpy as np


class ConfidenceCalibrator:
    def __init__(self, window: int = 100) -> None:
        self.memory: Deque[Tuple[float, int]] = deque(maxlen=window)
        self.bias: DefaultDict[str, float] = defaultdict(float)
        self.passive_traces: DefaultDict[str, float] = defaultdict(float)

    def update(self, confidence: float, result: str, symbol: str | None = None) -> None:
        outcome = 1 if (result or "").strip().upper() == "WIN" else 0
        self.memory.append((float(confidence), outcome))
        if symbol:
            adjustment = (outcome - 0.5) * 0.15
            self.bias[symbol] += adjustment

    def passive_update(self, symbol: str, confidence: float) -> None:
        trace = self.passive_traces[symbol]
        self.passive_traces[symbol] = trace * 0.9 + float(confidence) * 0.1
        self.bias[symbol] *= 0.995

    def rebalance(self) -> None:
        for symbol, value in list(self.bias.items()):
            self.bias[symbol] = value * 0.9

    def adjusted_confidence(self, current_confidence: float, symbol: str | None = None) -> float:
        if len(self.memory) < 10:
            baseline = float(current_confidence)
        else:
            confs, outcomes = zip(*self.memory)
            reliability = np.mean([
                outcome if conf > 0.6 else 1 - outcome
                for conf, outcome in zip(confs, outcomes)
            ])
            baseline = (float(current_confidence) * 0.7) + (float(reliability) * 0.3)

        if symbol:
            baseline += self.bias[symbol]

        return float(np.clip(baseline, 0.0, 1.0))
