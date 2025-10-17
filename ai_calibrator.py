# =========================================================
# MODULE: ai_calibrator.py
# Adaptive Bayesian Confidence Calibration
# =========================================================
from collections import deque
from typing import Deque, Tuple

import numpy as np


class ConfidenceCalibrator:
    def __init__(self, window: int = 100) -> None:
        self.memory: Deque[Tuple[float, int]] = deque(maxlen=window)

    def update(self, confidence: float, result: str) -> None:
        outcome = 1 if (result or "").strip().upper() == "WIN" else 0
        self.memory.append((float(confidence), outcome))

    def adjusted_confidence(self, current_confidence: float) -> float:
        if len(self.memory) < 10:
            return float(current_confidence)
        confs, outcomes = zip(*self.memory)
        reliability = np.mean([
            outcome if conf > 0.6 else 1 - outcome
            for conf, outcome in zip(confs, outcomes)
        ])
        blended = (float(current_confidence) * 0.7) + (float(reliability) * 0.3)
        return float(np.clip(blended, 0.0, 1.0))
