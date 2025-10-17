"""Adaptive learning helpers for strategy bias adjustment."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict

import numpy as np


@dataclass(slots=True)
class StrategyBiasTracker:
    """Maintain per-symbol biases for RSI and EMA strategies.

    Only learning speed is modified; the surrounding trading engine remains
    untouched, complying with the requirement to avoid GUI or threading
    changes.
    """

    rsi_bias: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))
    ema_bias: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))
    confidence_memory: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(lambda: 0.5))

    def update(self, symbol: str, final_confidence: float, result: str) -> None:
        outcome = (result or "").strip().upper()
        learning_rate = 0.02 + (0.03 * abs(final_confidence - 0.5))
        if outcome == "WIN":
            self.rsi_bias[symbol] += learning_rate
            self.ema_bias[symbol] += learning_rate
        elif outcome == "LOSS":
            self.rsi_bias[symbol] -= learning_rate
            self.ema_bias[symbol] -= learning_rate

        self.rsi_bias[symbol] *= 0.98
        self.ema_bias[symbol] *= 0.98

        if outcome == "WIN":
            self.confidence_memory[symbol] += 0.05
        elif outcome == "LOSS":
            self.confidence_memory[symbol] -= 0.05
        self.confidence_memory[symbol] = float(np.clip(self.confidence_memory[symbol], 0.0, 1.0))

    def bias_snapshot(self, symbol: str) -> dict[str, float]:
        return {
            "RSI": float(self.rsi_bias[symbol]),
            "EMA": float(self.ema_bias[symbol]),
        }

    def confidence_snapshot(self, symbol: str) -> float:
        return float(self.confidence_memory[symbol])


__all__ = ["StrategyBiasTracker"]
