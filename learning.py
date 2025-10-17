"""Adaptive learning helpers for strategy bias adjustment."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict


@dataclass(slots=True)
class StrategyBiasTracker:
    """Maintain per-symbol biases for RSI and EMA strategies.

    Only learning speed is modified; the surrounding trading engine remains
    untouched, complying with the requirement to avoid GUI or threading
    changes.
    """

    rsi_bias: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))
    ema_bias: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))

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

    def bias_snapshot(self, symbol: str) -> dict[str, float]:
        return {
            "RSI": float(self.rsi_bias[symbol]),
            "EMA": float(self.ema_bias[symbol]),
        }


__all__ = ["StrategyBiasTracker"]
