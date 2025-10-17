"""Core strategy data structures and helpers used by the trading engine.

This module intentionally focuses on modelling strategy outputs without
introducing any GUI or threading dependencies so that the adaptive engine
can evaluate signals in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence


SignalType = Optional[str]


@dataclass(slots=True)
class StrategyResult:
    """Container describing the outcome of a strategy evaluation.

    Attributes
    ----------
    name:
        Human readable strategy identifier (e.g. "RSI", "EMA").
    signal:
        Direction suggested by the strategy. ``None`` or "NONE" means the
        strategy remains neutral.
    confidence:
        Score within ``[0, 1]`` describing how reliable the signal is.
    weight:
        Optional base weight supplied by the strategy manager. This value is
        further adjusted inside the engine depending on the current market
        regime.
    reasons:
        Optional human readable explanations supporting the decision.
    metadata:
        Extra payload used by adaptive filters (e.g. EMA slope arrays or
        volatility snapshots).
    """

    name: str
    signal: SignalType
    confidence: float
    weight: float = 1.0
    reasons: Sequence[str] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        return self.signal not in (None, "NONE", "")

    def normalized_name(self) -> str:
        return self.name.strip().upper().replace(" ", "_")

    def strong_alignment(self) -> bool:
        if self.metadata.get("strong") is True:
            return True
        threshold = float(self.metadata.get("strong_threshold", 0.65))
        return abs(self.confidence) >= threshold and self.is_active()


@dataclass(slots=True)
class StrategySnapshot:
    """Aggregated snapshot returned by the engine."""

    symbol: str
    regime: str
    results: Sequence[StrategyResult]
    volatility: Optional[float]
    closes: Sequence[float]
    ema_short: Optional[Sequence[float]] = None
    rsi_value: Optional[float] = None
    ema_slope_value: Optional[float] = None
    base_stake: float = 1.0
    result_available: bool = False
    trade_result: Optional[str] = None

    def active_signals(self) -> Iterable[StrategyResult]:
        return (result for result in self.results if result.is_active())

    def strong_signals(self) -> Iterable[StrategyResult]:
        return (result for result in self.results if result.strong_alignment())


__all__ = ["StrategyResult", "StrategySnapshot"]
