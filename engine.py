"""Adaptive signal evaluation engine.

This module consolidates individual strategy outputs into a final trade
confidence while applying dynamic filters requested by the calibration
specification. Trading logic, GUI elements, and threading constructs live
elsewhere; here we only manipulate numerical confidences.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from ai_calibrator import ConfidenceCalibrator
from ai_regressor import LightRegressor
from market_memory import MarketMemory
from strategy_base import StrategyResult, StrategySnapshot

MIN_CONFIDENCE = 0.65
SOFT_LOWER_BOUND = 0.55
MOMENTUM_BOOST = 0.03
EMA_SLOPE_MIN = 0.0001
EMA_SLOPE_DAMP = 0.8
LOW_VOLATILITY_THRESHOLD = 0.0005
LOW_VOLATILITY_STAKE_FACTOR = 0.5

DEFAULT_BASE_WEIGHTS: Mapping[str, float] = {
    "RSI": 1.0,
    "EMA": 1.0,
    "EMA_TREND": 1.0,
    "PULLBACK": 1.0,
    "BOLL": 1.0,
    "BOLLINGER": 1.0,
    "BOLLINGER_REBOUND": 1.0,
    "DIVERGENCE": 1.0,
}

REGIME_WEIGHT_MAP: Mapping[str, Mapping[str, float]] = {
    "RANGING": {
        "RSI": 1.3,
        "DIVERGENCE": 1.2,
        "EMA": 0.8,
        "EMA_TREND": 0.8,
        "BOLL": 1.1,
        "BOLLINGER": 1.1,
        "BOLLINGER_REBOUND": 1.1,
    },
    "VOLATILE": {
        "RSI": 0.9,
        "EMA": 1.4,
        "EMA_TREND": 1.4,
        "PULLBACK": 1.3,
        "BOLL": 0.8,
        "BOLLINGER": 0.8,
        "BOLLINGER_REBOUND": 0.8,
    },
    "TREND": {
        "RSI": 1.2,
        "EMA": 1.5,
        "EMA_TREND": 1.5,
        "PULLBACK": 1.0,
        "BOLL": 0.9,
        "BOLLINGER": 0.9,
        "BOLLINGER_REBOUND": 0.9,
    },
}

STRONG_STRATEGY_KEYS = {
    "RSI",
    "EMA",
    "EMA_TREND",
    "PULLBACK",
    "BOLL",
    "BOLLINGER",
    "BOLLINGER_REBOUND",
    "DIVERGENCE",
}

calibrator = ConfidenceCalibrator()
market_memory = MarketMemory()
light_regressor = LightRegressor()


@dataclass(slots=True)
class EvaluationResult:
    final_confidence: float
    allow_trade: bool
    strong_signals: int
    regime_weights: Dict[str, float]
    momentum: float
    ema_slope: Optional[float]
    stake: float
    ai_prediction: float
    calibrated_confidence: float


def _normalise_key(name: str) -> str:
    return name.strip().upper().replace(" ", "_")


def _regime_weights(regime: str) -> Dict[str, float]:
    base = dict(DEFAULT_BASE_WEIGHTS)
    override = REGIME_WEIGHT_MAP.get(regime.strip().upper())
    if override:
        base.update(override)
    return base


def _weighted_confidence(results: Iterable[StrategyResult], weights: Mapping[str, float]) -> tuple[float, float, int]:
    total = 0.0
    weight_sum = 0.0
    strong_count = 0
    for result in results:
        key = _normalise_key(result.name)
        weight = weights.get(key, result.weight)
        total += float(result.confidence) * weight
        weight_sum += weight
        if key in STRONG_STRATEGY_KEYS and result.strong_alignment():
            strong_count += 1
    if weight_sum == 0:
        return 0.0, weight_sum, strong_count
    return total / weight_sum, weight_sum, strong_count


def _ema_slope(ema_short: Optional[Sequence[float]]) -> Optional[float]:
    if not ema_short or len(ema_short) < 5:
        return None
    reference = float(ema_short[-5])
    if reference == 0:
        return None
    latest = float(ema_short[-1])
    return (latest - reference) / reference


def _short_term_momentum(closes: Sequence[float]) -> float:
    if len(closes) < 3:
        return 0.0
    tail = np.array(closes[-3:], dtype=float)
    diffs = np.diff(tail)
    if diffs.size == 0:
        return 0.0
    return float(np.mean(diffs))


def _metadata_value(snapshot: StrategySnapshot, keys: Sequence[str], default: float) -> float:
    for result in snapshot.results:
        metadata = result.metadata
        for key in keys:
            if key in metadata:
                try:
                    return float(metadata[key])
                except (TypeError, ValueError):
                    continue
    return float(default)


def _build_features(symbol_snapshot: StrategySnapshot, momentum: float, ema_slope: Optional[float]) -> tuple[float, float, float, float]:
    if symbol_snapshot.rsi_value is not None:
        rsi_value = float(symbol_snapshot.rsi_value)
    else:
        rsi_value = _metadata_value(symbol_snapshot, ("rsi", "RSI"), 50.0)
    if symbol_snapshot.ema_slope_value is not None:
        ema_metric = float(symbol_snapshot.ema_slope_value)
    elif ema_slope is not None:
        ema_metric = float(ema_slope)
    else:
        ema_metric = _metadata_value(symbol_snapshot, ("ema_slope", "slope"), 0.0)
    volatility = float(symbol_snapshot.volatility or 0.0)
    return (
        rsi_value / 100.0,
        ema_metric,
        volatility,
        float(momentum),
    )


def evaluate_snapshot(snapshot: StrategySnapshot) -> EvaluationResult:
    weights = _regime_weights(snapshot.regime)
    active_results = list(snapshot.active_signals())
    base_confidence, _, strong_count = _weighted_confidence(active_results, weights)
    base_confidence = float(np.clip(base_confidence, 0.0, 1.0))

    momentum = _short_term_momentum(snapshot.closes)
    if momentum > 0:
        base_confidence += MOMENTUM_BOOST
    elif momentum < 0:
        base_confidence -= MOMENTUM_BOOST

    ema_slope = _ema_slope(snapshot.ema_short)
    if ema_slope is not None and abs(ema_slope) < EMA_SLOPE_MIN:
        base_confidence *= EMA_SLOPE_DAMP

    calibrated_confidence = float(np.clip(base_confidence, 0.0, 1.0))
    calibrated_confidence = calibrator.adjusted_confidence(calibrated_confidence)
    calibrated_confidence *= market_memory.bias(snapshot.symbol)

    features = _build_features(snapshot, momentum, ema_slope)
    ai_prediction = light_regressor.predict(features)
    final_confidence = (calibrated_confidence * 0.7) + (ai_prediction * 0.3)
    final_confidence = float(np.clip(final_confidence, 0.0, 1.0))

    stake = float(snapshot.base_stake)
    volatility = float(snapshot.volatility or 0.0)
    if volatility and volatility < LOW_VOLATILITY_THRESHOLD:
        stake *= LOW_VOLATILITY_STAKE_FACTOR
        logging.info(
            "⚠️ Low volatility (%s) → reducing stake to %.4f",
            f"{volatility:.5f}",
            stake,
        )

    allow_trade = final_confidence >= MIN_CONFIDENCE
    if not allow_trade and SOFT_LOWER_BOUND <= final_confidence < MIN_CONFIDENCE:
        allow_trade = strong_count >= 3

    if snapshot.result_available and snapshot.trade_result:
        calibrator.update(final_confidence, snapshot.trade_result)
        market_memory.update(snapshot.symbol, snapshot.trade_result)
        outcome_value = 1.0 if snapshot.trade_result.strip().upper() == "WIN" else 0.0
        light_regressor.train([features], [outcome_value])

    return EvaluationResult(
        final_confidence=final_confidence,
        allow_trade=allow_trade,
        strong_signals=strong_count,
        regime_weights=dict(weights),
        momentum=momentum,
        ema_slope=ema_slope,
        stake=stake,
        ai_prediction=ai_prediction,
        calibrated_confidence=calibrated_confidence,
    )


__all__ = [
    "EvaluationResult",
    "evaluate_snapshot",
    "MIN_CONFIDENCE",
    "SOFT_LOWER_BOUND",
]
