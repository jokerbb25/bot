"""Adaptive confidence evaluation engine.

This module computes the trade confidence by combining strategy alignment
with adaptive adjustments driven by volatility, market regime, and
historical performance. The surrounding trading logic, GUI, and threading
layers consume the resulting metrics without alteration.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ai_calibrator import ConfidenceCalibrator
from strategy_base import StrategyResult, StrategySnapshot

MIN_CONFIDENCE = 0.70
SOFT_LOWER_BOUND = 0.55
CONFIDENCE_THRESHOLD = 0.70
VOLATILITY_MIN = 0.0002
VOLATILITY_MAX = 0.002
VOLATILITY_LOW_INFLUENCE = 0.0005
VOLATILITY_HIGH_INFLUENCE = 0.0012
STAKE_REDUCTION_THRESHOLD = 0.0005
LOW_VOLATILITY_STAKE_FACTOR = 0.5
DEFAULT_SUCCESS_RATE = 0.5
CONFIDENCE_MEMORY_PATH = Path("confidence_memory.json")
MIN_ALIGNED_STRATEGIES = 3

calibrator = ConfidenceCalibrator()


def _load_confidence_memory() -> Dict[str, float]:
    if not CONFIDENCE_MEMORY_PATH.exists():
        return {}
    try:
        with CONFIDENCE_MEMORY_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    cleaned: Dict[str, float] = {}
    for symbol, value in data.items():
        try:
            cleaned[symbol] = float(np.clip(float(value), 0.0, 1.0))
        except (TypeError, ValueError):
            continue
    return cleaned


def _save_confidence_memory(memory: Dict[str, float]) -> None:
    try:
        with CONFIDENCE_MEMORY_PATH.open("w", encoding="utf-8") as handle:
            json.dump(memory, handle, ensure_ascii=False, indent=2)
    except OSError:
        pass


_confidence_memory: Dict[str, float] = _load_confidence_memory()


@dataclass(slots=True)
class EvaluationResult:
    final_confidence: float
    base_confidence: float
    allow_trade: bool
    strong_signals: int
    aligned_strategies: int
    active_strategies: int
    volatility_weight: float
    regime_weight: float
    historical_weight: float
    valid_volatility: bool
    consensus_direction: Optional[str]
    skip_reason: Optional[str]
    stake: float
    cooldown_until: Optional[float]
    perfect_alignment: bool
    rsi_strength: float
    ema_strength: float


def _consensus_direction(results: Sequence[StrategyResult]) -> Tuple[Optional[str], int]:
    call_count = sum(1 for result in results if result.normalized_signal() == "CALL")
    put_count = sum(1 for result in results if result.normalized_signal() == "PUT")
    if call_count == 0 and put_count == 0:
        return None, 0
    if call_count >= put_count:
        return "CALL", call_count
    return "PUT", put_count


def _volatility_weight(volatility: float) -> float:
    if volatility < VOLATILITY_LOW_INFLUENCE:
        return 0.9
    if volatility > VOLATILITY_HIGH_INFLUENCE:
        return 1.05
    return 1.0


def _regime_weight(regime: str) -> float:
    normalized = regime.strip().upper()
    if normalized in {"TRENDING", "TREND"}:
        return 1.05
    if normalized in {"RANGING", "RANGE"}:
        return 0.95
    return 1.0


def _success_rate(symbol: str) -> float:
    return float(_confidence_memory.get(symbol, DEFAULT_SUCCESS_RATE))


def _update_success_memory(symbol: str, result: str) -> None:
    normalized = (result or "").strip().upper()
    if normalized not in {"WIN", "LOSS"}:
        return
    current = _confidence_memory.get(symbol, DEFAULT_SUCCESS_RATE)
    if normalized == "WIN":
        updated = current * 1.02
    else:
        updated = current * 0.98
    updated = float(np.clip(updated, 0.0, 1.0))
    _confidence_memory[symbol] = updated
    _save_confidence_memory(_confidence_memory)


def _strong_signal_count(results: Iterable[StrategyResult]) -> int:
    return sum(1 for result in results if result.strong_alignment())


def _find_result(results: Sequence[StrategyResult], aliases: Iterable[str]) -> Optional[StrategyResult]:
    normalized_aliases = {alias.strip().upper().replace(" ", "_") for alias in aliases}
    for result in results:
        if result.normalized_name() in normalized_aliases:
            return result
    return None


def evaluate_snapshot(snapshot: StrategySnapshot) -> EvaluationResult:
    rsi_signal = None
    ema_signal = None
    bollinger_signal = None
    pullback_signal = None
    range_break_signal = None
    divergence_signal = None
    volatility_signal = None
    final_signal = None
    confidence = 0.0

    active_results: Sequence[StrategyResult] = []
    active_count = 0
    strong_count = 0
    consensus_direction: Optional[str] = None
    aligned_count = 0
    total_strategies = 1
    rsi_result: Optional[StrategyResult] = None
    ema_result: Optional[StrategyResult] = None
    pullback_result: Optional[StrategyResult] = None
    rsi_strength = 0.0
    ema_strength = 0.0
    volatility = float(snapshot.volatility or 0.0)
    volatility_mid_range = False
    has_contradiction = False
    perfect_alignment = False
    base_confidence = 0.0
    final_confidence = 0.0

    try:
        active_results = list(snapshot.active_signals())
        active_count = len(active_results)
        strong_count = _strong_signal_count(active_results)
        consensus_direction, aligned_count = _consensus_direction(active_results)
        total_strategies = active_count if active_count > 0 else 1

        rsi_result = _find_result(active_results, {"RSI"})
        ema_result = _find_result(active_results, {"EMA", "EMA_TREND"})
        pullback_result = _find_result(active_results, {"PULLBACK"})

        rsi_signal = rsi_result.normalized_signal() if rsi_result else None
        ema_signal = ema_result.normalized_signal() if ema_result else None
        pullback_signal = pullback_result.normalized_signal() if pullback_result else None

        rsi_strength = float(np.clip(rsi_result.confidence if rsi_result else 0.0, 0.0, 1.0))
        ema_strength = float(np.clip(ema_result.confidence if ema_result else 0.0, 0.0, 1.0))

        volatility = float(snapshot.volatility or 0.0)
        volatility_mid_range = 0.0005 <= volatility <= 0.002 if volatility else False

        if consensus_direction:
            has_contradiction = any(
                result.normalized_signal() not in {"NONE", consensus_direction}
                for result in active_results
            )

        perfect_alignment = (
            consensus_direction in {"CALL", "PUT"}
            and not has_contradiction
            and rsi_result is not None
            and ema_result is not None
            and pullback_result is not None
            and rsi_result.normalized_signal() == consensus_direction
            and ema_result.normalized_signal() == consensus_direction
            and pullback_result.normalized_signal() == consensus_direction
            and aligned_count == active_count
            and active_count > 0
            and volatility_mid_range
        )

        if perfect_alignment:
            final_confidence = 1.0
        else:
            base_confidence = float(aligned_count / total_strategies) if total_strategies else 0.0
            average_strength = (rsi_strength + ema_strength) / 2.0
            final_confidence = base_confidence * average_strength
            final_confidence = float(np.clip(final_confidence, 0.0, 0.95))

        if active_count == 0:
            base_confidence = 0.0
        else:
            base_confidence = float(aligned_count / total_strategies)

        final_signal = consensus_direction
        confidence = final_confidence
    except Exception as exc:
        logging.warning('[%s] Strategy evaluation error: %s', snapshot.symbol, str(exc))

    if volatility and volatility < 0.0005:
        logging.info('⚠️ Skipping low-volatility asset (%s) — volatility=%.6f', snapshot.symbol, volatility)
        return None

    if pullback_signal is None:
        pullback_signal = 'NONE'
    if bollinger_signal is None:
        bollinger_signal = 'NONE'
    if divergence_signal is None:
        divergence_signal = 'NONE'
    if range_break_signal is None:
        range_break_signal = 'NONE'
    if volatility_signal is None:
        volatility_signal = 'NONE'

    valid_volatility = False
    if volatility:
        valid_volatility = VOLATILITY_MIN < volatility < VOLATILITY_MAX

    stake = float(snapshot.base_stake)
    if valid_volatility and volatility < STAKE_REDUCTION_THRESHOLD:
        stake *= LOW_VOLATILITY_STAKE_FACTOR
        logging.info('⚠️ Low volatility (%s) → reducing stake to %.4f', f"{volatility:.5f}", stake)

    allow_trade = False
    skip_reason: Optional[str] = None

    if final_confidence < SOFT_LOWER_BOUND:
        skip_reason = 'confidence_floor'
    elif final_confidence < CONFIDENCE_THRESHOLD:
        calibrator.passive_update(snapshot.symbol, final_confidence)
        skip_reason = 'confidence_soft_zone'
    else:
        allow_trade = True

    if allow_trade and not valid_volatility:
        allow_trade = False
        skip_reason = 'invalid_volatility'

    if allow_trade and (consensus_direction is None or aligned_count < MIN_ALIGNED_STRATEGIES):
        allow_trade = False
        skip_reason = 'insufficient_alignment'

    current_time = snapshot.current_time or time.time()
    cooldown_until: Optional[float] = None

    if snapshot.cooldowns:
        existing = float(snapshot.cooldowns.get(snapshot.symbol, 0.0) or 0.0)
        if existing > current_time:
            allow_trade = False
            cooldown_until = existing
            skip_reason = skip_reason or 'cooldown_active'

    losses = 0
    if snapshot.consecutive_losses:
        losses = int(snapshot.consecutive_losses.get(snapshot.symbol, 0))
    if losses >= 3:
        cooldown_until = max(cooldown_until or 0.0, current_time + 120.0)
        allow_trade = False
        skip_reason = skip_reason or 'cooldown_triggered'

    if snapshot.trades_count and snapshot.trades_count % 30 == 0:
        calibrator.rebalance()

    if snapshot.result_available and snapshot.trade_result:
        calibrator.update(final_confidence, snapshot.trade_result, snapshot.symbol)
        _update_success_memory(snapshot.symbol, snapshot.trade_result)

    return EvaluationResult(
        final_confidence=final_confidence,
        base_confidence=base_confidence,
        allow_trade=allow_trade,
        strong_signals=strong_count,
        aligned_strategies=aligned_count,
        active_strategies=active_count,
        volatility_weight=1.0,
        regime_weight=1.0,
        historical_weight=1.0,
        valid_volatility=valid_volatility,
        consensus_direction=consensus_direction,
        skip_reason=skip_reason,
        stake=stake,
        cooldown_until=cooldown_until,
        perfect_alignment=perfect_alignment,
        rsi_strength=rsi_strength,
        ema_strength=ema_strength,
    )

__all__ = ["EvaluationResult", "evaluate_snapshot", "CONFIDENCE_THRESHOLD", "SOFT_LOWER_BOUND", "MIN_CONFIDENCE"]
