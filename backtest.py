"""Offline backtesting utilities for the Deriv bot."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class TradeRecord:
    timestamp: datetime
    symbol: str
    decision: str
    result: str
    confidence: float
    pnl: float


def _evaluate_trade(decision: str, entry: Dict[str, float], exit_: Dict[str, float]) -> str:
    if decision not in {"CALL", "PUT"}:
        return "SKIP"
    direction = exit_["close"] - entry["close"]
    if decision == "CALL":
        return "WIN" if direction >= 0 else "LOSS"
    return "WIN" if direction <= 0 else "LOSS"


def _sharpe_ratio(returns: List[float]) -> float:
    if not returns:
        return 0.0
    arr = np.array(returns, dtype=float)
    if arr.std() == 0:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))


def run_backtest(
    strategy_config: Dict[str, Any],
    data: Dict[str, List[Dict[str, float]]],
    technical_fn: Callable[[List[Dict[str, float]]], Dict[str, Any]],
    ml_predictor: Optional[Any] = None,
    costs: float = 0.0,
) -> Dict[str, Any]:
    """Run a simple backtest using historical candles."""
    initial_balance = strategy_config.get("initial_balance", 100.0)
    stake = strategy_config.get("stake", 1.0)
    ml_enabled = strategy_config.get("ml_enabled", False)
    ml_threshold = strategy_config.get("ml_threshold", 0.55)

    balance = initial_balance
    all_trades: List[TradeRecord] = []
    per_symbol: Dict[str, Dict[str, Any]] = {}

    for symbol, candles in data.items():
        if len(candles) < 70:
            continue
        per_symbol.setdefault(symbol, {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0})
        for idx in range(60, len(candles) - 1):
            window = candles[: idx + 1]
            signal_info = technical_fn(window)
            signal = signal_info["signal"]
            confidence = signal_info["confidence"]
            ctx = signal_info["context"]
            if signal == "NULL" or confidence < 0.35:
                continue
            if ml_enabled and ml_predictor is not None:
                ml_result = ml_predictor.predict(symbol, window, ctx, [])
                if ml_result.get("status") == "ok":
                    if signal == "CALL" and ml_result["prob_up"] >= ml_threshold:
                        confidence = min(0.99, confidence + 0.1)
                    elif signal == "PUT" and ml_result["prob_down"] >= ml_threshold:
                        confidence = min(0.99, confidence + 0.1)
                    elif signal == "CALL" and ml_result["prob_down"] >= ml_threshold + 0.1:
                        confidence = max(0.05, confidence - 0.15)
                    elif signal == "PUT" and ml_result["prob_up"] >= ml_threshold + 0.1:
                        confidence = max(0.05, confidence - 0.15)
            entry = candles[idx]
            exit_ = candles[idx + 1]
            result = _evaluate_trade(signal, entry, exit_)
            pnl = stake * (0.9 if result == "WIN" else -1)
            pnl -= costs
            balance += pnl
            if result == "WIN":
                per_symbol[symbol]["wins"] += 1
            elif result == "LOSS":
                per_symbol[symbol]["losses"] += 1
            per_symbol[symbol]["trades"] += 1
            per_symbol[symbol]["pnl"] += pnl
            all_trades.append(
                TradeRecord(
                    timestamp=datetime.fromtimestamp(exit_["epoch"]),
                    symbol=symbol,
                    decision=signal,
                    result=result,
                    confidence=float(confidence),
                    pnl=pnl,
                )
            )

    wins = sum(1 for t in all_trades if t.result == "WIN")
    losses = sum(1 for t in all_trades if t.result == "LOSS")
    returns = [t.pnl for t in all_trades]
    max_drawdown = 0.0
    peak = initial_balance
    equity = initial_balance
    for trade in all_trades:
        equity += trade.pnl
        peak = max(peak, equity)
        drawdown = (equity - peak)
        if drawdown < max_drawdown:
            max_drawdown = drawdown

    metrics = {
        "initial_balance": initial_balance,
        "final_balance": balance,
        "total_trades": len(all_trades),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / max(1, wins + losses),
        "max_drawdown": max_drawdown,
        "sharpe_like": _sharpe_ratio(returns),
        "per_symbol": per_symbol,
    }
    logging.info("Backtest completed: %s", metrics)
    return metrics


def export_trades_csv(trades: Iterable[TradeRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "symbol", "decision", "result", "confidence", "pnl"])
        for trade in trades:
            writer.writerow(
                [
                    trade.timestamp.isoformat(),
                    trade.symbol,
                    trade.decision,
                    trade.result,
                    f"{trade.confidence:.2f}",
                    f"{trade.pnl:.2f}",
                ]
            )
