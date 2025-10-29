from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from engine.state import BotState, trade_state

logger = logging.getLogger(__name__)

_status_callbacks: List[Callable[[str], None]] = []
_open_callbacks: List[Callable[[Dict[str, Any]], None]] = []
_close_callbacks: List[Callable[[Dict[str, Any]], None]] = []


def register_status_callback(callback: Callable[[str], None]) -> None:
    if callback not in _status_callbacks:
        _status_callbacks.append(callback)


def register_open_callback(callback: Callable[[Dict[str, Any]], None]) -> None:
    if callback not in _open_callbacks:
        _open_callbacks.append(callback)


def register_close_callback(callback: Callable[[Dict[str, Any]], None]) -> None:
    if callback not in _close_callbacks:
        _close_callbacks.append(callback)


def push_status(message: str) -> None:
    if trade_state.get("state") == BotState.OPEN and "NEUTRAL" in message.upper():
        return
    logger.info(message)
    for callback in list(_status_callbacks):
        try:
            callback(message)
        except Exception:
            logger.debug("Status callback failed", exc_info=True)


def mark_open(symbol: str, direction: str, confidence: Optional[float] = None, entry: Optional[float] = None, sl_pips: Optional[int] = None) -> None:
    payload: Dict[str, Any] = {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "entry": entry,
        "sl_pips": sl_pips,
    }
    for callback in list(_open_callbacks):
        try:
            callback(dict(payload))
        except Exception:
            logger.debug("Open callback failed", exc_info=True)


def mark_closed(ticket: int, result: Optional[str] = None, pnl: Optional[float] = None) -> None:
    payload: Dict[str, Any] = {
        "ticket": ticket,
        "result": result,
        "pnl": pnl,
    }
    for callback in list(_close_callbacks):
        try:
            callback(dict(payload))
        except Exception:
            logger.debug("Close callback failed", exc_info=True)
