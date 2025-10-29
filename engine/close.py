from __future__ import annotations

import logging
import time

import gui
from engine import state as bot_state

logger = logging.getLogger(__name__)


def on_order_closed(ticket: int, result: str, pnl: float) -> None:
    with bot_state.position_lock:
        if bot_state.trade_state.get("ticket") == ticket:
            bot_state.trade_state.update({"state": bot_state.BotState.COOLDOWN, "ticket": None})
            bot_state.last_close_ts = time.time()
    gui.mark_closed(ticket, result=result, pnl=pnl)
    logger.info(f"TRADE_CLOSED ticket={ticket} result={result} pnl={pnl:.2f}")
