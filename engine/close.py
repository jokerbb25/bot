from __future__ import annotations

import logging
import time

from engine import state

logger = logging.getLogger(__name__)


def on_order_closed(ticket: int, result: str, pnl: float) -> None:
    with state.position_lock:
        if state.trade_state.get("ticket") == ticket:
            state.trade_state.update(
                {
                    "state": state.BotState.COOLDOWN,
                    "ticket": None,
                    "symbol": None,
                    "direction": None,
                    "entry_price": None,
                    "open_time": None,
                }
            )
            state.last_close_ts = time.time()
    logger.info("TRADE_CLOSED ticket=%s result=%s pnl=%.2f", ticket, result, pnl)
