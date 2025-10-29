import logging
import time

from engine import state
from gui import gui


logger = logging.getLogger(__name__)


def on_order_closed(ticket: int, result: str, pnl: float) -> None:
    with state.position_lock:
        if state.trade_state.get("ticket") == ticket:
            state.trade_state.update({"state": state.BotState.COOLDOWN, "ticket": None})
            state.last_close_ts = time.time()
    gui.mark_closed(ticket, result=result, pnl=pnl)
    logger.info("TRADE_CLOSED ticket=%s result=%s pnl=%.2f", ticket, result, pnl)
