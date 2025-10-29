import time

from engine.state import BotState, last_close_ts, position_lock, trade_state
from gui import gui
from settings import RISK_CFG


def can_trade_now() -> bool:
    with position_lock:
        if trade_state["state"] in (BotState.OPEN, BotState.CLOSING):
            gui.push_status("BUSY_OPEN_TRADE")
            return False
        cooldown = RISK_CFG["reentry_cooldown_sec"]
        elapsed = time.time() - last_close_ts
        if cooldown and elapsed < cooldown:
            gui.push_status("COOLDOWN")
            return False
        if trade_state["state"] == BotState.COOLDOWN:
            trade_state["state"] = BotState.IDLE
    return True
