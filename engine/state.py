from enum import Enum, auto
import threading


class BotState(Enum):
    IDLE = auto()
    ARMED = auto()
    OPEN = auto()
    CLOSING = auto()
    COOLDOWN = auto()


trade_state = {
    "state": BotState.IDLE,
    "ticket": None,
    "symbol": None,
    "direction": None,
    "entry_price": None,
    "open_time": None,
    "magic": 20251029,
}

position_lock = threading.RLock()
last_close_ts = 0.0
