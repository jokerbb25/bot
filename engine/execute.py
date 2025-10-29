from __future__ import annotations

import logging
import threading
import time
from typing import Iterable

from engine.state import BotState, position_lock, trade_state
from risk.sl_calc import compute_sl_pips
from settings import RISK_CFG

logger = logging.getLogger(__name__)


def mt5_send_order(symbol: str, direction: str, lot: float, sl_price: float | None, tp_price: float | None, magic: int) -> int | None:
    logger.warning("mt5_send_order called without MT5 integration for symbol=%s", symbol)
    return None


def mt5_modify_sl(ticket: int, price: float) -> None:
    logger.warning("mt5_modify_sl called without MT5 integration ticket=%s", ticket)


def mt5_positions() -> Iterable[object]:  # pragma: no cover - placeholder for MT5 positions
    return []


def get_bid(symbol: str) -> float:
    raise RuntimeError("get_bid not implemented for symbol %s" % symbol)


def get_ask(symbol: str) -> float:
    raise RuntimeError("get_ask not implemented for symbol %s" % symbol)


def price_to_pips(symbol: str, delta: float) -> float:
    raise RuntimeError("price_to_pips not implemented for symbol %s" % symbol)


def pips_to_price(symbol: str, pips: float) -> float:
    raise RuntimeError("pips_to_price not implemented for symbol %s" % symbol)


def has_open_with_magic(symbol: str, magic: int) -> bool:
    for pos in mt5_positions():
        if getattr(pos, "symbol", None) == symbol and getattr(pos, "magic", None) == magic:
            return True
    return False


def execute_trade(symbol: str, direction: str, lot: float, entry_price: float, atr_pips: float) -> None:
    sl_pips = compute_sl_pips(atr_pips)
    sl_distance = pips_to_price(symbol, sl_pips)
    if direction.upper() == "BUY":
        sl_price = entry_price - sl_distance
    else:
        sl_price = entry_price + sl_distance

    with position_lock:
        if has_open_with_magic(symbol, trade_state.get("magic", 0)):
            logger.info("Duplicate trade prevented for %s", symbol)
            return
        ticket = mt5_send_order(
            symbol,
            direction,
            lot,
            sl_price=sl_price,
            tp_price=None,
            magic=trade_state.get("magic", 0),
        )
        if ticket is None:
            logger.error("ORDER_REJECTED")
            return
        trade_state.update(
            {
                "state": BotState.OPEN,
                "ticket": ticket,
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "open_time": time.time(),
            }
        )

    if RISK_CFG.get("slp_enable", False):
        threading.Thread(target=_slp_watcher, args=(ticket,), daemon=True).start()


def _slp_watcher(ticket: int) -> None:
    arm = RISK_CFG["slp_arm_after_pips"]
    be_off = RISK_CFG["slp_be_offset_pips"]
    trail = RISK_CFG["slp_trail_distance_pips"]
    guard_until = trade_state.get("open_time", 0.0) + RISK_CFG["sl_time_guard_seconds"]
    armed = False
    last_sl: float | None = None

    while True:
        with position_lock:
            if trade_state.get("ticket") != ticket or trade_state.get("state") != BotState.OPEN:
                return
            symbol = trade_state.get("symbol")
            direction = trade_state.get("direction")
            entry_price = trade_state.get("entry_price")
        if not symbol or not direction or entry_price is None:
            return
        price = get_bid(symbol) if direction.upper() == "SELL" else get_ask(symbol)
        if direction.upper() == "BUY":
            pnl_pips = price_to_pips(symbol, price - entry_price)
        else:
            pnl_pips = price_to_pips(symbol, entry_price - price)

        if not armed and pnl_pips >= arm and time.time() >= guard_until:
            be_price = entry_price + pips_to_price(symbol, be_off) if direction.upper() == "BUY" else entry_price - pips_to_price(symbol, be_off)
            mt5_modify_sl(ticket, be_price)
            armed = True
            last_sl = be_price
            logger.info("SLP armed for ticket %s", ticket)

        if armed:
            desired_sl = price - pips_to_price(symbol, trail) if direction.upper() == "BUY" else price + pips_to_price(symbol, trail)
            if direction.upper() == "BUY":
                should_move = last_sl is None or desired_sl > last_sl
            else:
                should_move = last_sl is None or desired_sl < last_sl
            if should_move:
                mt5_modify_sl(ticket, desired_sl)
                last_sl = desired_sl
        time.sleep(0.25)
