import logging
import threading
import time
from typing import Optional

import metatrader5 as mt5

from engine.state import BotState, position_lock, trade_state
from gui import gui
from risk.sl_calc import compute_sl_pips
from settings import RISK_CFG


logger = logging.getLogger(__name__)


def _pip_size(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if info is None or info.point == 0:
        raise RuntimeError(f"Invalid symbol info for {symbol}")
    digits = getattr(info, "digits", 0)
    if digits in (3, 5):
        return info.point * 10
    if digits in (2, 4):
        return info.point * 10
    return info.point


def pips_to_price(symbol: str, pips: float) -> float:
    return float(pips) * _pip_size(symbol)


def price_to_pips(symbol: str, price_diff: float) -> float:
    pip_value = _pip_size(symbol)
    if pip_value == 0:
        return 0.0
    return float(price_diff) / pip_value


def get_bid(symbol: str) -> float:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick data for {symbol}")
    return float(tick.bid)


def get_ask(symbol: str) -> float:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick data for {symbol}")
    return float(tick.ask)


def mt5_positions():
    positions = mt5.positions_get()
    if positions is None:
        return []
    return list(positions)


def has_open_with_magic(symbol: str, magic: int) -> bool:
    for position in mt5_positions():
        if getattr(position, "symbol", "") == symbol and getattr(position, "magic", 0) == magic:
            return True
    return False


def mt5_modify_sl(ticket: int, price: float) -> None:
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl": float(price),
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.debug("Failed to modify SL for %s: %s", ticket, getattr(result, "comment", "unknown"))


def mt5_send_order(
    symbol: str,
    direction: str,
    lot: float,
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None,
    magic: Optional[int] = None,
):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick data for {symbol}")
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    price = float(tick.ask if direction == "BUY" else tick.bid)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot),
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "magic": magic,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    return mt5.order_send(request)


def execute_trade(
    symbol: str,
    direction: str,
    lot: float,
    entry_price: Optional[float],
    atr_pips: float,
    confidence: Optional[float] = None,
):
    if entry_price is None:
        entry_price = get_ask(symbol) if direction == "BUY" else get_bid(symbol)
    atr_reference = float(atr_pips) if atr_pips > 0 else RISK_CFG["sl_min_pips"]
    sl_pips = compute_sl_pips(atr_reference)
    sl_distance = pips_to_price(symbol, sl_pips)
    if direction == "BUY":
        sl_price = entry_price - sl_distance
    else:
        sl_price = entry_price + sl_distance
    with position_lock:
        if trade_state["state"] in (BotState.OPEN, BotState.CLOSING):
            gui.push_status("BUSY_OPEN_TRADE")
            return None
        if has_open_with_magic(symbol, trade_state["magic"]):
            gui.push_status("BUSY_OPEN_TRADE")
            return None
        result = mt5_send_order(
            symbol,
            direction,
            lot,
            sl_price=sl_price,
            tp_price=None,
            magic=trade_state["magic"],
        )
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("ORDER_REJECTED %s", getattr(result, "comment", "NO_RESULT"))
            gui.push_status("ORDER_REJECTED")
            return result
        ticket = getattr(result, "order", None) or getattr(result, "deal", None)
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
        gui.mark_open(symbol, direction, confidence=confidence, entry=entry_price, sl_pips=sl_pips)
    if ticket is not None and RISK_CFG.get("slp_enable", False):
        threading.Thread(target=_slp_watcher, args=(ticket,), daemon=True).start()
    return result


def _slp_watcher(ticket: int) -> None:
    arm = RISK_CFG["slp_arm_after_pips"]
    be_offset = RISK_CFG["slp_be_offset_pips"]
    trail_distance = RISK_CFG["slp_trail_distance_pips"]
    armed = False
    last_sl = None
    guard_until = 0.0
    while True:
        with position_lock:
            if trade_state.get("ticket") != ticket or trade_state.get("state") != BotState.OPEN:
                return
            symbol = trade_state["symbol"]
            direction = trade_state["direction"]
            entry_price = trade_state["entry_price"]
            if guard_until == 0.0:
                open_time = trade_state.get("open_time") or time.time()
                guard_until = open_time + RISK_CFG["sl_time_guard_seconds"]
        try:
            price = get_bid(symbol) if direction == "SELL" else get_ask(symbol)
        except Exception as exc:
            logger.debug("SLP watcher price error: %s", exc)
            time.sleep(0.5)
            continue
        pnl_pips = price_to_pips(
            symbol,
            (price - entry_price) if direction == "BUY" else (entry_price - price),
        )
        if not armed and pnl_pips >= arm and time.time() >= guard_until:
            be_price = (
                entry_price + pips_to_price(symbol, be_offset)
                if direction == "BUY"
                else entry_price - pips_to_price(symbol, be_offset)
            )
            mt5_modify_sl(ticket, be_price)
            last_sl = be_price
            armed = True
            gui.push_status(f"SLP_ARMED_BE({arm}p ➜ +{be_offset}p)")
            logger.info("SLP: moved to BE+offset")
        if armed:
            desired_sl = (
                price - pips_to_price(symbol, trail_distance)
                if direction == "BUY"
                else price + pips_to_price(symbol, trail_distance)
            )
            if (direction == "BUY" and (last_sl is None or desired_sl > last_sl)) or (
                direction == "SELL" and (last_sl is None or desired_sl < last_sl)
            ):
                mt5_modify_sl(ticket, desired_sl)
                last_sl = desired_sl
                gui.push_status(f"SLP_TRAIL({trail_distance}p)")
        time.sleep(0.25)
