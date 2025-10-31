from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import metatrader5 as mt5

import gui
from engine.state import BotState, position_lock, trade_state
from risk.sl_calc import compute_sl_pips
from settings import RISK_CFG
from ui_bus import bridge

logger = logging.getLogger(__name__)


_SYMBOL_CACHE: Dict[str, str] = {}

_SLP_STATE: Dict[str, Any] = {
    "ticket": None,
    "guard_until": 0.0,
    "last_sl": None,
    "armed": False,
    "symbol": None,
    "direction": None,
}


def reset_slp_state() -> None:
    _SLP_STATE.update(
        {
            "ticket": None,
            "guard_until": 0.0,
            "last_sl": None,
            "armed": False,
            "symbol": None,
            "direction": None,
        }
    )


def resolve_symbol(symbol_name: str) -> str:
    base = str(symbol_name or "")
    cached = _SYMBOL_CACHE.get(base)
    if cached:
        return cached
    try:
        available = mt5.symbols_get()
    except Exception:
        available = None
    resolved = base
    if available:
        target = base.upper()
        direct_match = next(
            (
                str(getattr(entry, "name", "") or "")
                for entry in available
                if str(getattr(entry, "name", "") or "").upper() == target
            ),
            None,
        )
        if direct_match:
            resolved = direct_match
        else:
            partial_match = next(
                (
                    str(getattr(entry, "name", "") or "")
                    for entry in available
                    if target in str(getattr(entry, "name", "") or "").upper()
                ),
                None,
            )
            if partial_match:
                resolved = partial_match
    _SYMBOL_CACHE[base] = resolved
    if resolved != base:
        logger.info("Resolved symbol %s -> %s", base, resolved)
    return resolved


def mt5_positions():
    positions = mt5.positions_get()
    return positions or ()


def has_open_with_magic(symbol: str, magic: int) -> bool:
    resolved = resolve_symbol(symbol)
    for position in mt5_positions():
        name = getattr(position, "symbol", None)
        if name not in {symbol, resolved}:
            continue
        if getattr(position, "magic", None) == magic:
            return True
    return False


def _pip_size(symbol: str) -> float:
    info = mt5.symbol_info(resolve_symbol(symbol))
    if info is None:
        return 0.0
    point = float(getattr(info, "point", 0.0) or 0.0)
    digits = int(getattr(info, "digits", 0) or 0)
    if digits in (3, 5):
        return point * 10.0
    if digits in (1, 2, 4):
        return point
    if digits >= 6:
        return point * 10.0
    return point


def pips_to_price(symbol: str, pips: float) -> float:
    pip_value = _pip_size(symbol)
    return float(pips) * pip_value if pip_value else 0.0


def price_to_pips(symbol: str, price_delta: float) -> float:
    pip_value = _pip_size(symbol)
    if not pip_value:
        return 0.0
    return float(price_delta) / pip_value


def get_bid(symbol: str) -> float:
    mt5_symbol = resolve_symbol(symbol)
    tick = mt5.symbol_info_tick(mt5_symbol)
    if tick is None:
        raise RuntimeError(f"No tick data for {symbol}")
    return float(getattr(tick, "bid", 0.0))


def get_ask(symbol: str) -> float:
    mt5_symbol = resolve_symbol(symbol)
    tick = mt5.symbol_info_tick(mt5_symbol)
    if tick is None:
        raise RuntimeError(f"No tick data for {symbol}")
    return float(getattr(tick, "ask", 0.0))


def mt5_send_order(
    symbol: str,
    direction: str,
    lot: float,
    sl_price: Optional[float],
    tp_price: Optional[float],
    magic: int,
) -> Optional[object]:
    mt5_symbol = resolve_symbol(symbol)
    mt5.symbol_select(mt5_symbol, True)
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    price = get_ask(symbol) if direction == "BUY" else get_bid(symbol)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": mt5_symbol,
        "volume": float(lot),
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "magic": magic,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "comment": "axi-bot",
    }
    result = mt5.order_send(request)
    if result is None:
        logger.error("ORDER_SEND_FAILED | No response from MT5")
        return None
    return result


def mt5_modify_sl(ticket: int, sl_price: float) -> bool:
    position = mt5.positions_get(ticket=ticket)
    if not position:
        return False
    pos = position[0]
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": getattr(pos, "symbol", None),
        "position": ticket,
        "sl": sl_price,
        "tp": getattr(pos, "tp", None),
    }
    result = mt5.order_send(request)
    if result is None or getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
        logger.debug("Failed to modify SL", exc_info=True)
        return False
    return True


def can_trade_now() -> bool:
    from engine.state import last_close_ts  # lazy import to avoid circular updates

    now = time.time()
    with position_lock:
        state = trade_state.get("state")
        if state in (BotState.OPEN, BotState.CLOSING):
            gui.push_status("BUSY_OPEN_TRADE")
            return False
        cooldown = RISK_CFG.get("reentry_cooldown_sec", 0)
        if cooldown and now - last_close_ts < cooldown:
            gui.push_status("COOLDOWN")
            return False
        if state == BotState.COOLDOWN and (not cooldown or now - last_close_ts >= cooldown):
            trade_state["state"] = BotState.IDLE
    return True


def execute_trade(
    symbol: str,
    direction: str,
    lot: float,
    entry_price: float,
    atr_pips: float,
    confidence: Optional[float] = None,
) -> Tuple[Optional[object], Optional[int]]:
    sl_pips = compute_sl_pips(atr_pips)
    price_offset = pips_to_price(symbol, sl_pips)
    if direction == "BUY":
        sl_price = entry_price - price_offset
    else:
        sl_price = entry_price + price_offset
    result = mt5_send_order(
        symbol,
        direction,
        lot,
        sl_price=sl_price,
        tp_price=None,
        magic=trade_state["magic"],
    )
    if result is None:
        gui.push_status("ORDER_REJECTED")
        return None, None
    retcode = getattr(result, "retcode", None)
    if retcode != mt5.TRADE_RETCODE_DONE:
        comment = getattr(result, "comment", "")
        logger.error(f"ORDER_REJECTED ({retcode}) {comment}")
        gui.push_status("ORDER_REJECTED")
        return result, None
    ticket = getattr(result, "order", 0) or getattr(result, "deal", 0) or getattr(result, "position", 0)
    with position_lock:
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
    gui.push_status(f"TRADE_OPEN_{symbol}_{direction}")
    confidence_value = confidence if confidence is not None else 0.0
    logger.info(
        f"🚀 EXECUTED MARKET ORDER {symbol} → {direction} | Confidence={confidence_value:.2f}"
    )
    if RISK_CFG.get("slp_enable"):
        reset_slp_state()
        _SLP_STATE.update(
            {
                "ticket": ticket,
                "guard_until": time.time() + RISK_CFG.get("sl_time_guard_seconds", 0),
                "last_sl": None,
                "armed": False,
                "symbol": symbol,
                "direction": direction,
            }
        )
    return result, ticket


def slp_tick() -> None:
    if not RISK_CFG.get("slp_enable"):
        return
    with position_lock:
        ticket = trade_state.get("ticket")
        state = trade_state.get("state")
        if ticket is None or state != BotState.OPEN:
            if _SLP_STATE["ticket"] is not None:
                reset_slp_state()
            return
        symbol = trade_state.get("symbol")
        direction = trade_state.get("direction")
        entry = trade_state.get("entry_price")
        open_time = trade_state.get("open_time") or time.time()
        _SLP_STATE["ticket"] = ticket
        _SLP_STATE["symbol"] = symbol
        _SLP_STATE["direction"] = direction
    if symbol is None or direction is None or entry is None:
        return
    arm = RISK_CFG.get("slp_arm_after_pips", 0)
    be_offset = RISK_CFG.get("slp_be_offset_pips", 0)
    trail_distance = RISK_CFG.get("slp_trail_distance_pips", 0)
    guard_until = max(float(open_time) + RISK_CFG.get("sl_time_guard_seconds", 0), float(_SLP_STATE.get("guard_until", 0.0) or 0.0))
    _SLP_STATE["guard_until"] = guard_until
    try:
        price = get_bid(symbol) if direction == "SELL" else get_ask(symbol)
    except Exception:
        return
    if direction == "BUY":
        pnl_pips = price_to_pips(symbol, price - entry)
    else:
        pnl_pips = price_to_pips(symbol, entry - price)
    now = time.time()
    armed = bool(_SLP_STATE.get("armed"))
    last_sl = _SLP_STATE.get("last_sl")
    if not armed and pnl_pips >= arm and now >= guard_until:
        be_price = entry + pips_to_price(symbol, be_offset) if direction == "BUY" else entry - pips_to_price(symbol, be_offset)
        if mt5_modify_sl(ticket, be_price):
            _SLP_STATE.update({"last_sl": be_price, "armed": True})
            gui.push_status(f"SLP_ARMED_BE({arm}p ➜ +{be_offset}p)")
            logger.info("SLP: moved to BE+offset")
        armed = bool(_SLP_STATE.get("armed"))
        last_sl = _SLP_STATE.get("last_sl")
    if armed:
        desired_sl = price - pips_to_price(symbol, trail_distance) if direction == "BUY" else price + pips_to_price(symbol, trail_distance)
        should_update = False
        if direction == "BUY":
            should_update = last_sl is None or desired_sl > float(last_sl)
        else:
            should_update = last_sl is None or desired_sl < float(last_sl)
        if should_update and mt5_modify_sl(ticket, desired_sl):
            _SLP_STATE["last_sl"] = desired_sl
            gui.push_status(f"SLP_TRAIL({trail_distance}p)")


def maybe_execute(
    symbol: str,
    action: str,
    lot: float,
    atr_pips: float,
    entry_price: float,
    confidence: Optional[float],
) -> Tuple[Optional[object], Optional[int]]:
    log_buffer: List[str] = []

    def _finalize(result: Optional[object], ticket: Optional[int]) -> Tuple[Optional[object], Optional[int]]:
        conf_value = confidence if confidence is not None else 0.0
        summary = " | ".join(log_buffer) if log_buffer else "OK"
        if bridge is not None:
            try:
                bridge.log_signal.emit(
                    f"[{symbol}] conf={conf_value:.2f} | action={action} | {summary}"
                )
            except Exception:
                logger.debug("Failed to emit evaluation summary", exc_info=True)
        return result, ticket

    if action not in ("BUY", "SELL"):
        log_buffer.append(f"SKIPPED_NO_ACTION symbol={symbol}")
        return _finalize(None, None)
    if not can_trade_now():
        log_buffer.append("SKIPPED_CAN_TRADE_NOW")
        return _finalize(None, None)
    with position_lock:
        if has_open_with_magic(symbol, trade_state["magic"]):
            log_buffer.append("BUSY_OPEN_TRADE")
            return _finalize(None, None)
    result, ticket = execute_trade(symbol, action, lot, entry_price, atr_pips, confidence)
    if result is None or ticket is None:
        log_buffer.append("ORDER_REJECTED")
    else:
        log_buffer.append(f"ORDER_PLACED ticket={ticket}")
    return _finalize(result, ticket)
