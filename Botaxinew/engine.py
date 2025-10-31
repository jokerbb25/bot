from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import MetaTrader5 as mt5
import pandas as pd
import yaml

from utils.indicators import evaluate_indicators
from utils.logger import Logger
from utils.learning import LearningMemory

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None


class BotEngine:
    def __init__(
        self,
        gui_log: Optional[Callable[[str], None]] = None,
        config_path: Optional[Path] = None,
        memory_path: Optional[Path] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        confidence_callback: Optional[Callable[[float, str], None]] = None,
        order_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        stats_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        base_path = Path(__file__).resolve().parent
        self.config_path = config_path or base_path / "config.yaml"
        self.memory_path = memory_path or base_path / "memory.json"
        self.gui_log = gui_log
        self.logger = Logger(gui_log=self.gui_log)
        self.status_callback = status_callback
        self.confidence_callback = confidence_callback
        self.order_callback = order_callback
        self.stats_callback = stats_callback

        self.stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._thread_lock = threading.Lock()
        self.cycle_lock = threading.Lock()
        self._symbol_lock = threading.Lock()

        self.config = self._load_config()
        self.symbols: List[str] = list(self.config.get("symbols", []))
        self.selected_symbols: List[str] = list(self.symbols)
        self.focus_symbol: Optional[str] = None

        self.timeframe = self._resolve_timeframe(self.config.get("timeframe", "M1"))
        self.poll_interval = int(self.config.get("poll_interval", 5))
        self.history_bars = int(self.config.get("history_bars", 500))

        risk_config = self.config.get("risk", {})
        self.lot_high = float(risk_config.get("lot_high", 0.1))
        self.lot_low = float(risk_config.get("lot_low", 0.05))

        confidence_cfg = self.config.get("confidence", {})
        self.base_confidence = float(confidence_cfg.get("base_confidence", 0.7))
        self.lower_confidence = float(confidence_cfg.get("lower_confidence", 0.6))
        self.memory_boost = float(confidence_cfg.get("memory_boost", 0.1))

        self.memory = LearningMemory(self.memory_path)
        self.strategy_flags: Dict[str, bool] = {
            "rsi": True,
            "ema": True,
            "macd": True,
            "pullback": True,
            "memory": True,
        }

        self.last_operation_timestamp = 0.0
        self.trade_lock_seconds = 60

        self.stats: Dict[str, Any] = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "pnl": 0.0,
        }

    def start(self) -> bool:
        with self._thread_lock:
            if self._thread and self._thread.is_alive():
                self.logger.log("Engine already running.")
                return False
            self.stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            return True

    def stop(self) -> None:
        with self._thread_lock:
            if not self._thread:
                return
            self.stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.poll_interval + 1)
        self._thread = None

    def set_focus_symbol(self, symbol: str) -> None:
        with self._symbol_lock:
            if symbol in self.symbols:
                self.focus_symbol = symbol
            else:
                self.focus_symbol = None

    def update_selected_symbols(self, symbols: List[str]) -> None:
        with self._symbol_lock:
            valid = [s for s in symbols if s in self.symbols]
            self.selected_symbols = valid if valid else list(self.symbols)

    def update_risk(self, lot_high: float, lot_low: float) -> None:
        self.lot_high = float(max(0.0, lot_high))
        self.lot_low = float(max(0.0, lot_low))

    def update_confidence_levels(self, base: float, lower: float, memory_boost: float) -> None:
        self.base_confidence = float(min(max(base, 0.0), 1.0))
        self.lower_confidence = float(min(max(lower, 0.0), 1.0))
        self.memory_boost = float(min(max(memory_boost, 0.0), 1.0))

    def update_strategy(self, name: str, enabled: bool) -> None:
        if name in self.strategy_flags:
            self.strategy_flags[name] = bool(enabled)

    def get_memory_snapshot(self) -> List[Dict[str, Any]]:
        return self.memory.get_patterns()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _run_loop(self) -> None:
        if not mt5.initialize():
            self.logger.log("Failed to connect to MetaTrader5.")
            self._notify_status("Idle")
            return
        self.logger.log("✅ MT5 Connected")
        self.send_telegram("✅ Trading engine started")
        self._notify_status("Scanning")
        try:
            while not self.stop_event.is_set():
                symbols_to_scan = self._get_symbols_to_scan()
                if not symbols_to_scan:
                    self.logger.log("No symbols configured for scanning.")
                    self.stop_event.wait(self.poll_interval)
                    continue
                for symbol in symbols_to_scan:
                    if self.stop_event.is_set():
                        break
                    try:
                        self._execute_cycle(symbol)
                    except Exception as exc:  # pragma: no cover - runtime safeguard
                        self.logger.log(f"Error during cycle for {symbol}: {exc}")
                self.stop_event.wait(self.poll_interval)
        finally:
            mt5.shutdown()
            self.logger.log("🛑 Trading engine stopped — MT5 disconnected")
            self.send_telegram("🛑 Trading engine stopped — MT5 disconnected")
            self._notify_status("Idle")

    def _get_symbols_to_scan(self) -> List[str]:
        with self._symbol_lock:
            ordered = list(self.selected_symbols)
            if self.focus_symbol and self.focus_symbol in ordered:
                ordered.remove(self.focus_symbol)
                return [self.focus_symbol] + ordered
            return ordered

    def _execute_cycle(self, symbol: str) -> None:
        if not self.cycle_lock.acquire(blocking=False):
            return
        try:
            df = self._fetch_rates(symbol)
            if df is None or len(df) < 50:
                return
            analysis = evaluate_indicators(df, strategies=self.strategy_flags)
            pullback_value = analysis.get("pullback", False)
            if hasattr(pullback_value, "__len__") and not isinstance(pullback_value, (str, bytes)):
                try:
                    pullback_value = bool(list(pullback_value)[-1])
                except Exception:
                    pullback_value = bool(pullback_value)
            pullback_flag = bool(pullback_value)
            analysis["pullback"] = pullback_flag

            confidence = float(analysis.get("confidence", 0.0))
            if self.strategy_flags.get("memory", True) and analysis.get("direction") in {"CALL", "PUT"}:
                boosted_confidence, memory_applied = self.memory.apply_memory(
                    symbol,
                    float(analysis.get("rsi_value", 0.0)),
                    str(analysis.get("ema_trend", "flat")),
                    str(analysis.get("macd_signal", "NONE")),
                    pullback_flag,
                    confidence,
                    self.memory_boost,
                )
                if memory_applied:
                    self.logger.log(f"🧠 Memory match: +{self.memory_boost:.2f} confidence boost")
                confidence = boosted_confidence
            else:
                memory_applied = False
            analysis["confidence"] = confidence

            log_line = (
                f"[{symbol}] RSI {analysis.get('rsi_value', 0.0):.2f} | "
                f"EMA: {analysis.get('ema_trend', 'flat')} | "
                f"MACD: {analysis.get('macd_signal', 'NONE')} | "
                f"Pullback: {str(analysis.get('pullback', False)).lower()} | "
                f"Confidence {confidence:.2f}"
            )
            self.logger.log(log_line)
            direction = str(analysis.get("direction", "NONE"))
            self._notify_confidence(confidence, direction)

            last_price = float(df["close"].iloc[-1])
            if direction in {"CALL", "PUT"}:
                if confidence >= self.base_confidence:
                    executed = self._handle_market_order(symbol, direction, confidence, last_price, analysis)
                    if executed:
                        self._notify_status("Operation executed")
                        return
                elif pullback_flag and confidence >= self.lower_confidence:
                    executed = self._handle_pending_order(symbol, direction, confidence, last_price, analysis)
                    if executed:
                        self._notify_status("Operation executed")
                        return
            self._notify_status("Scanning")
        finally:
            self.cycle_lock.release()

    def _handle_market_order(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        last_price: float,
        analysis: Dict[str, Any],
    ) -> bool:
        if not self._acquire_trade_slot():
            return False
        lot = self.lot_high
        confirmation = self._send_market_order(symbol, direction, lot, last_price)
        if not confirmation:
            return False
        event = self._build_trade_event(
            symbol=symbol,
            decision=direction,
            confidence=confidence,
            result="PENDING",
            pnl=0.0,
            notes="Market order",
        )
        self._notify_order(event)
        trade_result = self._evaluate_trade_result(symbol, direction)
        pnl_value = self._estimate_pnl(trade_result, lot)
        self.memory.record_trade(
            symbol,
            float(analysis.get("rsi_value", 0.0)),
            str(analysis.get("ema_trend", "flat")),
            str(analysis.get("macd_signal", "NONE")),
            bool(analysis.get("pullback", False)),
            confidence,
            trade_result,
        )
        result_event = self._build_trade_event(
            symbol=symbol,
            decision=direction,
            confidence=confidence,
            result=trade_result,
            pnl=pnl_value,
            notes="Result",
        )
        result_message = f"Result {trade_result} — [{symbol}] {direction}"
        self.logger.log(result_message)
        self.send_telegram(result_message)
        self._notify_order(result_event)
        self._update_stats(trade_result, pnl_value)
        return True

    def _handle_pending_order(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        last_price: float,
        analysis: Dict[str, Any],
    ) -> bool:
        if not self._acquire_trade_slot():
            return False
        lot = self.lot_low
        confirmation = self._send_pending_order(symbol, direction, last_price, analysis, lot)
        if not confirmation:
            return False
        event = self._build_trade_event(
            symbol=symbol,
            decision=direction,
            confidence=confidence,
            result="PENDING",
            pnl=0.0,
            notes="Pending order",
        )
        self._notify_order(event)
        return True

    def _acquire_trade_slot(self) -> bool:
        now = time.time()
        if now - self.last_operation_timestamp < self.trade_lock_seconds:
            self.logger.log("⏳ Trade skipped due to candle lock.")
            return False
        self.last_operation_timestamp = now
        return True

    def _fetch_rates(self, symbol: str) -> Optional[pd.DataFrame]:
        rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, self.history_bars)
        if rates is None:
            self.logger.log(f"No data returned for {symbol}.")
            return None
        df = pd.DataFrame(rates)
        if df.empty:
            self.logger.log(f"Empty dataframe for {symbol}.")
            return None
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def _send_market_order(self, symbol: str, direction: str, lot: float, fallback_price: float) -> bool:
        mt5.symbol_select(symbol, True)
        order_type = mt5.ORDER_TYPE_BUY if direction == "CALL" else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(symbol)
        price = (tick.ask if direction == "CALL" else tick.bid) if tick else fallback_price
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "deviation": 10,
            "magic": 0,
            "comment": "axinew_market",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.log(f"Market order failed for {symbol}: {getattr(result, 'retcode', 'no response')}")
            return False
        message = f"✅ ORDER EXECUTED [{symbol}] → {direction} lot={lot:.2f}"
        self.logger.log(message)
        self.send_telegram(message)
        return True

    def _send_pending_order(
        self,
        symbol: str,
        direction: str,
        last_price: float,
        analysis: Dict[str, Any],
        lot: float,
    ) -> bool:
        mt5.symbol_select(symbol, True)
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "CALL" else mt5.ORDER_TYPE_SELL_LIMIT
        atr_value = float(analysis.get("atr", 0.0) or 0.0)
        price_offset = max(atr_value * 0.5, last_price * 0.001)
        price = last_price - price_offset if direction == "CALL" else last_price + price_offset
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "deviation": 10,
            "magic": 0,
            "comment": "axinew_pending",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            self.logger.log(f"Pending order failed for {symbol}: {getattr(result, 'retcode', 'no response')}")
            return False
        message = f"📌 PENDING ORDER SET [{symbol}] {direction} @ {price:.5f}"
        self.logger.log(message)
        self.send_telegram(message)
        return True

    def _evaluate_trade_result(self, symbol: str, direction: str) -> str:
        rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, 3)
        if not rates:
            return "UNKNOWN"
        df = pd.DataFrame(rates)
        if len(df) < 2:
            return "UNKNOWN"
        last_close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2])
        if direction == "CALL":
            return "WIN" if last_close >= prev_close else "LOSS"
        if direction == "PUT":
            return "WIN" if last_close <= prev_close else "LOSS"
        return "UNKNOWN"

    def _estimate_pnl(self, result: str, lot: float) -> float:
        if result == "WIN":
            return round(lot, 2)
        if result == "LOSS":
            return round(-lot, 2)
        return 0.0

    def _build_trade_event(
        self,
        symbol: str,
        decision: str,
        confidence: float,
        result: str,
        pnl: float,
        notes: str,
    ) -> Dict[str, Any]:
        timestamp = datetime.now().strftime("%H:%M:%S")
        return {
            "time": timestamp,
            "symbol": symbol,
            "decision": decision,
            "confidence": round(confidence, 2),
            "result": result,
            "pnl": pnl,
            "notes": notes,
        }

    def _update_stats(self, result: str, pnl: float) -> None:
        if result not in {"WIN", "LOSS"}:
            return
        self.stats["trades"] += 1
        if result == "WIN":
            self.stats["wins"] += 1
        else:
            self.stats["losses"] += 1
        self.stats["pnl"] = round(float(self.stats["pnl"]) + pnl, 2)
        self._notify_stats()

    def _notify_status(self, status: str) -> None:
        if self.status_callback:
            self.status_callback(status)

    def _notify_confidence(self, confidence: float, direction: str) -> None:
        if self.confidence_callback:
            self.confidence_callback(confidence, direction)

    def _notify_order(self, event: Dict[str, Any]) -> None:
        if self.order_callback:
            self.order_callback(event)

    def _notify_stats(self) -> None:
        if self.stats_callback:
            self.stats_callback(dict(self.stats))

    def send_telegram(self, message: str) -> None:
        telegram_cfg = self.config.get("telegram", {})
        if not telegram_cfg.get("enabled"):
            return
        token = telegram_cfg.get("token")
        chat_id = telegram_cfg.get("chat_id")
        if not token or not chat_id:
            return
        if requests is None:
            self.logger.log("Telegram notification skipped (requests not available).")
            return

        def _worker() -> None:
            try:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=5)
            except Exception as exc:  # pragma: no cover - network issues
                self.logger.log(f"Telegram send failed: {exc}")

        threading.Thread(target=_worker, daemon=True).start()

    def _resolve_timeframe(self, name: str):
        if hasattr(mt5, name):
            return getattr(mt5, name)
        return mt5.TIMEFRAME_M1
