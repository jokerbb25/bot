from __future__ import annotations

import threading
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
        order_callback: Optional[Callable[[str], None]] = None,
    ):
        base_path = Path(__file__).resolve().parent
        self.config_path = config_path or base_path / "config.yaml"
        self.memory_path = memory_path or base_path / "memory.json"
        self.gui_log = gui_log
        self.logger = Logger(gui_log=self.gui_log)
        self.status_callback = status_callback
        self.confidence_callback = confidence_callback
        self.order_callback = order_callback
        self.stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._thread_lock = threading.Lock()
        self.cycle_lock = threading.Lock()
        self.config = self._load_config()
        self.symbols: List[str] = list(self.config.get("symbols", []))
        self.active_symbol: Optional[str] = self.symbols[0] if self.symbols else None
        self.timeframe = self._resolve_timeframe(self.config.get("timeframe", "M1"))
        self.poll_interval = int(self.config.get("poll_interval", 5))
        self.history_bars = int(self.config.get("history_bars", 500))
        risk_config = self.config.get("risk_management", {})
        self.base_confidence = float(risk_config.get("base_confidence", 0.7))
        self.lower_confidence = float(risk_config.get("lower_confidence", 0.6))
        self.lot_high = float(risk_config.get("lot_high", 0.1))
        self.lot_low = float(risk_config.get("lot_low", 0.05))
        volatility = self.config.get("volatility", {}).get("atr_percent", {})
        self.atr_percentages = {
            "forex": float(volatility.get("forex", 0.15)),
            "metals": float(volatility.get("metals", 0.25)),
            "crypto": float(volatility.get("crypto", 0.4)),
        }
        self.memory = LearningMemory(self.memory_path)

    def set_active_symbol(self, symbol: str) -> None:
        if symbol in self.symbols:
            self.active_symbol = symbol

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

    def _load_config(self) -> Dict:
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
                symbols = [self.active_symbol] if self.active_symbol else list(self.symbols)
                if not symbols:
                    self.logger.log("No symbols configured for scanning.")
                    self.stop_event.wait(self.poll_interval)
                    continue
                for symbol in symbols:
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

    def _execute_cycle(self, symbol: str) -> None:
        if not self.cycle_lock.acquire(blocking=False):
            return
        try:
            df = self._fetch_rates(symbol)
            if df is None or len(df) < 50:
                return
            analysis = evaluate_indicators(df)
            memory_applied = False
            if analysis["direction"] != "NONE" and self.memory.match_pattern(
                symbol, analysis["rsi_signal"], analysis["ema_trend"], analysis["macd_signal"]
            ):
                analysis["confidence"] = min(analysis["confidence"] + 0.10, 1.0)
                memory_applied = True
            if memory_applied:
                self.logger.log("🧠 Memory match: boosting confidence +0.10")
            log_line = (
                f"[{symbol}] RSI: {analysis['rsi_value']:.2f} | "
                f"EMA trend: {analysis['ema_trend']} | "
                f"MACD: {analysis['macd_signal']} | "
                f"Pullback: {str(analysis['pullback']).lower()} | "
                f"Confidence: {analysis['confidence']:.2f} → {analysis['direction']}"
            )
            self.logger.log(log_line)
            self._notify_confidence(analysis["confidence"], analysis["direction"])
            action_taken = False
            last_price = float(df["close"].iloc[-1])
            if analysis["direction"] != "NONE":
                if analysis["confidence"] >= self.base_confidence:
                    action_taken = self._handle_market_order(symbol, analysis, last_price)
                elif analysis["pullback"] and analysis["confidence"] >= self.lower_confidence:
                    action_taken = self._handle_pending_order(symbol, analysis, last_price)
            if not action_taken:
                self._notify_status("Scanning")
        finally:
            self.cycle_lock.release()

    def _handle_market_order(self, symbol: str, analysis: Dict[str, Any], last_price: float) -> bool:
        lot = self.lot_high
        direction = analysis["direction"]
        executed = self.execute_market_order(symbol, direction, lot, last_price)
        if executed:
            self._notify_status("Operation executed")
            trade_result = self._evaluate_trade_result(symbol, direction)
            if trade_result in ("WIN", "LOSS"):
                self.memory.record_result(
                    symbol,
                    analysis["rsi_signal"],
                    analysis["ema_trend"],
                    analysis["macd_signal"],
                    trade_result,
                )
                result_message = f"Result {trade_result} — {executed}"
                self.logger.log(result_message)
                self.send_telegram(result_message)
            else:
                self.logger.log(f"Result {trade_result} — {executed}")
        return executed is not None

    def _handle_pending_order(self, symbol: str, analysis: Dict[str, Any], last_price: float) -> bool:
        lot = self.lot_low
        executed = self.execute_pending_order(symbol, analysis["direction"], last_price, analysis, lot)
        if executed:
            self._notify_status("Operation executed")
        return executed is not None

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

    def execute_market_order(self, symbol: str, direction: str, lot: float, fallback_price: float) -> Optional[str]:
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
            return None
        message = f"✅ ORDER EXECUTED [{symbol}] → {direction} lot={lot:.2f}"
        self.logger.log(message)
        self.send_telegram(message)
        self._notify_order(message)
        return message

    def execute_pending_order(
        self,
        symbol: str,
        direction: str,
        last_price: float,
        analysis: Dict[str, Any],
        lot: float,
    ) -> Optional[str]:
        mt5.symbol_select(symbol, True)
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "CALL" else mt5.ORDER_TYPE_SELL_LIMIT
        category = self._symbol_category(symbol)
        atr_percent = self.atr_percentages.get(category, 0.15)
        atr_value = analysis.get("atr", 0.0) or 0.0
        if atr_value != atr_value:
            atr_value = 0.0
        price_offset = max(atr_value * atr_percent, last_price * 0.001)
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
            return None
        message = f"📌 PENDING ORDER SET [{symbol}] {direction} @ {price:.5f}"
        self.logger.log(message)
        self.send_telegram(message)
        self._notify_order(message)
        return message

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

    def _notify_status(self, status: str) -> None:
        if self.status_callback:
            self.status_callback(status)

    def _notify_confidence(self, confidence: float, direction: str) -> None:
        if self.confidence_callback:
            self.confidence_callback(confidence, direction)

    def _notify_order(self, message: str) -> None:
        if self.order_callback:
            self.order_callback(message)

    def send_telegram(self, message: str) -> None:
        telegram_cfg = self.config.get("telegram", {})
        if not telegram_cfg.get("enabled"):
            return
        token = telegram_cfg.get("token")
        chat_id = telegram_cfg.get("chat_id")
        if not token or not chat_id or token == "YOUR_BOT_TOKEN" or chat_id == "YOUR_CHAT_ID":
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

    def _symbol_category(self, symbol: str) -> str:
        upper = symbol.upper()
        if upper.startswith("XAU") or upper.startswith("XAG") or "GOLD" in upper:
            return "metals"
        if any(token in upper for token in ("BTC", "ETH", "CRYPTO")):
            return "crypto"
        return "forex"
