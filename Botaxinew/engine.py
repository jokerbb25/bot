from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import pandas as pd
import yaml
from ta.trend import ADXIndicator
from ta.volatility import BollingerBands

from utils.indicators import calc_atr, evaluate_indicators
from utils.logger import Logger
from utils.learning import LearningMemory

try:
    import requests
except Exception:  # pragma: no cover
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

        self.running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._thread_lock = threading.Lock()
        self.cycle_lock = threading.Lock()
        self._symbol_lock = threading.Lock()

        self.config = self._load_config()
        self.symbols: List[str] = list(self.config.get("symbols", []))
        self.selected_symbols: List[str] = list(self.symbols)
        self.focus_symbol: Optional[str] = None

        self.timeframe = self._resolve_timeframe(self.config.get("timeframe", "M1"))
        self.poll_interval = float(self.config.get("poll_interval", 1.0))
        self.history_bars = int(self.config.get("history_bars", 500))

        risk_config = self.config.get("risk", {})
        self.lot_high = float(risk_config.get("lot_high", 0.1))
        self.lot_low = float(risk_config.get("lot_low", 0.05))

        confidence_cfg = self.config.get("confidence", {})
        self.base_confidence = float(confidence_cfg.get("base_confidence", 0.7))
        self.lower_confidence = float(confidence_cfg.get("lower_confidence", 0.6))
        self.memory_boost = float(confidence_cfg.get("memory_boost", 0.05))

        self.sl_tp_mode = "Fixed pips"
        self.sl_value = 7.5
        self.tp_value = 15.0
        self.apply_sl_tp_on_pending = True
        self._apply_sl_tp_config_defaults()

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
        self.last_confidence = 0.0

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
            self._apply_sl_tp_config_defaults()
            self.running.set()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            return True

    def stop(self) -> None:
        with self._thread_lock:
            if not self._thread:
                return
            self.running.clear()
            thread = self._thread
            self._thread = None
        if thread:
            thread.join(timeout=self.poll_interval + 2)

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

    def set_sl_tp_mode(self, mode: str) -> None:
        self.sl_tp_mode = str(mode) if mode else "Fixed pips"

    def update_sl_tp_values(self, sl: float, tp: float) -> None:
        self.sl_value = float(sl)
        self.tp_value = float(tp)

    def apply_sl_tp_to_pending(self, enabled: bool) -> None:
        self.apply_sl_tp_on_pending = bool(enabled)

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _apply_sl_tp_config_defaults(self) -> None:
        cfg = self.config.get("sl_tp", {})
        self.set_sl_tp_mode(cfg.get("mode", "Fixed pips"))
        self.update_sl_tp_values(cfg.get("sl_value", 7.5), cfg.get("tp_value", 15.0))
        self.apply_sl_tp_to_pending(cfg.get("apply_to_pending", True))

    def _run_loop(self) -> None:
        if not mt5.initialize():
            self.logger.log("Failed to connect to MetaTrader5.")
            self._notify_status("Idle")
            return
        self.logger.log("✅ MT5 Connected")
        self.send_telegram("✅ Trading engine started")
        self._notify_status("Scanning")
        try:
            while self.running.is_set():
                symbols_to_scan = self._get_symbols_to_scan()
                if not symbols_to_scan:
                    self.logger.log("No symbols configured for scanning.")
                    time.sleep(self.poll_interval)
                    continue
                for symbol in symbols_to_scan:
                    if not self.running.is_set():
                        break
                    try:
                        self.scan_symbol(symbol)
                    except Exception as exc:  # pragma: no cover
                        self.logger.log(f"Error in {symbol}: {exc}")
                    time.sleep(0.4)
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

    def scan_symbol(self, symbol: str) -> None:
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

            close_prices = df["close"].astype(float)
            open_prices = df["open"].astype(float)
            high_prices = df["high"].astype(float)
            low_prices = df["low"].astype(float)
            if "tick_volume" in df.columns:
                volumes = df["tick_volume"].astype(float)
            elif "real_volume" in df.columns:
                volumes = df["real_volume"].astype(float)
            else:
                volumes = pd.Series([0.0] * len(df), index=df.index)

            base_direction = str(analysis.get("direction", "NONE"))
            strategies: List[Tuple[str, str, float]] = []

            if self.strategy_flags.get("rsi", True):
                rsi_signal = str(analysis.get("rsi_signal", "NONE"))
                if rsi_signal in {"CALL", "PUT"}:
                    strategies.append(("RSI", rsi_signal, 0.20))

            if self.strategy_flags.get("ema", True):
                ema_trend = str(analysis.get("ema_trend", "flat"))
                ema_signal = "CALL" if ema_trend == "up" else "PUT" if ema_trend == "down" else "NONE"
                if ema_signal in {"CALL", "PUT"}:
                    strategies.append(("EMA Trend", ema_signal, 0.20))

            if self.strategy_flags.get("macd", True):
                macd_signal = str(analysis.get("macd_signal", "NONE"))
                if macd_signal in {"CALL", "PUT"}:
                    strategies.append(("MACD", macd_signal, 0.20))

            if self.strategy_flags.get("pullback", True) and pullback_flag:
                pullback_direction = base_direction if base_direction in {"CALL", "PUT"} else str(analysis.get("macd_signal", "NONE"))
                if pullback_direction in {"CALL", "PUT"}:
                    strategies.append(("Pullback", pullback_direction, 0.15))

            bollinger_position = str(analysis.get("bollinger_position", "middle"))
            if bollinger_position == "lower":
                strategies.append(("Bollinger Position", "CALL", 0.10))
            elif bollinger_position == "upper":
                strategies.append(("Bollinger Position", "PUT", 0.10))

            base_votes = [signal for (_, signal, _) in strategies if signal in {"CALL", "PUT"}]
            if base_direction in {"CALL", "PUT"}:
                main_signal = base_direction
            elif base_votes:
                main_signal = max(set(base_votes), key=base_votes.count)
            else:
                main_signal = "CALL"

            try:
                bb = BollingerBands(close=close_prices, window=20, window_dev=2)
                upper = float(bb.bollinger_hband().iloc[-1])
                lower = float(bb.bollinger_lband().iloc[-1])
                last_close = float(close_prices.iloc[-1])
                if last_close <= lower:
                    strategies.append(("Bollinger Rebound", "CALL", 0.20))
                elif last_close >= upper:
                    strategies.append(("Bollinger Rebound", "PUT", 0.20))
            except Exception:
                pass

            try:
                adx = ADXIndicator(high_prices, low_prices, close_prices, window=14)
                adx_val = float(adx.adx().iloc[-1])
                if adx_val >= 25 and main_signal in {"CALL", "PUT"}:
                    strategies.append(("ADX Trend", main_signal, 0.15))
            except Exception:
                pass

            try:
                avg_volume = float(volumes.iloc[-10:].mean()) if len(volumes) >= 10 else float(volumes.mean())
                if len(volumes) and volumes.iloc[-1] > avg_volume * 1.8 and main_signal in {"CALL", "PUT"}:
                    strategies.append(("Volume Spike", main_signal, 0.15))
            except Exception:
                pass

            try:
                prev_high = float(high_prices.iloc[-2])
                prev_low = float(low_prices.iloc[-2])
                last_close = float(close_prices.iloc[-1])
                if last_close > prev_high:
                    strategies.append(("Breakout High", "CALL", 0.25))
                elif last_close < prev_low:
                    strategies.append(("Breakout Low", "PUT", 0.25))
            except Exception:
                pass

            try:
                body = abs(float(open_prices.iloc[-1]) - float(close_prices.iloc[-1]))
                wick = abs(float(high_prices.iloc[-1]) - float(low_prices.iloc[-1]))
                if body > wick * 0.70:
                    momentum_direction = "CALL" if float(close_prices.iloc[-1]) > float(open_prices.iloc[-1]) else "PUT"
                    strategies.append(("Momentum Candle", momentum_direction, 0.20))
            except Exception:
                pass

            confidence = sum(weight for (_, _, weight) in strategies)
            confidence = min(confidence, 1.0)

            votes = [signal for (_, signal, _) in strategies if signal in {"CALL", "PUT"}]
            if votes:
                direction = max(set(votes), key=votes.count)
            else:
                direction = "NONE"

            for name, signal_name, weight in strategies:
                self.logger.log(f"[{symbol}] {name} → {signal_name} (weight={weight:.2f})")

            analysis["confidence"] = confidence
            analysis["direction"] = direction

            memory_applied = False
            if self.strategy_flags.get("memory", True) and direction in {"CALL", "PUT"}:
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
                confidence = min(float(boosted_confidence), 1.0)

            analysis["confidence"] = confidence
            self.last_confidence = confidence

            log_line = (
                f"[{symbol}] RSI {float(analysis.get('rsi_value', 0.0)):.2f} | "
                f"EMA: {analysis.get('ema_trend', 'flat')} | "
                f"MACD: {analysis.get('macd_signal', 'NONE')} | "
                f"Pullback: {str(analysis.get('pullback', False)).lower()} | "
                f"Confidence {confidence:.2f}"
            )
            self.logger.log(log_line)
            final_marker = " ✅" if direction in {"CALL", "PUT"} else ""
            self.logger.log(f"[{symbol}] CONFIDENCE FINAL: {confidence * 100:.0f}% → {direction}{final_marker}")
            self._notify_confidence(confidence, direction)

            last_price = float(df["close"].iloc[-1])
            atr_value = float(analysis.get("atr", calc_atr(df)))

            if direction in {"CALL", "PUT"}:
                if confidence >= self.base_confidence:
                    executed = self.execute_market_order(symbol, direction, confidence, df, atr_value, analysis)
                    if executed:
                        self._notify_status("Operation executed")
                        return
                elif pullback_flag and confidence >= self.lower_confidence:
                    pending_price = self._determine_pending_price(direction, last_price, atr_value)
                    executed = self.execute_pending_order(symbol, direction, pending_price, df, atr_value)
                    if executed:
                        self._notify_status("Operation executed")
                        return
            self._notify_status("Scanning")
        finally:
            self.cycle_lock.release()

    def execute_market_order(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        df: pd.DataFrame,
        atr_value: float,
        analysis: Dict[str, Any],
    ) -> bool:
        if not self._acquire_trade_slot():
            return False
        lot = self._select_lot_by_confidence()
        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            self.logger.log(f"No tick data available for {symbol}.")
            return False
        entry_price = tick.ask if direction == "CALL" else tick.bid
        sl_price, tp_price = self._calc_sl_tp_prices(symbol, direction, entry_price, df, atr_value)
        order = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if direction == "CALL" else mt5.ORDER_TYPE_SELL,
            "price": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 10,
            "magic": 123456,
            "comment": "AXINEW-market",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = mt5.order_send(order)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.log(f"Market order failed for {symbol}: {getattr(result, 'retcode', 'no response')}")
            return False
        message = f"✅ ORDER EXECUTED [{symbol}] → {direction} lot={lot:.2f}"
        self.logger.log(message)
        self.send_telegram(message)

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

    def execute_pending_order(
        self,
        symbol: str,
        direction: str,
        price: float,
        df: pd.DataFrame,
        atr_value: float,
    ) -> bool:
        if not self._acquire_trade_slot():
            return False
        lot = self._select_lot_by_confidence()
        mt5.symbol_select(symbol, True)
        if not self.apply_sl_tp_on_pending:
            sl_price, tp_price = 0.0, 0.0
        else:
            sl_price, tp_price = self._calc_sl_tp_prices(symbol, direction, price, df, atr_value)
        order = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY_LIMIT if direction == "CALL" else mt5.ORDER_TYPE_SELL_LIMIT,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 10,
            "magic": 123456,
            "comment": "AXINEW-pending",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        result = mt5.order_send(order)
        if result is None or result.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            self.logger.log(f"Pending order failed for {symbol}: {getattr(result, 'retcode', 'no response')}")
            return False
        message = f"📌 PENDING ORDER SET [{symbol}] {direction} @ {price:.5f}"
        self.logger.log(message)
        self.send_telegram(message)
        event = self._build_trade_event(
            symbol=symbol,
            decision=direction,
            confidence=self.last_confidence,
            result="PENDING",
            pnl=0.0,
            notes="Pending order",
        )
        self._notify_order(event)
        return True

    def _determine_pending_price(self, direction: str, last_price: float, atr_value: float) -> float:
        offset = max(atr_value * 0.5, last_price * 0.001)
        if direction == "CALL":
            return last_price - offset
        return last_price + offset

    def _select_lot_by_confidence(self) -> float:
        if self.last_confidence >= self.base_confidence:
            return round(self.lot_high, 2)
        return round(self.lot_low, 2)

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

    def _points_per_pip(self, symbol: str) -> float:
        info = mt5.symbol_info(symbol)
        if info and info.digits in (3, 5):
            return 10.0
        return 1.0

    def _calc_sl_tp_prices(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        df: pd.DataFrame,
        atr_value: float,
    ) -> Tuple[float, float]:
        info = mt5.symbol_info(symbol)
        point = info.point if info else 0.0001
        if self.sl_tp_mode == "Fixed pips":
            pip_points = self._points_per_pip(symbol)
            sl_delta = float(self.sl_value) * pip_points * point
            tp_delta = float(self.tp_value) * pip_points * point
        else:
            atr_used = atr_value if atr_value else calc_atr(df)
            sl_delta = float(self.sl_value) * atr_used
            tp_delta = float(self.tp_value) * atr_used
        if direction == "CALL":
            sl_price = entry_price - sl_delta
            tp_price = entry_price + tp_delta
        else:
            sl_price = entry_price + sl_delta
            tp_price = entry_price - tp_delta
        return float(sl_price), float(tp_price)

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
            except Exception as exc:  # pragma: no cover
                self.logger.log(f"Telegram send failed: {exc}")

        threading.Thread(target=_worker, daemon=True).start()

    def _resolve_timeframe(self, name: str):
        if hasattr(mt5, name):
            return getattr(mt5, name)
        return mt5.TIMEFRAME_M1
