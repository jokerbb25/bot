import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

import MetaTrader5 as mt5
import pandas as pd
import yaml

from utils.indicators import (
    calc_bollinger,
    calc_confidence,
    calc_ema,
    calc_macd,
    calc_rsi,
    detect_pullback,
)
from utils.logger import Logger


class BotEngine:
    def __init__(self, gui_log: Optional[Callable[[str], None]] = None, config_path: Optional[Path] = None):
        self.gui_log = gui_log
        self.config_path = config_path or Path(__file__).resolve().parent / "config.yaml"
        self.logger = Logger(gui_log=self.gui_log)
        self.stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.config = self._load_config()
        self.poll_interval = self.config.get("poll_interval", 5)

    def start(self) -> bool:
        with self._lock:
            if self._thread and self._thread.is_alive():
                self.logger.log("Engine already running.")
                return False
            self.stop_event.clear()
            self._thread = threading.Thread(target=self.run, daemon=True)
            self._thread.start()
            self.logger.log("Starting trading engine...")
            return True

    def stop(self) -> None:
        with self._lock:
            if not self._thread:
                return
            self.stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None

    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def run(self) -> None:
        if not mt5.initialize():
            self.logger.log("Failed to connect to MetaTrader5.")
            return
        self.logger.log("✅ MT5 Connected")
        symbols: List[str] = self.config.get("symbols", [])
        timeframe_name: str = self.config.get("timeframe", "M1")
        thresholds: Dict[str, float] = self.config.get("thresholds", {})

        timeframe = getattr(mt5, timeframe_name, mt5.TIMEFRAME_M1)

        try:
            while not self.stop_event.is_set():
                for symbol in symbols:
                    if self.stop_event.is_set():
                        break
                    try:
                        rates = self._fetch_rates(symbol, timeframe)
                        if rates is None:
                            continue
                        analysis_log = self._analyze_symbol(symbol, rates, thresholds)
                        if analysis_log:
                            self.logger.log(analysis_log)
                    except Exception as exc:
                        self.logger.log(f"Error analyzing {symbol}: {exc}")
                self.stop_event.wait(self.poll_interval)
        finally:
            mt5.shutdown()
            self.logger.log("🛑 Bot stopped — MT5 disconnected")

    def _fetch_rates(self, symbol: str, timeframe) -> Optional[pd.DataFrame]:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 500)
        if rates is None:
            self.logger.log(f"No data returned for {symbol}.")
            return None
        df = pd.DataFrame(rates)
        if df.empty:
            self.logger.log(f"Empty dataframe for {symbol}.")
            return None
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def _analyze_symbol(self, symbol: str, df: pd.DataFrame, thresholds: Dict[str, float]) -> Optional[str]:
        rsi_info = calc_rsi(df)
        ema_info = calc_ema(df)
        macd_info = calc_macd(df)
        bollinger_info = calc_bollinger(df)
        pullback_detected = detect_pullback(df)

        confidence = calc_confidence(
            rsi=rsi_info,
            ema=ema_info,
            macd=macd_info,
            pullback=pullback_detected,
            thresholds=thresholds,
        )

        direction = confidence.get("direction", "NONE")
        confidence_value = confidence.get("value", 0)

        return (
            f"[{symbol}] "
            f"RSI: {rsi_info['value']:.2f} | "
            f"EMA trend: {ema_info['trend']} | "
            f"MACD: {macd_info['signal']} | "
            f"Bollinger: {bollinger_info['position']} | "
            f"Pullback: {str(pullback_detected).lower()} | "
            f"Confidence: {confidence_value:.2f} → {direction}"
        )
