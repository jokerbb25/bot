import sys
import time
import json
import threading
import logging
import warnings
import csv
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from joblib import dump, load
except ImportError:  # pragma: no cover
    dump = None  # type: ignore
    load = None  # type: ignore
try:
    from skopt import gp_minimize
except ImportError:  # pragma: no cover
    gp_minimize = None  # type: ignore
try:  # pragma: no cover
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
except Exception:  # pragma: no cover
    tf = None  # type: ignore
    keras = None  # type: ignore
    layers = None  # type: ignore

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    import websocket  # type: ignore
except ImportError:  # pragma: no cover
    websocket = None

# ===============================================================
# CONFIG & CONSTANTS
# ===============================================================
APP_ID = "1089"
API_TOKEN = "dK57Ark9QreDexO"
SYMBOLS = ["R_25", "R_50", "R_75", "R_100"]
GRANULARITY = 60
CANDLE_COUNT = 200
STAKE = 1.0
PAYOUT = 0.9
TRADE_DELAY = 2.0
MAX_DAILY_TRADES = 200
MAX_DAILY_LOSS = -75.0
MAX_DAILY_PROFIT = 100.0
COOLDOWN_AFTER_LOSS = 60
MAX_DRAWDOWN = -150.0
MIN_TRADE_CONFIDENCE = 0.45

AI_ENABLED = True
AI_PASSIVE_MODE = True
AI_MODEL_PATH = "models/adaptive_ai.pkl"
AI_ENDPOINT = "http://localhost:11434/api/generate"
AI_TIMEOUT = 3
AI_SEMI_ACTIVE_THRESHOLD = 300
AI_AUTONOMOUS_THRESHOLD = 10000
AI_AUTONOMOUS_ACCURACY = 0.65

TRADES_LOG_PATH = "trades_log.csv"
ADAPTIVE_STATE_PATH = Path("adaptive_ai_state.npz")
STRATEGY_CONFIG_PATH = Path("strategies_config.json")
ADVISORY_INTERVAL_SEC = 180

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")

operation_active = False
TRADE_DURATION_SECONDS = 60
RESULT_POLL_INTERVAL = 5
RESUME_MESSAGE = "🔁 Reanudando análisis del mercado..."


# ===============================================================
# DATA STRUCTURES
# ===============================================================
@dataclass
class Candle:
    epoch: int
    open: float
    high: float
    low: float
    close: float


@dataclass
class StrategyResult:
    signal: str
    score: float
    reasons: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeRecord:
    timestamp: datetime
    symbol: str
    decision: str
    confidence: float
    result: Optional[str]
    pnl: float
    reasons: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===============================================================
# UTILITY FUNCTIONS
# ===============================================================
def to_dataframe(candles: List[Candle]) -> pd.DataFrame:
    data = {
        "open": [c.open for c in candles],
        "high": [c.high for c in candles],
        "low": [c.low for c in candles],
        "close": [c.close for c in candles],
    }
    return pd.DataFrame(data)


def ema(series: pd.Series, period: int, symbol: Optional[str] = None) -> pd.Series:
    window = int(period)
    gestor = globals().get("auto_learn")
    activo = symbol
    if activo is None and gestor is not None:
        try:
            activo = gestor.get_active_symbol()
        except Exception:
            activo = None
    if activo and gestor is not None:
        try:
            ajustes = gestor.get_symbol_snapshot(activo)
        except Exception:
            ajustes = {}
        if abs(period - 9) <= 3:
            window = int(max(2, round(ajustes.get("ema_fast", float(period)))))
        elif abs(period - 21) <= 5:
            window = int(max(2, round(ajustes.get("ema_slow", float(period)))))
    return EMAIndicator(close=series, window=window, fillna=False).ema_indicator()


def sma(series: pd.Series, period: int) -> pd.Series:
    return SMAIndicator(close=series, window=period, fillna=False).sma_indicator()


def rsi(series: pd.Series, period: int = 14, symbol: Optional[str] = None) -> pd.Series:
    window = int(period)
    gestor = globals().get("auto_learn")
    activo = symbol
    if activo is None and gestor is not None:
        try:
            activo = gestor.get_active_symbol()
        except Exception:
            activo = None
    if activo and gestor is not None:
        try:
            ajustes = gestor.get_symbol_snapshot(activo)
            window = int(max(3, round(ajustes.get("rsi_period", float(period)))))
        except Exception:
            window = int(period)
    return RSIIndicator(close=series, window=window, fillna=False).rsi()


def bollinger_bands(series: pd.Series, period: int = 20, num_dev: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    bb = BollingerBands(close=series, window=period, window_dev=num_dev, fillna=False)
    return bb.bollinger_lband(), bb.bollinger_hband()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=period, fillna=False).average_true_range()


def donchian_channels(df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series]:
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    return lower, upper


def log_trade(record: TradeRecord) -> None:
    try:
        row = {
            "timestamp": record.timestamp.isoformat(),
            "symbol": record.symbol,
            "decision": record.decision,
            "confidence": record.confidence,
            "result": record.result,
            "pnl": record.pnl,
            "reasons": "|".join(record.reasons),
        }
        df = pd.DataFrame([row])
        path = Path(TRADES_LOG_PATH)
        if not path.exists():
            df.to_csv(path, index=False)
        else:
            df.to_csv(path, mode="a", header=False, index=False)
    except Exception as exc:  # pragma: no cover
        logging.debug(f"Error al registrar operación: {exc}")


class auto_learning:
    def __init__(self) -> None:
        self.csv_path = Path("botderivcsv.csv")
        self.csv_fields = [
            "timestamp",
            "asset",
            "direction",
            "result",
            "signal_source",
            "signal_direction",
            "price_change",
            "rsi",
            "ema",
            "bollinger",
            "volatility",
            "confidence",
            "indicator_confidence",
            "ml_probability",
            "stake",
            "rsi_bias",
            "ema_bias",
            "rsi_adjusted",
            "ema_fast_adjusted",
            "ema_slow_adjusted",
        ]
        self.lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.bias_lock = threading.Lock()
        self.memory_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.asset_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"wins": 0, "total": 0})
        self.global_totals = {"wins": 0, "total": 0}
        self.last_trades: deque = deque(maxlen=50)
        self.training_data: List[List[float]] = []
        self.training_labels: List[int] = []
        self.trade_counter = 0
        self.model: Optional[RandomForestClassifier] = None
        self.biases: Dict[str, Dict[str, float]] = {
            symbol: {"RSI": 0.0, "EMA": 0.0} for symbol in SYMBOLS
        }
        self.last_signal: Dict[str, Dict[str, str]] = {
            symbol: {"source": "COMBINED"} for symbol in SYMBOLS
        }
        self.weights_lock = threading.Lock()
        self.weights: Dict[str, float] = {
            "RSI": 1.0,
            "EMA": 1.0,
            "BOLL": 1.0,
            "ADX": 0.8,
            "MACD": 0.8,
        }
        self.symbol_weights: Dict[str, float] = {symbol: 1.0 for symbol in SYMBOLS}
        self.batch_lock = threading.Lock()
        self.batch_results: List[Tuple[str, str, str, float, str]] = []
        self.batch_size = 10
        self.adx_prev: Dict[str, float] = {}
        self.last_prediction = 0.5
        self.memory: List[Dict[str, Any]] = []
        self.memory_limit = 5000
        self.recent_results: deque = deque(maxlen=200)
        self.learning_rate = 0.02
        self.min_confidence = MIN_TRADE_CONFIDENCE
        self.rsi_high_threshold = 70.0
        self.adx_min_threshold = 20.0
        self.predictive_model_path = Path("predictive_model.pkl")
        self.neural_model: Any = None
        self.neural_model_path = Path("neural_predictor.h5")
        self.reinforce_batches = 0
        self.optimize_batches = 0
        self.learning_event = threading.Event()
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        self._pending_context: Dict[str, Dict[str, float]] = {}
        self.symbol_profiles: Dict[str, Dict[str, Any]] = {
            symbol: self._default_symbol_profile() for symbol in SYMBOLS
        }
        for profile in self.symbol_profiles.values():
            profile["learning_rate"] = self.learning_rate
        self._current_symbol: Optional[str] = None
        self.current_regime: Optional[str] = None
        self.regime_baseline_weights: Dict[str, float] = dict(self.weights)
        self.regime_baseline_min_conf: float = self.min_confidence
        self._ensure_csv()

    def _ensure_csv(self) -> None:
        if self.csv_path.exists():
            return
        with self.lock:
            if self.csv_path.exists():
                return
            try:
                with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=self.csv_fields)
                    writer.writeheader()
            except Exception as exc:  # pragma: no cover
                logging.debug(f"No se pudo crear CSV de autoaprendizaje: {exc}")

    def _default_symbol_profile(self) -> Dict[str, Any]:
        return {
            "wins": 0,
            "total": 0,
            "recent": deque(maxlen=30),
            "loss_streak": 0,
            "win_streak": 0,
            "rsi_period": 14.0,
            "rsi_baseline": 14.0,
            "rsi_lower": 30.0,
            "rsi_upper": 70.0,
            "ema_fast": 9.0,
            "ema_slow": 21.0,
            "ema_fast_baseline": 9.0,
            "ema_slow_baseline": 21.0,
            "ema_tolerance": 0.0,
            "learning_rate": 0.02,
            "restore_rate": 0.01,
        }

    def _get_symbol_profile(self, asset: str) -> Dict[str, Any]:
        if asset not in self.symbol_profiles:
            self.symbol_profiles[asset] = self._default_symbol_profile()
        return self.symbol_profiles[asset]

    def set_active_symbol(self, asset: Optional[str]) -> None:
        with self.lock:
            self._current_symbol = asset

    def get_active_symbol(self) -> Optional[str]:
        with self.lock:
            return self._current_symbol

    def _symbol_snapshot(self, profile: Dict[str, Any]) -> Dict[str, float]:
        return {
            "rsi_period": float(profile.get("rsi_period", 14.0)),
            "rsi_lower": float(profile.get("rsi_lower", 30.0)),
            "rsi_upper": float(profile.get("rsi_upper", 70.0)),
            "ema_fast": float(profile.get("ema_fast", 9.0)),
            "ema_slow": float(profile.get("ema_slow", 21.0)),
            "ema_tolerance": float(profile.get("ema_tolerance", 0.0)),
        }

    def get_symbol_snapshot(self, asset: str) -> Dict[str, float]:
        with self.lock:
            profile = self._get_symbol_profile(asset)
            return self._symbol_snapshot(profile)

    def get_symbol_weight(self, asset: str) -> float:
        with self.lock:
            return float(self.symbol_weights.get(asset, 1.0))

    def normalize(self, value: float, min_val: float, max_val: float) -> float:
        if max_val == min_val:
            return 0.0
        scaled = (value - min_val) / (max_val - min_val)
        return float(max(0.0, min(1.0, scaled)))

    def calculate_confidence(
        self,
        rsi_value: float,
        ema_value: float,
        boll_value: float,
        adx_value: float,
        macd_value: float,
    ) -> float:
        rsi_n = self.normalize(rsi_value, 0.0, 100.0)
        ema_n = self.normalize(ema_value, -0.5, 0.5)
        boll_n = self.normalize(boll_value, 0.0, 1.0)
        adx_n = self.normalize(adx_value, 0.0, 50.0)
        macd_n = self.normalize(macd_value, -2.0, 2.0)
        with self.weights_lock:
            weights_snapshot = dict(self.weights)
        total_weight = float(sum(weights_snapshot.values())) or 1.0
        confidence = (
            rsi_n * weights_snapshot.get("RSI", 1.0)
            + ema_n * weights_snapshot.get("EMA", 1.0)
            + boll_n * weights_snapshot.get("BOLL", 1.0)
            + adx_n * weights_snapshot.get("ADX", 0.8)
            + macd_n * weights_snapshot.get("MACD", 0.8)
        ) / total_weight
        return float(max(0.0, min(1.0, confidence)))

    def calculate_adx(
        self,
        symbol: str,
        highs: Iterable[float],
        lows: Iterable[float],
        closes: Iterable[float],
        period: int = 14,
    ) -> float:
        highs_list = list(highs)
        lows_list = list(lows)
        closes_list = list(closes)
        if len(highs_list) < 2 or len(lows_list) < 2 or len(closes_list) < 2:
            return 0.0
        plus_dm = highs_list[-1] - highs_list[-2] if highs_list[-1] > highs_list[-2] else 0.0
        minus_dm = lows_list[-2] - lows_list[-1] if lows_list[-1] < lows_list[-2] else 0.0
        plus_dm = max(0.0, float(plus_dm))
        minus_dm = max(0.0, float(minus_dm))
        tr_components = [
            highs_list[-1] - lows_list[-1],
            abs(highs_list[-1] - closes_list[-2]),
            abs(lows_list[-1] - closes_list[-2]),
        ]
        true_range = max(tr_components) if tr_components else 0.0
        denominator = plus_dm + minus_dm + 1e-9
        dx = (abs(plus_dm - minus_dm) / denominator) * 100.0
        prev_adx = self.adx_prev.get(symbol)
        if prev_adx is None:
            adx = dx
        else:
            adx = (prev_adx * (period - 1) + dx) / period
        if true_range <= 0.0:
            adx = max(0.0, min(100.0, adx))
        self.adx_prev[symbol] = float(adx)
        return float(max(0.0, min(100.0, adx)))

    def calculate_macd(
        self,
        closes: Iterable[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> float:
        close_list = list(closes)
        if len(close_list) < 3:
            return 0.0
        series = pd.Series(close_list, dtype=float)
        ema_fast_series = series.ewm(span=fast, adjust=False).mean()
        ema_slow_series = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast_series - ema_slow_series
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = float(macd_line.iloc[-1] - signal_line.iloc[-1]) if not macd_line.empty else 0.0
        return macd_hist

    def _clamp_weights(self) -> None:
        with self.weights_lock:
            for key, value in list(self.weights.items()):
                self.weights[key] = float(np.clip(value, 0.2, 3.0))

    def reset_regime_baseline(self) -> None:
        with self.weights_lock:
            self.regime_baseline_weights = dict(self.weights)
        with self.lock:
            self.regime_baseline_min_conf = self.min_confidence
        self.current_regime = None

    def detect_market_regime(
        self,
        adx_value: float,
        bollinger_width: float,
        volatility: float,
    ) -> str:
        adx_value = float(adx_value)
        bollinger_width = float(bollinger_width)
        volatility = float(volatility)
        if adx_value > 25.0 and volatility < 0.02:
            return "TRENDING"
        if adx_value < 20.0 and volatility < 0.015 and bollinger_width < 0.015:
            return "RANGING"
        if volatility > 0.03 or bollinger_width > 0.03:
            return "VOLATILE"
        return "CALM"

    def apply_market_regime(self, regime: str) -> Dict[str, Any]:
        regime = str(regime).upper()
        if regime not in {"TRENDING", "RANGING", "VOLATILE", "CALM"}:
            regime = "CALM"
        if regime != self.current_regime:
            self.reset_regime_baseline()
            self.current_regime = regime
        with self.weights_lock:
            base_weights = dict(self.regime_baseline_weights)
            new_weights = dict(base_weights)
            if regime == "TRENDING":
                new_weights["EMA"] = base_weights.get("EMA", 1.0) * 1.05
                new_weights["RSI"] = base_weights.get("RSI", 1.0) * 0.95
            elif regime == "RANGING":
                new_weights["RSI"] = base_weights.get("RSI", 1.0) * 1.05
                new_weights["MACD"] = base_weights.get("MACD", 0.8) * 0.95
            elif regime == "VOLATILE":
                new_weights = dict(base_weights)
            else:
                new_weights = dict(base_weights)
            self.weights.update(new_weights)
            self._clamp_weights()
            weights_snapshot = dict(self.weights)
        with self.lock:
            base_min = float(self.regime_baseline_min_conf)
            if regime == "VOLATILE":
                self.min_confidence = float(np.clip(base_min + 0.05, 0.3, 0.95))
            elif regime == "CALM":
                self.min_confidence = float(np.clip(base_min - 0.02, 0.3, 0.95))
            else:
                self.min_confidence = base_min
            min_conf_snapshot = self.min_confidence
        return {"weights": weights_snapshot, "min_confidence": min_conf_snapshot, "regime": regime}

    def train_neural_predictor(self) -> None:
        if keras is None or layers is None:
            logging.debug("TensorFlow no disponible, se omite el entrenamiento de la red neuronal")
            return
        with self.memory_lock:
            if len(self.memory) < 500:
                return
            data = pd.DataFrame(self.memory)
        try:
            X = data[["RSI", "EMA", "MACD", "ADX", "hour"]].values
            y = (data["result"].str.upper() == "WIN").astype(int).values
            model = keras.Sequential(
                [
                    layers.Input(shape=(5,)),
                    layers.Dense(32, activation="relu"),
                    layers.Dense(16, activation="relu"),
                    layers.Dense(1, activation="sigmoid"),
                ]
            )
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            model.fit(X, y, epochs=15, batch_size=32, verbose=0)
            model.save(self.neural_model_path)
            self.neural_model = model
            logging.info("🤖 Neural predictor trained and saved.")
        except Exception as exc:  # pragma: no cover
            logging.debug(f"No se pudo entrenar el predictor neuronal: {exc}")

    def _load_neural_model(self) -> None:
        if keras is None:
            return
        if self.neural_model is not None:
            return
        if not self.neural_model_path.exists():
            return
        try:
            self.neural_model = keras.models.load_model(self.neural_model_path)
        except Exception as exc:  # pragma: no cover
            logging.debug(f"No se pudo cargar el predictor neuronal: {exc}")
            self.neural_model = None

    def predict_with_neural(
        self,
        rsi: float,
        ema: float,
        macd: float,
        adx: float,
        hour: int,
    ) -> float:
        if keras is None or layers is None:
            return 0.5
        self._load_neural_model()
        if self.neural_model is None:
            return 0.5
        try:
            X = np.array([[float(rsi), float(ema), float(macd), float(adx), float(hour)]], dtype=float)
            prob = float(self.neural_model.predict(X, verbose=0)[0][0])
            return float(max(0.0, min(1.0, prob)))
        except Exception:  # pragma: no cover
            return 0.5

    def record_result(
        self,
        symbol: str,
        result: str,
        signal_direction: str,
        price_change: float,
        source: str,
    ) -> None:
        entry = (
            symbol,
            str(result).upper(),
            str(signal_direction).upper(),
            float(price_change),
            source,
        )
        with self.batch_lock:
            self.batch_results.append(entry)
            if len(self.batch_results) >= self.batch_size:
                batch = list(self.batch_results)
                self.batch_results.clear()
            else:
                batch = None
        if batch:
            self.executor.submit(self.process_batch, batch)

    def process_batch(self, entries: Optional[List[Tuple[str, str, str, float, str]]] = None) -> None:
        if entries is None:
            with self.batch_lock:
                entries = list(self.batch_results)
                self.batch_results.clear()
        if not entries:
            return
        results_by_symbol: Dict[str, List[Tuple[str, str, float, str]]] = defaultdict(list)
        for symbol, result, direction, price_change, source in entries:
            if result not in {"WIN", "LOSS"}:
                continue
            results_by_symbol[symbol].append((result, direction, price_change, source))
        for symbol, symbol_entries in results_by_symbol.items():
            total = len(symbol_entries)
            if total == 0:
                continue
            wins = sum(1 for result, _, _, _ in symbol_entries if result == "WIN")
            win_rate = wins / total
            with self.lock:
                current_weight = self.symbol_weights.get(symbol, 1.0)
                if win_rate < 0.45:
                    current_weight *= 0.97
                elif win_rate > 0.65:
                    current_weight *= 1.03
                self.symbol_weights[symbol] = float(np.clip(current_weight, 0.5, 1.5))
            with self.weights_lock:
                if win_rate < 0.55:
                    self.weights["RSI"] *= 0.99
                    self.weights["EMA"] *= 0.99
                    self.weights["BOLL"] *= 0.99
                    self.weights["ADX"] *= 0.98
                    self.weights["MACD"] *= 0.98
                elif win_rate > 0.65:
                    self.weights["RSI"] *= 1.01
                    self.weights["EMA"] *= 1.01
                    self.weights["BOLL"] *= 1.01
                    self.weights["ADX"] *= 1.02
                    self.weights["MACD"] *= 1.02
            self._clamp_weights()
            self.reset_regime_baseline()
            with self.weights_lock:
                weights_snapshot = dict(self.weights)
            logging.info(
                f"Batch update for {symbol}: win_rate={win_rate:.2f}, weights={weights_snapshot}, symbol_weight={self.get_symbol_weight(symbol):.2f}"
            )

    def backtest_adaptive(self, historical_data: Iterable[Dict[str, Any]], window: int = 1000) -> None:
        data_list = list(historical_data)
        if not data_list or window <= 0 or len(data_list) <= window:
            return
        step = 250
        for start in range(0, len(data_list) - window, step):
            train_slice = data_list[start : start + window - step]
            test_slice = data_list[start + window - step : start + window]
            if not train_slice or not test_slice:
                continue
            self.optimize_on_data(train_slice)
            win_rate = self.simulate_trades(test_slice)
            logging.info(
                f"Walk-forward segment {start}–{start + window}: winrate={win_rate:.2f}"
            )

    def optimize_on_data(self, data_slice: Iterable[Dict[str, Any]]) -> None:
        results = [str(item.get("result", "")).upper() for item in data_slice if str(item.get("result", "")).upper() in {"WIN", "LOSS"}]
        total = len(results)
        if total == 0:
            return
        wins = sum(1 for item in results if item == "WIN")
        ratio = wins / total
        adjustment = 1.0 + (ratio - 0.5) * 0.02
        with self.weights_lock:
            for key in list(self.weights.keys()):
                self.weights[key] *= adjustment
        self._clamp_weights()

    def simulate_trades(self, data_slice: Iterable[Dict[str, Any]]) -> float:
        outcomes = [str(item.get("result", "")).upper() for item in data_slice if str(item.get("result", "")).upper() in {"WIN", "LOSS"}]
        if not outcomes:
            return 0.0
        wins = sum(1 for outcome in outcomes if outcome == "WIN")
        total = len(outcomes)
        return wins / total if total else 0.0


    def _apply_symbol_learning(
        self,
        asset: str,
        etiqueta: Optional[int],
        direction: str,
        rsi_value: float,
        ema_value: float,
        volatility_value: float,
    ) -> Dict[str, float]:
        profile = self._get_symbol_profile(asset)
        if etiqueta is None:
            return self._symbol_snapshot(profile)
        lr = float(profile.get("learning_rate", 0.02))
        restore_rate = float(profile.get("restore_rate", 0.01))
        profile.setdefault("recent", deque(maxlen=30)).append(etiqueta)
        if etiqueta == 0:
            profile["loss_streak"] = int(profile.get("loss_streak", 0)) + 1
            profile["win_streak"] = 0
            loss_factor = 1.0 + max(0.01, min(0.03, lr))
            profile["rsi_period"] = float(
                np.clip(
                    profile.get("rsi_period", profile["rsi_baseline"])
                    * loss_factor,
                    profile["rsi_baseline"] * 0.8,
                    profile["rsi_baseline"] * 1.6,
                )
            )
            ema_loss_factor = 1.0 + max(0.005, min(0.025, lr * 0.8))
            profile["ema_fast"] = float(
                np.clip(
                    profile.get("ema_fast", profile["ema_fast_baseline"])
                    * ema_loss_factor,
                    profile["ema_fast_baseline"] * 0.7,
                    profile["ema_fast_baseline"] * 1.6,
                )
            )
            profile["ema_slow"] = float(
                np.clip(
                    profile.get("ema_slow", profile["ema_slow_baseline"])
                    * ema_loss_factor,
                    profile["ema_slow_baseline"] * 0.7,
                    profile["ema_slow_baseline"] * 1.6,
                )
            )
            if direction == "CALL" and rsi_value < 40.0:
                profile["rsi_lower"] = float(
                    np.clip(
                        profile.get("rsi_lower", 30.0) + max(0.2, lr * 10.0),
                        25.0,
                        45.0,
                    )
                )
            elif direction == "PUT" and rsi_value > 60.0:
                profile["rsi_upper"] = float(
                    np.clip(
                        profile.get("rsi_upper", 70.0) - max(0.2, lr * 10.0),
                        55.0,
                        75.0,
                    )
                )
            spread_sensitivity = float(abs(ema_value))
            tolerance_increment = max(0.05, min(0.5, spread_sensitivity * 0.1))
            profile["ema_tolerance"] = float(
                np.clip(
                    profile.get("ema_tolerance", 0.0) + tolerance_increment,
                    0.0,
                    5.0,
                )
            )
            if profile.get("loss_streak", 0) >= 2:
                logging.info(f"🧠 Recalibrating RSI/EMA for {asset} after {profile['loss_streak']} losses")
        else:
            profile["win_streak"] = int(profile.get("win_streak", 0)) + 1
            profile["loss_streak"] = 0
            profile["rsi_period"] = float(
                profile.get("rsi_period", profile["rsi_baseline"])
                - (profile.get("rsi_period", profile["rsi_baseline"]) - profile["rsi_baseline"]) * restore_rate
            )
            profile["ema_fast"] = float(
                profile.get("ema_fast", profile["ema_fast_baseline"])
                - (profile.get("ema_fast", profile["ema_fast_baseline"]) - profile["ema_fast_baseline"]) * restore_rate
            )
            profile["ema_slow"] = float(
                profile.get("ema_slow", profile["ema_slow_baseline"])
                - (profile.get("ema_slow", profile["ema_slow_baseline"]) - profile["ema_slow_baseline"]) * restore_rate
            )
            profile["ema_tolerance"] = float(
                max(0.0, profile.get("ema_tolerance", 0.0) - max(0.05, restore_rate * 10.0))
            )
            if profile.get("win_streak", 0) >= 3:
                logging.info(f"📈 Restoring default EMA for {asset} (performance recovered)")
            perfil_accuracy = 0.0
            if profile.get("total", 0) > 0:
                perfil_accuracy = profile.get("wins", 0) / max(1, profile.get("total", 1))
            if perfil_accuracy > 0.6:
                ajuste = max(0.1, restore_rate * 10.0)
                profile["rsi_lower"] = float(
                    np.clip(
                        profile.get("rsi_lower", 30.0) - ajuste,
                        25.0,
                        35.0,
                    )
                )
                profile["rsi_upper"] = float(
                    np.clip(
                        profile.get("rsi_upper", 70.0) + ajuste,
                        65.0,
                        75.0,
                    )
                )
        return self._symbol_snapshot(profile)

    def register_trade_context(
        self,
        asset: str,
        direction: str,
        confidence: float,
        entry_price: float,
        signal_source: str,
        indicator_confidence: float = 0.0,
        ml_probability: float = 0.0,
        stake: float = 0.0,
    ) -> None:
        with self.lock:
            self._get_symbol_profile(asset)
            self._pending_context[asset] = {
                "direction": direction,
                "confidence": float(confidence),
                "entry_price": float(entry_price),
                "signal_source": signal_source,
                "indicator_confidence": float(indicator_confidence),
                "ml_probability": float(ml_probability),
                "stake": float(stake),
            }

    def record_signal_source(self, asset: str, source: str) -> None:
        with self.lock:
            state = self.last_signal.setdefault(asset, {})
            state["source"] = source

    def update_history(
        self,
        result: str,
        asset: str,
        rsi_value: float,
        ema_value: float,
        boll_value: float,
        volatility_value: float,
        price_change: float,
    ) -> None:
        self.executor.submit(
            self._update_history_sync,
            result,
            asset,
            float(rsi_value),
            float(ema_value),
            float(boll_value),
            float(volatility_value),
            float(price_change),
        )

    def _update_history_sync(
        self,
        result: str,
        asset: str,
        rsi_value: float,
        ema_value: float,
        boll_value: float,
        volatility_value: float,
        price_change: float,
    ) -> None:
        context = self._pop_context(asset)
        direction = context.get("direction", "NONE")
        confidence = context.get("confidence", 0.0)
        indicator_confidence = context.get("indicator_confidence", 0.0)
        ml_probability = context.get("ml_probability", 0.0)
        stake_value = context.get("stake", 0.0)
        signal_source = context.get("signal_source") or self.last_signal.get(asset, {}).get("source", "COMBINED")
        timestamp_now = datetime.now(timezone.utc).isoformat()
        resultado = str(result).upper()
        etiqueta = 1 if resultado == "WIN" else 0 if resultado == "LOSS" else None
        self._ensure_csv()
        should_train = False
        adjusted_rsi_value = float(rsi_value)
        adjusted_ema_value = float(ema_value)
        with self.lock:
            profile = self._get_symbol_profile(asset)
            snapshot = self._symbol_snapshot(profile)
            if etiqueta is not None:
                stats = self.asset_stats[asset]
                stats["total"] += 1
                if etiqueta == 1:
                    stats["wins"] += 1
                perfil = self._get_symbol_profile(asset)
                perfil["total"] = int(perfil.get("total", 0)) + 1
                if etiqueta == 1:
                    perfil["wins"] = int(perfil.get("wins", 0)) + 1
                self.global_totals["total"] += 1
                if etiqueta == 1:
                    self.global_totals["wins"] += 1
            ajustes = self._apply_symbol_learning(
                asset,
                etiqueta,
                direction,
                rsi_value,
                ema_value,
                volatility_value,
            )
            symbol_params = ajustes if ajustes else snapshot
            adjusted_rsi_value = float(rsi_value + (symbol_params.get("rsi_period", 14.0) - 14.0) * 0.5)
            adjusted_ema_value = float(ema_value + symbol_params.get("ema_tolerance", 0.0) * 0.1)
            with self.bias_lock:
                bias_state = self.biases.setdefault(asset, {"RSI": 0.0, "EMA": 0.0})
                rsi_bias = float(bias_state.get("RSI", 0.0))
                ema_bias = float(bias_state.get("EMA", 0.0))
            trade_entry = {
                "asset": asset,
                "direction": direction,
                "result": resultado,
                "rsi": rsi_value,
                "ema": ema_value,
                "boll": boll_value,
                "volatility": volatility_value,
                "confidence": confidence,
                "indicator_confidence": indicator_confidence,
                "ml_probability": ml_probability,
                "stake": stake_value,
                "label": etiqueta,
                "rsi_adjusted": ajustes.get("rsi_period", snapshot.get("rsi_period", 14.0)) if ajustes else snapshot.get("rsi_period", 14.0),
                "ema_fast": ajustes.get("ema_fast", snapshot.get("ema_fast", 9.0)) if ajustes else snapshot.get("ema_fast", 9.0),
                "ema_slow": ajustes.get("ema_slow", snapshot.get("ema_slow", 21.0)) if ajustes else snapshot.get("ema_slow", 21.0),
                "rsi_bias": rsi_bias,
                "ema_bias": ema_bias,
                "price_change": price_change,
                "signal_source": signal_source,
            }
            self.last_trades.append(trade_entry)
            features = [
                float(adjusted_rsi_value + rsi_bias),
                float(adjusted_ema_value + ema_bias),
                float(boll_value),
                float(volatility_value),
            ]
            if etiqueta is not None:
                self.training_data.append(features)
                self.training_labels.append(etiqueta)
                if len(self.training_data) > 5000:
                    self.training_data = self.training_data[-5000:]
                    self.training_labels = self.training_labels[-5000:]
            self.trade_counter += 1
            if self.trade_counter % 100 == 0 and self.trade_counter > 0:
                should_train = True
            row_payload = {
                "timestamp": timestamp_now,
                "asset": asset,
                "direction": direction,
                "result": resultado,
                "signal_source": signal_source,
                "signal_direction": direction,
                "price_change": f"{price_change:.6f}",
                "rsi": f"{rsi_value:.6f}",
                "ema": f"{ema_value:.6f}",
                "bollinger": f"{boll_value:.6f}",
                "volatility": f"{volatility_value:.6f}",
                "confidence": f"{confidence:.6f}",
                "indicator_confidence": f"{indicator_confidence:.6f}",
                "ml_probability": f"{ml_probability:.6f}",
                "stake": f"{stake_value:.6f}",
                "rsi_bias": f"{rsi_bias:.6f}",
                "ema_bias": f"{ema_bias:.6f}",
                "rsi_adjusted": f"{trade_entry['rsi_adjusted']:.6f}",
                "ema_fast_adjusted": f"{trade_entry['ema_fast']:.6f}",
                "ema_slow_adjusted": f"{trade_entry['ema_slow']:.6f}",
            }
        try:
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.csv_fields)
                writer.writerow(row_payload)
        except Exception as exc:  # pragma: no cover
            logging.debug(f"No se pudo registrar historial de autoaprendizaje: {exc}")
        if should_train:
            self.executor.submit(self._train_model)
        self.learning_event.set()

    def _pop_context(self, asset: str) -> Dict[str, float]:
        with self.lock:
            return self._pending_context.pop(asset, {}).copy()

    def adjust_bias(self, symbol: str, result: str, signal_direction: str, price_change: float) -> None:
        outcome = str(result).upper()
        direction = str(signal_direction).upper()
        source = self.last_signal.get(symbol, {}).get("source", "COMBINED")
        price_delta = float(price_change)
        self.record_result(symbol, outcome, direction, price_delta, source)
        if outcome != "LOSS":
            return
        direction_error = False
        if direction == "CALL" and price_delta < 0:
            direction_error = True
        elif direction == "PUT" and price_delta > 0:
            direction_error = True
        with self.bias_lock:
            bias_state = self.biases.setdefault(symbol, {"RSI": 0.0, "EMA": 0.0})
            if source == "RSI":
                bias_state["RSI"] += -0.02 if direction_error else 0.02
            elif source == "EMA":
                bias_state["EMA"] += -0.02 if direction_error else 0.02
            else:
                delta = -0.01 if direction_error else 0.01
                bias_state["RSI"] += delta
                bias_state["EMA"] += delta
            bias_state["RSI"] = float(np.clip(bias_state["RSI"], -2.0, 2.0))
            bias_state["EMA"] = float(np.clip(bias_state["EMA"], -2.0, 2.0))
            rsi_bias = float(bias_state["RSI"])
            ema_bias = float(bias_state["EMA"])
        logging.info(
            f"🧠 Bias adjusted [{symbol}] Source={source} RSI={rsi_bias:.2f} EMA={ema_bias:.2f}"
        )
        self.log_bias_update(symbol, outcome, direction, price_delta, source)
        self.learning_event.set()

    def log_bias_update(
        self,
        symbol: str,
        result: str,
        signal_direction: str,
        price_change: float,
        source: Optional[str] = None,
    ) -> None:
        timestamp_now = datetime.now(timezone.utc).isoformat()
        source_label = source or self.last_signal.get(symbol, {}).get("source", "COMBINED")
        with self.bias_lock:
            bias_state = self.biases.setdefault(symbol, {"RSI": 0.0, "EMA": 0.0})
            rsi_bias = float(bias_state.get("RSI", 0.0))
            ema_bias = float(bias_state.get("EMA", 0.0))
        self._ensure_csv()
        row = {field: "" for field in self.csv_fields}
        row.update(
            {
                "timestamp": timestamp_now,
                "asset": symbol,
                "result": str(result).upper(),
                "signal_source": source_label,
                "direction": signal_direction,
                "signal_direction": signal_direction,
                "price_change": f"{float(price_change):.6f}",
                "rsi_bias": f"{rsi_bias:.6f}",
                "ema_bias": f"{ema_bias:.6f}",
            }
        )
        try:
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.csv_fields)
                writer.writerow(row)
        except Exception as exc:  # pragma: no cover
            logging.debug(f"No se pudo registrar ajuste de sesgos: {exc}")

    def store_context(
        self,
        symbol: str,
        rsi: float,
        ema: float,
        macd: float,
        adx: float,
        hour: int,
        result: str,
    ) -> None:
        entry = {
            "symbol": symbol,
            "RSI": float(rsi),
            "EMA": float(ema),
            "MACD": float(macd),
            "ADX": float(adx),
            "hour": int(hour),
            "result": str(result).upper(),
        }
        trigger_reinforce = False
        trigger_optimize = False
        with self.memory_lock:
            self.memory.append(entry)
            if len(self.memory) > self.memory_limit:
                self.memory = self.memory[-self.memory_limit :]
            outcome_flag = 1 if entry["result"] == "WIN" else 0 if entry["result"] == "LOSS" else None
            if outcome_flag is not None:
                self.recent_results.append(outcome_flag)
            length = len(self.memory)
            reinforce_batches = length // 50
            optimize_batches = length // 200
            trigger_reinforce = reinforce_batches > self.reinforce_batches
            trigger_optimize = optimize_batches > self.optimize_batches
            if trigger_reinforce:
                self.reinforce_batches = reinforce_batches
            if trigger_optimize:
                self.optimize_batches = optimize_batches
        if trigger_reinforce or trigger_optimize:
            self.executor.submit(self._run_periodic_learning, trigger_optimize)

    def _run_periodic_learning(self, optimize: bool) -> None:
        try:
            self.reinforce_patterns()
            recent, historical = self.compute_winrates()
            self.stability_guard(recent, historical)
            if optimize:
                self.train_predictive_model()
                self.train_neural_predictor()
                self.optimize_thresholds()
        except Exception as exc:  # pragma: no cover
            logging.debug(f"Error en aprendizaje periódico: {exc}")

    def compute_winrates(self) -> Tuple[float, float]:
        with self.memory_lock:
            recent_values = list(self.recent_results)
        recent_winrate = sum(recent_values) / len(recent_values) if recent_values else 0.0
        with self.lock:
            total = self.global_totals.get("total", 0)
            wins = self.global_totals.get("wins", 0)
        historical = (wins / total) if total else 0.0
        return float(recent_winrate), float(historical)

    def reinforce_patterns(self) -> None:
        with self.memory_lock:
            if not self.memory:
                return
            snapshot = list(self.memory)
        stats: Dict[Tuple[str, int, int], Dict[str, int]] = {}
        for item in snapshot:
            symbol = item.get("symbol")
            rsi_bucket = int(round(float(item.get("RSI", 0.0)) / 10.0) * 10)
            hour = int(item.get("hour", 0))
            key = (symbol, rsi_bucket, hour)
            entry = stats.setdefault(key, {"w": 0, "l": 0})
            if str(item.get("result", "")).upper() == "WIN":
                entry["w"] += 1
            else:
                entry["l"] += 1
        adjustments_made = False
        for data in stats.values():
            total = data["w"] + data["l"]
            if total < 5:
                continue
            win_rate = data["w"] / total
            with self.weights_lock:
                if win_rate > 0.65:
                    self.weights["RSI"] *= 1.01
                    self.weights["MACD"] *= 1.01
                    adjustments_made = True
                elif win_rate < 0.45:
                    self.weights["RSI"] *= 0.99
                    self.weights["MACD"] *= 0.99
                    adjustments_made = True
        if adjustments_made:
            self._clamp_weights()
            self.reset_regime_baseline()

    def stability_guard(self, recent_winrate: float, historical_winrate: float) -> None:
        if recent_winrate - historical_winrate <= 0.20:
            return
        logging.warning("⚠️ Overfitting detected – lowering learning rate")
        self.learning_rate = max(0.005, self.learning_rate * 0.8)
        with self.lock:
            for profile in self.symbol_profiles.values():
                profile_rate = float(profile.get("learning_rate", self.learning_rate))
                profile["learning_rate"] = max(0.005, profile_rate * 0.8)

    def train_predictive_model(self) -> None:
        with self.memory_lock:
            if len(self.memory) < 200:
                return
            data_frame = pd.DataFrame(self.memory)
        try:
            features = data_frame[["RSI", "EMA", "MACD", "ADX", "hour"]]
            labels = (data_frame["result"].str.upper() == "WIN").astype(int)
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            model.fit(features, labels)
            with self.model_lock:
                self.model = model
            if dump is not None:
                try:
                    dump(model, self.predictive_model_path)
                except Exception:
                    pass
            logging.info("🧠 Predictive model trained and saved.")
        except Exception as exc:  # pragma: no cover
            logging.debug(f"Fallo al entrenar modelo predictivo: {exc}")

    def predict_win_probability(
        self,
        rsi: float,
        ema: float,
        macd: float,
        adx: float,
        hour: int,
    ) -> float:
        probabilities: List[float] = []
        rf_prob = 0.5
        with self.model_lock:
            model = self.model
        if model is None and load is not None and self.predictive_model_path.exists():
            try:
                model = load(self.predictive_model_path)
                with self.model_lock:
                    self.model = model
            except Exception:  # pragma: no cover
                model = None
        if model is not None:
            try:
                rf_prob = float(model.predict_proba([[rsi, ema, macd, adx, hour]])[0][1])
                probabilities.append(rf_prob)
            except Exception:  # pragma: no cover
                rf_prob = 0.5
        neural_prob = self.predict_with_neural(rsi, ema, macd, adx, hour)
        if self.neural_model is not None or (
            keras is not None and self.neural_model_path.exists()
        ):
            probabilities.append(neural_prob)
        if not probabilities:
            combined = 0.5
        elif len(probabilities) == 1:
            combined = probabilities[0]
        else:
            combined = float(np.clip(np.mean(probabilities), 0.0, 1.0))
        self.last_prediction = combined
        return combined

    def optimize_thresholds(self) -> None:
        if gp_minimize is None:
            return
        with self.memory_lock:
            if len(self.memory) < 200:
                return

        def objective(params: Tuple[float, float, float]) -> float:
            min_conf, rsi_high, adx_min = params
            winrate = self.simulate_trades_thresholds(min_conf, rsi_high, adx_min)
            return -float(winrate)

        try:
            result = gp_minimize(
                objective,
                [(0.5, 0.9), (60.0, 80.0), (10.0, 40.0)],
                n_calls=20,
                random_state=42,
            )
        except Exception as exc:  # pragma: no cover
            logging.debug(f"Bayesian optimization failed: {exc}")
            return
        self.min_confidence = float(np.clip(result.x[0], 0.4, 0.95))
        self.rsi_high_threshold = float(np.clip(result.x[1], 55.0, 85.0))
        self.adx_min_threshold = float(np.clip(result.x[2], 5.0, 60.0))
        self.reset_regime_baseline()
        logging.info(
            f"🎯 Bayesian optimization finished: min_conf={self.min_confidence:.2f}, RSI_high={self.rsi_high_threshold:.2f}, ADX_min={self.adx_min_threshold:.2f}"
        )

    def simulate_trades_thresholds(
        self,
        min_confidence: float,
        rsi_high: float,
        adx_min: float,
    ) -> float:
        with self.memory_lock:
            snapshot = list(self.memory)
        if not snapshot:
            return 0.0
        considered = 0
        wins = 0
        for item in snapshot:
            rsi_value = float(item.get("RSI", 0.0))
            ema_value = float(item.get("EMA", 0.0))
            macd_value = float(item.get("MACD", 0.0))
            adx_value = float(item.get("ADX", 0.0))
            confidence = self.calculate_confidence(rsi_value, ema_value, 0.5, adx_value, macd_value)
            if confidence < min_confidence:
                continue
            if adx_value < adx_min:
                continue
            if rsi_value > rsi_high and macd_value > 0:
                continue
            considered += 1
            if str(item.get("result", "")).upper() == "WIN":
                wins += 1
        if considered == 0:
            return 0.0
        return wins / considered

    def get_min_confidence(self) -> float:
        return float(self.min_confidence)

    def get_adx_min_threshold(self) -> float:
        return float(self.adx_min_threshold)

    def get_rsi_high_threshold(self) -> float:
        return float(self.rsi_high_threshold)

    def _learning_loop(self) -> None:
        while True:
            self.learning_event.wait()
            self.learning_event.clear()
            try:
                self._compute_biases()
            except Exception as exc:  # pragma: no cover
                logging.debug(f"Error en bucle de autoaprendizaje: {exc}")
            time.sleep(0.2)

    def _compute_biases(self) -> None:
        with self.bias_lock:
            # Mantener el bucle activo sin realizar ajustes adicionales.
            for bias_state in self.biases.values():
                bias_state["RSI"] = float(np.clip(bias_state["RSI"], -2.0, 2.0))
                bias_state["EMA"] = float(np.clip(bias_state["EMA"], -2.0, 2.0))

    def _train_model(self) -> None:
        with self.lock:
            datos = np.array(self.training_data, dtype=float) if self.training_data else np.empty((0, 4), dtype=float)
            etiquetas = np.array(self.training_labels, dtype=int) if self.training_labels else np.empty((0,), dtype=int)
        if datos.size == 0 or datos.shape[0] < 50:
            return
        try:
            modelo = RandomForestClassifier(n_estimators=80, random_state=42)
            modelo.fit(datos, etiquetas)
            with self.model_lock:
                self.model = modelo
        except Exception as exc:  # pragma: no cover
            logging.debug(f"No se pudo entrenar RandomForest interno: {exc}")

    def predict_next(self, rsi_value: float, ema_value: float, boll_value: float, volatility_value: float) -> float:
        simbolo = self.get_active_symbol()
        if simbolo:
            try:
                params = self.get_symbol_snapshot(simbolo)
                rsi_value = float(rsi_value + (params.get("rsi_period", 14.0) - 14.0) * 0.5)
                ema_value = float(ema_value + params.get("ema_tolerance", 0.0) * 0.1)
            except Exception:
                pass
        with self.bias_lock:
            if simbolo and simbolo in self.biases:
                bias_state = self.biases[simbolo]
            else:
                bias_state = {"RSI": 0.0, "EMA": 0.0}
            rsi_bias = float(bias_state.get("RSI", 0.0))
            ema_bias = float(bias_state.get("EMA", 0.0))
        features = np.array(
            [
                float(rsi_value + rsi_bias),
                float(ema_value + ema_bias),
                float(boll_value),
                float(volatility_value),
            ],
            dtype=float,
        ).reshape(1, -1)
        with self.model_lock:
            modelo = self.model
        if modelo is None:
            self.last_prediction = 0.5
            return 0.5
        try:
            proba = float(modelo.predict_proba(features)[0][1])
        except Exception:
            proba = 0.5
        self.last_prediction = proba
        return proba

    def get_summary(self) -> Dict[str, Any]:
        with self.lock:
            per_asset = {asset: (values["wins"] / values["total"] * 100.0) if values["total"] else 0.0 for asset, values in self.asset_stats.items()}
            for nombre in SYMBOLS:
                if nombre not in per_asset:
                    per_asset[nombre] = 0.0
            total_operaciones = self.global_totals["total"]
            global_accuracy = (self.global_totals["wins"] / total_operaciones * 100.0) if total_operaciones else 0.0
            historial = list(self.last_trades)[-5:]
            tuning = {nombre: self._symbol_snapshot(self._get_symbol_profile(nombre)) for nombre in SYMBOLS}
        with self.bias_lock:
            for nombre in SYMBOLS:
                if nombre not in self.biases:
                    self.biases[nombre] = {"RSI": 0.0, "EMA": 0.0}
            bias_snapshot = {
                symbol: {
                    "RSI": float(state.get("RSI", 0.0)),
                    "EMA": float(state.get("EMA", 0.0)),
                }
                for symbol, state in self.biases.items()
            }
        with self.weights_lock:
            weights_snapshot = {key: float(value) for key, value in self.weights.items()}
        with self.lock:
            symbol_weight_snapshot = {key: float(value) for key, value in self.symbol_weights.items()}
        resumen = {
            "per_asset": per_asset,
            "global_accuracy": global_accuracy,
            "last_trades": list(reversed(historial)),
            "status": self.learning_thread.is_alive(),
            "biases": bias_snapshot,
            "last_prediction": self.last_prediction,
            "symbol_tuning": tuning,
            "indicator_weights": weights_snapshot,
            "symbol_weights": symbol_weight_snapshot,
        }
        return resumen

    def reset_history(self) -> None:
        self.executor.submit(self._reset_history_sync)

    def _reset_history_sync(self) -> None:
        with self.lock:
            self.asset_stats = defaultdict(lambda: {"wins": 0, "total": 0})
            self.global_totals = {"wins": 0, "total": 0}
            self.last_trades.clear()
            self.training_data.clear()
            self.training_labels.clear()
            self.trade_counter = 0
            self._pending_context.clear()
            self.symbol_profiles = {symbol: self._default_symbol_profile() for symbol in SYMBOLS}
            self._current_symbol = None
            try:
                if self.csv_path.exists():
                    self.csv_path.unlink()
            except Exception as exc:  # pragma: no cover
                logging.debug(f"No se pudo eliminar CSV de autoaprendizaje: {exc}")
        with self.bias_lock:
            self.biases = {symbol: {"RSI": 0.0, "EMA": 0.0} for symbol in SYMBOLS}
        with self.model_lock:
            self.model = None
        self.neural_model = None
        self.last_prediction = 0.5
        with self.memory_lock:
            self.memory = []
            self.recent_results.clear()
            self.reinforce_batches = 0
            self.optimize_batches = 0
        with self.weights_lock:
            self.weights = {
                "RSI": 1.0,
                "EMA": 1.0,
                "BOLL": 1.0,
                "ADX": 0.8,
                "MACD": 0.8,
            }
        with self.lock:
            self.min_confidence = MIN_TRADE_CONFIDENCE
            self.symbol_weights = {symbol: 1.0 for symbol in SYMBOLS}
        if self.neural_model_path.exists():
            try:
                self.neural_model_path.unlink()
            except Exception:  # pragma: no cover
                pass
        if self.predictive_model_path.exists():
            try:
                self.predictive_model_path.unlink()
            except Exception:  # pragma: no cover
                pass
        self.reset_regime_baseline()
        self._ensure_csv()


auto_learn = auto_learning()


# ===============================================================
# STRATEGIES
# ===============================================================
def strategy_rsi(df: pd.DataFrame) -> StrategyResult:
    if len(df) < 15:
        return StrategyResult('NONE', 0.0, ['RSI sin suficientes datos'], {'rsi': None})
    rsi_series = rsi(df['close'])
    valor = float(rsi_series.iloc[-1])
    gestor = globals().get("auto_learn")
    simbolo = None
    if gestor is not None:
        try:
            simbolo = gestor.get_active_symbol()
        except Exception:
            simbolo = None
    inferior = 30.0
    superior = 70.0
    if simbolo and gestor is not None:
        try:
            perfil = gestor.get_symbol_snapshot(simbolo)
            inferior = float(perfil.get("rsi_lower", inferior))
            superior = float(perfil.get("rsi_upper", superior))
        except Exception:
            pass
    extra = {
        'rsi': valor,
        'strong_call': valor < max(20.0, inferior - 5.0),
        'strong_put': valor > min(80.0, superior + 5.0),
    }
    if valor < inferior:
        return StrategyResult('CALL', 2.0, [f"RSI {valor:.2f} sobrevendido → señal CALL"], extra)
    if valor > superior:
        return StrategyResult('PUT', -2.0, [f"RSI {valor:.2f} sobrecomprado → señal PUT"], extra)
    return StrategyResult('NONE', 0.0, [f"RSI {valor:.2f} sin sesgo claro"], extra)


def strategy_ema_trend(df: pd.DataFrame) -> StrategyResult:
    if len(df) < 25:
        return StrategyResult('NONE', 0.0, ['EMAs sin datos suficientes'], {})
    ema_corto = ema(df['close'], 9)
    ema_largo = ema(df['close'], 21)
    gestor = globals().get("auto_learn")
    tolerancia = 0.0
    if gestor is not None:
        try:
            simbolo = gestor.get_active_symbol()
        except Exception:
            simbolo = None
        else:
            if simbolo:
                try:
                    perfil = gestor.get_symbol_snapshot(simbolo)
                    tolerancia = float(perfil.get("ema_tolerance", 0.0))
                except Exception:
                    tolerancia = 0.0
    threshold = tolerancia / 1000.0
    diff_prev = float(ema_corto.iloc[-2] - ema_largo.iloc[-2])
    diff_curr = float(ema_corto.iloc[-1] - ema_largo.iloc[-1])
    cruz_alcista = diff_prev <= threshold and diff_curr > threshold
    cruz_bajista = diff_prev >= -threshold and diff_curr < -threshold
    if cruz_alcista:
        return StrategyResult('CALL', 1.5, ['Cruce alcista EMA9 sobre EMA21 → CALL'], {'ema_short': float(ema_corto.iloc[-1]), 'ema_long': float(ema_largo.iloc[-1])})
    if cruz_bajista:
        return StrategyResult('PUT', -1.5, ['Cruce bajista EMA9 bajo EMA21 → PUT'], {'ema_short': float(ema_corto.iloc[-1]), 'ema_long': float(ema_largo.iloc[-1])})
    return StrategyResult('NONE', 0.0, ['EMAs paralelas sin cruce'], {'ema_short': float(ema_corto.iloc[-1]), 'ema_long': float(ema_largo.iloc[-1])})


def strategy_bollinger_rebound(df: pd.DataFrame) -> StrategyResult:
    if len(df) < 22:
        return StrategyResult('NONE', 0.0, ['Bollinger sin historial suficiente'], {})
    lower, upper = bollinger_bands(df['close'])
    rsi_series = rsi(df['close'])
    precio = float(df['close'].iloc[-1])
    rsi_actual = float(rsi_series.iloc[-1])
    rsi_prev = float(rsi_series.iloc[-2])
    extra = {'rsi': rsi_actual, 'lower': float(lower.iloc[-1]), 'upper': float(upper.iloc[-1])}
    if precio <= float(lower.iloc[-1]) and rsi_actual > rsi_prev:
        return StrategyResult('CALL', 1.2, ['Precio en banda inferior y RSI repunta → CALL'], extra)
    if precio >= float(upper.iloc[-1]) and rsi_actual < rsi_prev:
        return StrategyResult('PUT', -1.2, ['Precio en banda superior y RSI cae → PUT'], extra)
    return StrategyResult('NONE', 0.0, ['Sin rebote claro en Bollinger'], extra)


def strategy_pullback(df: pd.DataFrame) -> StrategyResult:
    if len(df) < 25:
        return StrategyResult('NONE', 0.0, ['Pullback sin datos suficientes'], {})
    closes = df['close']
    rsi_series = rsi(closes)
    tramo = closes.iloc[-5:]
    rsi_tramo = rsi_series.iloc[-5:]
    extra = {'rsi_trend': float(rsi_tramo.iloc[-1] - rsi_tramo.iloc[-2])}
    if tramo.iloc[0] > tramo.iloc[1] > tramo.iloc[2] and tramo.iloc[3] >= tramo.iloc[2] and tramo.iloc[4] > tramo.iloc[2] and rsi_tramo.iloc[-1] > rsi_tramo.iloc[-2]:
        return StrategyResult('CALL', 1.0, ['Pullback alcista con RSI recuperándose → CALL'], extra)
    if tramo.iloc[0] < tramo.iloc[1] < tramo.iloc[2] and tramo.iloc[3] <= tramo.iloc[2] and tramo.iloc[4] < tramo.iloc[2] and rsi_tramo.iloc[-1] < rsi_tramo.iloc[-2]:
        return StrategyResult('PUT', -1.0, ['Pullback bajista con RSI cayendo → PUT'], extra)
    return StrategyResult('NONE', 0.0, ['Sin pullback definido'], extra)


def strategy_range_breakout(df: pd.DataFrame) -> StrategyResult:
    if len(df) < 30:
        return StrategyResult('NONE', 0.0, ['Ruptura sin datos suficientes'], {})
    ventana_altas = df['high'].iloc[-21:-1]
    ventana_bajas = df['low'].iloc[-21:-1]
    if ventana_altas.empty or ventana_bajas.empty:
        return StrategyResult('NONE', 0.0, ['Rango sin referencias claras'], {})
    resistencia = float(ventana_altas.max())
    soporte = float(ventana_bajas.min())
    cierre = float(df['close'].iloc[-1])
    rsi_actual = float(rsi(df['close']).iloc[-1])
    extra = {'resistencia': resistencia, 'soporte': soporte, 'rsi': rsi_actual}
    if cierre > resistencia and rsi_actual > 50:
        return StrategyResult('CALL', 1.0, ['Cierre rompe resistencia reciente → CALL'], extra)
    if cierre < soporte and rsi_actual < 50:
        return StrategyResult('PUT', -1.0, ['Cierre perfora soporte reciente → PUT'], extra)
    return StrategyResult('NONE', 0.0, ['Sin ruptura relevante'], extra)


def strategy_divergence(df: pd.DataFrame) -> StrategyResult:
    if len(df) < 12:
        return StrategyResult('NONE', 0.0, ['Divergencias sin datos suficientes'], {})
    rsi_series = rsi(df['close'])
    window = min(10, len(df))
    rsi_segment = rsi_series.iloc[-window:]
    price_segment = df['close'].iloc[-window:]
    rsi_delta = float(rsi_segment.iloc[-1] - rsi_segment.iloc[-3]) if window >= 3 else float(rsi_segment.iloc[-1] - rsi_segment.iloc[0])
    price_delta = float(price_segment.iloc[-1] - price_segment.iloc[-3]) if window >= 3 else float(price_segment.iloc[-1] - price_segment.iloc[0])
    extra = {'rsi_delta': rsi_delta, 'price_delta': price_delta}
    if price_delta < 0 and rsi_delta > 0:
        return StrategyResult('CALL', 1.5, ['Divergencia alcista RSI vs precio → CALL'], {**extra, 'strong_call': True})
    if price_delta > 0 and rsi_delta < 0:
        return StrategyResult('PUT', -1.5, ['Divergencia bajista RSI vs precio → PUT'], {**extra, 'strong_put': True})
    return StrategyResult('NONE', 0.0, ['Sin divergencias claras'], extra)


def strategy_volatility_filter(df: pd.DataFrame) -> StrategyResult:
    if len(df) < 25:
        return StrategyResult('NONE', 0.0, ['Volatilidad sin datos suficientes'], {'volatility': None, 'low': False})
    retornos = df['close'].pct_change().dropna()
    if retornos.empty:
        return StrategyResult('NONE', 0.0, ['Volatilidad no calculable'], {'volatility': None, 'low': True})
    reciente = retornos.iloc[-20:]
    volatilidad = float(reciente.std())
    umbral = 0.0007
    if volatilidad < umbral:
        return StrategyResult('NONE', 0.0, [f"Volatilidad {volatilidad:.4f} baja → confianza limitada"], {'volatility': volatilidad, 'low': True})
    return StrategyResult('NONE', 0.0, [f"Volatilidad {volatilidad:.4f} adecuada"], {'volatility': volatilidad, 'low': False})


STRATEGY_FUNCTIONS: List[Tuple[str, Callable[[pd.DataFrame], StrategyResult]]] = [
    ('RSI', strategy_rsi),
    ('EMA Trend', strategy_ema_trend),
    ('Bollinger Rebound', strategy_bollinger_rebound),
    ('Pullback', strategy_pullback),
    ('Range Breakout', strategy_range_breakout),
]

STRATEGY_WEIGHTS: Dict[str, float] = {
    'RSI': 2.0,
    'EMA Trend': 1.5,
    'Bollinger Rebound': 1.2,
    'Pullback': 1.0,
    'Range Breakout': 1.0,
    'Divergence': 1.5,
    'Volatility Filter': 0.5,
}

TOTAL_STRATEGY_COUNT = len(STRATEGY_WEIGHTS)

STRATEGY_DISPLAY_NAMES: Dict[str, str] = {
    'RSI': 'RSI',
    'EMA Trend': 'Tendencia EMA',
    'Bollinger Rebound': 'Rebote Bollinger',
    'Pullback': 'Pullback',
    'Range Breakout': 'Ruptura de rango',
    'Divergence': 'Divergencia',
    'Volatility Filter': 'Filtro de volatilidad',
}


# ===============================================================
# SIGNAL COMBINER
# ===============================================================
def _clasificar_confianza(valor: float) -> str:
    if valor >= 0.75:
        return 'Alta'
    if valor >= 0.45:
        return 'Media'
    return 'Baja'


def combine_signals(results: List[Tuple[str, StrategyResult]], active_states: Dict[str, bool]) -> Dict[str, Any]:
    reasons: List[str] = []
    agreements: Dict[str, str] = {}
    pesos_direccion = {'CALL': 0.0, 'PUT': 0.0}
    conteo_direccion = {'CALL': 0, 'PUT': 0}
    total_weight_signals = 0.0
    total_weight_active = 0.0
    override_signal: Optional[str] = None
    override_reason = ''
    low_volatility = False
    volatility_value: Optional[float] = None
    active_count = 0
    for name, res in results:
        if not active_states.get(name, True):
            continue
        weight = STRATEGY_WEIGHTS.get(name, 1.0)
        total_weight_active += weight
        active_count += 1
        agreements[name] = res.signal
        reasons.extend(res.reasons)
        if name == 'Volatility Filter':
            volatility_value = res.extra.get('volatility')
            if res.extra.get('low'):
                low_volatility = True
        if name == 'RSI':
            rsi_val = res.extra.get('rsi')
            if res.extra.get('strong_call'):
                override_signal = 'CALL'
                override_reason = f"RSI extremo {rsi_val:.2f}" if rsi_val is not None else 'RSI extremo'
            elif res.extra.get('strong_put'):
                override_signal = 'PUT'
                override_reason = f"RSI extremo {rsi_val:.2f}" if rsi_val is not None else 'RSI extremo'
        if name == 'Divergence':
            if res.extra.get('strong_call'):
                override_signal = 'CALL'
                override_reason = 'Divergencia confirmada'
            elif res.extra.get('strong_put'):
                override_signal = 'PUT'
                override_reason = 'Divergencia confirmada'
        if res.signal in {'CALL', 'PUT'}:
            pesos_direccion[res.signal] += weight
            conteo_direccion[res.signal] += 1
            total_weight_signals += weight
    if total_weight_signals == 0.0:
        main_reason = '⚠️ Ninguna de las estrategias activas generó señal'
        return {
            'signal': 'NONE',
            'confidence': 0.0,
            'reasons': reasons,
            'agreements': agreements,
            'counts': conteo_direccion,
            'weights': pesos_direccion,
            'active': active_count,
            'aligned': 0,
            'signals': 0,
            'confidence_label': 'Baja',
            'low_volatility': low_volatility,
            'volatility_value': volatility_value,
            'override': override_signal is not None,
            'override_reason': override_reason,
            'main_reason': main_reason,
            'dominant': 'NONE',
            'total_weight': total_weight_signals,
            'active_weight': total_weight_active,
        }
    dominante = 'CALL' if pesos_direccion['CALL'] >= pesos_direccion['PUT'] else 'PUT'
    confianza = pesos_direccion[dominante] / total_weight_signals
    signal = 'NONE'
    aligned = conteo_direccion[dominante]
    if confianza >= 0.45:
        signal = dominante
    if override_signal:
        signal = override_signal
        confianza = max(confianza, 0.45)
        aligned = conteo_direccion.get(override_signal, 0)
    if low_volatility and signal != 'NONE':
        confianza *= 0.85
        reasons.append('Volatilidad baja → confianza limitada')
    confianza = min(confianza, 0.98)
    if signal != 'NONE':
        confianza = max(confianza, 0.35)
    confianza_label = _clasificar_confianza(confianza) if signal != 'NONE' else 'Baja'
    motivos_alineados = [
        razon
        for nombre, res in results
        for razon in res.reasons
        if res.signal == signal and signal != 'NONE'
    ]
    main_reason = override_reason or (
        motivos_alineados[0] if motivos_alineados else (reasons[0] if reasons else 'Señal compuesta')
    )
    if low_volatility and signal != 'NONE' and 'volatilidad' not in main_reason.lower():
        main_reason = f"{main_reason} + baja volatilidad"
    return {
        'signal': signal,
        'confidence': confianza,
        'reasons': reasons,
        'agreements': agreements,
        'counts': conteo_direccion,
        'weights': pesos_direccion,
        'active': active_count,
        'aligned': aligned,
        'signals': conteo_direccion['CALL'] + conteo_direccion['PUT'],
        'confidence_label': confianza_label,
        'low_volatility': low_volatility,
        'volatility_value': volatility_value,
        'override': override_signal is not None,
        'override_reason': override_reason,
        'main_reason': main_reason,
        'dominant': dominante,
        'total_weight': total_weight_signals,
        'active_weight': total_weight_active,
    }


# ===============================================================
# RISK MANAGEMENT
# ===============================================================
class RiskManager:
    def __init__(self, daily_loss_limit: float = MAX_DAILY_LOSS, daily_profit_target: float = MAX_DAILY_PROFIT, max_drawdown: float = MAX_DRAWDOWN) -> None:
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_trade_time = datetime.now(timezone.utc) - timedelta(seconds=TRADE_DELAY)
        self.max_daily_loss = daily_loss_limit
        self.daily_profit_target = daily_profit_target
        self.max_drawdown_allowed = max_drawdown

    def can_trade(self, confidence: float) -> bool:
        now = datetime.now(timezone.utc)
        if self.daily_trades >= MAX_DAILY_TRADES:
            logging.info("Se alcanzó el máximo de operaciones diarias")
            return False
        if self.daily_pnl <= self.max_daily_loss:
            logging.info("Se alcanzó el límite diario de pérdida")
            return False
        if self.daily_pnl >= self.daily_profit_target:
            logging.info("Se alcanzó el objetivo diario de ganancia")
            return False
        if self.total_pnl <= self.max_drawdown_allowed:
            logging.info("Se alcanzó el drawdown máximo")
            return False
        if (now - self.last_trade_time).total_seconds() < TRADE_DELAY:
            return False
        if self.consecutive_losses > 0:
            cooldown = COOLDOWN_AFTER_LOSS * self.consecutive_losses
            if (now - self.last_trade_time).total_seconds() < cooldown:
                return False
        return confidence >= 0.35

    def register_trade(self, pnl: float) -> None:
        self.daily_trades += 1
        self.daily_pnl += pnl
        self.total_pnl += pnl
        self.last_trade_time = datetime.now(timezone.utc)
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def reset_daily(self) -> None:
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0

    def set_daily_loss_limit(self, value: float) -> None:
        self.max_daily_loss = value

    def set_daily_profit_target(self, value: float) -> None:
        self.daily_profit_target = value

    def set_max_drawdown(self, value: float) -> None:
        self.max_drawdown_allowed = value


# ===============================================================
# ADAPTIVE AI MODULE
# ===============================================================
class AdaptiveAIManager:
    def __init__(self) -> None:
        self.enabled = True
        self.passive = AI_PASSIVE_MODE
        self.model: Optional[LogisticRegression] = None
        self.trade_counter = 0
        self.win_counter = 0
        self.lock = threading.Lock()
        self.feature_cache: List[np.ndarray] = []
        self.result_cache: List[int] = []
        self.state_path = ADAPTIVE_STATE_PATH
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.learning_rate = 0.05
        self.learning_decay = 0.999
        self.offline_thread = threading.Thread(target=self._advisory_loop, daemon=True)
        self._load_state()
        self.offline_thread.start()

    def _phase(self) -> str:
        if self.trade_counter >= AI_AUTONOMOUS_THRESHOLD and self.accuracy() >= AI_AUTONOMOUS_ACCURACY:
            return "autonomous"
        if self.trade_counter >= AI_SEMI_ACTIVE_THRESHOLD:
            return "semi-active"
        return "passive"

    def accuracy(self) -> float:
        with self.lock:
            if self.trade_counter == 0:
                return 0.0
            return self.win_counter / self.trade_counter

    def log_trade(self, features: np.ndarray, result: int) -> None:
        with self.lock:
            self.feature_cache.append(features)
            self.result_cache.append(result)
            if result == 1:
                self.win_counter += 1
            self.trade_counter += 1
            if len(self.feature_cache) > 5000:
                self.feature_cache = self.feature_cache[-5000:]
                self.result_cache = self.result_cache[-5000:]
        self._online_update(features, result)
        self._save_state()

    def _train_model(self) -> None:
        with self.lock:
            if len(self.feature_cache) < 200:
                return
            X = np.vstack(self.feature_cache)
            y = np.array(self.result_cache)
        try:
            model = LogisticRegression(max_iter=200)
            model.fit(X, y)
            with self.lock:
                self.model = model
                self.weights = model.coef_[0].astype(float)
                self.bias = float(model.intercept_[0])
                self.learning_rate = max(self.learning_rate * 0.9, 0.001)
            self._save_state()
        except Exception as exc:  # pragma: no cover
            logging.debug(f"Error al entrenar modelo adaptativo: {exc}")

    def predict(self, features: np.ndarray) -> Tuple[float, List[str]]:
        if not self.enabled:
            return 0.5, []
        phase = self._phase()
        weights = None
        bias = 0.0
        with self.lock:
            if self.weights is not None and self.weights.size == features.size:
                weights = self.weights.copy()
                bias = self.bias
        if weights is None:
            if phase == "passive":
                return 0.5, ["Aprendizaje pasivo"]
            return 0.5, ["Modelo en calentamiento"]
        logit = float(np.dot(features, weights) + bias)
        logit = float(np.clip(logit, -50.0, 50.0))
        prob = 1.0 / (1.0 + np.exp(-logit))
        if phase == "semi-active":
            return prob, ["Asesoría semi-activa"]
        return prob, ["IA autónoma"]

    def fuse_with_technical(self, technical_conf: float, ai_prob: float) -> float:
        phase = self._phase()
        if phase == "passive":
            return technical_conf
        if phase == "semi-active":
            return min(0.98, technical_conf * 0.8 + ai_prob * 0.2)
        return min(0.98, 0.5 + (ai_prob - 0.5) * 0.8)

    def _advisory_loop(self) -> None:
        while True:
            time.sleep(ADVISORY_INTERVAL_SEC)
            try:
                acc = self.accuracy() * 100
                phase = self._phase()
                phase_text = {
                    "passive": "pasiva",
                    "semi-active": "semi-activa",
                    "autonomous": "autónoma",
                }.get(phase, phase)
                logging.info(f"📊 Aviso IA → fase={phase_text} precisión={acc:.2f}% operaciones={self.trade_counter}")
                self._train_model()
            except Exception as exc:  # pragma: no cover
                logging.debug(f"Error en bucle de avisos IA: {exc}")

    def _ensure_weight_dim(self, dim: int) -> None:
        with self.lock:
            if self.weights is None or self.weights.size != dim:
                self.weights = np.zeros(dim, dtype=float)
                self.bias = 0.0
                self.learning_rate = 0.05

    def _online_update(self, features: np.ndarray, result: int) -> None:
        if not self.enabled:
            return
        self._ensure_weight_dim(features.size)
        with self.lock:
            if self.weights is None:
                return
            logit = float(np.dot(features, self.weights) + self.bias)
            logit = float(np.clip(logit, -50.0, 50.0))
            prob = 1.0 / (1.0 + np.exp(-logit))
            error = float(result) - prob
            self.weights += self.learning_rate * error * features
            self.bias += self.learning_rate * error
            self.learning_rate = max(self.learning_rate * self.learning_decay, 0.001)

    def _save_state(self) -> None:
        if not self.enabled:
            return
        try:
            if self.state_path.parent != Path("."):
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
            weights = self.weights if self.weights is not None else np.array([], dtype=float)
            np.savez(
                self.state_path,
                weights=weights,
                bias=self.bias,
                trade_counter=self.trade_counter,
                win_counter=self.win_counter,
                learning_rate=self.learning_rate,
            )
        except Exception as exc:  # pragma: no cover
            logging.debug(f"Error al guardar estado adaptativo: {exc}")

    def _load_state(self) -> None:
        if not self.enabled or not self.state_path.exists():
            return
        try:
            data = np.load(self.state_path, allow_pickle=True)
            weights = data.get("weights")
            bias = data.get("bias")
            trade_counter = data.get("trade_counter")
            win_counter = data.get("win_counter")
            learning_rate = data.get("learning_rate")
            if weights is not None and weights.size:
                self.weights = weights.astype(float)
            if bias is not None:
                self.bias = float(bias)
            if learning_rate is not None:
                self.learning_rate = float(learning_rate)
            if trade_counter is not None:
                self.trade_counter = int(trade_counter)
            if win_counter is not None:
                self.win_counter = int(win_counter)
        except Exception as exc:  # pragma: no cover
            logging.debug(f"Error al cargar estado adaptativo: {exc}")


def train_adaptive_model(data_path: str, model_path: str) -> None:
    try:
        df = pd.read_csv(data_path)
        feature_cols = [col for col in df.columns if col.startswith("f_")]
        X = df[feature_cols].values
        y = df["label"].values
        model = LogisticRegression(max_iter=300)
        model.fit(X, y)
        with open(model_path, "wb") as handle:
            import pickle

            pickle.dump(model, handle)
        logging.info(f"Adaptive model trained and saved to {model_path}")
    except Exception as exc:  # pragma: no cover
        logging.warning(f"Adaptive model training failed: {exc}")


# ===============================================================
# AI BACKEND
# ===============================================================
def query_ai_backend(features: np.ndarray) -> Optional[float]:
    if not AI_ENABLED:
        return None
    if not AI_ENDPOINT:
        return None
    feature_snapshot = {
        "latest": float(features[-1]) if features.size else 0.0,
        "mean": float(np.mean(features)) if features.size else 0.0,
        "variance": float(np.var(features)) if features.size else 0.0,
        "vector": features.tolist(),
    }
    payload = {
        "model": "phi3:mini",
        "prompt": (
            "Analyze this market data and return only CALL or PUT based on current trend: "
            f"{json.dumps(feature_snapshot)}"
        ),
        "stream": False,
    }
    start = time.perf_counter()
    try:
        response = requests.post(AI_ENDPOINT, json=payload, timeout=AI_TIMEOUT)
        response.raise_for_status()
        content = response.json()
    except Exception as exc:
        logging.debug(f"Fallo en solicitud IA externa: {exc}")
        return None
    text = str(content.get("response", "")).lower()
    latency_ms = int((time.perf_counter() - start) * 1000)
    if "call" in text and "put" not in text:
        logging.info(f"IA backend=ollama latencia={latency_ms}ms prob_alza=0.80")
        return 0.8
    if "put" in text and "call" not in text:
        logging.info(f"IA backend=ollama latencia={latency_ms}ms prob_alza=0.20")
        return 0.2
    if text:
        logging.info(f"IA backend=ollama latencia={latency_ms}ms prob_alza=0.50")
        return 0.5
    logging.debug("La IA externa devolvió una respuesta vacía")
    return None


# ===============================================================
# FEATURE ENGINEERING
# ===============================================================
def build_feature_vector(df: pd.DataFrame, reasons: List[str], results: List[Tuple[str, StrategyResult]]) -> np.ndarray:
    ema9 = ema(df["close"], 9).iloc[-1]
    ema21 = ema(df["close"], 21).iloc[-1]
    ema50 = ema(df["close"], 50).iloc[-1]
    ema100 = ema(df["close"], 100).iloc[-1]
    rsi_val = rsi(df["close"], 14).iloc[-1]
    lower_bb, upper_bb = bollinger_bands(df["close"])
    atr_val = atr(df, 14).iloc[-1]
    total_score = sum(res.score for _, res in results)
    features = np.array(
        [
            df["close"].iloc[-1],
            ema9,
            ema21,
            ema50,
            ema100,
            rsi_val,
            lower_bb.iloc[-1],
            upper_bb.iloc[-1],
            atr_val,
            total_score,
            len(reasons),
        ],
        dtype=float,
    )
    return features


# ===============================================================
# DERIV CONNECTION
# ===============================================================
class DerivWebSocket:
    def __init__(self) -> None:
        if websocket is None:
            raise RuntimeError("websocket-client not available")
        self.socket: Optional[websocket.WebSocket] = None
        self.lock = threading.Lock()
        self.req_id = 1

    def connect(self) -> None:
        while True:
            try:
                self.socket = websocket.create_connection(
                    f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}", timeout=30
                )
                self._send({"authorize": API_TOKEN})
                self._recv()
                logging.info("Deriv conectado")
                return
            except Exception as exc:
                logging.warning(f"Error de conexión con Deriv: {exc}")
                time.sleep(2)

    def _send(self, payload: Dict[str, Any]) -> int:
        if self.socket is None:
            raise RuntimeError("Socket closed")
        with self.lock:
            payload["req_id"] = self.req_id
            self.req_id += 1
            self.socket.send(json.dumps(payload))
            return payload["req_id"]

    def _recv(self) -> Dict[str, Any]:
        if self.socket is None:
            raise RuntimeError("Socket closed")
        return json.loads(self.socket.recv())

    def fetch_candles(self, symbol: str) -> List[Candle]:
        req_id = self._send(
            {
                "ticks_history": symbol,
                "granularity": GRANULARITY,
                "count": CANDLE_COUNT,
                "end": "latest",
                "style": "candles",
            }
        )
        while True:
            msg = self._recv()
            if msg.get("req_id") == req_id:
                candles = []
                for item in msg.get("candles", []):
                    candles.append(
                        Candle(
                            epoch=int(item["epoch"]),
                            open=float(item["open"]),
                            high=float(item["high"]),
                            low=float(item["low"]),
                            close=float(item["close"]),
                        )
                    )
                return candles

    def buy(self, symbol: str, direction: str, amount: float) -> Tuple[Optional[int], int]:
        req_id = self._send(
            {
                "proposal": 1,
                "amount": amount,
                "basis": "stake",
                "contract_type": direction,
                "currency": "USD",
                "duration": 1,
                "duration_unit": "m",
                "symbol": symbol,
            }
        )
        proposal = None
        while True:
            msg = self._recv()
            if msg.get("req_id") == req_id:
                if "error" in msg:
                    logging.warning(f"Error al generar propuesta: {msg['error']}")
                    return None, 0.0
                proposal = msg["proposal"]
                break
        buy_id = self._send({"buy": proposal["id"], "price": proposal["ask_price"]})
        while True:
            msg = self._recv()
            if msg.get("req_id") == buy_id:
                if "error" in msg:
                    logging.warning(f"Error al comprar contrato: {msg['error']}")
                    return None, 0
                duration_val = int(proposal.get("duration", 1))
                unit = str(proposal.get("duration_unit", "m")).lower()
                if unit == "s":
                    duration_seconds = duration_val
                elif unit == "h":
                    duration_seconds = duration_val * 3600
                elif unit == "d":
                    duration_seconds = duration_val * 86400
                else:
                    duration_seconds = duration_val * 60
                return msg["buy"]["contract_id"], duration_seconds

    def check_trade_result(self, contract_id: int, retries: int = 6, delay: float = 3.0) -> str:
        for attempt in range(retries):
            try:
                req_id = self._send({"proposal_open_contract": 1, "contract_id": contract_id})
                while True:
                    msg = self._recv()
                    if msg.get("req_id") == req_id:
                        if "error" in msg:
                            raise RuntimeError(msg["error"].get("message", "Error desconocido"))
                        data = msg.get("proposal_open_contract", {})
                        status = str(data.get("status", "")).lower()
                        if status in {"won", "lost"}:
                            return status
                        if data.get("is_sold"):
                            profit_raw = data.get("profit")
                            try:
                                profit_val = float(profit_raw)
                            except (TypeError, ValueError):
                                profit_val = 0.0
                            return "won" if profit_val > 0 else "lost"
                        break
            except Exception as exc:
                logging.warning(f"[{contract_id}] Error al consultar resultado: {exc}")
            time.sleep(delay)
        return "unknown"


# ===============================================================
# TRADING ENGINE
# ===============================================================
class TradingEngine:
    def __init__(self) -> None:
        self.api = DerivWebSocket()
        self.risk = RiskManager()
        self.ai = AdaptiveAIManager()
        self.trade_history: List[TradeRecord] = []
        self.lock = threading.Lock()
        self.running = threading.Event()
        self.win_count = 0
        self.loss_count = 0
        self._trade_listeners: List[Callable[[TradeRecord, Dict[str, float]], None]] = []
        self._status_listeners: List[Callable[[str], None]] = []
        self._summary_listeners: List[Callable[[str, Dict[str, Any]], None]] = []
        self._trade_state_listeners: List[Callable[[str], None]] = []
        self._strategy_lock = threading.Lock()
        self.strategy_states: Dict[str, bool] = {name: True for name, _ in STRATEGY_FUNCTIONS}
        self.strategy_states["Divergence"] = True
        self.strategy_states["Volatility Filter"] = True
        self.trade_amount = STAKE
        self.auto_shutdown_enabled = False
        self.auto_shutdown_limit = 0
        self.auto_shutdown_triggered = False
        self.active_trade_symbol: Optional[str] = None
        self.kelly_enabled = False
        self.initial_balance = 1000.0
        self.base_trade_amount = STAKE

    def add_trade_listener(self, callback: Callable[[TradeRecord, Dict[str, float]], None]) -> None:
        self._trade_listeners.append(callback)

    def add_status_listener(self, callback: Callable[[str], None]) -> None:
        self._status_listeners.append(callback)

    def add_summary_listener(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        self._summary_listeners.append(callback)

    def add_trade_state_listener(self, callback: Callable[[str], None]) -> None:
        self._trade_state_listeners.append(callback)

    def set_trade_amount(self, value: float) -> None:
        self.trade_amount = max(0.1, float(value))
        self.base_trade_amount = self.trade_amount

    def set_daily_loss_limit(self, value: float) -> None:
        self.risk.set_daily_loss_limit(float(value))

    def set_daily_profit_target(self, value: float) -> None:
        self.risk.set_daily_profit_target(float(value))

    def set_max_drawdown(self, value: float) -> None:
        self.risk.set_max_drawdown(float(value))

    def configure_auto_shutdown(self, enabled: bool, limit: int) -> None:
        self.auto_shutdown_enabled = enabled
        self.auto_shutdown_limit = max(0, int(limit))
        if not enabled:
            self.auto_shutdown_triggered = False

    def set_strategy_state(self, name: str, enabled: bool) -> None:
        with self._strategy_lock:
            if name in self.strategy_states:
                self.strategy_states[name] = enabled

    def get_strategy_states(self) -> Dict[str, bool]:
        with self._strategy_lock:
            return dict(self.strategy_states)

    def set_kelly_enabled(self, enabled: bool) -> None:
        self.kelly_enabled = bool(enabled)

    def _estimate_balance(self) -> float:
        return max(100.0, self.initial_balance + self.risk.total_pnl)

    def _calculate_kelly_stake(self, probability: float) -> float:
        if not self.kelly_enabled:
            return self.trade_amount
        balance = self._estimate_balance()
        payout_ratio = PAYOUT if PAYOUT > 0 else 1.0
        edge = float(probability)
        f_star = edge - (1.0 - edge) / payout_ratio
        f_star = max(0.01, min(f_star, 0.05))
        stake = balance * f_star
        stake = float(np.clip(stake, 0.1, balance * 0.1))
        return stake

    def _notify_trade(self, record: TradeRecord) -> None:
        stats = {
            "operations": float(self.win_count + self.loss_count),
            "wins": float(self.win_count),
            "losses": float(self.loss_count),
            "pnl": float(self.risk.total_pnl),
            "daily_pnl": float(self.risk.daily_pnl),
            "accuracy": float((self.win_count / max(1, self.win_count + self.loss_count)) * 100.0),
        }
        for callback in list(self._trade_listeners):
            try:
                callback(record, stats)
            except Exception as exc:
                logging.debug(f"Error en escucha de operaciones: {exc}")

    def _notify_status(self, status: str) -> None:
        for callback in list(self._status_listeners):
            try:
                callback(status)
            except Exception as exc:
                logging.debug(f"Error en escucha de estado: {exc}")

    def _notify_summary(self, symbol: str, data: Dict[str, Any]) -> None:
        for callback in list(self._summary_listeners):
            try:
                callback(symbol, data)
            except Exception as exc:
                logging.debug(f"Error en escucha de resumen: {exc}")

    def _notify_trade_state(self, state: str) -> None:
        for callback in list(self._trade_state_listeners):
            try:
                callback(state)
            except Exception as exc:
                logging.debug(f"Error en escucha de estado de operación: {exc}")

    def _evaluate_strategies(self, df: pd.DataFrame) -> Tuple[List[Tuple[str, StrategyResult]], Dict[str, Any]]:
        with self._strategy_lock:
            active_entries = [(name, func) for name, func in STRATEGY_FUNCTIONS if self.strategy_states.get(name, True)]
            divergence_enabled = self.strategy_states.get('Divergence', True)
            volatility_enabled = self.strategy_states.get('Volatility Filter', True)
            active_states = dict(self.strategy_states)
        results: List[Tuple[str, StrategyResult]] = []
        for name, func in active_entries:
            results.append((name, func(df)))
        if divergence_enabled:
            results.append(('Divergence', strategy_divergence(df)))
        if volatility_enabled:
            results.append(('Volatility Filter', strategy_volatility_filter(df)))
        consensus = combine_signals(results, active_states)
        return results, consensus

    def _handle_auto_shutdown(self) -> None:
        if self.auto_shutdown_triggered:
            return
        self.auto_shutdown_triggered = True
        logging.warning("⚠️ Límite de pérdidas consecutivas alcanzado — bot detenido automáticamente")
        self.risk.consecutive_losses = 0
        self._notify_status("auto_shutdown")
        self.stop()


    def _wait_for_contract_result(self, contract_id: int, duration_seconds: int) -> str:
        end_time = datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)
        logging.info(f"⏳ Esperando resultado real del contrato #{contract_id}...")
        while True:
            now = datetime.now(timezone.utc)
            remaining = int((end_time - now).total_seconds())
            if remaining <= 0:
                break
            logging.info(f"⌛ Contrato #{contract_id} — {remaining}s restantes...")
            time.sleep(min(RESULT_POLL_INTERVAL, max(1, remaining)))
        status = self.api.check_trade_result(contract_id)
        if status == "won":
            logging.info(f"✅ Contrato #{contract_id} GANADO")
        elif status == "lost":
            logging.info(f"❌ Contrato #{contract_id} PERDIDO")
        else:
            logging.info(f"⚠️ Contrato #{contract_id} sin resultado confirmado, reintentando...")
            status = self.api.check_trade_result(contract_id, retries=3, delay=5.0)
            if status == "won":
                logging.info(f"✅ Contrato #{contract_id} GANADO")
            elif status == "lost":
                logging.info(f"❌ Contrato #{contract_id} PERDIDO")
            else:
                logging.info(f"⚠️ Contrato #{contract_id} continúa sin resultado tras múltiples intentos")
        return status

    def _resolve_trade_result(self, status: str, stake: float) -> Tuple[str, float]:
        if status == "won":
            return "WIN", float(stake) * PAYOUT
        if status == "lost":
            return "LOSS", -float(stake)
        return "UNKNOWN", 0.0


    def _determine_signal_source(
        self,
        signal: str,
        consensus: Dict[str, Any],
        results: List[Tuple[str, StrategyResult]],
    ) -> str:
        if signal not in {"CALL", "PUT"}:
            return "COMBINED"
        if consensus.get("override"):
            reason = str(consensus.get("override_reason", "")).upper()
            if "RSI" in reason:
                return "RSI"
            if "EMA" in reason:
                return "EMA"
        matching = [name for name, outcome in results if outcome.signal == signal]
        if not matching:
            return "COMBINED"
        if matching == ["RSI"]:
            return "RSI"
        if matching == ["EMA Trend"]:
            return "EMA"
        if "RSI" in matching and all(name == "RSI" for name in matching):
            return "RSI"
        if "EMA Trend" in matching and all(name == "EMA Trend" for name in matching):
            return "EMA"
        return "COMBINED"

    def _evaluate_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        auto_learn.set_active_symbol(symbol)
        try:
            candles = self.api.fetch_candles(symbol)
            df = to_dataframe(candles)
            results, consensus = self._evaluate_strategies(df)
            self._notify_summary(symbol, consensus)
            signal = consensus['signal']
            confidence = consensus['confidence']
            reasons = consensus['reasons']
            for nombre, resultado in results:
                etiqueta = STRATEGY_DISPLAY_NAMES.get(nombre, nombre)
                mensaje = resultado.reasons[0] if resultado.reasons else 'Sin comentario'
                logging.info(f'[{symbol}] {etiqueta}: {mensaje} (señal {resultado.signal})')
            active_total = consensus['active']
            signals_total = consensus['signals']
            etiqueta_conf = consensus['confidence_label'].lower()
            if active_total == 0:
                logging.info('⚠️ Sin estrategias activas configuradas')
            elif signals_total == 0:
                logging.info('⚠️ Ninguna de las estrategias activas generó señal')
                logging.info(f"Estrategias activas: {active_total}/{TOTAL_STRATEGY_COUNT}")
            elif signal == 'NONE':
                logging.info(f"⚠️ Confianza {etiqueta_conf} ({confidence:.2f}) → {consensus['main_reason']}")
                logging.info(f"Estrategias activas: {active_total}/{TOTAL_STRATEGY_COUNT}")
                if consensus['low_volatility']:
                    valor_vol = consensus['volatility_value']
                    detalle = f" ({valor_vol:.4f})" if valor_vol is not None else ''
                    logging.info(f"⚠️ Volatilidad baja detectada{detalle}")
            else:
                logging.info(f"📊 Confianza {etiqueta_conf} ({confidence:.2f}) → {consensus['main_reason']}")
                logging.info(f"Estrategias alineadas: {consensus['aligned']}/{signals_total}")
                logging.info(f"Estrategias activas: {active_total}/{TOTAL_STRATEGY_COUNT}")
                if consensus['low_volatility']:
                    valor_vol = consensus['volatility_value']
                    detalle = f" ({valor_vol:.4f})" if valor_vol is not None else ''
                    if consensus['override']:
                        if 'RSI' in consensus['override_reason']:
                            logging.info('🚫 Volatilidad baja pero señal fuerte RSI → operación anticipada')
                        else:
                            logging.info(f"🚫 Volatilidad baja pero {consensus['override_reason'].lower()} → operación anticipada")
                    else:
                        logging.info(f"⚠️ Volatilidad baja detectada{detalle}")
                logging.info(
                    f"✅ Señal final: {signal} | Confianza {confidence:.2f} | Estrategias activas: {active_total}/{TOTAL_STRATEGY_COUNT}"
                )
            entry_price = float(df['close'].iloc[-1]) if not df.empty else 0.0
            evaluation: Dict[str, Any] = {
                'symbol': symbol,
                'signal': signal,
                'base_confidence': confidence,
                'ai_confidence': confidence,
                'final_confidence': 0.0,
                'reasons': reasons,
                'ai_notes': [],
                'consensus': consensus,
                'results': results,
                'features': None,
                'entry_price': entry_price,
                'latest_rsi': 0.0,
                'ema_spread': 0.0,
                'boll_width': 0.0,
                'volatility': 0.0,
                'signal_source': 'COMBINED',
                'ema_diff': float(df['close'].iloc[-1] - df['close'].iloc[-2]) if len(df) > 1 else 0.0,
                'indicator_confidence': 0.0,
                'adx': 0.0,
                'macd': 0.0,
                'predicted_probability': 0.5,
                'confluence_direction': 'NONE',
                'confluence_confirmed': False,
                'eligible': False,
                'rsi_signal': 'NONE',
                'ema_signal': 'NONE',
                'macd_signal': 'NONE',
            }
            latest_rsi_series = rsi(df['close'])
            latest_rsi = float(latest_rsi_series.iloc[-1]) if not latest_rsi_series.empty else 0.0
            ema_corto_valor = float(ema(df['close'], 9).iloc[-1]) if len(df) else 0.0
            ema_largo_valor = float(ema(df['close'], 21).iloc[-1]) if len(df) else 0.0
            ema_spread = ema_corto_valor - ema_largo_valor
            bandas_inferior, bandas_superior = bollinger_bands(df['close'])
            boll_width = float(bandas_superior.iloc[-1] - bandas_inferior.iloc[-1]) if len(bandas_superior) else 0.0
            volatilidad_serie = df['close'].pct_change().rolling(20).std()
            volatilidad_actual = float(np.nan_to_num(volatilidad_serie.iloc[-1], nan=0.0)) if len(volatilidad_serie) else 0.0
            adx_value = auto_learn.calculate_adx(symbol, df['high'].values, df['low'].values, df['close'].values)
            macd_value = auto_learn.calculate_macd(df['close'].values)
            price_reference = entry_price if entry_price else 1.0
            boll_ratio = boll_width / price_reference if price_reference else 0.0
            rsi_signal = next((out.signal for name, out in results if name == 'RSI'), 'NONE')
            ema_signal = next((out.signal for name, out in results if name == 'EMA Trend'), 'NONE')
            macd_signal = 'CALL' if macd_value > 0 else 'PUT' if macd_value < 0 else 'NONE'
            regime = auto_learn.detect_market_regime(adx_value, boll_ratio, volatilidad_actual)
            regime_snapshot = auto_learn.apply_market_regime(regime)
            logging.info(
                f"📊 Market Regime: {regime} | min_conf={regime_snapshot['min_confidence']:.2f}"
            )
            indicator_conf = auto_learn.calculate_confidence(
                latest_rsi,
                ema_spread,
                max(0.0, min(1.0, boll_ratio)),
                adx_value,
                macd_value,
            )
            evaluation.update(
                {
                    'latest_rsi': latest_rsi,
                    'ema_spread': ema_spread,
                    'boll_width': boll_width,
                    'volatility': volatilidad_actual,
                    'indicator_confidence': indicator_conf,
                    'adx': adx_value,
                    'macd': macd_value,
                    'rsi_signal': rsi_signal,
                    'ema_signal': ema_signal,
                    'macd_signal': macd_signal,
                    'market_regime': regime,
                    'regime_min_confidence': regime_snapshot['min_confidence'],
                }
            )
            signals = [rsi_signal, ema_signal, macd_signal]
            if signals.count('CALL') >= 2:
                evaluation['confluence_direction'] = 'CALL'
                evaluation['confluence_confirmed'] = True
            elif signals.count('PUT') >= 2:
                evaluation['confluence_direction'] = 'PUT'
                evaluation['confluence_confirmed'] = True
            current_hour = datetime.now(timezone.utc).hour
            ml_probability = auto_learn.predict_win_probability(
                latest_rsi,
                ema_spread,
                macd_value,
                adx_value,
                current_hour,
            )
            evaluation['predicted_probability'] = ml_probability
            ai_confidence = confidence
            ai_notes: List[str] = []
            features: Optional[List[float]] = None
            if signal in {'CALL', 'PUT'}:
                features = build_feature_vector(df, reasons, results)
                ai_prob = None
                if AI_ENABLED:
                    api_prob = query_ai_backend(features)
                    if api_prob is not None:
                        ai_prob = api_prob
                internal_prob, internal_notes = self.ai.predict(features)
                if internal_notes:
                    ai_notes.extend(internal_notes)
                if ai_prob is not None:
                    if internal_prob != 0.5:
                        ai_prob = (ai_prob + internal_prob) / 2
                        ai_notes.append('Mezcla adaptativa aplicada')
                    fused = self.ai.fuse_with_technical(confidence, ai_prob)
                    ai_confidence = fused
                    ai_notes.append(f'Mezcla IA {ai_prob:.2f}')
                elif internal_prob != 0.5:
                    fused = self.ai.fuse_with_technical(confidence, internal_prob)
                    ai_confidence = fused
                    ai_notes.append(f'Núcleo adaptativo {internal_prob:.2f}')
                signal_source = self._determine_signal_source(signal, consensus, results)
                auto_learn.record_signal_source(symbol, signal_source)
                evaluation['signal_source'] = signal_source
            symbol_weight = auto_learn.get_symbol_weight(symbol)
            final_mix = (ai_confidence + indicator_conf + ml_probability) / 3.0
            final_confidence = float(np.clip(final_mix * symbol_weight, 0.0, 1.0))
            min_confidence = auto_learn.get_min_confidence()
            adx_threshold = auto_learn.get_adx_min_threshold()
            rsi_high_threshold = auto_learn.get_rsi_high_threshold()
            call_floor = max(0.0, 100.0 - rsi_high_threshold)
            eligible = (
                final_confidence >= min_confidence
                and ml_probability >= 0.55
                and adx_value >= adx_threshold
            )
            if signal == 'PUT' and latest_rsi < rsi_high_threshold:
                eligible = False
            if signal == 'CALL' and latest_rsi > call_floor:
                eligible = False
            if signal not in {'CALL', 'PUT'}:
                eligible = False
            if not eligible:
                final_confidence = 0.0
            evaluation.update(
                {
                    'ai_confidence': ai_confidence,
                    'final_confidence': final_confidence,
                    'ai_notes': ai_notes,
                    'features': features,
                    'eligible': eligible,
                }
            )
            return evaluation
        except Exception as exc:
            logging.warning(f"Error en ciclo para {symbol}: {exc}")
            return None
        finally:
            auto_learn.set_active_symbol(None)

    def confirm_and_execute(self, evaluation: Dict[str, Any]) -> bool:
        direction = evaluation.get('confluence_direction')
        if direction not in {'CALL', 'PUT'}:
            return False
        symbol = evaluation.get('symbol', 'UNKNOWN')
        logging.info(f"✅ Confluence confirmed for {symbol}: {direction}")
        enriched = dict(evaluation)
        enriched['signal'] = direction
        reasons = list(enriched.get('reasons', []))
        reasons.append('Multi-indicator confirmation')
        enriched['reasons'] = reasons
        consensus_snapshot = dict(enriched.get('consensus', {}))
        consensus_snapshot['confidence_label'] = 'Alta'
        consensus_snapshot['confidence'] = 1.0
        consensus_snapshot['main_reason'] = 'multi-confirmation'
        consensus_snapshot['override'] = True
        consensus_snapshot['override_reason'] = 'multi-confirmation'
        enriched['consensus'] = consensus_snapshot
        enriched['ai_confidence'] = max(float(enriched.get('ai_confidence', 0.0)), 1.0)
        enriched['final_confidence'] = 1.0
        enriched['eligible'] = True
        probability = float(enriched.get('predicted_probability', 0.6))
        enriched['stake'] = self._calculate_kelly_stake(probability)
        return self._execute_selected_trade(enriched)

    def _fetch_exit_price(self, symbol: str, fallback: float) -> float:
        try:
            candles = self.api.fetch_candles(symbol)
            if candles:
                return float(candles[-1].close)
        except Exception as exc:
            logging.debug(f"No se pudo obtener precio de salida para {symbol}: {exc}")
        return fallback

    def _execute_selected_trade(self, evaluation: Dict[str, Any]) -> bool:
        global operation_active
        symbol = evaluation['symbol']
        signal = evaluation['signal']
        ai_confidence = float(evaluation['ai_confidence'])
        reasons = evaluation['reasons']
        ai_notes = evaluation['ai_notes']
        consensus = evaluation['consensus']
        features = evaluation['features']
        entry_price = float(evaluation['entry_price'])
        signal_source = evaluation['signal_source']
        if 'stake' not in evaluation:
            evaluation['stake'] = self._calculate_kelly_stake(evaluation.get('predicted_probability', 0.5))
        stake_amount = float(max(0.1, evaluation.get('stake', self.trade_amount)))
        ml_probability = float(evaluation.get('predicted_probability', 0.5))
        trade_initiated = False
        if not self.risk.can_trade(ai_confidence):
            return False
        if operation_active:
            return False
        auto_learn.register_trade_context(
            symbol,
            signal,
            ai_confidence,
            entry_price,
            signal_source,
            evaluation.get('indicator_confidence', 0.0),
            ml_probability,
            stake_amount,
        )
        contract_id, duration_seconds = self.api.buy(symbol, signal, stake_amount)
        if contract_id is None:
            logging.warning('No se pudo abrir la operación, se reanuda el análisis.')
            self._notify_trade_state("ready")
            return False
        operation_active = True
        self.active_trade_symbol = symbol
        self._notify_trade_state('active')
        dur_seconds = duration_seconds if duration_seconds > 0 else TRADE_DURATION_SECONDS
        logging.info(f"🟢 Operación abierta — Contrato #{contract_id} | Duración: {dur_seconds}s")
        trade_initiated = True
        try:
            result_status = self._wait_for_contract_result(contract_id, dur_seconds)
            trade_result, pnl = self._resolve_trade_result(result_status, stake_amount)
            self.risk.register_trade(pnl)
            if features is not None:
                self.ai.log_trade(features, 1 if trade_result == 'WIN' else 0)
            record_metadata = {
                'confidence_label': consensus['confidence_label'],
                'confidence_value': consensus['confidence'],
                'aligned': consensus['aligned'],
                'active': consensus['active'],
                'signals': consensus['signals'],
                'direction': signal,
                'main_reason': consensus['main_reason'],
                'low_volatility': consensus['low_volatility'],
                'volatility_value': consensus['volatility_value'],
                'override': consensus['override'],
                'override_reason': consensus['override_reason'],
                'dominant': consensus['dominant'],
                'total_weight': consensus['total_weight'],
                'active_weight': consensus.get('active_weight', 0.0),
                'contract_id': contract_id,
                'stake': stake_amount,
                'ml_probability': ml_probability,
                'market_regime': evaluation.get('market_regime', 'UNKNOWN'),
            }
            record = TradeRecord(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                decision=signal,
                confidence=ai_confidence,
                result=trade_result,
                pnl=pnl,
                reasons=reasons + ai_notes,
                metadata=record_metadata,
            )
            log_trade(record)
            if trade_result == 'WIN':
                self.win_count += 1
            elif trade_result == 'LOSS':
                self.loss_count += 1
            with self.lock:
                self.trade_history.append(record)
            latest_rsi = float(evaluation['latest_rsi'])
            ema_spread = float(evaluation['ema_spread'])
            boll_width = float(evaluation['boll_width'])
            volatilidad_actual = float(evaluation['volatility'])
            exit_price = self._fetch_exit_price(symbol, entry_price)
            price_change = float(exit_price - entry_price)
            auto_learn.update_history(
                trade_result,
                symbol,
                latest_rsi,
                ema_spread,
                boll_width,
                volatilidad_actual,
                price_change,
            )
            auto_learn.adjust_bias(symbol, trade_result, signal, price_change)
            auto_learn.store_context(
                symbol,
                latest_rsi,
                ema_spread,
                evaluation.get('macd', 0.0),
                evaluation.get('adx', 0.0),
                datetime.now(timezone.utc).hour,
                trade_result,
            )
            auto_learn.predict_next(latest_rsi, ema_spread, boll_width, volatilidad_actual)
            logging.info(
                f"{record.timestamp:%Y-%m-%d %H:%M:%S} INFO: [{symbol}] {signal} @{ai_confidence:.2f} | Stake:{stake_amount:.2f} | EMA:{evaluation['ema_diff']:.2f} RSI:{latest_rsi:.2f} | Regime:{evaluation.get('market_regime', 'UNKNOWN')} | Motivos: {'; '.join(reasons)}"
            )
            if ai_notes:
                logging.info(f"📊 Aviso IA → {'; '.join(ai_notes)}")
            self._notify_trade(record)
            if (
                self.auto_shutdown_enabled
                and not self.auto_shutdown_triggered
                and trade_result == 'LOSS'
                and self.auto_shutdown_limit > 0
                and self.risk.consecutive_losses >= self.auto_shutdown_limit
            ):
                self._handle_auto_shutdown()
        except Exception as exc:
            logging.warning(f"Error al gestionar la operación #{contract_id}: {exc}")
        finally:
            operation_active = False
            self.active_trade_symbol = None
            logging.info(RESUME_MESSAGE)
            self._notify_trade_state("ready")
        return trade_initiated

    def run(self) -> None:
        global operation_active
        self.running.set()
        self._notify_status("connecting")
        self.api.connect()
        self._notify_status("running")
        operation_active = False
        self._notify_trade_state("ready")
        try:
            while self.running.is_set():
                evaluations: List[Dict[str, Any]] = []
                confidence_lines: List[str] = []
                trade_executed = False
                for symbol in SYMBOLS:
                    if not self.running.is_set():
                        break
                    evaluation = self._evaluate_symbol(symbol)
                    if evaluation is not None:
                        probability = float(evaluation.get('predicted_probability', 0.5))
                        evaluation['stake'] = self._calculate_kelly_stake(probability)
                        confidence_lines.append(
                            f"{symbol}: {float(evaluation['final_confidence']):.2f} (p={probability:.2f})"
                        )
                        if evaluation.get('confluence_confirmed'):
                            if self.confirm_and_execute(evaluation):
                                trade_executed = True
                                break
                        evaluations.append(evaluation)
                    if not self.running.is_set():
                        break
                    time.sleep(1)
                if trade_executed:
                    continue
                if not self.running.is_set():
                    break
                if not evaluations:
                    continue
                best = max(evaluations, key=lambda item: float(item['final_confidence']))
                summary_line = " | ".join(confidence_lines)
                selected_text = (
                    f"{best['symbol']} ({float(best['final_confidence']):.2f})"
                    if best['signal'] in {'CALL', 'PUT'}
                    else "ninguno"
                )
                if summary_line:
                    logging.info(f"{summary_line} → Selected: {selected_text}")
                min_required = auto_learn.get_min_confidence()
                if (
                    best['signal'] not in {'CALL', 'PUT'}
                    or float(best['final_confidence']) < min_required
                    or not best.get('eligible', False)
                ):
                    logging.info(
                        f"⚠️ Confianza máxima {float(best['final_confidence']):.2f} inferior al mínimo {min_required:.2f} → sin operación"
                    )
                    continue
                self._execute_selected_trade(best)
        finally:
            self._notify_status("stopped")

    def stop(self) -> None:
        global operation_active
        self.running.clear()
        operation_active = False
        self.active_trade_symbol = None
        self._notify_trade_state("ready")
        try:
            if self.api.socket is not None:
                self.api.socket.close()
        except Exception:
            pass
        self.api.socket = None


# ===============================================================
# GUI LAYER
# ===============================================================


class EngineBridge(QtCore.QObject):
    trade = QtCore.pyqtSignal(object, dict)
    status = QtCore.pyqtSignal(str)
    summary = QtCore.pyqtSignal(str, dict)
    trade_state = QtCore.pyqtSignal(str)


class LogEmitter(QtCore.QObject):
    message = QtCore.pyqtSignal(str)


class QtLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.emitter = LogEmitter()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        self.emitter.message.emit(message)


class TradingThread(QtCore.QThread):
    def __init__(self, engine: TradingEngine) -> None:
        super().__init__()
        self.engine = engine

    def run(self) -> None:  # type: ignore[override]
        self.engine.run()

    def stop(self) -> None:
        self.engine.stop()


class BotWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Bot Deriv Pro Trader")
        self.resize(1320, 780)
        self.setStyleSheet(
            """
            QWidget { background: #0a0f1a; color: #b3e5ff; font: 11pt 'Consolas'; }
            QPushButton { background: #2196f3; color: white; border-radius: 6px; padding: 8px 14px; }
            QPushButton:disabled { background: #1c3c5d; color: #7aa8c7; }
            QPushButton:hover:!disabled { background: #64b5f6; }
            QTabWidget::pane { border: 1px solid #1c3c5d; }
            QTabBar::tab { background: #102235; padding: 8px 18px; margin: 2px; border-radius: 4px; }
            QTabBar::tab:selected { background: #1976d2; }
            QTableWidget { background: #101820; color: #b3e5ff; gridline-color: #25455e; }
            QHeaderView::section { background: #1976d2; color: white; padding: 6px; border: none; }
            QGroupBox { border: 1px solid #1c3c5d; border-radius: 4px; margin-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 8px; }
            QPlainTextEdit { background: #0f1724; border: 1px solid #1c3c5d; }
            QListWidget { background: #101820; border: 1px solid #1c3c5d; }
            QLabel.section-title { font: 12pt 'Consolas'; color: #90caf9; }
            """
        )

        self.bridge = EngineBridge()
        self.bridge.trade.connect(self._on_trade)
        self.bridge.status.connect(self._on_status)
        self.bridge.summary.connect(self._on_summary)
        self.bridge.trade_state.connect(self._on_trade_state)

        self.log_handler = QtLogHandler()
        self.log_handler.emitter.message.connect(self._append_log)
        logging.getLogger().addHandler(self.log_handler)

        self.engine = TradingEngine()
        self.engine.add_trade_listener(lambda record, stats: self.bridge.trade.emit(record, stats))
        self.engine.add_status_listener(lambda status: self.bridge.status.emit(status))
        self.engine.add_summary_listener(lambda symbol, data: self.bridge.summary.emit(symbol, data))
        self.engine.add_trade_state_listener(lambda state: self.bridge.trade_state.emit(state))
        self.strategy_initial_state = self._load_strategy_config()
        for name, enabled in self.strategy_initial_state.items():
            self.engine.set_strategy_state(name, enabled)

        self.thread: Optional[TradingThread] = None
        self.latest_stats: Dict[str, float] = {
            "operations": 0.0,
            "wins": 0.0,
            "losses": 0.0,
            "pnl": 0.0,
            "daily_pnl": 0.0,
            "accuracy": 0.0,
        }
        self.strategy_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
        self.asset_summary_labels: Dict[str, Dict[str, QtWidgets.QLabel]] = {}
        self.auto_shutdown_active = False
        self.history_accuracy_labels: Dict[str, QtWidgets.QLabel] = {}
        self.history_global_label: Optional[QtWidgets.QLabel] = None
        self.history_status_label: Optional[QtWidgets.QLabel] = None
        self.history_list: Optional[QtWidgets.QListWidget] = None
        self.history_bias_labels: Dict[str, QtWidgets.QLabel] = {}
        self.history_prediction_label: Optional[QtWidgets.QLabel] = None

        self._build_ui()

        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setInterval(1500)
        self.refresh_timer.timeout.connect(self._refresh_phase)
        self.refresh_timer.start()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)
        self._build_general_tab()
        self._build_strategies_tab()
        self._build_asset_summary_tab()
        self._build_settings_tab()
        self._build_history_tab()
        self._initialize_asset_summary()

    def _build_general_tab(self) -> None:
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "General")
        vbox = QtWidgets.QVBoxLayout(tab)

        control_layout = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("▶️ Iniciar")
        self.stop_button = QtWidgets.QPushButton("⏹️ Detener")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_trading)
        self.stop_button.clicked.connect(self.stop_trading)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addStretch()
        self.status_label = QtWidgets.QLabel("Estado: Inactivo")
        self.ai_mode_label = QtWidgets.QLabel("Modo IA: Pasivo")
        self.trade_state_label = QtWidgets.QLabel("Estado: Listo")
        control_layout.addWidget(self.status_label)
        control_layout.addWidget(self.ai_mode_label)
        control_layout.addWidget(self.trade_state_label)
        vbox.addLayout(control_layout)

        monto_layout = QtWidgets.QHBoxLayout()
        monto_label = QtWidgets.QLabel("Monto por operación:")
        self.trade_amount_spin = QtWidgets.QDoubleSpinBox()
        self.trade_amount_spin.setRange(0.1, 1000.0)
        self.trade_amount_spin.setDecimals(2)
        self.trade_amount_spin.setSingleStep(0.1)
        self.trade_amount_spin.setValue(self.engine.trade_amount)
        self.trade_amount_spin.valueChanged.connect(self._on_trade_amount_changed)
        monto_layout.addWidget(monto_label)
        monto_layout.addWidget(self.trade_amount_spin)
        monto_layout.addStretch()
        vbox.addLayout(monto_layout)

        stats_group = QtWidgets.QGroupBox("Desempeño")
        stats_layout = QtWidgets.QGridLayout(stats_group)
        labels = [
            ("Operaciones", "0"),
            ("Ganadas", "0"),
            ("Perdidas", "0"),
            ("Ganancia", "$0.00"),
            ("Ganancia diaria", "$0.00"),
            ("Precisión", "0.0%"),
        ]
        self.stats_values: Dict[str, QtWidgets.QLabel] = {}
        for index, (title, initial) in enumerate(labels):
            title_label = QtWidgets.QLabel(title)
            value_label = QtWidgets.QLabel(initial)
            title_label.setProperty("class", "section-title")
            stats_layout.addWidget(title_label, index // 3, (index % 3) * 2)
            stats_layout.addWidget(value_label, index // 3, (index % 3) * 2 + 1)
            self.stats_values[title] = value_label
        vbox.addWidget(stats_group)

        self.trade_table = QtWidgets.QTableWidget(0, 7)
        self.trade_table.setHorizontalHeaderLabels([
            "Hora",
            "Símbolo",
            "Decisión",
            "Confianza",
            "Resultado",
            "PnL",
            "Notas",
        ])
        self.trade_table.horizontalHeader().setStretchLastSection(True)
        self.trade_table.verticalHeader().setVisible(False)
        self.trade_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        vbox.addWidget(self.trade_table, 1)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        vbox.addWidget(self.log_view, 1)

    def _build_strategies_tab(self) -> None:
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Estrategias")
        layout = QtWidgets.QVBoxLayout(tab)
        layout.addWidget(QtWidgets.QLabel("Activar o desactivar estrategias:"))
        strategy_descriptions = {
            'RSI': 'RSI extremo → busca sobreventa (<30) o sobrecompra (>70)',
            'EMA Trend': 'Cruce de EMA → confirmar dirección de corto contra largo plazo',
            'Bollinger Rebound': 'Rebote Bollinger → aprovechar extremos con impulso del RSI',
            'Pullback': 'Pullback → retroceso controlado con recuperación del RSI',
            'Range Breakout': 'Ruptura de rango → validar cierres sobre resistencia o bajo soporte',
            'Divergence': 'Divergencia → alerta cuando el RSI contradice al precio',
            'Volatility Filter': 'Volatilidad → exigir movimiento mínimo para operar',
        }
        strategy_labels = {
            'RSI': 'RSI',
            'EMA Trend': 'Tendencia EMA',
            'Bollinger Rebound': 'Rebote Bollinger',
            'Pullback': 'Pullback',
            'Range Breakout': 'Ruptura',
            'Divergence': 'Divergencia',
            'Volatility Filter': 'Filtro de volatilidad',
        }
        states = self.engine.get_strategy_states()
        strategy_names = [name for name, _ in STRATEGY_FUNCTIONS] + ['Divergence', 'Volatility Filter']
        for name in strategy_names:
            checkbox = QtWidgets.QCheckBox(strategy_labels.get(name, name))
            checkbox.setChecked(states.get(name, True))
            checkbox.setToolTip(strategy_descriptions.get(name, ""))
            checkbox.stateChanged.connect(lambda state, n=name: self._handle_strategy_toggle(n, state))
            layout.addWidget(checkbox)
            self.strategy_checkboxes[name] = checkbox
        layout.addStretch(1)

    def _build_asset_summary_tab(self) -> None:
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Resumen de activo")
        grid = QtWidgets.QGridLayout(tab)
        columnas = 2
        for indice, symbol in enumerate(SYMBOLS):
            caja = QtWidgets.QGroupBox(symbol)
            caja_layout = QtWidgets.QVBoxLayout(caja)
            confianza = QtWidgets.QLabel("Confianza: -")
            activas = QtWidgets.QLabel(f"Estrategias activas: 0/{TOTAL_STRATEGY_COUNT}")
            alineadas = QtWidgets.QLabel("Estrategias alineadas: 0/0")
            direccion = QtWidgets.QLabel("Dirección dominante: -")
            motivo = QtWidgets.QLabel("Motivo principal: -")
            motivo.setWordWrap(True)
            for etiqueta in (confianza, activas, alineadas, direccion, motivo):
                caja_layout.addWidget(etiqueta)
            caja_layout.addStretch(1)
            fila = indice // columnas
            columna = indice % columnas
            grid.addWidget(caja, fila, columna)
            self.asset_summary_labels[symbol] = {
                'confidence': confianza,
                'active': activas,
                'aligned': alineadas,
                'direction': direccion,
                'reason': motivo,
            }
        for columna in range(columnas):
            grid.setColumnStretch(columna, 1)
        grid.setRowStretch((len(SYMBOLS) + columnas - 1) // columnas, 1)

    def _build_settings_tab(self) -> None:
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Configuración")
        form = QtWidgets.QFormLayout(tab)
        self.ai_phase_value = QtWidgets.QLabel("Pasivo")
        self.ai_accuracy_value = QtWidgets.QLabel("0.0%")
        self.daily_loss_spin = QtWidgets.QDoubleSpinBox()
        self.daily_loss_spin.setRange(-10000.0, 0.0)
        self.daily_loss_spin.setDecimals(2)
        self.daily_loss_spin.setSingleStep(1.0)
        self.daily_loss_spin.setValue(self.engine.risk.max_daily_loss)
        self.daily_loss_spin.valueChanged.connect(lambda valor: self.engine.set_daily_loss_limit(valor))
        self.take_profit_spin = QtWidgets.QDoubleSpinBox()
        self.take_profit_spin.setRange(0.0, 10000.0)
        self.take_profit_spin.setDecimals(2)
        self.take_profit_spin.setSingleStep(1.0)
        self.take_profit_spin.setValue(self.engine.risk.daily_profit_target)
        self.take_profit_spin.valueChanged.connect(lambda valor: self.engine.set_daily_profit_target(valor))
        self.drawdown_spin = QtWidgets.QDoubleSpinBox()
        self.drawdown_spin.setRange(-10000.0, 0.0)
        self.drawdown_spin.setDecimals(2)
        self.drawdown_spin.setSingleStep(1.0)
        self.drawdown_spin.setValue(self.engine.risk.max_drawdown_allowed)
        self.drawdown_spin.valueChanged.connect(lambda valor: self.engine.set_max_drawdown(valor))
        self.ml_state_label = QtWidgets.QLabel("IA adaptativa lista")
        form.addRow("Fase IA", self.ai_phase_value)
        form.addRow("Precisión IA", self.ai_accuracy_value)
        form.addRow("Límite diario de pérdida", self.daily_loss_spin)
        form.addRow("Objetivo diario de ganancia", self.take_profit_spin)
        form.addRow("Máx. drawdown", self.drawdown_spin)
        form.addRow("Motor de aprendizaje", self.ml_state_label)

        control_group = QtWidgets.QGroupBox("Control de apagado automático")
        control_layout = QtWidgets.QGridLayout(control_group)
        self.auto_shutdown_checkbox = QtWidgets.QCheckBox("Activar apagado automático")
        control_layout.addWidget(self.auto_shutdown_checkbox, 0, 0, 1, 2)
        etiqueta_limite = QtWidgets.QLabel("Apagar bot tras perder X operaciones seguidas:")
        self.auto_shutdown_spin = QtWidgets.QSpinBox()
        self.auto_shutdown_spin.setRange(1, 50)
        self.auto_shutdown_spin.setValue(3)
        control_layout.addWidget(etiqueta_limite, 1, 0)
        control_layout.addWidget(self.auto_shutdown_spin, 1, 1)
        form.addRow(control_group)
        self.auto_shutdown_checkbox.toggled.connect(self._update_auto_shutdown)
        self.auto_shutdown_spin.valueChanged.connect(self._update_auto_shutdown)
        self._update_auto_shutdown()

        self.kelly_checkbox = QtWidgets.QCheckBox("Enable Kelly Fraction")
        self.kelly_checkbox.setChecked(False)
        self.kelly_checkbox.toggled.connect(self._on_kelly_toggled)
        form.addRow("Gestión de capital", self.kelly_checkbox)

    def _build_history_tab(self) -> None:
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "History & Learning")
        layout = QtWidgets.QVBoxLayout(tab)

        accuracy_group = QtWidgets.QGroupBox("Precisión por activo")
        accuracy_layout = QtWidgets.QGridLayout(accuracy_group)
        for index, symbol in enumerate(SYMBOLS):
            name_label = QtWidgets.QLabel(symbol)
            name_label.setProperty("class", "section-title")
            value_label = QtWidgets.QLabel("0.00%")
            value_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            accuracy_layout.addWidget(name_label, index, 0)
            accuracy_layout.addWidget(value_label, index, 1)
            self.history_accuracy_labels[symbol] = value_label
        layout.addWidget(accuracy_group)

        global_group = QtWidgets.QGroupBox("Rendimiento global")
        global_layout = QtWidgets.QHBoxLayout(global_group)
        global_label = QtWidgets.QLabel("Precisión global:")
        self.history_global_label = QtWidgets.QLabel("0.00%")
        status_label = QtWidgets.QLabel("Aprendizaje:")
        self.history_status_label = QtWidgets.QLabel("OFF")
        self.history_status_label.setStyleSheet("color: #ef5350;")
        prediction_label = QtWidgets.QLabel("Última predicción:")
        self.history_prediction_label = QtWidgets.QLabel("0.50")
        global_layout.addWidget(global_label)
        global_layout.addWidget(self.history_global_label)
        global_layout.addSpacing(12)
        global_layout.addWidget(status_label)
        global_layout.addWidget(self.history_status_label)
        global_layout.addSpacing(12)
        global_layout.addWidget(prediction_label)
        global_layout.addWidget(self.history_prediction_label)
        global_layout.addStretch(1)
        layout.addWidget(global_group)

        bias_group = QtWidgets.QGroupBox("Sesgos por activo")
        bias_layout = QtWidgets.QGridLayout(bias_group)
        for index, symbol in enumerate(SYMBOLS):
            name_label = QtWidgets.QLabel(symbol)
            name_label.setProperty("class", "section-title")
            value_label = QtWidgets.QLabel("RSI +0.00 | EMA +0.00")
            value_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            bias_layout.addWidget(name_label, index, 0)
            bias_layout.addWidget(value_label, index, 1)
            self.history_bias_labels[symbol] = value_label
        layout.addWidget(bias_group)

        trades_group = QtWidgets.QGroupBox("Últimas operaciones")
        trades_layout = QtWidgets.QVBoxLayout(trades_group)
        self.history_list = QtWidgets.QListWidget()
        self.history_list.setMaximumHeight(200)
        trades_layout.addWidget(self.history_list)
        layout.addWidget(trades_group)

        button_layout = QtWidgets.QHBoxLayout()
        self.reset_history_button = QtWidgets.QPushButton("Reset History")
        button_layout.addStretch(1)
        button_layout.addWidget(self.reset_history_button)
        layout.addLayout(button_layout)
        self.reset_history_button.clicked.connect(self._reset_history_data)
        self._update_history_tab()

    def start_trading(self) -> None:
        if self.thread is not None and self.thread.isRunning():
            return
        self.auto_shutdown_active = False
        self.engine.auto_shutdown_triggered = False
        self.thread = TradingThread(self.engine)
        self.thread.finished.connect(self._on_thread_finished)
        self.thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Estado: Iniciando...")

    def stop_trading(self) -> None:
        if self.thread is None:
            if self.auto_shutdown_active:
                self.auto_shutdown_active = False
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.status_label.setText("Estado: Detenido")
                self._on_trade_state("ready")
            return
        self.thread.stop()
        self.thread.wait(2000)
        self.thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Estado: Detenido")
        self.auto_shutdown_active = False
        self._on_trade_state("ready")

    def _on_trade_amount_changed(self, value: float) -> None:
        self.engine.set_trade_amount(value)

    def _handle_strategy_toggle(self, name: str, state: int) -> None:
        enabled = state == QtCore.Qt.Checked
        self.engine.set_strategy_state(name, enabled)
        self._save_strategy_config()

    def _update_auto_shutdown(self) -> None:
        habilitado = self.auto_shutdown_checkbox.isChecked()
        limite = self.auto_shutdown_spin.value()
        self.engine.configure_auto_shutdown(habilitado, limite)
        if not habilitado and self.auto_shutdown_active:
            self.auto_shutdown_active = False
            if self.thread is None:
                self.start_button.setEnabled(True)

    def _on_kelly_toggled(self, checked: bool) -> None:
        self.engine.set_kelly_enabled(bool(checked))

    def _reset_history_data(self) -> None:
        auto_learn.reset_history()
        QtCore.QTimer.singleShot(350, self._update_history_tab)

    def _on_trade(self, record: TradeRecord, stats: Dict[str, float]) -> None:
        self.latest_stats = stats
        self.trade_table.insertRow(0)
        entries = [
            record.timestamp.strftime("%H:%M:%S"),
            record.symbol,
            record.decision,
            f"{record.confidence:.2f}",
            record.result or "-",
            f"{record.pnl:.2f}",
            "; ".join(record.reasons),
        ]
        for column, text in enumerate(entries):
            self.trade_table.setItem(0, column, QtWidgets.QTableWidgetItem(text))
        if self.trade_table.rowCount() > 250:
            self.trade_table.removeRow(self.trade_table.rowCount() - 1)
        self._update_stats_labels(stats)
        metadata = record.metadata
        summary_data = {
            'confidence': metadata.get('confidence_value', record.confidence),
            'confidence_label': metadata.get('confidence_label', 'Baja'),
            'active': metadata.get('active', 0),
            'signals': metadata.get('signals', 0),
            'aligned': metadata.get('aligned', 0),
            'signal': metadata.get('direction', record.decision),
            'dominant': metadata.get('dominant', record.decision),
            'main_reason': metadata.get('main_reason', 'Motivo no disponible'),
        }
        self._update_asset_summary_from_dict(record.symbol, summary_data)
        self._update_history_tab()

    def _on_status(self, status: str) -> None:
        if status == "auto_shutdown":
            self.auto_shutdown_active = True
            self.status_label.setText("Estado: Apagado automático")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            return
        mapping = {
            "connecting": "Estado: Conectando...",
            "running": "Estado: Ejecutando",
            "stopped": "Estado: Detenido",
        }
        self.status_label.setText(mapping.get(status, f"Estado: {status}"))
        if status == "running":
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        if status == "stopped":
            if self.auto_shutdown_active:
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
            else:
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)

    def _on_trade_state(self, state: str) -> None:
        mapping = {
            "active": "Estado: Operando",
            "waiting": "Estado: Esperando cierre de operación...",
            "ready": "Estado: Listo",
        }
        texto = mapping.get(state, "Estado: Listo")
        if hasattr(self, "trade_state_label"):
            self.trade_state_label.setText(texto)

    def _on_thread_finished(self) -> None:
        self.thread = None
        if self.auto_shutdown_active:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Estado: Apagado automático")
        else:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_label.setText("Estado: Detenido")
        self._on_trade_state("ready")

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _update_stats_labels(self, stats: Dict[str, float]) -> None:
        self.stats_values["Operaciones"].setText(str(int(stats.get("operations", 0.0))))
        self.stats_values["Ganadas"].setText(str(int(stats.get("wins", 0.0))))
        self.stats_values["Perdidas"].setText(str(int(stats.get("losses", 0.0))))
        self.stats_values["Ganancia"].setText(f"${stats.get('pnl', 0.0):.2f}")
        self.stats_values["Ganancia diaria"].setText(f"${stats.get('daily_pnl', 0.0):.2f}")
        self.stats_values["Precisión"].setText(f"{stats.get('accuracy', 0.0):.1f}%")

    def _on_summary(self, symbol: str, data: Dict[str, Any]) -> None:
        resumen = {
            'confidence': data.get('confidence', 0.0),
            'confidence_label': data.get('confidence_label', 'Baja'),
            'active': data.get('active', 0),
            'signals': data.get('signals', 0),
            'aligned': data.get('aligned', 0),
            'signal': data.get('signal', 'NONE'),
            'dominant': data.get('dominant', data.get('signal', 'NONE')),
            'main_reason': data.get('main_reason', 'Motivo no disponible'),
        }
        self._update_asset_summary_from_dict(symbol, resumen)

    def _update_asset_summary_from_dict(self, symbol: str, data: Dict[str, Any]) -> None:
        etiquetas = self.asset_summary_labels.get(symbol)
        if not etiquetas:
            return
        confianza_valor = float(data.get('confidence', 0.0))
        confianza_label = str(data.get('confidence_label', 'Baja'))
        etiquetas['confidence'].setText(f"Confianza: {confianza_label} ({confianza_valor:.2f})")
        activos = int(data.get('active', 0))
        etiquetas['active'].setText(f"Estrategias activas: {activos}/{TOTAL_STRATEGY_COUNT}")
        señales = int(data.get('signals', 0))
        alineadas = int(data.get('aligned', 0))
        divisor = señales if señales else activos
        if divisor <= 0:
            divisor = 1
        etiquetas['aligned'].setText(f"Estrategias alineadas: {alineadas}/{divisor}")
        direccion = str(data.get('signal', 'NONE'))
        if direccion == 'NONE':
            direccion = str(data.get('dominant', 'NONE'))
        etiquetas['direction'].setText(f"Dirección dominante: {direccion}")
        self._apply_direction_style(etiquetas['direction'], direccion)
        motivo = str(data.get('main_reason', 'Motivo no disponible'))
        etiquetas['reason'].setText(f"Motivo principal: {motivo}")

    def _apply_direction_style(self, label: QtWidgets.QLabel, direction: str) -> None:
        direction_upper = direction.upper()
        if direction_upper == 'CALL':
            label.setStyleSheet("color: #66bb6a;")
        elif direction_upper == 'PUT':
            label.setStyleSheet("color: #ef5350;")
        else:
            label.setStyleSheet("color: #b0bec5;")

    def _initialize_asset_summary(self) -> None:
        estados = self.engine.get_strategy_states()
        activos = sum(1 for habilitada in estados.values() if habilitada)
        for symbol in SYMBOLS:
            self._update_asset_summary_from_dict(
                symbol,
                {
                    'confidence': 0.0,
                    'confidence_label': 'Baja',
                    'active': activos,
                    'signals': 0,
                    'aligned': 0,
                    'signal': 'NONE',
                    'dominant': 'NONE',
                    'main_reason': 'Aún sin datos',
                },
            )

    def _refresh_phase(self) -> None:
        raw_phase = self.engine.ai._phase()
        phase_map = {
            "passive": "Pasivo",
            "semi-active": "Semi-activo",
            "autonomous": "Autónomo",
        }
        phase = phase_map.get(raw_phase, raw_phase.title())
        accuracy = self.engine.ai.accuracy() * 100.0
        self.ai_mode_label.setText(f"Modo IA: {phase}")
        self.ai_phase_value.setText(phase)
        self.ai_accuracy_value.setText(f"{accuracy:.2f}%")
        self.ml_state_label.setText("IA adaptativa lista" if self.engine.ai.enabled else "IA adaptativa deshabilitada")
        states = self.engine.get_strategy_states()
        for name, checkbox in self.strategy_checkboxes.items():
            desired = states.get(name, True)
            if checkbox.isChecked() != desired:
                blocker = QtCore.QSignalBlocker(checkbox)
                checkbox.setChecked(desired)
        self._update_history_tab()

    def _load_strategy_config(self) -> Dict[str, bool]:
        estados: Dict[str, bool] = {}
        try:
            if STRATEGY_CONFIG_PATH.exists():
                with STRATEGY_CONFIG_PATH.open('r', encoding='utf-8') as handle:
                    raw = json.load(handle)
                if isinstance(raw, dict):
                    alias = {
                        'RSI+EMA': 'RSI',
                        'Trend Filter': 'EMA Trend',
                        'Breakout': 'Range Breakout',
                    }
                    for nombre, valor in raw.items():
                        clave = alias.get(nombre, nombre)
                        estados[clave] = bool(valor)
        except Exception as exc:
            logging.debug(f"No se pudo cargar configuración de estrategias: {exc}")
        return estados

    def _save_strategy_config(self) -> None:
        try:
            estados = self.engine.get_strategy_states()
            if STRATEGY_CONFIG_PATH.parent != Path("."):
                STRATEGY_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with STRATEGY_CONFIG_PATH.open("w", encoding="utf-8") as handle:
                json.dump(estados, handle, ensure_ascii=False, indent=2)
        except Exception as exc:
            logging.debug(f"No se pudo guardar configuración de estrategias: {exc}")

    def _update_history_tab(self) -> None:
        summary = auto_learn.get_summary()
        per_asset = summary.get('per_asset', {})
        for symbol in SYMBOLS:
            label = self.history_accuracy_labels.get(symbol)
            if label is not None:
                valor = float(per_asset.get(symbol, 0.0))
                label.setText(f"{valor:.2f}%")
        if self.history_global_label is not None:
            self.history_global_label.setText(f"{float(summary.get('global_accuracy', 0.0)):.2f}%")
        estado_activo = bool(summary.get('status', False))
        if self.history_status_label is not None:
            self.history_status_label.setText("ON" if estado_activo else "OFF")
            self.history_status_label.setStyleSheet("color: #66bb6a;" if estado_activo else "color: #ef5350;")
        biases = summary.get('biases', {})
        for symbol in SYMBOLS:
            label = self.history_bias_labels.get(symbol)
            if label is None:
                continue
            state = biases.get(symbol, {})
            rsi_bias = float(state.get('RSI', 0.0))
            ema_bias = float(state.get('EMA', 0.0))
            label.setText(f"RSI {rsi_bias:+.2f} | EMA {ema_bias:+.2f}")
        if self.history_prediction_label is not None:
            self.history_prediction_label.setText(f"{float(summary.get('last_prediction', 0.5)):.2f}")
        if self.history_list is not None:
            self.history_list.clear()
            for trade in summary.get('last_trades', []):
                asset = trade.get('asset', '-')
                direction = trade.get('direction', '-')
                result = trade.get('result', '-')
                confidence = float(trade.get('confidence', 0.0))
                text = f"{asset} {direction} {result} {confidence:.2f}"
                item = QtWidgets.QListWidgetItem(text)
                color = QtGui.QColor('#b0bec5')
                if result == 'WIN':
                    color = QtGui.QColor('#66bb6a')
                elif result == 'LOSS':
                    color = QtGui.QColor('#ef5350')
                item.setForeground(color)
                self.history_list.addItem(item)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(2000)
        logging.getLogger().removeHandler(self.log_handler)
        super().closeEvent(event)


# ===============================================================
# ENTRY POINT
# ===============================================================
def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = BotWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
