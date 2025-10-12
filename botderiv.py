import sys
import time
import json
import threading
import logging
import warnings
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


@dataclass
class TradeRecord:
    timestamp: datetime
    symbol: str
    decision: str
    confidence: float
    result: Optional[str]
    pnl: float
    reasons: List[str]


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


def ema(series: pd.Series, period: int) -> pd.Series:
    return EMAIndicator(close=series, window=period, fillna=False).ema_indicator()


def sma(series: pd.Series, period: int) -> pd.Series:
    return SMAIndicator(close=series, window=period, fillna=False).sma_indicator()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    return RSIIndicator(close=series, window=period, fillna=False).rsi()


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


# ===============================================================
# STRATEGIES
# ===============================================================
def strategy_rsi_ema(df: pd.DataFrame) -> StrategyResult:
    rsi_series = rsi(df["close"])
    ema9 = ema(df["close"], 9)
    ema21 = ema(df["close"], 21)
    signal = "NULL"
    score = 0.0
    reasons: List[str] = []
    if len(df) < 25:
        return StrategyResult(signal, score, reasons)
    if rsi_series.iloc[-1] < 30 and ema9.iloc[-1] > ema21.iloc[-1]:
        signal = "CALL"
        score = 1.0
        reasons.append("RSI oversold + EMA9>EMA21")
    elif rsi_series.iloc[-1] > 70 and ema9.iloc[-1] < ema21.iloc[-1]:
        signal = "PUT"
        score = -1.0
        reasons.append("RSI overbought + EMA9<EMA21")
    return StrategyResult(signal, score, reasons)


def strategy_bollinger_rebound(df: pd.DataFrame) -> StrategyResult:
    if len(df) < 22:
        return StrategyResult("NULL", 0.0, [])
    lower, upper = bollinger_bands(df["close"])
    rsi_series = rsi(df["close"])
    price = df["close"].iloc[-1]
    signal = "NULL"
    score = 0.0
    reasons: List[str] = []
    if price <= lower.iloc[-1] * 1.01 and rsi_series.iloc[-1] > rsi_series.iloc[-2]:
        signal = "CALL"
        score = 0.8
        reasons.append("Bollinger rebound lower")
    elif price >= upper.iloc[-1] * 0.99 and rsi_series.iloc[-1] < rsi_series.iloc[-2]:
        signal = "PUT"
        score = -0.8
        reasons.append("Bollinger rebound upper")
    return StrategyResult(signal, score, reasons)


def strategy_trend_filter(df: pd.DataFrame) -> StrategyResult:
    if len(df) < 210:
        return StrategyResult("NULL", 0.0, [])
    sma200 = sma(df["close"], 200)
    ema50 = ema(df["close"], 50)
    ema100 = ema(df["close"], 100)
    reasons: List[str] = []
    if df["close"].iloc[-1] > sma200.iloc[-1] and ema50.iloc[-1] > ema100.iloc[-1]:
        reasons.append("Uptrend filter")
        return StrategyResult("CALL", 0.6, reasons)
    if df["close"].iloc[-1] < sma200.iloc[-1] and ema50.iloc[-1] < ema100.iloc[-1]:
        reasons.append("Downtrend filter")
        return StrategyResult("PUT", -0.6, reasons)
    return StrategyResult("NULL", 0.0, [])


def strategy_pullback(df: pd.DataFrame) -> StrategyResult:
    if len(df) < 25:
        return StrategyResult("NULL", 0.0, [])
    ema21 = ema(df["close"], 21)
    close = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    signal = "NULL"
    score = 0.0
    reasons: List[str] = []
    if close > ema21.iloc[-1] and prev_close < ema21.iloc[-2]:
        signal = "CALL"
        score = 0.7
        reasons.append("Bullish pullback")
    elif close < ema21.iloc[-1] and prev_close > ema21.iloc[-2]:
        signal = "PUT"
        score = -0.7
        reasons.append("Bearish pullback")
    return StrategyResult(signal, score, reasons)


def strategy_breakout(df: pd.DataFrame) -> StrategyResult:
    if len(df) < 25:
        return StrategyResult("NULL", 0.0, [])
    lower, upper = donchian_channels(df, 20)
    rsi_series = rsi(df["close"])
    close = df["close"].iloc[-1]
    signal = "NULL"
    score = 0.0
    reasons: List[str] = []
    if close > upper.iloc[-1] and rsi_series.iloc[-1] > 50:
        signal = "CALL"
        score = 0.9
        reasons.append("Donchian breakout up")
    elif close < lower.iloc[-1] and rsi_series.iloc[-1] < 50:
        signal = "PUT"
        score = -0.9
        reasons.append("Donchian breakout down")
    return StrategyResult(signal, score, reasons)


def strategy_divergence_block(df: pd.DataFrame) -> StrategyResult:
    if len(df) < 30:
        return StrategyResult("NULL", 0.0, [])
    rsi_series = rsi(df["close"])
    price = df["close"]
    rsi_change = rsi_series.iloc[-1] - rsi_series.iloc[-5]
    price_change = price.iloc[-1] - price.iloc[-5]
    reasons: List[str] = []
    if rsi_change > 0 and price_change < 0:
        reasons.append("Bullish divergence")
        return StrategyResult("CALL", 0.0, reasons)
    if rsi_change < 0 and price_change > 0:
        reasons.append("Bearish divergence")
        return StrategyResult("PUT", 0.0, reasons)
    return StrategyResult("NULL", 0.0, [])


def strategy_volatility_filter(df: pd.DataFrame) -> StrategyResult:
    if len(df) < 21:
        return StrategyResult("NULL", 0.0, [])
    atr_values = atr(df, 14)
    latest_atr = atr_values.iloc[-1]
    mean_atr = atr_values.iloc[-14:].mean()
    reasons: List[str] = []
    if latest_atr < mean_atr * 0.6:
        reasons.append("Volatility too low")
        return StrategyResult("NULL", -0.5, reasons)
    if latest_atr > mean_atr * 1.6:
        reasons.append("Volatility too high")
        return StrategyResult("NULL", -0.5, reasons)
    return StrategyResult("NULL", 0.0, [])


STRATEGY_FUNCTIONS: List[Tuple[str, Callable[[pd.DataFrame], StrategyResult]]] = [
    ("RSI+EMA", strategy_rsi_ema),
    ("Bollinger Rebound", strategy_bollinger_rebound),
    ("Trend Filter", strategy_trend_filter),
    ("Pullback", strategy_pullback),
    ("Breakout", strategy_breakout),
]


# ===============================================================
# SIGNAL COMBINER
# ===============================================================
def combine_signals(results: List[Tuple[str, StrategyResult]], total_active: int) -> Tuple[str, float, List[str], Dict[str, str], Dict[str, int]]:
    reasons: List[str] = []
    agreements: Dict[str, str] = {}
    votes = {"CALL": 0, "PUT": 0}
    for name, res in results:
        reasons.extend(res.reasons)
        if res.signal in {"CALL", "PUT"}:
            votes[res.signal] += 1
            agreements[name] = res.signal
        else:
            agreements[name] = "NULL"
    majority_needed = total_active // 2 + 1 if total_active else 0
    selected_signal = "NULL"
    if votes["CALL"] >= majority_needed and votes["CALL"] > votes["PUT"]:
        selected_signal = "CALL"
    elif votes["PUT"] >= majority_needed and votes["PUT"] > votes["CALL"]:
        selected_signal = "PUT"
    if selected_signal == "NULL" or total_active == 0:
        return "NULL", 0.0, reasons, agreements, votes
    ratio = votes[selected_signal] / total_active
    confidence = min(0.98, 0.4 + 0.4 * ratio)
    reasons.append(f"Consenso de {votes[selected_signal]}/{total_active} estrategias")
    return selected_signal, confidence, reasons, agreements, votes


# ===============================================================
# RISK MANAGEMENT
# ===============================================================
class RiskManager:
    def __init__(self) -> None:
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_trade_time = datetime.now(timezone.utc) - timedelta(seconds=TRADE_DELAY)

    def can_trade(self, confidence: float) -> bool:
        now = datetime.now(timezone.utc)
        if self.daily_trades >= MAX_DAILY_TRADES:
            logging.info("Se alcanzó el máximo de operaciones diarias")
            return False
        if self.daily_pnl <= MAX_DAILY_LOSS:
            logging.info("Se alcanzó el límite diario de pérdida")
            return False
        if self.daily_pnl >= MAX_DAILY_PROFIT:
            logging.info("Se alcanzó el objetivo diario de ganancia")
            return False
        if self.total_pnl <= MAX_DRAWDOWN:
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

    def buy(self, symbol: str, direction: str, amount: float) -> Tuple[Optional[int], float]:
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
                    return None, 0.0
                return msg["buy"]["contract_id"], float(proposal["ask_price"])


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
        self._strategy_lock = threading.Lock()
        self.strategy_states: Dict[str, bool] = {name: True for name, _ in STRATEGY_FUNCTIONS}
        self.strategy_states["Divergence"] = True
        self.strategy_states["Volatility Filter"] = True

    def add_trade_listener(self, callback: Callable[[TradeRecord, Dict[str, float]], None]) -> None:
        self._trade_listeners.append(callback)

    def add_status_listener(self, callback: Callable[[str], None]) -> None:
        self._status_listeners.append(callback)

    def set_strategy_state(self, name: str, enabled: bool) -> None:
        with self._strategy_lock:
            if name in self.strategy_states:
                self.strategy_states[name] = enabled

    def get_strategy_states(self) -> Dict[str, bool]:
        with self._strategy_lock:
            return dict(self.strategy_states)

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

    def _evaluate_strategies(self, df: pd.DataFrame) -> Tuple[str, float, List[str], List[Tuple[str, StrategyResult]], Dict[str, str], Dict[str, int], int]:
        with self._strategy_lock:
            active_entries = [(name, func) for name, func in STRATEGY_FUNCTIONS if self.strategy_states.get(name, True)]
            divergence_enabled = self.strategy_states.get("Divergence", True)
            volatility_enabled = self.strategy_states.get("Volatility Filter", True)
            total_active = len(active_entries) + int(divergence_enabled) + int(volatility_enabled)
        results: List[Tuple[str, StrategyResult]] = []
        for name, func in active_entries:
            results.append((name, func(df)))
        if divergence_enabled:
            results.append(("Divergence", strategy_divergence_block(df)))
        if volatility_enabled:
            results.append(("Volatility Filter", strategy_volatility_filter(df)))
        signal, confidence, reasons, agreements, votes = combine_signals(results, total_active)
        return signal, confidence, reasons, results, agreements, votes, total_active

    def _simulate_result(self) -> Tuple[str, float]:
        outcome = np.random.rand() > 0.5
        pnl = STAKE * PAYOUT if outcome else -STAKE
        return ("WIN" if outcome else "LOSS"), pnl

    def execute_cycle(self, symbol: str) -> None:
        candles = self.api.fetch_candles(symbol)
        df = to_dataframe(candles)
        signal, confidence, reasons, results, agreements, votes, total_active = self._evaluate_strategies(df)
        if total_active > 0:
            if signal != "NULL":
                logging.info(f"✅ {votes[signal]}/{total_active} estrategias confirman {signal}")
            else:
                direccion = "CALL" if votes["CALL"] >= votes["PUT"] else "PUT"
                if max(votes.values()) > 0:
                    logging.info(f"⚠️ {votes[direccion]}/{total_active} estrategias en desacuerdo")
                else:
                    logging.info(f"⚠️ Ninguna de las {total_active} estrategias activas generó señal")
        if signal == "NULL":
            return
        features = build_feature_vector(df, reasons, results)
        ai_prob = None
        if AI_ENABLED:
            api_prob = query_ai_backend(features)
            if api_prob is not None:
                ai_prob = api_prob
        ai_confidence = confidence
        ai_notes: List[str] = []
        internal_prob, internal_notes = self.ai.predict(features)
        if internal_notes:
            ai_notes.extend(internal_notes)
        if ai_prob is not None:
            if internal_prob != 0.5:
                ai_prob = (ai_prob + internal_prob) / 2
                ai_notes.append("Mezcla adaptativa aplicada")
            fused = self.ai.fuse_with_technical(confidence, ai_prob)
            ai_confidence = fused
            ai_notes.append(f"Mezcla IA {ai_prob:.2f}")
        elif internal_prob != 0.5:
            fused = self.ai.fuse_with_technical(confidence, internal_prob)
            ai_confidence = fused
            ai_notes.append(f"Núcleo adaptativo {internal_prob:.2f}")
        if not self.risk.can_trade(ai_confidence):
            return
        contract_id, price = self.api.buy(symbol, signal, STAKE)
        if contract_id is None:
            return
        result, pnl = self._simulate_result()
        self.risk.register_trade(pnl)
        self.ai.log_trade(features, 1 if result == "WIN" else 0)
        record = TradeRecord(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            decision=signal,
            confidence=ai_confidence,
            result=result,
            pnl=pnl,
            reasons=reasons + ai_notes,
        )
        log_trade(record)
        if result == "WIN":
            self.win_count += 1
        else:
            self.loss_count += 1
        with self.lock:
            self.trade_history.append(record)
        ema_diff = df["close"].iloc[-1] - df["close"].iloc[-2]
        latest_rsi = rsi(df["close"]).iloc[-1]
        logging.info(
            f"{record.timestamp:%Y-%m-%d %H:%M:%S} INFO: [{symbol}] {signal} @{ai_confidence:.2f} | EMA:{ema_diff:.2f} RSI:{latest_rsi:.2f} | Motivos: {'; '.join(reasons)}"
        )
        if ai_notes:
            logging.info(f"📊 Aviso IA → {'; '.join(ai_notes)}")
        self._notify_trade(record)

    def run(self) -> None:
        self.running.set()
        self._notify_status("connecting")
        self.api.connect()
        self._notify_status("running")
        try:
            while self.running.is_set():
                for symbol in SYMBOLS:
                    if not self.running.is_set():
                        break
                    try:
                        self.execute_cycle(symbol)
                    except Exception as exc:
                        logging.warning(f"Error en ciclo para {symbol}: {exc}")
                    if not self.running.is_set():
                        break
                    time.sleep(1)
        finally:
            self._notify_status("stopped")

    def stop(self) -> None:
        self.running.clear()
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

        self.log_handler = QtLogHandler()
        self.log_handler.emitter.message.connect(self._append_log)
        logging.getLogger().addHandler(self.log_handler)

        self.engine = TradingEngine()
        self.engine.add_trade_listener(lambda record, stats: self.bridge.trade.emit(record, stats))
        self.engine.add_status_listener(lambda status: self.bridge.status.emit(status))
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
        self._build_settings_tab()

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
        control_layout.addWidget(self.status_label)
        control_layout.addWidget(self.ai_mode_label)
        vbox.addLayout(control_layout)

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
            "RSI+EMA": "Cruce RSI + EMA → buscar sobreventa o sobrecompra alineada",
            "Bollinger Rebound": "Rebote en Bollinger → aprovechar extremos con confirmación RSI",
            "Trend Filter": "Filtro de tendencia → seguir la SMA200 y estructura EMA",
            "Pullback": "Pullback → retroceso hacia EMA21 con vela de confirmación",
            "Breakout": "Ruptura Donchian → seguir nuevos máximos/mínimos con RSI",
            "Divergence": "Bloqueo por divergencia → evitar operaciones si RSI discrepa",
            "Volatility Filter": "Filtro de volatilidad → evitar ATR demasiado bajo o alto",
        }
        strategy_labels = {
            "RSI+EMA": "RSI + EMA",
            "Bollinger Rebound": "Rebote Bollinger",
            "Trend Filter": "Filtro de tendencia",
            "Pullback": "Pullback",
            "Breakout": "Ruptura Donchian",
            "Divergence": "Bloqueo por divergencia",
            "Volatility Filter": "Filtro de volatilidad",
        }
        states = self.engine.get_strategy_states()
        strategy_names = [name for name, _ in STRATEGY_FUNCTIONS] + ["Divergence", "Volatility Filter"]
        for name in strategy_names:
            checkbox = QtWidgets.QCheckBox(strategy_labels.get(name, name))
            checkbox.setChecked(states.get(name, True))
            checkbox.setToolTip(strategy_descriptions.get(name, ""))
            checkbox.stateChanged.connect(lambda state, n=name: self._handle_strategy_toggle(n, state))
            layout.addWidget(checkbox)
            self.strategy_checkboxes[name] = checkbox
        layout.addStretch(1)

    def _build_settings_tab(self) -> None:
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Configuración")
        form = QtWidgets.QFormLayout(tab)
        self.ai_phase_value = QtWidgets.QLabel("Pasivo")
        self.ai_accuracy_value = QtWidgets.QLabel("0.0%")
        self.daily_limit_value = QtWidgets.QLabel(f"{MAX_DAILY_LOSS:.2f}")
        self.take_profit_value = QtWidgets.QLabel(f"{MAX_DAILY_PROFIT:.2f}")
        self.drawdown_value = QtWidgets.QLabel(f"{MAX_DRAWDOWN:.2f}")
        self.ml_state_label = QtWidgets.QLabel("IA adaptativa lista")
        form.addRow("Fase IA", self.ai_phase_value)
        form.addRow("Precisión IA", self.ai_accuracy_value)
        form.addRow("Límite diario de pérdida", self.daily_limit_value)
        form.addRow("Objetivo diario de ganancia", self.take_profit_value)
        form.addRow("Máx. drawdown", self.drawdown_value)
        form.addRow("Motor de aprendizaje", self.ml_state_label)

    def start_trading(self) -> None:
        if self.thread is not None and self.thread.isRunning():
            return
        self.thread = TradingThread(self.engine)
        self.thread.finished.connect(self._on_thread_finished)
        self.thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Estado: Iniciando...")

    def stop_trading(self) -> None:
        if self.thread is None:
            return
        self.thread.stop()
        self.thread.wait(2000)
        self.thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Estado: Detenido")

    def _handle_strategy_toggle(self, name: str, state: int) -> None:
        enabled = state == QtCore.Qt.Checked
        self.engine.set_strategy_state(name, enabled)
        self._save_strategy_config()

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

    def _on_status(self, status: str) -> None:
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
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def _on_thread_finished(self) -> None:
        self.thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Estado: Detenido")

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

    def _load_strategy_config(self) -> Dict[str, bool]:
        estados: Dict[str, bool] = {}
        try:
            if STRATEGY_CONFIG_PATH.exists():
                with STRATEGY_CONFIG_PATH.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                if isinstance(raw, dict):
                    for nombre, valor in raw.items():
                        estados[nombre] = bool(valor)
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
