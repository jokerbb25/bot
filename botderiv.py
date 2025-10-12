import time
import json
import threading
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.linear_model import LogisticRegression

try:
    import websocket  # type: ignore
except ImportError:  # pragma: no cover
    websocket = None

# ===============================================================
# CONFIG & CONSTANTS
# ===============================================================
APP_ID = "1089"
API_TOKEN = "dK57Ark9QreDexO"
SYMBOLS = ["R_25", "R_50"]
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
ADVISORY_INTERVAL_SEC = 180

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


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
        logging.debug(f"Trade log error: {exc}")


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


STRATEGY_FUNCTIONS = [
    strategy_rsi_ema,
    strategy_bollinger_rebound,
    strategy_trend_filter,
    strategy_pullback,
    strategy_breakout,
]


# ===============================================================
# SIGNAL COMBINER
# ===============================================================
def combine_signals(results: List[StrategyResult], divergence_block: StrategyResult, volatility_filter: StrategyResult) -> Tuple[str, float, List[str]]:
    total_score = 0.0
    reasons: List[str] = []
    for res in results:
        total_score += res.score
        reasons.extend(res.reasons)
    threshold = 1.2
    signal = "NULL"
    if divergence_block.signal in {"CALL", "PUT"}:
        reasons.append("Divergence detected → skip")
        return "NULL", 0.0, reasons
    if volatility_filter.score < 0:
        reasons.extend(volatility_filter.reasons)
        return "NULL", 0.0, reasons
    if total_score > threshold:
        signal = "CALL"
    elif total_score < -threshold:
        signal = "PUT"
    confidence = min(0.98, 0.4 + 0.1 * abs(total_score)) if signal != "NULL" else 0.0
    return signal, confidence, reasons


# ===============================================================
# RISK MANAGEMENT
# ===============================================================
class RiskManager:
    def __init__(self) -> None:
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_trade_time = datetime.min

    def can_trade(self, confidence: float) -> bool:
        now = datetime.utcnow()
        if self.daily_trades >= MAX_DAILY_TRADES:
            logging.info("Max daily trades reached")
            return False
        if self.daily_pnl <= MAX_DAILY_LOSS:
            logging.info("Daily loss limit reached")
            return False
        if self.daily_pnl >= MAX_DAILY_PROFIT:
            logging.info("Daily profit target reached")
            return False
        if self.total_pnl <= MAX_DRAWDOWN:
            logging.info("Max drawdown reached")
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
        self.last_trade_time = datetime.utcnow()
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
        self.enabled = AI_ENABLED
        self.passive = AI_PASSIVE_MODE
        self.model: Optional[LogisticRegression] = None
        self.trade_counter = 0
        self.win_counter = 0
        self.lock = threading.Lock()
        self.feature_cache: List[np.ndarray] = []
        self.result_cache: List[int] = []
        self.offline_thread = threading.Thread(target=self._advisory_loop, daemon=True)
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
        except Exception as exc:  # pragma: no cover
            logging.debug(f"Adaptive model train error: {exc}")

    def predict(self, features: np.ndarray) -> Tuple[float, List[str]]:
        if not self.enabled:
            return 0.5, []
        phase = self._phase()
        if phase == "passive" or self.model is None:
            return 0.5, ["Passive learning"]
        prob = float(self.model.predict_proba(features.reshape(1, -1))[0][1])
        if phase == "semi-active":
            return prob, ["Semi-active advisory"]
        return prob, ["Autonomous AI"]

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
                logging.info(f"📊 AI Advisory → phase={phase} accuracy={acc:.2f}% trades={self.trade_counter}")
                self._train_model()
            except Exception as exc:  # pragma: no cover
                logging.debug(f"Advisory loop error: {exc}")


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
    payload = {
        "model": "phi3:mini",
        "prompt": "Analyze this market data and return only CALL or PUT based on current trend:",
        "stream": False,
    }
    attempt = 0
    start = time.perf_counter()
    while attempt < 3:
        try:
            response = requests.post(AI_ENDPOINT, json=payload, timeout=AI_TIMEOUT)
            response.raise_for_status()
            content = response.json()
            text = str(content.get("response", "")).lower()
            if "call" in text and "put" not in text:
                return 0.8
            if "put" in text and "call" not in text:
                return 0.2
            if text:
                return 0.5
            raise ValueError("empty response")
        except Exception as exc:
            attempt += 1
            if attempt >= 3 or (time.perf_counter() - start) > AI_TIMEOUT:
                logging.warning("⚙️ Using technical fallback (AI not responding)")
                return None
            time.sleep(0.5)
    logging.warning("⚙️ Using technical fallback (AI not responding)")
    return None


# ===============================================================
# FEATURE ENGINEERING
# ===============================================================
def build_feature_vector(df: pd.DataFrame, reasons: List[str], results: List[StrategyResult]) -> np.ndarray:
    ema9 = ema(df["close"], 9).iloc[-1]
    ema21 = ema(df["close"], 21).iloc[-1]
    ema50 = ema(df["close"], 50).iloc[-1]
    ema100 = ema(df["close"], 100).iloc[-1]
    rsi_val = rsi(df["close"], 14).iloc[-1]
    lower_bb, upper_bb = bollinger_bands(df["close"])
    atr_val = atr(df, 14).iloc[-1]
    total_score = sum(res.score for res in results)
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
                logging.info("Deriv connected")
                return
            except Exception as exc:
                logging.warning(f"Deriv connection error: {exc}")
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
                    logging.warning(f"Proposal error: {msg['error']}")
                    return None, 0.0
                proposal = msg["proposal"]
                break
        buy_id = self._send({"buy": proposal["id"], "price": proposal["ask_price"]})
        while True:
            msg = self._recv()
            if msg.get("req_id") == buy_id:
                if "error" in msg:
                    logging.warning(f"Buy error: {msg['error']}")
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

    def _evaluate_strategies(self, df: pd.DataFrame) -> Tuple[str, float, List[str], List[StrategyResult]]:
        results = [func(df) for func in STRATEGY_FUNCTIONS]
        divergence_res = strategy_divergence_block(df)
        volatility_res = strategy_volatility_filter(df)
        signal, confidence, reasons = combine_signals(results, divergence_res, volatility_res)
        if divergence_res.reasons:
            reasons.extend(divergence_res.reasons)
        if volatility_res.reasons:
            reasons.extend(volatility_res.reasons)
        return signal, confidence, reasons, results

    def _simulate_result(self) -> Tuple[str, float]:
        outcome = np.random.rand() > 0.5
        pnl = STAKE * PAYOUT if outcome else -STAKE
        return ("WIN" if outcome else "LOSS"), pnl

    def execute_cycle(self, symbol: str) -> None:
        candles = self.api.fetch_candles(symbol)
        df = to_dataframe(candles)
        signal, confidence, reasons, results = self._evaluate_strategies(df)
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
        if ai_prob is not None:
            fused = self.ai.fuse_with_technical(confidence, ai_prob)
            ai_confidence = fused
            ai_notes.append(f"AI blend {ai_prob:.2f}")
        if not self.risk.can_trade(ai_confidence):
            return
        contract_id, price = self.api.buy(symbol, signal, STAKE)
        if contract_id is None:
            return
        result, pnl = self._simulate_result()
        self.risk.register_trade(pnl)
        self.ai.log_trade(features, 1 if result == "WIN" else 0)
        record = TradeRecord(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            decision=signal,
            confidence=ai_confidence,
            result=result,
            pnl=pnl,
            reasons=reasons + ai_notes,
        )
        log_trade(record)
        with self.lock:
            self.trade_history.append(record)
        ema_diff = df['close'].iloc[-1] - df['close'].iloc[-2]
        latest_rsi = rsi(df['close']).iloc[-1]
        logging.info(
            f"{record.timestamp:%Y-%m-%d %H:%M:%S} INFO: [{symbol}] {signal} @{ai_confidence:.2f} | EMA:{ema_diff:.2f} RSI:{latest_rsi:.2f} | {'+'.join(reasons)}"
        )
        if ai_notes:
            logging.info(f"📊 AI Advisory → {'; '.join(ai_notes)}")

    def run(self) -> None:
        self.api.connect()
        while True:
            for symbol in SYMBOLS:
                try:
                    self.execute_cycle(symbol)
                except Exception as exc:
                    logging.warning(f"Cycle error for {symbol}: {exc}")
                time.sleep(1)


# ===============================================================
# ENTRY POINT
# ===============================================================
def main() -> None:
    engine = TradingEngine()
    engine.run()


if __name__ == "__main__":
    main()