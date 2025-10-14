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
def strategy_rsi(df: pd.DataFrame) -> StrategyResult:
    if len(df) < 15:
        return StrategyResult('NONE', 0.0, ['RSI sin suficientes datos'], {'rsi': None})
    rsi_series = rsi(df['close'])
    valor = float(rsi_series.iloc[-1])
    extra = {
        'rsi': valor,
        'strong_call': valor < 25,
        'strong_put': valor > 75,
    }
    if valor < 30:
        return StrategyResult('CALL', 2.0, [f"RSI {valor:.2f} sobrevendido → señal CALL"], extra)
    if valor > 70:
        return StrategyResult('PUT', -2.0, [f"RSI {valor:.2f} sobrecomprado → señal PUT"], extra)
    return StrategyResult('NONE', 0.0, [f"RSI {valor:.2f} sin sesgo claro"], extra)


def strategy_ema_trend(df: pd.DataFrame) -> StrategyResult:
    if len(df) < 25:
        return StrategyResult('NONE', 0.0, ['EMAs sin datos suficientes'], {})
    ema_corto = ema(df['close'], 9)
    ema_largo = ema(df['close'], 21)
    cruz_alcista = ema_corto.iloc[-2] <= ema_largo.iloc[-2] and ema_corto.iloc[-1] > ema_largo.iloc[-1]
    cruz_bajista = ema_corto.iloc[-2] >= ema_largo.iloc[-2] and ema_corto.iloc[-1] < ema_largo.iloc[-1]
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

    def _resolve_trade_result(self, status: str) -> Tuple[str, float]:
        if status == "won":
            return "WIN", self.trade_amount * PAYOUT
        if status == "lost":
            return "LOSS", -self.trade_amount
        return "UNKNOWN", 0.0


    def execute_cycle(self, symbol: str) -> None:
        global operation_active
        now = datetime.now(timezone.utc)
        if operation_active:
            time.sleep(2)
            return
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
        if active_total == 0:
            logging.info('⚠️ Sin estrategias activas configuradas')
            self._notify_trade_state("ready")
            return
        if signals_total == 0:
            logging.info('⚠️ Ninguna de las estrategias activas generó señal')
            logging.info(f"Estrategias activas: {active_total}/{TOTAL_STRATEGY_COUNT}")
            self._notify_trade_state("ready")
            return
        logging.info(f"Estrategias con señal: {signals_total}/{active_total}")
        etiqueta_conf = consensus['confidence_label'].lower()
        if signal == 'NONE':
            logging.info(f"⚠️ Confianza {etiqueta_conf} ({confidence:.2f}) → {consensus['main_reason']}")
            logging.info(f"Estrategias activas: {active_total}/{TOTAL_STRATEGY_COUNT}")
            if consensus['low_volatility']:
                valor_vol = consensus['volatility_value']
                detalle = f" ({valor_vol:.4f})" if valor_vol is not None else ''
                logging.info(f"⚠️ Volatilidad baja detectada{detalle}")
            self._notify_trade_state("ready")
            return
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
        logging.info(f"✅ Señal final: {signal} | Confianza {confidence:.2f} | Estrategias activas: {active_total}/{TOTAL_STRATEGY_COUNT}")
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
                ai_notes.append('Mezcla adaptativa aplicada')
            fused = self.ai.fuse_with_technical(confidence, ai_prob)
            ai_confidence = fused
            ai_notes.append(f'Mezcla IA {ai_prob:.2f}')
        elif internal_prob != 0.5:
            fused = self.ai.fuse_with_technical(confidence, internal_prob)
            ai_confidence = fused
            ai_notes.append(f'Núcleo adaptativo {internal_prob:.2f}')
        if not self.risk.can_trade(ai_confidence):
            self._notify_trade_state("ready")
            return
        if operation_active:
            return
        contract_id, duration_seconds = self.api.buy(symbol, signal, self.trade_amount)
        if contract_id is None:
            logging.warning('No se pudo abrir la operación, se reanuda el análisis.')
            self._notify_trade_state("ready")
            return
        operation_active = True
        self.active_trade_symbol = symbol
        self._notify_trade_state('active')
        dur_seconds = duration_seconds if duration_seconds > 0 else TRADE_DURATION_SECONDS
        logging.info(f"🟢 Operación abierta — Contrato #{contract_id} | Duración: {dur_seconds}s")
        try:
            result_status = self._wait_for_contract_result(contract_id, dur_seconds)
            trade_result, pnl = self._resolve_trade_result(result_status)
            self.risk.register_trade(pnl)
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
            ema_diff = df['close'].iloc[-1] - df['close'].iloc[-2]
            latest_rsi = float(rsi(df['close']).iloc[-1])
            logging.info(
                f"{record.timestamp:%Y-%m-%d %H:%M:%S} INFO: [{symbol}] {signal} @{ai_confidence:.2f} | EMA:{ema_diff:.2f} RSI:{latest_rsi:.2f} | Motivos: {'; '.join(reasons)}"
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
