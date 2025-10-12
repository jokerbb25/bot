"""Local machine learning predictor for Deriv bot.

This module provides a lightweight predictor that can be extended to load
neural network models while keeping a deterministic NumPy-based fallback so
that the trading workflow never breaks if heavy ML frameworks are missing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional

import numpy as np
import requests

try:
    # Optional dependencies: only used if available.
    import tensorflow as tf  # type: ignore

    TF_AVAILABLE = True
except Exception:  # pragma: no cover - optional path
    TF_AVAILABLE = False

MODEL_DIR = Path("models")
FEATURE_WINDOW = 60
CACHE_MAX_ITEMS = 128


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _ema(values: Iterable[float], period: int) -> float:
    values = list(values)
    if not values:
        return 0.0
    if len(values) < period:
        return float(np.mean(values))
    k = 2 / (period + 1)
    ema_val = float(values[0])
    for price in values[1:]:
        ema_val = float(price) * k + ema_val * (1 - k)
    return ema_val


def _pattern_flags(candles: List[Dict[str, float]]) -> Dict[str, int]:
    flags = {"engulfing": 0, "doji": 0, "hammer": 0}
    if len(candles) < 2:
        return flags
    last = candles[-1]
    prev = candles[-2]
    body = abs(last["close"] - last["open"])
    high_low = last["high"] - last["low"]
    body_prev = abs(prev["close"] - prev["open"])

    if body_prev > 0 and body > body_prev * 1.1:
        flags["engulfing"] = 1
    if high_low > 0 and body / high_low < 0.1:
        flags["doji"] = 1
    upper_wick = last["high"] - max(last["close"], last["open"])
    lower_wick = min(last["close"], last["open"]) - last["low"]
    if body > 0 and lower_wick > body * 1.5 and upper_wick < body * 0.3:
        flags["hammer"] = 1
    return flags


def build_features(
    candles: List[Dict[str, float]],
    indicators_ctx: Dict[str, Any],
    recent_trades: Iterable[Dict[str, Any]],
    window: int = FEATURE_WINDOW,
) -> np.ndarray:
    """Convert recent market information into a ML feature vector."""
    if len(candles) < window:
        window = len(candles)
    if window == 0:
        return np.zeros(10, dtype=float)

    closes = np.array([c["close"] for c in candles[-window:]], dtype=float)
    opens = np.array([c["open"] for c in candles[-window:]], dtype=float)
    highs = np.array([c["high"] for c in candles[-window:]], dtype=float)
    lows = np.array([c["low"] for c in candles[-window:]], dtype=float)

    returns = np.diff(closes) / np.clip(closes[:-1], 1e-6, None)
    returns = returns if returns.size else np.zeros(1)
    short_ema = _ema(closes, 9)
    long_ema = _ema(closes, 21)
    previous_window = closes[:-1] if closes.size > 1 else closes
    ema_slope = short_ema - _ema(previous_window, 9)
    ema_variance = float(np.var(closes[-10:])) if closes.size >= 10 else 0.0
    rsi_value = float(indicators_ctx.get("rsi", 50.0))
    rsi_trend = 1.0 if indicators_ctx.get("rsi_trend") == "up" else -1.0 if indicators_ctx.get("rsi_trend") == "down" else 0.0
    volatility_short = float(np.std(returns[-10:])) if returns.size >= 10 else float(np.std(returns))
    volatility_long = float(np.std(returns))

    bodies = closes - opens
    wicks_up = highs - np.maximum(closes, opens)
    wicks_down = np.minimum(closes, opens) - lows
    avg_body = float(np.mean(np.abs(bodies))) if bodies.size else 0.0
    avg_wick_up = float(np.mean(np.abs(wicks_up))) if wicks_up.size else 0.0
    avg_wick_down = float(np.mean(np.abs(wicks_down))) if wicks_down.size else 0.0

    flags = _pattern_flags(candles[-5:])

    recent = list(recent_trades)[-5:]
    wins = sum(1 for t in recent if t.get("result") == "WIN")
    losses = sum(1 for t in recent if t.get("result") == "LOSS")
    accuracy = wins / max(1, len(recent))

    feature_vector = np.array(
        [
            closes[-1],
            returns[-1] if returns.size else 0.0,
            float(np.mean(returns)) if returns.size else 0.0,
            float(np.std(returns)) if returns.size else 0.0,
            short_ema,
            long_ema,
            ema_slope,
            ema_variance,
            rsi_value,
            rsi_trend,
            volatility_short,
            volatility_long,
            avg_body,
            avg_wick_up,
            avg_wick_down,
            flags["engulfing"],
            flags["doji"],
            flags["hammer"],
            accuracy,
            wins,
            losses,
            float(indicators_ctx.get("bull_score", 0.0)),
            float(indicators_ctx.get("bear_score", 0.0)),
        ],
        dtype=float,
    )
    return feature_vector


class _LRFallbackModel:
    """Simple logistic regression fallback to keep predictions available."""

    def __init__(self, feature_dim: int) -> None:
        rng = np.random.default_rng(42)
        self.weights = rng.normal(0, 0.05, feature_dim)
        self.bias = 0.0

    def predict_proba(self, features: np.ndarray) -> float:
        logit = float(np.dot(features, self.weights) + self.bias)
        return _sigmoid(logit)


class MLPredictor:
    """Lightweight ML predictor with caching and timeout control."""

    def __init__(
        self,
        version: str,
        timeout_ms: int,
        enabled: bool = True,
        use_api: bool = False,
        external_model_url: Optional[str] = None,
    ) -> None:
        self.version = version
        self.timeout_ms = timeout_ms
        self.enabled = enabled
        self.use_api = use_api
        self.external_model_url = external_model_url
        self.model: Any = None
        self.model_backend = "numpy"
        self.cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.cache_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.ready = False
        if self.enabled:
            try:
                self.load_model(version)
            except Exception as exc:  # pragma: no cover - defensive
                logging.warning(f"MLPredictor load failed: {exc}")
                self.enabled = False

    def _cache_key(self, features: np.ndarray) -> str:
        digest = hashlib.sha1(features.tobytes()).hexdigest()
        return digest

    def _cache_put(self, key: str, value: Dict[str, Any]) -> None:
        with self.cache_lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            while len(self.cache) > CACHE_MAX_ITEMS:
                self.cache.popitem(last=False)

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.cache_lock:
            value = self.cache.get(key)
            if value:
                self.cache.move_to_end(key)
            return value

    def load_model(self, version: str) -> None:
        """Load a model version. Falls back to logistic regression if missing."""
        with self.model_lock:
            model_path = MODEL_DIR / f"ml_model_{version}.h5"
            if TF_AVAILABLE and model_path.exists():
                self.model = tf.keras.models.load_model(model_path)  # type: ignore[attr-defined]
                self.model_backend = "tensorflow"
                self.ready = True
                logging.info(f"MLPredictor loaded TensorFlow model {model_path}")
                return
            # Fallback: random logistic regression baseline.
            self.model = _LRFallbackModel(feature_dim=23)
            self.model_backend = "numpy"
            self.ready = True
            logging.info("MLPredictor using NumPy logistic regression fallback")

    def predict(
        self,
        symbol: str,
        candles: List[Dict[str, float]],
        indicators_ctx: Dict[str, Any],
        recent_trades: Deque[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return probability estimates for the next movement."""
        if not self.enabled or not self.ready:
            return {
                "prob_up": 0.5,
                "prob_down": 0.5,
                "latency_ms": 0,
                "model_version": self.version,
                "status": "disabled",
                "timed_out": False,
            }

        features = build_features(candles, indicators_ctx, recent_trades)
        key = self._cache_key(features)
        cached = self._cache_get(key)
        if cached:
            cached["cached"] = True
            logging.info(
                f"[{symbol}] AI backend=cache latency={cached['latency_ms']}ms prob_up={cached['prob_up']:.2f}"
            )
            return cached

        backend_used = self.model_backend
        prob_up = 0.5
        latency_ms = 0
        timed_out = False

        if self.use_api and self.external_model_url:
            backend_used = "ollama"
            start_api = time.perf_counter()
            response_text = ""
            api_success = False
            last_error = ""
            for attempt in range(3):
                if attempt:
                    time.sleep(0.5)
                try:
                    payload = {
                        "model": "phi3:mini",
                        "prompt": (
                            "Analyze this market data and return only CALL or PUT based on current trend: "
                            f"features={features.tolist()}"
                        ),
                        "stream": False,
                    }
                    resp = requests.post(
                        self.external_model_url,
                        json=payload,
                        timeout=3,
                    )
                    resp.raise_for_status()
                    resp_json = resp.json()
                    response_text = str(resp_json.get("response", ""))
                    if response_text:
                        api_success = True
                        break
                    last_error = "missing response"
                except Exception as exc:  # pragma: no cover - network path
                    last_error = str(exc)
                if (time.perf_counter() - start_api) > 3:
                    break
            latency_ms = int((time.perf_counter() - start_api) * 1000)
            if api_success:
                lowered = response_text.lower()
                if "call" in lowered and "put" not in lowered:
                    prob_up = 0.8
                elif "put" in lowered and "call" not in lowered:
                    prob_up = 0.2
                elif "call" in lowered and "put" in lowered:
                    prob_up = 0.5
                else:
                    api_success = False
                    last_error = "unrecognized response"
            if not api_success:
                logging.warning("⚙️ Using technical fallback (AI not responding)")
                backend_used = self.model_backend
                if self.model_backend == "tensorflow":  # pragma: no cover - optional path
                    arr = np.expand_dims(features, axis=0)
                    prediction = self.model.predict(arr, verbose=0)[0]
                    prob_up = float(prediction[0]) if prediction.size else 0.5
                else:
                    prob_up = float(self.model.predict_proba(features))
                latency_ms = int((time.perf_counter() - start_api) * 1000)
        else:
            start_local = time.perf_counter()
            if self.model_backend == "tensorflow":  # pragma: no cover - optional path
                arr = np.expand_dims(features, axis=0)
                prediction = self.model.predict(arr, verbose=0)[0]
                prob_up = float(prediction[0]) if prediction.size else 0.5
            else:
                prob_up = float(self.model.predict_proba(features))
            latency_ms = int((time.perf_counter() - start_local) * 1000)

        timed_out = latency_ms > self.timeout_ms
        if timed_out:
            logging.warning(f"[{symbol}] ML predictor timeout {latency_ms}ms > {self.timeout_ms}ms")
            return {
                "prob_up": 0.5,
                "prob_down": 0.5,
                "latency_ms": latency_ms,
                "model_version": self.version,
                "status": "timeout",
                "timed_out": True,
            }

        recent = list(recent_trades)[-50:]
        if recent:
            valid = [item for item in recent if item.get("result") in {"WIN", "LOSS"}]
            wins = sum(1 for item in valid if item.get("result") == "WIN")
            accuracy = wins / max(1, len(valid))
            multiplier = 1.0
            if accuracy > 0.7:
                multiplier = 1.05
                logging.info(f"[{symbol}] Adaptive bias applied (+5%) based on recent accuracy")
            elif accuracy < 0.4:
                multiplier = 0.95
                logging.info(f"[{symbol}] Adaptive bias applied (-5%) based on recent accuracy")
            if multiplier != 1.0:
                prob_up = 0.5 + (prob_up - 0.5) * multiplier
                prob_up = min(max(prob_up, 0.0), 1.0)

        prob_down = 1.0 - prob_up
        result = {
            "prob_up": prob_up,
            "prob_down": prob_down,
            "latency_ms": latency_ms,
            "model_version": self.version,
            "status": "ok",
            "timed_out": False,
        }
        self._cache_put(key, result.copy())
        logging.info(f"[{symbol}] AI backend={backend_used} latency={latency_ms}ms prob_up={prob_up:.2f}")
        return result

    def ml_healthcheck(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready": self.ready,
            "model_version": self.version,
            "backend": self.model_backend,
        }


def ml_healthcheck(predictor: Optional[MLPredictor]) -> Dict[str, Any]:
    if predictor is None:
        return {"enabled": False, "ready": False, "model_version": None, "backend": None}
    return predictor.ml_healthcheck()


def serialize_features(candles: List[Dict[str, Any]], ctx: Dict[str, Any], trades: Iterable[Dict[str, Any]]) -> str:
    """Utility helper for caching or debugging."""
    vector = build_features(candles, ctx, trades)
    return json.dumps(vector.tolist())
