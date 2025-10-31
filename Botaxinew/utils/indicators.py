from typing import Dict, Union

import numpy as np
import pandas as pd

IndicatorResult = Dict[str, Union[float, str]]


def calc_rsi(df: pd.DataFrame, period: int = 14) -> IndicatorResult:
    close = df["close"].astype(float)
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    latest_rsi = rsi.iloc[-1] if not rsi.empty else np.nan

    if latest_rsi >= 70:
        signal = "PUT"
    elif latest_rsi <= 30:
        signal = "CALL"
    else:
        signal = "NONE"

    return {"value": float(latest_rsi), "signal": signal}


def calc_ema(df: pd.DataFrame, short_period: int = 12, long_period: int = 26) -> IndicatorResult:
    close = df["close"].astype(float)
    ema_short = close.ewm(span=short_period, adjust=False).mean()
    ema_long = close.ewm(span=long_period, adjust=False).mean()
    trend = "side"
    signal = "NONE"
    if ema_short.iloc[-1] > ema_long.iloc[-1]:
        trend = "up"
        signal = "CALL"
    elif ema_short.iloc[-1] < ema_long.iloc[-1]:
        trend = "down"
        signal = "PUT"
    return {"trend": trend, "signal": signal, "short": float(ema_short.iloc[-1]), "long": float(ema_long.iloc[-1])}


def calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_period: int = 9) -> IndicatorResult:
    close = df["close"].astype(float)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    latest_hist = histogram.iloc[-1]
    signal = "CALL" if latest_hist > 0 else "PUT" if latest_hist < 0 else "NONE"
    return {"value": float(macd_line.iloc[-1]), "signal": signal, "histogram": float(latest_hist)}


def calc_bollinger(df: pd.DataFrame, period: int = 20, deviation: float = 2.0) -> IndicatorResult:
    close = df["close"].astype(float)
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper_band = sma + deviation * std
    lower_band = sma - deviation * std
    price = close.iloc[-1]
    position = "middle"
    if price >= upper_band.iloc[-1]:
        position = "upper"
    elif price <= lower_band.iloc[-1]:
        position = "lower"
    return {
        "upper": float(upper_band.iloc[-1]),
        "lower": float(lower_band.iloc[-1]),
        "middle": float(sma.iloc[-1]),
        "position": position,
    }


def detect_pullback(df: pd.DataFrame, lookback: int = 10, threshold: float = 0.002) -> bool:
    close = df["close"].astype(float)
    recent = close.iloc[-lookback:]
    max_price = recent.max()
    min_price = recent.min()
    current_price = recent.iloc[-1]
    pullback_range = max_price - min_price
    if pullback_range == 0:
        return False
    retracement = (max_price - current_price) / pullback_range
    return retracement > threshold


def calc_confidence(rsi: IndicatorResult, ema: IndicatorResult, macd: IndicatorResult, pullback: bool, thresholds: Dict[str, float]) -> IndicatorResult:
    base_confidence = thresholds.get("base_confidence", 0.6)
    range_confidence = thresholds.get("range_confidence", 0.7)
    strong_trend = thresholds.get("strong_trend", 0.65)

    score = base_confidence
    votes = []

    if rsi["signal"] != "NONE":
        score += 0.1
        votes.append(rsi["signal"])

    if ema["signal"] != "NONE":
        score += 0.1
        votes.append(ema["signal"])

    if macd["signal"] != "NONE":
        score += 0.1
        votes.append(macd["signal"])

    if pullback:
        score += 0.05

    direction = "NONE"
    if votes:
        direction = max(set(votes), key=votes.count)
        if score >= strong_trend:
            score += 0.05

    score = min(score, 1.0)

    if score < range_confidence:
        direction = "NONE"

    return {"value": float(score), "direction": direction}
