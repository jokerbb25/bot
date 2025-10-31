from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def calc_rsi(df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
    close = df["close"].astype(float)
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    value = float(rsi.iloc[-1]) if not rsi.empty else float("nan")
    if value >= 70:
        signal = "PUT"
    elif value <= 30:
        signal = "CALL"
    else:
        signal = "NONE"
    return {"value": value, "signal": signal}


def calc_ema(df: pd.DataFrame, short_period: int = 12, long_period: int = 26) -> Dict[str, Any]:
    close = df["close"].astype(float)
    ema_short = close.ewm(span=short_period, adjust=False).mean()
    ema_long = close.ewm(span=long_period, adjust=False).mean()
    short_value = float(ema_short.iloc[-1])
    long_value = float(ema_long.iloc[-1])
    if short_value > long_value:
        trend = "up"
        signal = "CALL"
    elif short_value < long_value:
        trend = "down"
        signal = "PUT"
    else:
        trend = "flat"
        signal = "NONE"
    return {
        "short": short_value,
        "long": long_value,
        "trend": trend,
        "signal": signal,
    }


def calc_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Dict[str, Any]:
    close = df["close"].astype(float)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    hist_value = float(histogram.iloc[-1])
    if hist_value > 0:
        signal = "CALL"
    elif hist_value < 0:
        signal = "PUT"
    else:
        signal = "NONE"
    return {
        "macd": float(macd_line.iloc[-1]),
        "signal_line": float(signal_line.iloc[-1]),
        "histogram": hist_value,
        "signal": signal,
    }


def calc_bollinger(
    df: pd.DataFrame,
    period: int = 20,
    deviation: float = 2.0,
) -> Dict[str, Any]:
    close = df["close"].astype(float)
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper_band = sma + deviation * std
    lower_band = sma - deviation * std
    price = float(close.iloc[-1])
    if price >= float(upper_band.iloc[-1]):
        position = "upper"
    elif price <= float(lower_band.iloc[-1]):
        position = "lower"
    else:
        position = "middle"
    return {
        "upper": float(upper_band.iloc[-1]),
        "lower": float(lower_band.iloc[-1]),
        "middle": float(sma.iloc[-1]),
        "position": position,
        "price": price,
    }


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr_components = pd.DataFrame(
        {
            "high_low": high - low,
            "high_close": (high - prev_close).abs(),
            "low_close": (low - prev_close).abs(),
        }
    )
    tr = tr_components.max(axis=1)
    atr = tr.rolling(window=period).mean()
    return float(atr.iloc[-1])


def detect_pullback(df: pd.DataFrame, bollinger: Dict[str, Any], lookback: int = 5) -> bool:
    close = df["close"].astype(float)
    recent = close.iloc[-lookback:]
    price = float(recent.iloc[-1])
    max_price = float(recent.max())
    min_price = float(recent.min())
    range_span = max_price - min_price
    if range_span == 0:
        return False
    retracement = (max_price - price) / range_span if bollinger["position"] == "upper" else (price - min_price) / range_span
    return retracement > 0.5


def evaluate_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    rsi = calc_rsi(df)
    ema = calc_ema(df)
    macd = calc_macd(df)
    bollinger = calc_bollinger(df)
    atr = calc_atr(df)
    pullback = detect_pullback(df, bollinger)

    votes = []
    if rsi["signal"] in ("CALL", "PUT"):
        votes.append(rsi["signal"])
    if ema["signal"] in ("CALL", "PUT"):
        votes.append(ema["signal"])
    if macd["signal"] in ("CALL", "PUT"):
        votes.append(macd["signal"])

    if votes:
        direction = max(set(votes), key=votes.count)
    else:
        direction = "NONE"

    score = 0.5
    if rsi["signal"] in ("CALL", "PUT"):
        score += 0.15
    if ema["signal"] in ("CALL", "PUT"):
        score += 0.15
    if macd["signal"] in ("CALL", "PUT"):
        score += 0.15
    if pullback:
        score += 0.1
    if bollinger["position"] in ("upper", "lower"):
        score += 0.1

    score = max(0.0, min(score, 1.0))

    if direction == "NONE" and score < 0.65:
        score = min(score, 0.6)

    return {
        "rsi_signal": rsi["signal"],
        "ema_trend": ema["trend"],
        "macd_signal": macd["signal"],
        "pullback": pullback,
        "confidence": float(score),
        "direction": direction,
        "bollinger_position": bollinger["position"],
        "atr": atr,
        "rsi_value": rsi["value"],
    }
