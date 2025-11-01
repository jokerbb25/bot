from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def calc_rsi(df: pd.DataFrame, period: int = 14) -> Dict[str, float]:
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


def calc_ema(df: pd.DataFrame, short_period: int = 12, long_period: int = 26) -> Dict[str, float]:
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
) -> Dict[str, float]:
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
) -> Dict[str, float]:
    close = df["close"].astype(float)
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper_band = sma + deviation * std
    lower_band = sma - deviation * std
    price = float(close.iloc[-1])
    upper_value = float(upper_band.iloc[-1]) if not np.isnan(upper_band.iloc[-1]) else price
    lower_value = float(lower_band.iloc[-1]) if not np.isnan(lower_band.iloc[-1]) else price
    middle_value = float(sma.iloc[-1]) if not np.isnan(sma.iloc[-1]) else price
    if price >= upper_value:
        position = "upper"
    elif price <= lower_value:
        position = "lower"
    else:
        position = "middle"
    return {
        "upper": upper_value,
        "lower": lower_value,
        "middle": middle_value,
        "position": position,
        "price": price,
    }


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr_high_low = high - low
    tr_high_close = (high - prev_close).abs()
    tr_low_close = (low - prev_close).abs()
    tr = pd.concat([tr_high_low, tr_high_close, tr_low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return float(atr.iloc[-1]) if not atr.empty else 0.0


def detect_pullback(df: pd.DataFrame, bollinger: Dict[str, float], lookback: int = 5) -> bool:
    close = df["close"].astype(float)
    recent = close.iloc[-lookback:]
    if recent.empty:
        return False
    price = float(recent.iloc[-1])
    max_price = float(recent.max())
    min_price = float(recent.min())
    range_span = max_price - min_price
    if range_span <= 0:
        return False
    if bollinger["position"] == "upper":
        retracement = (max_price - price) / range_span
    elif bollinger["position"] == "lower":
        retracement = (price - min_price) / range_span
    else:
        return False
    return bool(retracement > 0.5)


def evaluate_indicators(
    df: pd.DataFrame,
    strategies: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    flags = {
        "rsi": True,
        "ema": True,
        "macd": True,
        "pullback": True,
    }
    if strategies:
        for key in flags:
            flags[key] = bool(strategies.get(key, True))

    rsi_data = calc_rsi(df)
    ema_data = calc_ema(df)
    macd_data = calc_macd(df)
    bollinger = calc_bollinger(df)
    atr_value = calc_atr(df)
    pullback_active = detect_pullback(df, bollinger) if flags["pullback"] else False

    votes = []
    confidence = 0.0
    if flags["rsi"] and rsi_data["signal"] in ("CALL", "PUT"):
        votes.append(rsi_data["signal"])
        confidence += 0.25
    if flags["ema"] and ema_data["signal"] in ("CALL", "PUT"):
        votes.append(ema_data["signal"])
        confidence += 0.25
    if flags["macd"] and macd_data["signal"] in ("CALL", "PUT"):
        votes.append(macd_data["signal"])
        confidence += 0.25
    if flags["pullback"] and pullback_active:
        confidence += 0.15
    if bollinger["position"] in ("upper", "lower"):
        confidence += 0.1

    if votes:
        direction = max(set(votes), key=votes.count)
    else:
        direction = "NONE"

    confidence = max(0.0, min(confidence, 1.0))

    return {
        "rsi_signal": rsi_data["signal"],
        "rsi_value": rsi_data["value"],
        "ema_trend": ema_data["trend"],
        "ema_short": ema_data["short"],
        "ema_long": ema_data["long"],
        "macd_signal": macd_data["signal"],
        "macd_hist": macd_data["histogram"],
        "pullback": bool(pullback_active),
        "bollinger_position": bollinger["position"],
        "atr": float(atr_value),
        "confidence": float(confidence),
        "direction": direction,
    }
