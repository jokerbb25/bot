import sys
import time
import json
import threading
import logging
import warnings
import csv
import math
import shutil
import os
import subprocess
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
import MetaTrader5 as mt5
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, ADXIndicator, MACD
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

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from aprendizaje import Aprendizaje
from telegram_bot import BOT_ACTIVE, telegram_listener

# ===============================================================
# CONFIG & CONSTANTS
# ===============================================================
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "USDCAD"]
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
MIN_CONFIDENCE = 0.0
MIN_VOLATILITY = 0.0
LOT_SIZE = 0.01  # configurable, equivalent to $1 per pip depending on broker leverage
MIN_CONFLUENCE = 2

WEIGHT_BOOST = {
    "XAUUSD": {"trend": 0.03, "range": -0.02},
    "USDJPY": {"trend": 0.01, "range": -0.01},
}

# === STRICT MODE PARAMETERS (Disable for Forex) ===
STRICT_MODE_ENABLED = False  # was True
CONFIDENCE_MIN = 0.80        # legacy (unused after patch)
LOW_VOL_THRESHOLD = 0.0006   # legacy (unused after patch)
LOW_VOL_CONFIDENCE = 0.85    # legacy (unused after patch)
NEUTRAL_RSI_BAND = (45.0, 55.0)
NEUTRAL_RSI_CONF = 0.95
MIN_ALIGNED_STRATEGIES = 3
POST_LOSS_COOLDOWN_SEC = 120
MAX_TRADES_PER_HOUR = 20
KEEP_WIN_MIN_CONF = 0.70
MAINTENANCE_EVERY = 50

# === TELEGRAM BOT CONFIGURATION ===
TELEGRAM_TOKEN = "8300367826:AAGzaMCJRY6pzZEqzjqgzAaRUXC_19KcB60"
TELEGRAM_CHAT_ID = "8364256476"

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

print("🧠 Axi module active — running strategies with MT5 market feed.")

logger = logging.getLogger(__name__)


def connect_axi(account_id: int, password: str, server: str) -> None:
    if not mt5.initialize(login=account_id, password=password, server=server):
        raise Exception(f"❌ Axi connection failed → {mt5.last_error()}")
    print("✅ Connected to Axi MT5")


def execute_market_order(symbol: str, action: str, atr_value: float, lot_size: float = LOT_SIZE):
    """
    Executes a market order on MT5 (instant order).
    SL/TP are dynamic based on ATR.
    """

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        raise RuntimeError(f"Symbol {symbol} is not available in MT5")

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick data for {symbol}")

    direction = str(action or "CALL").upper()
    order_type = mt5.ORDER_TYPE_BUY if direction == "CALL" else mt5.ORDER_TYPE_SELL
    price = tick.ask if direction == "CALL" else tick.bid
    point = float(getattr(symbol_info, "point", 0.0) or 0.0)
    lot = float(lot_size)
    atr = float(max(atr_value, 0.0))

    if point <= 0.0:
        raise RuntimeError(f"Invalid point size for {symbol}")

    atr_pips = atr / point if point else 0.0

    # ==========================================================
    # ✅ ATR VOLATILITY FILTER (PIPS BASED)
    # ==========================================================

    ATR_LIMITS = {
        "XAUUSD": (30, 350),
    }

    MIN_ATR, MAX_ATR = ATR_LIMITS.get(symbol.upper(), (2, 40))

    if not (MIN_ATR <= atr_pips <= MAX_ATR):
        logger.info(f"❌ Skip {symbol} - ATR {atr_pips:.1f} pips not in [{MIN_ATR}-{MAX_ATR}]")
        return None

    # ==========================================================
    # ✅ AUTO-SL/TP (NO MORE "Invalid Stops")
    # ==========================================================

    stop_level = int(getattr(symbol_info, "trade_stops_level", 0) or 0)

    sl_points = max(stop_level, int(max(1.0, atr_pips * 1.2)))
    tp_points = max(stop_level, int(max(1.0, atr_pips * 2.5)))

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if direction == "CALL" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": price - sl_points * point if direction == "CALL" else price + sl_points * point,
        "tp": price + tp_points * point if direction == "CALL" else price - tp_points * point,
        "magic": 1001,
        "comment": "axi-bot-auto",
    }

    result = mt5.order_send(request)

    if result is None:
        logger.error("❌ Order failed: No response from MT5")
        if BOT_ACTIVE:
            telegram_send(
                f"❌ ORDER FAILED\n{symbol}\nReason: No response from MT5"
            )
        return None

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"❌ Order failed: {getattr(result, 'comment', 'Unknown error')}")
        if BOT_ACTIVE:
            telegram_send(
                "❌ ORDER FAILED\n"
                f"{symbol}\nReason: {getattr(result, 'comment', 'Unknown error')}"
            )
    else:
        logger.info(f"✅ Order placed! #{getattr(result, 'order', 0)} {symbol} → {direction}")
        if BOT_ACTIVE:
            telegram_send(
                "✅ MARKET ORDER EXECUTED\n"
                f"{symbol}\nAction: {direction}\nLot: {lot}"
            )

    return result


def get_candles(symbol: str, timeframe: int = mt5.TIMEFRAME_M1, count: int = 100) -> List[Dict[str, Any]]:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        print(f"[WARN] No candles for {symbol}")
        return []
    return [
        {
            "time": rate["time"],
            "open": rate["open"],
            "high": rate["high"],
            "low": rate["low"],
            "close": rate["close"],
        }
        for rate in rates
    ]


def fetch_axi_candles(symbol: str, timeframe: int = mt5.TIMEFRAME_M1, count: int = 200) -> List[Dict[str, Any]]:
    return get_candles(symbol, timeframe=timeframe, count=count)



operation_active = False
TRADE_DURATION_SECONDS = 60
RESULT_POLL_INTERVAL = 5
RESUME_MESSAGE = "🔁 Reanudando análisis del mercado..."

CSV_LOGGED_CONTRACTS: Set[int] = set()
BIAS_MEMORY_PATH = Path("bias_memory.json")
WINNER_MEMORY_PATH = Path("sesgos_ganadores.json")
LEARNING_MEMORY_PATH = Path("learning_memory.json")
LEARNING_MEMORY_BACKUP_PATH = Path("learning_memory_backup.json")
STATS_DATA_PATH = Path("stats.json")


def send_telegram_message(text: str) -> None:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        requests.post(url, json=payload, timeout=5)
    except Exception as exc:
        logging.warning(f"⚠️ Telegram message failed: {exc}")


def telegram_send(text: str) -> None:
    send_telegram_message(text)



def safe_restart_windows() -> None:
    """Safely restart the bot on Windows, even if Python path contains spaces."""
    try:
        python_path = sys.executable
        script_path = os.path.abspath(sys.argv[0])
        logging.info(f"♻️ Restarting bot using: {python_path} {script_path}")
        subprocess.Popen([python_path, script_path], close_fds=True)
        logging.info("✅ Restart command executed successfully. Exiting current process.")
        os._exit(0)
    except Exception as exc:
        logging.error(f"❌ Failed to restart bot: {exc}")


def load_biases() -> Dict[str, Dict[str, Any]]:
    if not BIAS_MEMORY_PATH.exists():
        return {}
    try:
        with BIAS_MEMORY_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        normalized: Dict[str, Dict[str, Any]] = {}
        for symbol, state in data.items():
            symbol_key = str(symbol)
            snapshot = dict(state)
            snapshot.setdefault("RSI", 0.0)
            snapshot.setdefault("EMA", 0.0)
            snapshot.setdefault("last_result", "NONE")
            snapshot.setdefault("confidence", 0.0)
            try:
                snapshot["confidence"] = float(snapshot.get("confidence", 0.0))
            except (TypeError, ValueError):
                snapshot["confidence"] = 0.0
            normalized[symbol_key] = snapshot
        return normalized
    except Exception as exc:  # pragma: no cover
        logging.debug(f"No se pudo cargar sesgos almacenados: {exc}")
        return {}


def save_biases(biases: Dict[str, Dict[str, Any]]) -> None:
    payload: Dict[str, Dict[str, Any]] = {}
    for symbol, state in biases.items():
        payload[symbol] = {
            "RSI": float(state.get("RSI", 0.0)),
            "EMA": float(state.get("EMA", 0.0)),
            "last_result": state.get("last_result", "NONE"),
            "confidence": float(state.get("confidence", 0.0)),
        }
    try:
        with BIAS_MEMORY_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except Exception as exc:  # pragma: no cover
        logging.debug(f"No se pudo guardar sesgos: {exc}")


def load_winner_biases() -> List[Dict[str, Any]]:
    if not WINNER_MEMORY_PATH.exists():
        return []
    entries: List[Dict[str, Any]] = []
    try:
        with WINNER_MEMORY_PATH.open("r", encoding="utf-8") as handle:
            raw_data = json.load(handle)
        candidates: Iterable[Any]
        if isinstance(raw_data, list):
            candidates = raw_data
        elif isinstance(raw_data, dict):
            candidates = raw_data.values()
        else:
            candidates = []
        for item in candidates:
            record = dict(item) if isinstance(item, dict) else {}
            symbol = str(record.get("symbol", ""))
            direction = str(record.get("direction", "NONE")).upper()
            confidence_value = float(record.get("confidence", 0.0) or 0.0)
            timestamp_value = str(record.get("timestamp", datetime.now(timezone.utc).isoformat()))
            weight_value = float(record.get("weight", 1.0) or 0.0)
            entries.append(
                {
                    "symbol": symbol,
                    "direction": direction,
                    "confidence": confidence_value,
                    "timestamp": timestamp_value,
                    "rsi": float(record.get("rsi", 0.0) or 0.0),
                    "ema": float(record.get("ema", 0.0) or 0.0),
                    "volatility": float(record.get("volatility", 0.0) or 0.0),
                    "regime": record.get("regime", "DESCONOCIDO"),
                    "weight": float(max(weight_value, 0.0)),
                }
            )
    except Exception as exc:  # pragma: no cover
        logging.debug(f"No se pudo cargar sesgos ganadores: {exc}")
        return []
    entries.sort(key=lambda entry: entry.get("weight", 0.0), reverse=True)
    return entries


def save_winner_biases(entries: List[Dict[str, Any]]) -> None:
    serializable: List[Dict[str, Any]] = []
    for entry in entries:
        serializable.append(
            {
                "symbol": entry.get("symbol", ""),
                "direction": entry.get("direction", "NONE"),
                "confidence": float(entry.get("confidence", 0.0) or 0.0),
                "timestamp": entry.get(
                    "timestamp", datetime.now(timezone.utc).isoformat()
                ),
                "rsi": float(entry.get("rsi", 0.0) or 0.0),
                "ema": float(entry.get("ema", 0.0) or 0.0),
                "volatility": float(entry.get("volatility", 0.0) or 0.0),
                "regime": entry.get("regime", "DESCONOCIDO"),
                "weight": float(entry.get("weight", 0.0) or 0.0),
            }
        )
    try:
        with WINNER_MEMORY_PATH.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, ensure_ascii=False, indent=2)
    except Exception as exc:  # pragma: no cover
        logging.debug(f"No se pudo guardar sesgos ganadores: {exc}")


def build_learning_pattern_key(symbol: str, action: str, rsi_value: float, ema_value: float) -> str:
    return (
        f"{symbol.upper()}|{action.upper()}|RSI:{round(float(rsi_value), 1)}|EMA:{round(float(ema_value), 1)}"
    )


def is_similar_pattern(
    current_signals: Dict[str, str],
    stored_signals: Optional[Dict[str, Any]],
) -> Tuple[int, int]:
    if not isinstance(stored_signals, dict):
        stored_signals = {}
    matches = 0
    considered = 0
    for name, current_value in current_signals.items():
        current_signal = str(current_value or "NONE").upper()
        if current_signal == "NONE":
            continue
        stored_signal = str(stored_signals.get(name, "NONE") or "NONE").upper()
        considered += 1
        if stored_signal == current_signal:
            matches += 1
    return matches, considered


def _normalize_learning_entry(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    try:
        symbol = str(item.get("symbol", ""))
        action_value = str(item.get("action", item.get("direction", "NONE"))).upper()
        rsi_value = round(float(item.get("rsi", 0.0)), 1)
        ema_value = round(float(item.get("ema", 0.0)), 1)
        result_value = str(item.get("result", "LOSS")).upper()
        confidence_value = float(item.get("confidence", 0.0))
        timestamp_raw = item.get("timestamp")
        if isinstance(timestamp_raw, str):
            try:
                timestamp_value = datetime.fromisoformat(timestamp_raw).timestamp()
            except ValueError:
                timestamp_value = time.time()
        else:
            timestamp_value = float(timestamp_raw or time.time())
        weight_value = float(item.get("weight", 1.0))
        wins_row = int(item.get("wins_in_a_row", 0) or 0)
        losses_row = int(item.get("loss_streak", 0) or 0)
        signals_raw = item.get("signals", {})
        normalized_signals: Dict[str, str] = {}
        if isinstance(signals_raw, dict):
            for key, value in signals_raw.items():
                normalized_signals[str(key)] = str(value or "NONE").upper()
    except (TypeError, ValueError):
        return None
    pattern_key = build_learning_pattern_key(symbol, action_value, rsi_value, ema_value)
    return {
        "symbol": symbol,
        "action": action_value,
        "direction": action_value,
        "rsi": rsi_value,
        "ema": ema_value,
        "result": result_value,
        "confidence": confidence_value,
        "timestamp": float(timestamp_value),
        "weight": max(1.0, weight_value),
        "wins_in_a_row": max(0, wins_row),
        "loss_streak": max(0, losses_row),
        "pattern_key": pattern_key,
        "signals": normalized_signals,
    }


def prune_learning_memory(
    entries: Dict[str, Dict[str, Any]], max_patterns: int = 200
) -> Dict[str, Dict[str, Any]]:
    normalized_entries: Dict[str, Dict[str, Any]] = {}
    for key, entry in entries.items():
        normalized = _normalize_learning_entry(entry)
        if normalized is None:
            continue
        pattern_key = normalized.get("pattern_key")
        if not pattern_key:
            continue
        normalized_entries[pattern_key] = normalized
    if len(normalized_entries) <= max_patterns:
        return normalized_entries
    winners = [
        item
        for item in normalized_entries.values()
        if item.get("result") == "WIN"
    ]
    losses = [
        item
        for item in normalized_entries.values()
        if item.get("result") != "WIN"
    ]
    winners.sort(
        key=lambda entry: (
            float(entry.get("weight", 1.0)),
            int(entry.get("wins_in_a_row", 0)),
        ),
        reverse=True,
    )
    losses.sort(
        key=lambda entry: (
            float(entry.get("weight", 1.0)),
            -int(entry.get("loss_streak", 0)),
        ),
        reverse=True,
    )
    max_winners = min(len(winners), min(max_patterns, 150))
    remaining_slots = max(0, max_patterns - max_winners)
    max_losses = min(len(losses), remaining_slots)
    pruned_list = winners[:max_winners] + losses[:max_losses]
    logging.info(
        "🧹 Pruned learning memory to %d patrones (favoring winners)",
        len(pruned_list),
    )
    pruned_entries: Dict[str, Dict[str, Any]] = {}
    for entry in pruned_list:
        pattern_key = entry.get("pattern_key")
        if not pattern_key:
            continue
        pruned_entries[pattern_key] = entry
    return pruned_entries


def load_learning_memory() -> Dict[str, Dict[str, Any]]:
    try:
        if LEARNING_MEMORY_PATH.exists():
            with LEARNING_MEMORY_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                candidate_map: Dict[str, Dict[str, Any]] = {}
                for item in data:
                    normalized_entry = _normalize_learning_entry(item)
                    if normalized_entry is None:
                        continue
                    key = normalized_entry.get("pattern_key")
                    if key:
                        candidate_map[key] = normalized_entry
                data = candidate_map
            if isinstance(data, dict):
                normalized = prune_learning_memory(data, max_patterns=200)
                total = len(normalized)
                win_count = sum(
                    1 for item in normalized.values() if item.get("result") == "WIN"
                )
                loss_count = sum(
                    1 for item in normalized.values() if item.get("result") == "LOSS"
                )
                logging.info(
                    "🧠 Learning memory loaded (%d WIN / %d LOSS, total=%d)",
                    win_count,
                    loss_count,
                    total,
                )
                try:
                    shutil.copy(str(LEARNING_MEMORY_PATH), str(LEARNING_MEMORY_BACKUP_PATH))
                except Exception as exc:
                    logging.debug(f"Could not backup learning memory: {exc}")
                return normalized
            logging.warning("⚠️ Memory file is not a dictionary, attempting backup restore...")
        else:
            logging.info("🧠 No previous learning memory found, creating a new one.")
        if LEARNING_MEMORY_BACKUP_PATH.exists():
            with LEARNING_MEMORY_BACKUP_PATH.open("r", encoding="utf-8") as handle:
                backup_data = json.load(handle)
            if isinstance(backup_data, list):
                backup_map: Dict[str, Dict[str, Any]] = {}
                for item in backup_data:
                    normalized_entry = _normalize_learning_entry(item)
                    if normalized_entry is None:
                        continue
                    key = normalized_entry.get("pattern_key")
                    if key:
                        backup_map[key] = normalized_entry
                backup_data = backup_map
            if isinstance(backup_data, dict):
                restored = prune_learning_memory(backup_data, max_patterns=200)
                total = len(restored)
                win_count = sum(
                    1 for item in restored.values() if item.get("result") == "WIN"
                )
                loss_count = sum(
                    1 for item in restored.values() if item.get("result") == "LOSS"
                )
                try:
                    shutil.copy(str(LEARNING_MEMORY_BACKUP_PATH), str(LEARNING_MEMORY_PATH))
                except Exception as exc:
                    logging.debug(f"Could not restore learning memory backup: {exc}")
                logging.info(
                    "♻️ Restored learning memory from backup (%d WIN / %d LOSS, total=%d)",
                    win_count,
                    loss_count,
                    total,
                )
                return restored
    except Exception as exc:
        logging.error(f"❌ Error loading learning memory: {exc}")
    return {}


def save_learning_memory(mem: Dict[str, Dict[str, Any]]) -> None:
    try:
        pruned = prune_learning_memory(mem, max_patterns=200)
        if not pruned:
            logging.warning("⚠️ Attempted to save empty memory, skipping save.")
            return
        with LEARNING_MEMORY_PATH.open("w", encoding="utf-8") as handle:
            json.dump(pruned, handle, ensure_ascii=False, indent=2)
        try:
            shutil.copy(str(LEARNING_MEMORY_PATH), str(LEARNING_MEMORY_BACKUP_PATH))
        except Exception as exc:
            logging.debug(f"Could not backup learning memory: {exc}")
        win_count = sum(1 for item in pruned.values() if item.get("result") == "WIN")
        loss_count = sum(1 for item in pruned.values() if item.get("result") == "LOSS")
        logging.info(
            "💾 Learning memory saved (%d WIN / %d LOSS, total=%d)",
            win_count,
            loss_count,
            len(pruned),
        )
    except Exception as exc:
        logging.error(f"❌ Error saving learning memory: {exc}")


def _load_stats_data() -> Dict[str, int]:
    if not STATS_DATA_PATH.exists():
        return {"total_ops": 0, "wins": 0, "losses": 0}
    try:
        with STATS_DATA_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {
            "total_ops": int(data.get("total_ops", 0)),
            "wins": int(data.get("wins", 0)),
            "losses": int(data.get("losses", data.get("loss", 0))),
        }
    except Exception as exc:
        logging.debug(f"No se pudieron cargar estadísticas persistentes: {exc}")
        return {"total_ops": 0, "wins": 0, "losses": 0}


def _save_stats_data(stats: Dict[str, int]) -> None:
    try:
        with STATS_DATA_PATH.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=2)
    except Exception as exc:
        logging.warning(f"⚠️ Could not persist stats: {exc}")


def safe_startup_accuracy(session_wins: int, session_losses: int) -> Tuple[str, int]:
    session_total = session_wins + session_losses
    if session_total > 0:
        accuracy = (session_wins / session_total) * 100.0
        return f"{accuracy:.2f}%", session_total
    stats = _load_stats_data()
    total_ops = stats.get("total_ops", 0)
    if total_ops <= 0:
        return "— (sin operaciones previas)", 0
    wins = stats.get("wins", 0)
    accuracy = (wins / total_ops) * 100.0 if total_ops > 0 else 0.0
    return f"{accuracy:.2f}%", total_ops


def update_stats_persist(win: bool) -> None:
    stats = _load_stats_data()
    stats["total_ops"] = int(stats.get("total_ops", 0)) + 1
    stats["losses"] = int(stats.get("losses", stats.get("loss", 0)))
    if win:
        stats["wins"] = int(stats.get("wins", 0)) + 1
    else:
        stats["losses"] = int(stats.get("losses", 0)) + 1
        stats.pop("loss", None)
    _save_stats_data(stats)


global_engine = None


def telegram_bot() -> None:
    global BOT_ACTIVE, global_engine
    logging.info("🤖 Bot de Telegram activo y escuchando comandos...")

    def _send_text(text: str) -> None:
        send_telegram_message(text)

    offset: Optional[int] = None
    while True:
        try:
            params: Dict[str, Any] = {"timeout": 10}
            if offset is not None:
                params["offset"] = offset
            response = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
                params=params,
                timeout=15,
            )
            payload = response.json()
            if not payload.get("ok"):
                time.sleep(5)
                continue
            for update in payload.get("result", []):
                offset = update.get("update_id", 0) + 1
                message = (update.get("message") or {}).get("text", "")
                if not message:
                    continue
                command = message.lower()
                if any(keyword in command for keyword in ("pause", "stop", "pausar")):
                    BOT_ACTIVE = False
                    _send_text("⏸ Bot detenido manualmente.")
                elif any(keyword in command for keyword in ("resume", "start", "reanudar")):
                    BOT_ACTIVE = True
                    _send_text("▶️ Bot reanudado.")
                elif any(keyword in command for keyword in ("status", "estado")):
                    engine_ref = global_engine
                    if engine_ref is not None:
                        precision = engine_ref.get_accuracy()
                        operations = engine_ref.total_operations
                        _send_text(
                            f"📊 Precisión: {precision:.2f}%\nOperaciones: {int(operations)}"
                        )
                    else:
                        _send_text("ℹ️ Motor no disponible todavía.")
                elif any(keyword in command for keyword in ("help", "ayuda")):
                    _send_text("🧠 Comandos:\n- pausar\n- reanudar\n- estado\n- info")
                elif "info" in command:
                    engine_ref = global_engine
                    if engine_ref is not None:
                        _send_text(f"📄 Último ticket: {engine_ref.get_last_contract_info()}")
                    else:
                        _send_text("ℹ️ No hay información de tickets disponible.")
        except Exception as exc:
            logging.error(f"❌ Telegram bot error: {exc}")
            time.sleep(5)
        time.sleep(3)


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


def detect_regime(adx_value: float, bb_width: float) -> str:
    adx_val = float(adx_value)
    bb_val = float(bb_width)
    if adx_val >= 25.0 or bb_val >= 0.015:
        return "TREND"
    return "RANGE"


def min_confidence_for(regime: str) -> float:
    normalized_regime = (regime or "RANGE").upper()
    return 0.65 if normalized_regime == "TREND" else 0.75


def compute_atr_for_symbol(symbol: str, period: int = 14, count: int = CANDLE_COUNT) -> float:
    try:
        candles = fetch_axi_candles(symbol, timeframe=mt5.TIMEFRAME_M1, count=max(count, period + 2))
        if not candles:
            return 0.0
        candle_objects = [
            Candle(
                epoch=int(entry["time"]),
                open=float(entry["open"]),
                high=float(entry["high"]),
                low=float(entry["low"]),
                close=float(entry["close"]),
            )
            for entry in candles
        ]
        df = to_dataframe(candle_objects)
        atr_series = atr(df, period)
        if atr_series.empty:
            return 0.0
        value = float(atr_series.iloc[-1])
        return 0.0 if math.isnan(value) else value
    except Exception as exc:
        logger.debug(f"Failed to compute ATR for {symbol}: {exc}")
        return 0.0


def donchian_channels(df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series]:
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    return lower, upper


def log_trade(record: TradeRecord) -> None:
    contract_id: Optional[int] = None
    if record.metadata:
        raw_id = record.metadata.get('contract_id')
        try:
            contract_id = int(raw_id) if raw_id is not None else None
        except (TypeError, ValueError):
            contract_id = None
    logged_contracts = CSV_LOGGED_CONTRACTS
    if contract_id is not None:
        if contract_id in logged_contracts:
            logging.debug(f"Duplicate ticket {contract_id} ignored.")
            return
        logged_contracts.add(contract_id)
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
        strategy_snapshot = record.metadata.get('strategy_details', {}) if record.metadata else {}
        confidence_snapshot = record.metadata.get('strategy_confidences', {}) if record.metadata else {}
        row["strategies"] = json.dumps(strategy_snapshot)
        row["strategy_confidences"] = json.dumps(confidence_snapshot)
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
        self.biases: Dict[str, Dict[str, Any]] = {
            symbol: {"RSI": 0.0, "EMA": 0.0, "last_result": "NONE"}
            for symbol in SYMBOLS
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
        self.weights_recalibrated = False
        self.strategy_accuracy_snapshot: Dict[str, float] = {}
        self._last_recalibration_total = 0
        self.min_confidence = MIN_TRADE_CONFIDENCE
        self.rsi_high_threshold = 70.0
        self.adx_min_threshold = 20.0
        self.predictive_model_path = Path("predictive_model.pkl")
        self.model_initialized = False
        self.reinforce_batches = 0
        self.optimize_batches = 0
        self.learning_event = threading.Event()
        self.learning_thread: Optional[threading.Thread] = None
        self._pending_context: Dict[str, Dict[str, float]] = {}
        self.symbol_profiles: Dict[str, Dict[str, Any]] = {
            symbol: self._default_symbol_profile() for symbol in SYMBOLS
        }
        for profile in self.symbol_profiles.values():
            profile["learning_rate"] = self.learning_rate
        self._current_symbol: Optional[str] = None
        self._ensure_csv()
        self._load_bias_storage()
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

    def _load_bias_storage(self) -> None:
        stored = load_biases()
        if not stored:
            save_biases(self.biases)
            return
        with self.bias_lock:
            for symbol, state in self.biases.items():
                if symbol in stored:
                    data = stored[symbol]
                    state["RSI"] = float(data.get("RSI", state.get("RSI", 0.0)))
                    state["EMA"] = float(data.get("EMA", state.get("EMA", 0.0)))
                    state["last_result"] = data.get("last_result", "WIN")
                    state["confidence"] = float(data.get("confidence", state.get("confidence", 0.0)))
                else:
                    state["RSI"] = 0.0
                    state["EMA"] = 0.0
                    state["last_result"] = "LOSS"
                    state["confidence"] = 0.0
        save_biases(self.biases)

    def _persist_biases(self) -> None:
        with self.bias_lock:
            save_biases(self.biases)

    def start_background_services(self) -> None:
        if self.learning_thread is None or not self.learning_thread.is_alive():
            self.learning_thread = threading.Thread(
                target=self._learning_loop,
                daemon=True,
            )
            self.learning_thread.start()
        if self._telegram_thread is None or not self._telegram_thread.is_alive():
            self._telegram_thread = threading.Thread(
                target=telegram_listener,
                args=(self,),
                daemon=True,
            )
            self._telegram_thread.start()

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
                bias_state = self.biases.setdefault(
                    asset,
                    {"RSI": 0.0, "EMA": 0.0, "last_result": "NONE"},
                )
                rsi_bias = float(bias_state.get("RSI", 0.0))
                ema_bias = float(bias_state.get("EMA", 0.0))
                bias_state["last_result"] = resultado
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
            self._persist_biases()
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
        total_trades = self.global_totals['total']
        if total_trades and total_trades % 150 == 0:
            accuracy = self.global_totals['wins'] / max(1, total_trades)
            adjustment = float(np.clip((accuracy - 0.55) * 0.1, -0.03, 0.03))
            with self.weights_lock:
                for clave, valor in list(self.weights.items()):
                    self.weights[clave] = float(np.clip(valor * (1.0 + adjustment), 0.2, 3.0))
            self.weights_recalibrated = True
        if total_trades >= 100 and total_trades % 100 == 0 and total_trades != self._last_recalibration_total:
            self._adaptive_recalibration(total_trades)
        self.learning_event.set()

    def _pop_context(self, asset: str) -> Dict[str, float]:
        with self.lock:
            return self._pending_context.pop(asset, {}).copy()

    def _adaptive_recalibration(self, trade_total: int) -> None:
        path = Path(TRADES_LOG_PATH)
        if not path.exists():
            self._last_recalibration_total = trade_total
            return
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover
            logging.debug(f"No se pudo leer el log de operaciones para recalibración: {exc}")
            self._last_recalibration_total = trade_total
            return
        if df.empty or 'strategies' not in df.columns:
            self._last_recalibration_total = trade_total
            return
        accuracy_data: Dict[str, Dict[str, int]] = {}
        for _, row in df.iterrows():
            strategies_payload = row.get('strategies')
            if not isinstance(strategies_payload, str) or not strategies_payload:
                continue
            try:
                strategies_dict = json.loads(strategies_payload)
            except Exception:
                continue
            decision = str(row.get('decision', '')).upper()
            result = str(row.get('result', '')).upper()
            if decision not in {'CALL', 'PUT'} or result not in {'WIN', 'LOSS'}:
                continue
            if decision == 'CALL':
                opposite = 'PUT'
            else:
                opposite = 'CALL'
            actual_direction = decision if result == 'WIN' else opposite
            for strat_name, strat_info in strategies_dict.items():
                signal = str(strat_info.get('signal', '')).upper()
                if signal not in {'CALL', 'PUT'}:
                    continue
                stats = accuracy_data.setdefault(strat_name, {'wins': 0, 'total': 0})
                stats['total'] += 1
                if signal == actual_direction:
                    stats['wins'] += 1
        if not accuracy_data:
            self._last_recalibration_total = trade_total
            return
        indicator_map = {
            'RSI': 'RSI',
            'EMA Trend': 'EMA',
            'Bollinger Rebound': 'BOLL',
            'Range Breakout': 'ADX',
            'Divergence': 'MACD',
        }
        updated = False
        new_indicator_weights: Dict[str, float] = {}
        with self.weights_lock:
            new_indicator_weights = dict(self.weights)
        for strat_name, stats in accuracy_data.items():
            total = stats['total']
            if total == 0:
                continue
            accuracy_ratio = stats['wins'] / total
            previous_accuracy = self.strategy_accuracy_snapshot.get(strat_name)
            self.strategy_accuracy_snapshot[strat_name] = accuracy_ratio
            if previous_accuracy is None:
                continue
            change = accuracy_ratio - previous_accuracy
            if abs(change) < 0.01:
                continue
            adjustment_ratio = min(0.02, abs(change) * 0.5)
            if adjustment_ratio <= 0.0:
                continue
            factor = 1.0 + adjustment_ratio if change > 0 else 1.0 - adjustment_ratio
            if strat_name in STRATEGY_WEIGHTS:
                with self.lock:
                    STRATEGY_WEIGHTS[strat_name] = float(
                        np.clip(STRATEGY_WEIGHTS[strat_name] * factor, 0.3, 3.0)
                    )
                updated = True
            key = indicator_map.get(strat_name)
            if key:
                current_weight = new_indicator_weights.get(key)
                if current_weight is not None:
                    new_indicator_weights[key] = float(
                        np.clip(current_weight * factor, 0.2, 3.0)
                    )
                    updated = True
        if updated:
            with self.weights_lock:
                self.weights.update(new_indicator_weights)
                self.weights_recalibrated = True
        with self.weights_lock:
            indicator_snapshot = {
                'RSI': float(self.weights.get('RSI', 1.0)),
                'EMA': float(self.weights.get('EMA', 1.0)),
                'BOLL': float(self.weights.get('BOLL', 1.0)),
                'ADX': float(self.weights.get('ADX', 0.8)),
                'MACD': float(self.weights.get('MACD', 0.8)),
            }
        logging.info(
            "📈 Adaptive weights updated: "
            + ", ".join(
                [
                    f"RSI={indicator_snapshot['RSI']:.2f}",
                    f"EMA={indicator_snapshot['EMA']:.2f}",
                    f"Bollinger={indicator_snapshot['BOLL']:.2f}",
                    f"ADX={indicator_snapshot['ADX']:.2f}",
                    f"MACD={indicator_snapshot['MACD']:.2f}",
                ]
            )
        )
        self._last_recalibration_total = trade_total

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
            bias_state = self.biases.setdefault(
                symbol,
                {"RSI": 0.0, "EMA": 0.0, "last_result": "NONE"},
            )
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
            bias_state["last_result"] = outcome
        logging.info(
            f"🧠 Bias adjusted [{symbol}] Source={source} RSI={rsi_bias:.2f} EMA={ema_bias:.2f}"
        )
        self.log_bias_update(symbol, outcome, direction, price_delta, source)
        self._persist_biases()
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
            bias_state = self.biases.setdefault(
                symbol,
                {"RSI": 0.0, "EMA": 0.0, "last_result": "NONE"},
            )
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
            self.model_initialized = True
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
        with self.memory_lock:
            memory_length = len(self.memory)
        if not self.model_initialized and memory_length >= 200:
            self.train_predictive_model()
        with self.model_lock:
            model = self.model
        if model is None and load is not None and self.predictive_model_path.exists():
            try:
                model = load(self.predictive_model_path)
                with self.model_lock:
                    self.model = model
                self.model_initialized = True
            except Exception:  # pragma: no cover
                model = None
        if model is None:
            self.last_prediction = 0.5
            return 0.5
        try:
            proba = float(model.predict_proba([[rsi, ema, macd, adx, hour]])[0][1])
        except Exception:
            proba = 0.5
        self.last_prediction = proba
        return proba

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
                bias_state = {"RSI": 0.0, "EMA": 0.0, "last_result": "NONE"}
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
                    self.biases[nombre] = {
                        "RSI": 0.0,
                        "EMA": 0.0,
                        "last_result": "NONE",
                    }
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
            "status": bool(self.learning_thread and self.learning_thread.is_alive()),
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
            self.biases = {
                symbol: {"RSI": 0.0, "EMA": 0.0, "last_result": "NONE"}
                for symbol in SYMBOLS
            }
        self._persist_biases()
        with self.model_lock:
            self.model = None
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
        if self.predictive_model_path.exists():
            try:
                self.predictive_model_path.unlink()
            except Exception:  # pragma: no cover
                pass
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
    variation = 0.0
    if len(df) > 10:
        serie_variacion = df['close'].pct_change().rolling(10).std()
        variation = float(np.nan_to_num(serie_variacion.iloc[-1], nan=0.0)) if len(serie_variacion) else 0.0
    stable_threshold = 0.0012
    high_threshold = 0.003
    margin = 0.0
    if variation <= stable_threshold:
        ratio = 1.0 - variation / max(stable_threshold, 1e-9)
        margin = ratio * 2.5
    elif variation >= high_threshold:
        ratio = min(1.0, (variation - high_threshold) / max(high_threshold, 1e-9))
        margin = -ratio * 2.0
    inferior_dinamico = float(np.clip(inferior + margin, 5.0, 45.0))
    superior_dinamico = float(np.clip(superior - margin, 55.0, 95.0))
    extra = {
        'rsi': valor,
        'strong_call': valor < max(20.0, inferior_dinamico - 5.0),
        'strong_put': valor > min(80.0, superior_dinamico + 5.0),
        'rsi_lower_dynamic': inferior_dinamico,
        'rsi_upper_dynamic': superior_dinamico,
        'price_variation': variation,
    }
    if valor < inferior_dinamico:
        return StrategyResult('CALL', 2.0, [f"RSI {valor:.2f} sobrevendido → señal CALL"], extra)
    if valor > superior_dinamico:
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
    diff_series = ema_corto - ema_largo
    recent_diff = diff_series.iloc[-5:] if len(diff_series) >= 5 else diff_series
    diff_std = float(np.nan_to_num(recent_diff.diff().abs().mean(), nan=0.0)) if len(recent_diff) > 1 else 0.0
    price_volatility = 0.0
    if len(df) > 10:
        window_vol = df['close'].pct_change().rolling(10).std()
        price_volatility = float(np.nan_to_num(window_vol.iloc[-1], nan=0.0)) if len(window_vol) else 0.0
    baseline_threshold = tolerancia / 1000.0
    slope_floor = max(0.0001, baseline_threshold, diff_std * 0.5, price_volatility * 0.3)
    threshold = baseline_threshold
    diff_prev = float(ema_corto.iloc[-2] - ema_largo.iloc[-2])
    diff_curr = float(ema_corto.iloc[-1] - ema_largo.iloc[-1])
    cruz_alcista = diff_prev <= threshold and diff_curr > threshold
    cruz_bajista = diff_prev >= -threshold and diff_curr < -threshold
    if cruz_alcista:
        return StrategyResult('CALL', 1.5, ['Cruce alcista EMA9 sobre EMA21 → CALL'], {'ema_short': float(ema_corto.iloc[-1]), 'ema_long': float(ema_largo.iloc[-1])})
    if cruz_bajista:
        return StrategyResult('PUT', -1.5, ['Cruce bajista EMA9 bajo EMA21 → PUT'], {'ema_short': float(ema_corto.iloc[-1]), 'ema_long': float(ema_largo.iloc[-1])})
    consistent_direction = False
    if len(recent_diff) > 0:
        signs = [np.sign(val) for val in recent_diff if abs(val) > 1e-9]
        if signs:
            consistent_direction = all(sign == signs[0] for sign in signs)
    minor_threshold = slope_floor * 0.8 if slope_floor else 0.0001
    slope_drift = float(recent_diff.iloc[-1] - recent_diff.iloc[0]) if len(recent_diff) > 1 else diff_curr
    if consistent_direction and abs(diff_curr) >= minor_threshold and abs(slope_drift) <= abs(diff_curr) * 1.5:
        direccion = 'CALL' if diff_curr > 0 else 'PUT'
        puntaje = 1.1 if direccion == 'CALL' else -1.1
        tendencia = 'alcista' if direccion == 'CALL' else 'bajista'
        extra = {
            'ema_short': float(ema_corto.iloc[-1]),
            'ema_long': float(ema_largo.iloc[-1]),
            'diff_mean': float(recent_diff.mean()) if len(recent_diff) else diff_curr,
            'slope_floor': slope_floor,
        }
        return StrategyResult(direccion, puntaje, [f"Pendiente EMA consistente {tendencia} → {direccion}"], extra)
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
    ema_short_series = ema(closes, 9)
    ema_long_series = ema(closes, 21)
    ema_support_diff = float(ema_short_series.iloc[-1] - ema_long_series.iloc[-1]) if len(ema_short_series) and len(ema_long_series) else 0.0
    extra = {
        'rsi_trend': float(rsi_tramo.iloc[-1] - rsi_tramo.iloc[-2]) if len(rsi_tramo) > 1 else 0.0,
        'ema_support': ema_support_diff,
    }
    if tramo.iloc[0] > tramo.iloc[1] > tramo.iloc[2] and tramo.iloc[3] >= tramo.iloc[2] and tramo.iloc[4] > tramo.iloc[2] and rsi_tramo.iloc[-1] > rsi_tramo.iloc[-2]:
        return StrategyResult('CALL', 1.0, ['Pullback alcista con RSI recuperándose → CALL'], extra)
    if tramo.iloc[0] < tramo.iloc[1] < tramo.iloc[2] and tramo.iloc[3] <= tramo.iloc[2] and tramo.iloc[4] < tramo.iloc[2] and rsi_tramo.iloc[-1] < rsi_tramo.iloc[-2]:
        return StrategyResult('PUT', -1.0, ['Pullback bajista con RSI cayendo → PUT'], extra)
    soporte_rsi_alcista = len(rsi_tramo) > 1 and rsi_tramo.iloc[-1] > rsi_tramo.iloc[-2]
    soporte_rsi_bajista = len(rsi_tramo) > 1 and rsi_tramo.iloc[-1] < rsi_tramo.iloc[-2]
    soporte_ema_alcista = ema_support_diff > 0
    soporte_ema_bajista = ema_support_diff < 0
    if len(tramo) >= 3:
        recuperacion_rapida = tramo.iloc[-3] >= tramo.iloc[-2] and tramo.iloc[-1] > tramo.iloc[-2]
        caida_rapida = tramo.iloc[-3] <= tramo.iloc[-2] and tramo.iloc[-1] < tramo.iloc[-2]
        if recuperacion_rapida and (soporte_rsi_alcista or soporte_ema_alcista):
            respaldos = []
            if soporte_rsi_alcista:
                respaldos.append('RSI')
            if soporte_ema_alcista:
                respaldos.append('EMA')
            texto_respaldo = ' y '.join(respaldos)
            return StrategyResult('CALL', 0.9, [f"Pullback rápido confirmado ({texto_respaldo}) → CALL"], extra)
        if caida_rapida and (soporte_rsi_bajista or soporte_ema_bajista):
            respaldos = []
            if soporte_rsi_bajista:
                respaldos.append('RSI')
            if soporte_ema_bajista:
                respaldos.append('EMA')
            texto_respaldo = ' y '.join(respaldos)
            return StrategyResult('PUT', -0.9, [f"Pullback rápido confirmado ({texto_respaldo}) → PUT"], extra)
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


def strategy_adx(df: pd.DataFrame) -> StrategyResult:
    if len(df) < 20:
        return StrategyResult('NONE', 0.0, ['ADX sin datos suficientes'], {'adx': None})
    try:
        adx_indicator = ADXIndicator(df['high'], df['low'], df['close'], window=14, fillna=False)
        adx_series = adx_indicator.adx()
        plus_di_series = adx_indicator.adx_pos()
        minus_di_series = adx_indicator.adx_neg()
        last_adx = float(np.nan_to_num(adx_series.iloc[-1], nan=0.0))
        last_plus = float(np.nan_to_num(plus_di_series.iloc[-1], nan=0.0))
        last_minus = float(np.nan_to_num(minus_di_series.iloc[-1], nan=0.0))
    except Exception as exc:
        return StrategyResult('NONE', 0.0, [f"ADX error: {exc}"], {'adx': None})
    extra = {
        'adx': last_adx,
        'plus_di': last_plus,
        'minus_di': last_minus,
    }
    if last_adx > 25 and last_plus > last_minus:
        return StrategyResult('CALL', 1.4, [f"ADX {last_adx:.2f} strong uptrend"], {**extra, 'strong_call': True})
    if last_adx > 25 and last_minus > last_plus:
        return StrategyResult('PUT', -1.4, [f"ADX {last_adx:.2f} strong downtrend"], {**extra, 'strong_put': True})
    return StrategyResult('NONE', 0.0, [f"ADX {last_adx:.2f} weak or mixed trend"], extra)


def strategy_macd(df: pd.DataFrame) -> StrategyResult:
    if len(df) < 26:
        return StrategyResult('NONE', 0.0, ['MACD sin datos suficientes'], {'macd': None, 'signal': None})
    try:
        macd_indicator = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
        macd_line = float(np.nan_to_num(macd_indicator.macd().iloc[-1], nan=0.0))
        signal_line = float(np.nan_to_num(macd_indicator.macd_signal().iloc[-1], nan=0.0))
        histogram = float(np.nan_to_num(macd_indicator.macd_diff().iloc[-1], nan=0.0))
    except Exception as exc:
        return StrategyResult('NONE', 0.0, [f"MACD error: {exc}"], {'macd': None, 'signal': None})
    extra = {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram,
    }
    if macd_line > signal_line:
        return StrategyResult('CALL', 1.3, ['MACD above signal line → bullish momentum'], extra)
    if macd_line < signal_line:
        return StrategyResult('PUT', -1.3, ['MACD below signal line → bearish momentum'], extra)
    return StrategyResult('NONE', 0.0, ['MACD neutral'], extra)


def strategy_candle_momentum(df: pd.DataFrame) -> StrategyResult:
    if len(df) < 2:
        return StrategyResult('NONE', 0.0, ['Candle momentum sin datos suficientes'], {})
    try:
        last_candle = df.iloc[-1]
    except Exception as exc:
        return StrategyResult('NONE', 0.0, [f"Candle error: {exc}"], {})
    body = float(abs(last_candle['close'] - last_candle['open']))
    high = float(last_candle['high'])
    low = float(last_candle['low'])
    range_candle = float(high - low)
    if range_candle <= 0:
        return StrategyResult('NONE', 0.0, ['Candle momentum sin rango definido'], {'body_ratio': 0.0})
    upper_wick = float(high - max(last_candle['close'], last_candle['open']))
    lower_wick = float(min(last_candle['close'], last_candle['open']) - low)
    body_ratio = body / range_candle
    extra = {
        'body_ratio': body_ratio,
        'upper_wick': upper_wick,
        'lower_wick': lower_wick,
        'range': range_candle,
    }
    if (
        last_candle['close'] > last_candle['open']
        and body_ratio > 0.6
        and upper_wick < body * 0.2
    ):
        return StrategyResult('CALL', 1.1, ['Strong bullish candle momentum'], {**extra, 'direction': 'bullish'})
    if (
        last_candle['close'] < last_candle['open']
        and body_ratio > 0.6
        and lower_wick < body * 0.2
    ):
        return StrategyResult('PUT', -1.1, ['Strong bearish candle momentum'], {**extra, 'direction': 'bearish'})
    return StrategyResult('NONE', 0.0, ['Low momentum or indecision'], extra)


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
        return StrategyResult('NONE', 0.0, ['Volatilidad sin datos suficientes'], {'volatility': None})
    retornos = df['close'].pct_change().dropna()
    if retornos.empty:
        return StrategyResult('NONE', 0.0, ['Volatilidad no calculable'], {'volatility': None})
    reciente = retornos.iloc[-20:]
    volatilidad = float(np.nan_to_num(reciente.std(), nan=0.0))
    return StrategyResult('NONE', 0.0, [f"Volatilidad informativa {volatilidad:.4f}"], {'volatility': volatilidad})


STRATEGY_FUNCTIONS: List[Tuple[str, Callable[[pd.DataFrame], StrategyResult]]] = [
    ('RSI', strategy_rsi),
    ('EMA Trend', strategy_ema_trend),
    ('Bollinger Rebound', strategy_bollinger_rebound),
    ('ADX', strategy_adx),
    ('MACD', strategy_macd),
    ('Candle Momentum', strategy_candle_momentum),
    ('Pullback', strategy_pullback),
    ('Range Breakout', strategy_range_breakout),
]

STRATEGY_WEIGHTS: Dict[str, float] = {
    'RSI': 2.3,
    'EMA Trend': 1.8,
    'Bollinger Rebound': 1.4,
    'ADX': 1.6,
    'MACD': 1.7,
    'Candle Momentum': 1.2,
    'Pullback': 1.1,
    'Range Breakout': 1.0,
    'Divergence': 1.7,
    'Volatility Filter': 0.6,
}

TOTAL_STRATEGY_COUNT = len(STRATEGY_WEIGHTS)

MAX_STRATEGY_SCORE = 3.0

STRATEGY_DISPLAY_NAMES: Dict[str, str] = {
    'RSI': 'RSI',
    'EMA Trend': 'Tendencia EMA',
    'Bollinger Rebound': 'Rebote Bollinger',
    'ADX': 'ADX',
    'MACD': 'MACD',
    'Candle Momentum': 'Candle Momentum',
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


def _compute_strategy_confidence(score: float) -> float:
    magnitude = abs(float(score))
    normalized = magnitude / MAX_STRATEGY_SCORE if MAX_STRATEGY_SCORE else magnitude
    return float(max(0.0, min(1.0, normalized)))


def _compute_volatility_factor(volatility: Optional[float]) -> float:
    if volatility is None:
        return 1.0
    vol = max(0.0, float(volatility))
    low_threshold = 0.0005
    high_threshold = 0.003
    if vol < low_threshold:
        ratio = vol / max(low_threshold, 1e-9)
        reduction = (1.0 - ratio) * 0.15
        return float(max(0.7, 1.0 - reduction))
    if vol > high_threshold:
        ratio_high = min(1.0, (vol - high_threshold) / max(high_threshold, 1e-9))
        boost = ratio_high * 0.05
        return float(min(1.1, 1.0 + boost))
    return 1.0


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
    direction_confidence_sum = {'CALL': 0.0, 'PUT': 0.0}
    direction_weight_sum = {'CALL': 0.0, 'PUT': 0.0}
    strategy_confidences: Dict[str, float] = {}
    for name, res in results:
        if not active_states.get(name, True):
            continue
        weight = STRATEGY_WEIGHTS.get(name, 1.0)
        total_weight_active += weight
        active_count += 1
        agreements[name] = res.signal
        reasons.extend(res.reasons)
        strategy_conf = _compute_strategy_confidence(res.score)
        strategy_confidences[name] = strategy_conf
        logging.debug(
            f"[Signals] {name}: señal={res.signal} score={res.score:.3f} confianza={strategy_conf:.3f}"
        )
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
            direction_confidence_sum[res.signal] += strategy_conf * weight
            direction_weight_sum[res.signal] += weight
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
            'strong': 0,
            'strategy_confidences': strategy_confidences,
        }
    call_support = direction_confidence_sum['CALL']
    put_support = direction_confidence_sum['PUT']
    if call_support == put_support:
        dominante = 'CALL' if pesos_direccion['CALL'] >= pesos_direccion['PUT'] else 'PUT'
    else:
        dominante = 'CALL' if call_support > put_support else 'PUT'
    dominant_weight = direction_weight_sum[dominante]
    weighted_confidence = (
        direction_confidence_sum[dominante] / dominant_weight if dominant_weight else 0.0
    )
    participation_factor = (
        pesos_direccion[dominante] / total_weight_active if total_weight_active else 0.0
    )
    confianza = max(weighted_confidence, participation_factor)
    signal = 'NONE'
    aligned = conteo_direccion[dominante]
    if confianza >= 0.45:
        signal = dominante
    if override_signal:
        signal = override_signal
        confianza = max(confianza, 0.45)
        aligned = conteo_direccion.get(override_signal, 0)
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
    strong_flag = 1 if signal != 'NONE' and confianza >= 0.75 else 0
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
        'strong': strong_flag,
        'strategy_confidences': strategy_confidences,
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
        self.stop_event = threading.Event()
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
        interval = max(30, ADVISORY_INTERVAL_SEC)
        while not self.stop_event.is_set():
            try:
                acc = self.accuracy() * 100
                phase = self._phase()
                phase_text = {
                    "passive": "pasiva",
                    "semi-active": "semi-activa",
                    "autonomous": "autónoma",
                }.get(phase, phase)
                logging.info(
                    f"📊 Aviso IA → fase={phase_text} precisión={acc:.2f}% operaciones={self.trade_counter}"
                )
                self._train_model()
            except Exception as exc:  # pragma: no cover
                logging.error(exc)
            if self.stop_event.wait(interval):
                break

    def shutdown(self) -> None:
        self.stop_event.set()
        if self.offline_thread.is_alive():
            self.offline_thread.join(timeout=1.0)

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


def fetch_axi_candle_objects(symbol: str, count: int = CANDLE_COUNT, timeframe: int = mt5.TIMEFRAME_M1) -> List[Candle]:
    raw_candles = get_candles(symbol, timeframe=timeframe, count=count)
    candles: List[Candle] = []
    for item in raw_candles:
        candles.append(
            Candle(
                epoch=int(item["time"]),
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=float(item["close"]),
            )
        )
    return candles


# ===============================================================
# TRADING ENGINE
# ===============================================================
class TradingEngine:
    def __init__(self) -> None:
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
        self._result_listeners: List[Callable[[Dict[str, Any]], None]] = []
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
        self.aprendizaje = Aprendizaje()
        self.learning_enabled = True
        self.learning_confidences: deque = deque(maxlen=100)
        self.last_learning_feedback: Optional[float] = None
        self.regime_cycle_count = 0
        with auto_learn.weights_lock:
            self._baseline_weights = dict(auto_learn.weights)
        self._indicator_cache: Dict[str, Dict[str, Any]] = {symbol: {} for symbol in SYMBOLS}
        self._volatility_history: Dict[str, deque] = {symbol: deque(maxlen=5) for symbol in SYMBOLS}
        self._cycle_id_counter = 0
        self._last_regime_cycle = -1
        self._current_cycle_id = 0
        self._latest_regime_inputs: Dict[str, Any] = {}
        self._active_regime: Optional[str] = None
        self._processed_lock = threading.Lock()
        self._processed_contracts: Set[int] = set()
        self._closed_lock = threading.Lock()
        self._closed_contracts: Set[int] = set()
        self._next_cycle_time = 0.0
        self._post_loss_cooldown_until: float = 0.0
        self._trade_timestamps: deque = deque()
        self.total_operations = 0
        self.win_operations = 0
        self.last_contract_id: Optional[int] = None
        self.last_symbol: Optional[str] = None
        self.last_result: Optional[str] = None
        self.last_confidence: float = 0.0
        self.bias_memory: Dict[str, Dict[str, Any]] = load_biases()
        self.winner_biases: List[Dict[str, Any]] = load_winner_biases()
        raw_learning_entries = load_learning_memory()
        self.learning_memory = dict(raw_learning_entries)
        wins_loaded = sum(
            1
            for item in self.learning_memory.values()
            if str(item.get('result', '')).upper() == 'WIN'
        )
        losses_loaded = sum(
            1
            for item in self.learning_memory.values()
            if str(item.get('result', '')).upper() == 'LOSS'
        )
        total_loaded = len(self.learning_memory)
        logging.info(
            "🧠 Memory loaded with %d WIN and %d LOSS patterns (%d total).",
            wins_loaded,
            losses_loaded,
            total_loaded,
        )
        self.session_total = 0
        self.session_wins = 0
        self.session_losses = 0
        accuracy_text, reference_ops = safe_startup_accuracy(
            self.session_wins,
            self.session_losses,
        )
        logging.info(
            "📊 Inicio de sesión → precisión histórica %s (registros=%d)",
            accuracy_text,
            reference_ops,
        )
        startup_message = (
            f"🤖 Bot iniciado correctamente y conectado a Axi MT5\n"
            f"🧠 Memoria cargada con {wins_loaded} ganadoras y {losses_loaded} perdedoras"
        )
        try:
            send_telegram_message(startup_message)
        except Exception:
            logging.debug("No se pudo notificar la carga de memoria de aprendizaje")
        self._telegram_thread: Optional[threading.Thread] = None
        self.telegram_bot = telegram_bot
        self.last_volatility: float = 0.0
        self.failed_candle_count = 0

    def add_trade_listener(self, callback: Callable[[TradeRecord, Dict[str, float]], None]) -> None:
        self._trade_listeners.append(callback)

    def add_status_listener(self, callback: Callable[[str], None]) -> None:
        self._status_listeners.append(callback)

    def add_summary_listener(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        self._summary_listeners.append(callback)

    def add_trade_state_listener(self, callback: Callable[[str], None]) -> None:
        self._trade_state_listeners.append(callback)

    def add_result_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._result_listeners.append(callback)

    def remove_result_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        try:
            self._result_listeners.remove(callback)
        except ValueError:
            pass

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

    def _store_winner_bias(
        self,
        symbol: str,
        direction: str,
        rsi_value: float,
        ema_spread: float,
        volatility: float,
        regime: Optional[str],
        confidence_value: float,
    ) -> None:
        direction_value = direction if direction in {"CALL", "PUT"} else "NONE"
        timestamp_value = datetime.now(timezone.utc).isoformat()
        weight_increment = max(confidence_value, 0.0)
        updated = False
        for entry in self.winner_biases:
            if (
                entry.get("symbol") == symbol
                and entry.get("direction") == direction_value
            ):
                entry["confidence"] = max(
                    float(entry.get("confidence", 0.0) or 0.0),
                    float(confidence_value),
                )
                entry["timestamp"] = timestamp_value
                entry["rsi"] = float(rsi_value)
                entry["ema"] = float(ema_spread)
                entry["volatility"] = float(volatility)
                entry["regime"] = regime or entry.get("regime", "DESCONOCIDO")
                entry["weight"] = float(entry.get("weight", 0.0) or 0.0) + weight_increment
                entry["rsi_direction"] = self._classify_rsi_direction(rsi_value)
                entry["ema_direction"] = self._classify_ema_direction(ema_spread)
                entry["volatility_level"] = self._classify_volatility_level(volatility)
                updated = True
                break
        if not updated:
            self.winner_biases.append(
                {
                    "symbol": symbol,
                    "direction": direction_value,
                    "confidence": float(confidence_value),
                    "timestamp": timestamp_value,
                    "rsi": float(rsi_value),
                    "ema": float(ema_spread),
                    "volatility": float(volatility),
                    "regime": regime or "DESCONOCIDO",
                    "weight": max(weight_increment, 1.0),
                    "rsi_direction": self._classify_rsi_direction(rsi_value),
                    "ema_direction": self._classify_ema_direction(ema_spread),
                    "volatility_level": self._classify_volatility_level(volatility),
                }
            )
        self.winner_biases.sort(key=lambda item: item.get("weight", 0.0), reverse=True)
        save_winner_biases(self.winner_biases)

    def _persist_learning_memory(self) -> None:
        self.learning_memory = prune_learning_memory(
            self.learning_memory, max_patterns=200
        )
        save_learning_memory(self.learning_memory)

    def _append_learning_memory_entry(
        self,
        symbol: str,
        action: str,
        rsi_value: float,
        ema_spread: float,
        volatility: float,
        result: str,
        confidence_value: float,
        signals_map: Optional[Dict[str, str]] = None,
    ) -> None:
        rounded_rsi = round(float(rsi_value), 1)
        rounded_ema = round(float(ema_spread), 1)
        pattern_key = build_learning_pattern_key(symbol, action, rounded_rsi, rounded_ema)
        timestamp_value = time.time()
        normalized_result = result.upper()
        volatility_value = float(volatility)
        context_snapshot = self._build_context_snapshot(
            rsi_value,
            ema_spread,
            volatility_value,
        )
        existing = self.learning_memory.get(pattern_key)
        wins_row = 0
        losses_row = 0
        weight_value = 1.0
        if existing is not None:
            wins_row = int(existing.get("wins_in_a_row", 0) or 0)
            losses_row = int(existing.get("loss_streak", 0) or 0)
            weight_value = float(existing.get("weight", 1.0) or 1.0)
        signals_snapshot: Dict[str, str] = {}
        if isinstance(signals_map, dict):
            signals_snapshot = {
                str(key): str(value or "NONE").upper()
                for key, value in signals_map.items()
            }
        entry = {
            "symbol": symbol,
            "action": action,
            "direction": action,
            "rsi": rounded_rsi,
            "ema": rounded_ema,
            "result": normalized_result,
            "confidence": float(confidence_value),
            "timestamp": timestamp_value,
            "weight": weight_value,
            "wins_in_a_row": wins_row,
            "loss_streak": losses_row,
            "pattern_key": pattern_key,
            "signals": signals_snapshot,
            "volatility": volatility_value,
            "rsi_direction": context_snapshot['RSI_direction'],
            "ema_direction": context_snapshot['EMA_direction'],
            "volatility_level": context_snapshot['volatility_level'],
        }
        self.learning_memory[pattern_key] = entry
        self._persist_learning_memory()

    def _update_learning_weights(
        self,
        symbol: str,
        action: str,
        rsi_value: float,
        ema_spread: float,
        volatility: float,
        result: str,
        signals_map: Optional[Dict[str, str]] = None,
    ) -> None:
        rounded_rsi = round(float(rsi_value), 1)
        rounded_ema = round(float(ema_spread), 1)
        pattern_key = build_learning_pattern_key(symbol, action, rounded_rsi, rounded_ema)
        entry = self.learning_memory.get(pattern_key)
        if entry is None:
            return
        weight_value = float(entry.get("weight", 1.0) or 1.0)
        normalized_result = result.upper()
        volatility_value = float(volatility)
        if normalized_result == "WIN":
            weight_value += 1.0
            entry["wins_in_a_row"] = int(entry.get("wins_in_a_row", 0) or 0) + 1
            entry["loss_streak"] = 0
        else:
            weight_value = max(1.0, weight_value - 0.3)
            entry["wins_in_a_row"] = 0
            entry["loss_streak"] = int(entry.get("loss_streak", 0) or 0) + 1
        if isinstance(signals_map, dict):
            entry["signals"] = {
                str(key): str(value or "NONE").upper()
                for key, value in signals_map.items()
            }
        entry["weight"] = weight_value
        entry["result"] = normalized_result
        entry["timestamp"] = time.time()
        entry["volatility"] = volatility_value
        context_snapshot = self._build_context_snapshot(
            rsi_value,
            ema_spread,
            volatility_value,
        )
        entry["rsi_direction"] = context_snapshot['RSI_direction']
        entry["ema_direction"] = context_snapshot['EMA_direction']
        entry["volatility_level"] = context_snapshot['volatility_level']
        self.learning_memory[pattern_key] = entry
        self._persist_learning_memory()

    def start_background_services(self) -> None:
        try:
            if hasattr(self, "telegram_bot") and callable(self.telegram_bot):
                if self._telegram_thread is None or not self._telegram_thread.is_alive():
                    self._telegram_thread = threading.Thread(
                        target=self.telegram_bot,
                        daemon=True,
                    )
                    self._telegram_thread.start()
                    logging.info("📨 Telegram bot service started in background.")
            else:
                logging.info("ℹ️ Telegram bot not configured in this version.")
        except Exception as exc:
            logging.error(f"❌ Error starting background services: {exc}")

    def save_learning_data(self) -> None:
        try:
            self.aprendizaje.save_data()
        except Exception:
            pass

    def save_learning_memory(self) -> None:
        try:
            self._persist_learning_memory()
        except Exception:
            pass

    def start_engine(self) -> None:
        global operation_active
        if self.running.is_set():
            return
        self.running.set()
        self._notify_status("connecting")
        try:
            if not mt5.initialize():
                error_info = mt5.last_error()
                raise RuntimeError(f"No se pudo inicializar MetaTrader5: {error_info}")
            try:
                send_telegram_message("🤖 Bot iniciado correctamente y conectado a Axi MT5")
            except Exception:
                logging.debug("No se pudo enviar mensaje de inicio a Telegram")
        except Exception:
            self.running.clear()
            self._notify_status("stopped")
            raise
        self._notify_status("running")
        operation_active = False
        self._notify_trade_state("ready")
        with self._processed_lock:
            self._processed_contracts.clear()
        with self._closed_lock:
            self._closed_contracts.clear()
        CSV_LOGGED_CONTRACTS.clear()

    def is_running(self) -> bool:
        return self.running.is_set()

    def set_strategy_state(self, name: str, enabled: bool) -> None:
        with self._strategy_lock:
            if name in self.strategy_states:
                self.strategy_states[name] = enabled

    def get_strategy_states(self) -> Dict[str, bool]:
        with self._strategy_lock:
            return dict(self.strategy_states)

    def set_kelly_enabled(self, enabled: bool) -> None:
        self.kelly_enabled = bool(enabled)

    def set_learning_enabled(self, enabled: bool) -> None:
        self.learning_enabled = bool(enabled)

    def get_accuracy(self) -> float:
        total = self.win_count + self.loss_count
        if total == 0:
            return 0.0
        return (self.win_count / total) * 100.0

    def get_last_contract_info(self) -> str:
        if self.last_contract_id is None:
            return "Sin operaciones registradas."
        symbol = self.last_symbol or "N/A"
        result = self.last_result or "N/A"
        return (
            f"#{self.last_contract_id} | {symbol} | {result} | Confianza {self.last_confidence:.2f}"
        )

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

    def _notify_trade_result(self, data: Dict[str, Any]) -> None:
        for callback in list(self._result_listeners):
            try:
                callback(dict(data))
            except Exception as exc:
                logging.debug(f"Error en escucha de resultados: {exc}")

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

    def detect_market_regime(self, adx_value: float, bollinger_width: float, volatility: float) -> str:
        ema_slope = 0.0
        symbol = None
        if isinstance(self._latest_regime_inputs, dict):
            ema_slope = float(self._latest_regime_inputs.get('ema_slope', 0.0))
            symbol = self._latest_regime_inputs.get('symbol')
        slope_strength = abs(ema_slope)
        wide_bands = bollinger_width > 0.03
        strong_trend = adx_value > 25.0 or slope_strength > 0.0005
        calm_volatility = volatility < 0.02
        if strong_trend and calm_volatility and not wide_bands:
            regime = "TRENDING"
        elif adx_value < 20.0 and slope_strength < 0.0003 and volatility < 0.015:
            regime = "RANGING"
        elif volatility > 0.03 or wide_bands:
            regime = "VOLATILE"
        else:
            regime = "CALM"
        logging.info(f"📊 Market Regime Detected: {regime}{' @ ' + symbol if symbol else ''}")
        return regime

    def _apply_regime_adjustments(self, regime: str) -> None:
        if not regime:
            return
        cycle_id = getattr(self, '_current_cycle_id', 0)
        if cycle_id == self._last_regime_cycle and regime == self._active_regime:
            return
        if getattr(auto_learn, 'weights_recalibrated', False):
            with auto_learn.weights_lock:
                self._baseline_weights = dict(auto_learn.weights)
            auto_learn.weights_recalibrated = False
        update_weights = False
        with auto_learn.weights_lock:
            weights_snapshot = dict(auto_learn.weights)
            if regime == "TRENDING":
                weights_snapshot["EMA"] = weights_snapshot.get("EMA", 1.0) * 1.05
                weights_snapshot["RSI"] = weights_snapshot.get("RSI", 1.0) * 0.95
                update_weights = True
            elif regime == "RANGING":
                weights_snapshot["RSI"] = weights_snapshot.get("RSI", 1.0) * 1.05
                weights_snapshot["MACD"] = weights_snapshot.get("MACD", 1.0) * 0.95
                update_weights = True
            elif regime == "VOLATILE" or regime == "CALM":
                update_weights = False
            if update_weights:
                for key, value in list(weights_snapshot.items()):
                    weights_snapshot[key] = float(np.clip(value, 0.2, 3.0))
                auto_learn.weights = weights_snapshot
        with auto_learn.lock:
            current_conf = float(auto_learn.min_confidence)
            if regime == "VOLATILE":
                current_conf += 0.05
            elif regime == "CALM":
                current_conf -= 0.02
            current_conf = float(np.clip(current_conf, 0.3, 0.95))
            auto_learn.min_confidence = current_conf
        self.regime_cycle_count += 1
        if self.regime_cycle_count % 200 == 0:
            with auto_learn.weights_lock:
                auto_learn.weights = dict(self._baseline_weights)
            logging.info("🔄 Resetting indicator weights to baseline.")
        self._last_regime_cycle = cycle_id
        self._active_regime = regime

    def _handle_auto_shutdown(self) -> None:
        if self.auto_shutdown_triggered:
            return
        self.auto_shutdown_triggered = True
        logging.warning("⚠️ Límite de pérdidas consecutivas alcanzado — bot detenido automáticamente")
        self.risk.consecutive_losses = 0
        self._notify_status("auto_shutdown")
        self.stop()


    def _resolve_trade_result(self, order_result: Any, stake: float, signal: str) -> Tuple[str, float]:
        if order_result is None:
            return "UNKNOWN", 0.0
        profit_value = float(getattr(order_result, "profit", 0.0) or 0.0)
        if profit_value > 0:
            return "WIN", profit_value
        if profit_value < 0:
            return "LOSS", profit_value
        retcode = getattr(order_result, "retcode", None)
        if retcode == mt5.TRADE_RETCODE_DONE:
            direction = signal.upper()
            logging.info(f"✅ Orden {direction} ejecutada correctamente en MT5")
            return "WIN", float(stake) * PAYOUT
        logging.info("❌ Orden MT5 no ejecutada correctamente")
        return "LOSS", -float(stake)

    def _register_processed_contract(self, contract_id: int) -> bool:
        with self._processed_lock:
            if contract_id in self._processed_contracts:
                return False
            self._processed_contracts.add(contract_id)
            if len(self._processed_contracts) > 5000:
                self._processed_contracts.pop()
            return True

    def _register_contract_closure(self, contract_id: Optional[int]) -> bool:
        if contract_id is None:
            return True
        with self._closed_lock:
            if contract_id in self._closed_contracts:
                return False
            self._closed_contracts.add(contract_id)
            if len(self._closed_contracts) > 5000:
                self._closed_contracts.pop()
            return True


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
        rsi_signal = 'NONE'
        ema_signal = 'NONE'
        bollinger_signal = 'NONE'
        adx_signal = 'NONE'
        macd_signal = 'NONE'
        candle_signal = 'NONE'
        adx_last_value = None
        pullback_signal = 'NONE'
        range_break_signal = 'NONE'
        divergence_signal = 'NONE'
        volatility_signal = 'NONE'
        final_signal = 'NONE'
        combined_base_action = 'NONE'
        primary_signals: List[str] = []
        skip_reason = None
        confidence = 0.0
        stake = 0.0
        try:
            try:
                candles = fetch_axi_candle_objects(symbol)
            except Exception as exc:
                logging.warning(f"Error al obtener velas de {symbol}: {exc}")
                candles = []
            candle_data = candles
            failed_candle_count = getattr(self, "failed_candle_count", 0)
            if not candle_data or "Error" in str(candle_data):
                self.failed_candle_count = failed_candle_count + 1
            else:
                self.failed_candle_count = 0
            if self.failed_candle_count >= 4:
                logging.error("❌ 4 consecutive candle fetch errors — triggering safe restart")
                try:
                    self.save_learning_data()
                    self.save_learning_memory()
                    send_telegram_message(
                        "⚠️ Error de conexión detectado en los 4 símbolos.\n♻️ Reinicio automático del bot ejecutado correctamente."
                    )
                except Exception as e:
                    logging.error(f"Error during pre-restart saving: {e}")

                time.sleep(2)
                safe_restart_windows()
            if not candles:
                logging.warning(f"Sin velas disponibles para {symbol}, se omite del ciclo")
                return None
            df = to_dataframe(candles)
            atr_series_calc = atr(df, 14) if not df.empty else pd.Series(dtype=float)
            atr_latest_value = float(atr_series_calc.iloc[-1]) if not atr_series_calc.empty else 0.0
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
                logging.info(f"Estrategias activas: {active_total}/{TOTAL_STRATEGY_COUNT}")
            else:
                logging.info(f"Estrategias alineadas: {consensus['aligned']}/{MIN_ALIGNED_STRATEGIES}")
                logging.info(f"Estrategias activas: {active_total}/{TOTAL_STRATEGY_COUNT}")
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
                'strategy_confidences': consensus.get('strategy_confidences', {}),
                'features': None,
                'entry_price': entry_price,
                'latest_rsi': 0.0,
                'ema_spread': 0.0,
                'boll_width': 0.0,
                'bb_width': 0.0,
                'volatility': 0.0,
                'atr': atr_latest_value,
                'signal_source': 'COMBINED',
                'ema_diff': float(df['close'].iloc[-1] - df['close'].iloc[-2]) if len(df) > 1 else 0.0,
                'indicator_confidence': 0.0,
                'adx': 0.0,
                'macd': 0.0,
                'recent_closes': [],
                'recent_highs': [],
                'recent_lows': [],
                'predicted_probability': 0.5,
                'confluence_direction': 'NONE',
                'confluence_confirmed': False,
                'eligible': False,
                'rsi_signal': 'NONE',
                'ema_signal': 'NONE',
                'macd_signal': 'NONE',
                'strong': 0,
            }
            last_epoch = candles[-1].epoch if candles else 0
            cache = self._indicator_cache.setdefault(symbol, {})
            cached_epoch = cache.get('epoch')
            if cached_epoch == last_epoch:
                latest_rsi = float(cache.get('latest_rsi', 0.0))
                ema_corto_valor = float(cache.get('ema_fast', 0.0))
                ema_largo_valor = float(cache.get('ema_slow', 0.0))
                boll_width = float(cache.get('boll_width', 0.0))
                ema_prev_fast = float(cache.get('ema_fast_prev', ema_corto_valor))
            else:
                latest_rsi_series = rsi(df['close'])
                latest_rsi = float(latest_rsi_series.iloc[-1]) if not latest_rsi_series.empty else 0.0
                ema_fast_series = ema(df['close'], 9) if len(df) else pd.Series(dtype=float)
                ema_slow_series = ema(df['close'], 21) if len(df) else pd.Series(dtype=float)
                ema_corto_valor = float(ema_fast_series.iloc[-1]) if not ema_fast_series.empty else 0.0
                ema_largo_valor = float(ema_slow_series.iloc[-1]) if not ema_slow_series.empty else 0.0
                bandas_inferior, bandas_superior = bollinger_bands(df['close'])
                boll_width = float(bandas_superior.iloc[-1] - bandas_inferior.iloc[-1]) if len(bandas_superior) else 0.0
                ema_prev_fast = float(cache.get('ema_fast', ema_corto_valor))
                cache.update({
                    'epoch': last_epoch,
                    'latest_rsi': latest_rsi,
                    'ema_fast_prev': cache.get('ema_fast', ema_corto_valor),
                    'ema_fast': ema_corto_valor,
                    'ema_slow': ema_largo_valor,
                    'boll_width': boll_width,
                })
            ema_slope = ema_corto_valor - ema_prev_fast
            cache['ema_fast_prev'] = ema_corto_valor
            cache['ema_fast'] = ema_corto_valor
            cache['ema_slow'] = ema_largo_valor
            cache['boll_width'] = boll_width
            cache['latest_rsi'] = latest_rsi
            cache['epoch'] = last_epoch
            price_reference = entry_price if entry_price else 1.0
            boll_ratio = boll_width / price_reference if price_reference else 0.0
            volatilidad_serie = df['close'].pct_change().rolling(20).std()
            volatilidad_actual = float(np.nan_to_num(volatilidad_serie.iloc[-1], nan=0.0)) if len(volatilidad_serie) else 0.0
            historial_vol = self._volatility_history.setdefault(symbol, deque(maxlen=5))
            historial_vol.append(volatilidad_actual)
            smoothed_volatility = float(sum(historial_vol) / len(historial_vol)) if historial_vol else volatilidad_actual
            adx_value = auto_learn.calculate_adx(symbol, df['high'].values, df['low'].values, df['close'].values)
            macd_value = auto_learn.calculate_macd(df['close'].values)
            rsi_signal = next((out.signal for name, out in results if name == 'RSI'), 'NONE')
            ema_signal = next((out.signal for name, out in results if name == 'EMA Trend'), 'NONE')
            bollinger_result = next(
                (out for name, out in results if name == 'Bollinger Rebound'),
                None,
            )
            if bollinger_result is not None:
                bollinger_signal = bollinger_result.signal or 'NONE'
            adx_result = next(
                (out for name, out in results if name == 'ADX'),
                None,
            )
            if adx_result is not None:
                adx_signal = adx_result.signal or 'NONE'
                adx_meta = getattr(adx_result, 'metadata', {}) or {}
                if adx_meta and adx_last_value is None:
                    raw_adx_value = adx_meta.get('adx')
                    if raw_adx_value is not None:
                        try:
                            adx_last_value = float(raw_adx_value)
                        except (TypeError, ValueError):
                            adx_last_value = None
            macd_result = next(
                (out for name, out in results if name == 'MACD'),
                None,
            )
            if macd_result is not None:
                macd_signal = macd_result.signal or 'NONE'
            candle_result = next(
                (out for name, out in results if name == 'Candle Momentum'),
                None,
            )
            if candle_result is not None:
                candle_signal = candle_result.signal or 'NONE'
            range_break_result = next(
                (out for name, out in results if name == 'Range Breakout'),
                None,
            )
            if range_break_result is not None:
                range_break_signal = range_break_result.signal or 'NONE'
            divergence_result = next(
                (out for name, out in results if name == 'Divergence'),
                None,
            )
            if divergence_result is not None:
                divergence_signal = divergence_result.signal or 'NONE'
            ema_spread = ema_corto_valor - ema_largo_valor
            indicator_conf = auto_learn.calculate_confidence(
                latest_rsi,
                ema_spread,
                max(0.0, min(1.0, boll_ratio)),
                adx_value,
                macd_value,
            )
            trend_bias = abs(ema_slope) > 0.0005 or adx_value > 25.0
            volatility_factor = _compute_volatility_factor(smoothed_volatility)
            adjusted_indicator_factor = volatility_factor
            if volatility_factor < 1.0 and trend_bias:
                adjusted_indicator_factor = 1.0 - (1.0 - volatility_factor) * 0.5
            logging.debug(
                f"[Fusion] Volatilidad suavizada: {smoothed_volatility:.6f} factor={volatility_factor:.3f}"
            )
            if volatility_factor < 1.0:
                indicator_conf *= adjusted_indicator_factor
            elif volatility_factor > 1.0:
                indicator_conf *= volatility_factor
            indicator_conf = float(np.clip(indicator_conf, 0.0, 1.0))
            ema50_series = ema(df['close'], 50, symbol)
            ema50_current = (
                float(ema50_series.iloc[-1]) if not ema50_series.empty else 0.0
            )
            ema50_previous = (
                float(ema50_series.iloc[-2])
                if len(ema50_series) > 1
                else ema50_current
            )
            evaluation.update(
                {
                    'latest_rsi': latest_rsi,
                    'ema_spread': ema_spread,
                    'boll_width': boll_width,
                    'bb_width': boll_ratio,
                    'volatility': smoothed_volatility,
                    'indicator_confidence': indicator_conf,
                    'adx': adx_value,
                    'adx_last_value': adx_last_value,
                    'macd': macd_value,
                    'rsi_signal': rsi_signal,
                    'ema_signal': ema_signal,
                    'pullback_signal': pullback_signal,
                    'base_action': combined_base_action,
                    'primary_signals': primary_signals,
                    'latest_prices': df['close'].tail(5).tolist(),
                    'macd_signal': macd_signal,
                    'bollinger_signal': bollinger_signal,
                    'adx_signal': adx_signal,
                    'candle_signal': candle_signal,
                    'aligned': consensus['aligned'],
                    'breakout_detected': bool(
                        range_break_signal in {'CALL', 'PUT'}
                    ),
                    'breakout_signal': range_break_signal,
                    'divergence_detected': bool(
                        divergence_signal in {'CALL', 'PUT'}
                    ),
                    'divergence_signal': divergence_signal,
                    'ema50_current': ema50_current,
                    'ema50_previous': ema50_previous,
                    'ema50_slope': ema50_current - ema50_previous,
                    'recent_closes': df['close'].tail(120).tolist(),
                    'recent_highs': df['high'].tail(120).tolist(),
                    'recent_lows': df['low'].tail(120).tolist(),
                }
            )
            evaluation['current_context'] = self._build_context_snapshot(
                latest_rsi,
                ema_spread,
                smoothed_volatility,
            )
            self._latest_regime_inputs = {'symbol': symbol, 'ema_slope': ema_slope, 'boll_width': boll_width}
            regime = self.detect_market_regime(adx_value, boll_width, smoothed_volatility)
            self._apply_regime_adjustments(regime)
            evaluation['regime'] = regime
            signals = [rsi_signal, ema_signal, macd_signal]
            if signals.count('CALL') >= 2:
                evaluation['confluence_direction'] = 'CALL'
                evaluation['confluence_confirmed'] = True
            elif signals.count('PUT') >= 2:
                evaluation['confluence_direction'] = 'PUT'
                evaluation['confluence_confirmed'] = True
            strategy_conf_map = consensus.get('strategy_confidences', {})
            rsi_strength = float(np.clip(strategy_conf_map.get('RSI', 0.0), 0.0, 1.0))
            if rsi_strength == 0.0:
                rsi_strength = float(
                    np.clip(strategy_conf_map.get('RSI Strategy', 0.0), 0.0, 1.0)
                )
            if rsi_strength == 0.0:
                rsi_outcome = next(
                    (
                        outcome
                        for name, outcome in results
                        if name.lower() == 'rsi'
                    ),
                    None,
                )
                if rsi_outcome is not None:
                    rsi_strength = float(np.clip(abs(rsi_outcome.score), 0.0, 1.0))
            ema_strength = float(
                np.clip(
                    strategy_conf_map.get('EMA Trend', strategy_conf_map.get('EMA', 0.0)),
                    0.0,
                    1.0,
                )
            )
            if ema_strength == 0.0:
                ema_outcome = next(
                    (
                        outcome
                        for name, outcome in results
                        if name.lower() == 'ema trend' or name.lower() == 'ema'
                    ),
                    None,
                )
                if ema_outcome is not None:
                    ema_strength = float(np.clip(abs(ema_outcome.score), 0.0, 1.0))
            active_signals = [
                outcome for _, outcome in results if outcome.signal in {'CALL', 'PUT'}
            ]
            total_active = len(active_signals)
            aligned_strategies = sum(
                1 for outcome in active_signals if outcome.signal == signal
            ) if signal in {'CALL', 'PUT'} else 0
            contradictory = any(
                outcome.signal not in {signal}
                for outcome in active_signals
            ) if signal in {'CALL', 'PUT'} else False
            pullback_signal = next(
                (
                    outcome.signal
                    for name, outcome in results
                    if name.lower() == 'pullback'
                ),
                'NONE',
            )
            primary_signals = [rsi_signal, ema_signal, pullback_signal]
            call_votes = primary_signals.count('CALL')
            put_votes = primary_signals.count('PUT')
            if call_votes > put_votes:
                combined_base_action = 'CALL'
            elif put_votes > call_votes:
                combined_base_action = 'PUT'
            else:
                combined_base_action = signal if signal in {'CALL', 'PUT'} else 'NONE'
            volatility_mid_range = 0.0005 <= smoothed_volatility <= 0.002
            perfect_alignment = (
                signal in {'CALL', 'PUT'}
                and not contradictory
                and pullback_signal == signal
                and rsi_signal == signal
                and ema_signal == signal
                and total_active > 0
                and aligned_strategies == total_active
                and volatility_mid_range
            )
            if total_active > 0:
                average_strength = (rsi_strength + ema_strength) / 2.0
                strict_confidence = (aligned_strategies / total_active) * average_strength
            else:
                strict_confidence = 0.0
            strict_confidence = float(np.clip(strict_confidence, 0.0, 0.95))
            if perfect_alignment:
                strict_confidence = 1.0
            confidence = strict_confidence
            consensus['confidence'] = confidence
            evaluation['base_confidence'] = confidence
            evaluation['strict_confidence'] = confidence
            evaluation['perfect_alignment'] = perfect_alignment
            evaluation['aligned'] = aligned_strategies
            evaluation['active'] = total_active
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
            base_components: List[float] = []
            component_weights: List[float] = []
            for value, weight in ((ai_confidence, 0.4), (indicator_conf, 0.35), (confidence, 0.25)):
                if value is None:
                    continue
                val = float(value)
                if val <= 0.0 or math.isnan(val):
                    continue
                base_components.append(val)
                component_weights.append(weight)
            if base_components:
                weight_sum = sum(component_weights) or 1.0
                normalized = [w / weight_sum for w in component_weights]
                weighted_average = sum(v * w for v, w in zip(base_components, normalized))
                combined_confidence = float(max(weighted_average, max(base_components)))
            else:
                combined_confidence = 0.6
            adjusted_combined_factor = volatility_factor
            if volatility_factor < 1.0 and trend_bias:
                adjusted_combined_factor = 1.0 - (1.0 - volatility_factor) * 0.5
            if volatility_factor < 1.0:
                combined_confidence *= adjusted_combined_factor
            elif volatility_factor > 1.0:
                combined_confidence *= volatility_factor
            combined_confidence = float(np.clip(combined_confidence, 0.0, 1.0))
            if combined_confidence == 0.0 or math.isnan(combined_confidence):
                combined_confidence = 0.6
            if ml_probability == 0.0 or math.isnan(ml_probability):
                ml_probability = 0.55
            symbol_weight = auto_learn.get_symbol_weight(symbol)
            min_confidence = auto_learn.get_min_confidence()
            fused_probability = (combined_confidence * 0.6) + (ml_probability * 0.4)
            final_confidence = max(combined_confidence * ml_probability, fused_probability)
            if final_confidence < min_confidence:
                final_confidence = min_confidence
            final_confidence *= symbol_weight
            final_confidence = float(np.clip(final_confidence, 0.0, 1.0))
            if perfect_alignment:
                final_confidence = 1.0
            else:
                final_confidence = float(min(final_confidence, 0.95))
            logging.debug(
                f"[Fusion] base={confidence:.3f} indicador={indicator_conf:.3f} IA={ai_confidence:.3f} final={final_confidence:.3f}"
            )
            if final_confidence >= 0.75:
                evaluation['strong'] = 1
            adx_threshold = auto_learn.get_adx_min_threshold()
            rsi_high_threshold = auto_learn.get_rsi_high_threshold()
            call_floor = max(0.0, 100.0 - rsi_high_threshold)
            eligible = (
                signal in {'CALL', 'PUT'}
                and final_confidence >= min_confidence
                and ml_probability >= 0.55
                and adx_value >= adx_threshold
            )
            if signal == 'PUT' and latest_rsi < rsi_high_threshold:
                eligible = False
            if signal == 'CALL' and latest_rsi > call_floor:
                eligible = False
            evaluation.update(
                {
                    'ai_confidence': ai_confidence,
                    'final_confidence': final_confidence,
                    'ai_notes': ai_notes,
                    'features': features,
                    'eligible': eligible,
                    'base_confidence': combined_confidence,
                }
            )
            return evaluation
        except Exception as exc:
            logging.warning(f"Error en ciclo para {symbol}: {exc}")
            return None
        finally:
            if combined_base_action is None:
                combined_base_action = 'NONE'
            if pullback_signal is None:
                pullback_signal = 'NONE'
            if bollinger_signal is None:
                bollinger_signal = 'NONE'
            if adx_signal is None:
                adx_signal = 'NONE'
            if macd_signal is None:
                macd_signal = 'NONE'
            if candle_signal is None:
                candle_signal = 'NONE'
            if divergence_signal is None:
                divergence_signal = 'NONE'
            if range_break_signal is None:
                range_break_signal = 'NONE'
            if volatility_signal is None:
                volatility_signal = 'NONE'
            if final_signal is None:
                final_signal = 'NONE'
            if primary_signals is None:
                primary_signals = []
            auto_learn.set_active_symbol(None)

    @staticmethod
    def _classify_rsi_direction(rsi_value: float) -> str:
        try:
            value = float(rsi_value)
        except (TypeError, ValueError):
            value = 0.0
        if value > NEUTRAL_RSI_BAND[1]:
            return 'UP'
        if value < NEUTRAL_RSI_BAND[0]:
            return 'DOWN'
        return 'FLAT'

    @staticmethod
    def _classify_ema_direction(ema_spread: float) -> str:
        try:
            spread = float(ema_spread)
        except (TypeError, ValueError):
            spread = 0.0
        if spread > 0:
            return 'UP'
        if spread < 0:
            return 'DOWN'
        return 'FLAT'

    @staticmethod
    def _classify_volatility_level(volatility: float) -> str:
        try:
            value = float(volatility)
        except (TypeError, ValueError):
            return 'UNKNOWN'
        if value < MIN_VOLATILITY:
            return 'LOW'
        if value > 0.002:
            return 'HIGH'
        return 'MEDIUM'

    def _build_context_snapshot(
        self,
        rsi_value: float,
        ema_spread: float,
        volatility: float,
    ) -> Dict[str, str]:
        return {
            'RSI_direction': self._classify_rsi_direction(rsi_value),
            'EMA_direction': self._classify_ema_direction(ema_spread),
            'volatility_level': self._classify_volatility_level(volatility),
        }


    
    def _passes_confluence_validation(self, evaluation: Dict[str, Any]) -> bool:
        symbol = evaluation.get('symbol', 'UNKNOWN')
        try:
            results = evaluation.get('results', []) or []
            signals_map: Dict[str, str] = {}
            last_adx_value = evaluation.get('adx_last_value', evaluation.get('adx'))
            if last_adx_value is not None:
                try:
                    last_adx_value = float(last_adx_value)
                except (TypeError, ValueError):
                    last_adx_value = None
            adx_signal = str(evaluation.get('adx_signal', 'NONE') or 'NONE').upper()
            macd_signal = str(evaluation.get('macd_signal', 'NONE') or 'NONE').upper()
            candle_signal = str(evaluation.get('candle_signal', 'NONE') or 'NONE').upper()
            for name, outcome in results:
                if outcome is None:
                    continue
                simple_key = None
                normalized_name = str(name).strip().upper()
                raw_signal = getattr(outcome, 'signal', None)
                normalized_signal = str(raw_signal or 'NONE').strip().upper()
                if 'ADX' in normalized_name:
                    simple_key = 'ADX'
                    metadata = getattr(outcome, 'metadata', {}) or {}
                    if metadata and last_adx_value is None:
                        raw_adx = metadata.get('adx')
                        if raw_adx is not None:
                            try:
                                last_adx_value = float(raw_adx)
                            except (TypeError, ValueError):
                                last_adx_value = None
                elif 'MACD' in normalized_name:
                    simple_key = 'MACD'
                elif 'CANDLE' in normalized_name:
                    simple_key = 'Candle'
                if simple_key:
                    signals_map[simple_key] = normalized_signal
            if last_adx_value is None:
                raw_fallback = evaluation.get('adx')
                if raw_fallback is not None:
                    try:
                        last_adx_value = float(raw_fallback)
                    except (TypeError, ValueError):
                        last_adx_value = None
            if 'ADX' in signals_map and adx_signal in {'CALL', 'PUT'}:
                try:
                    adx_threshold = 22.0
                    symbol_upper = str(symbol).upper()
                    if 'R_25' in symbol_upper:
                        adx_threshold = 19.0
                    elif 'R_50' in symbol_upper:
                        adx_threshold = 20.0
                    elif 'R_75' in symbol_upper:
                        adx_threshold = 22.0
                    elif 'R_100' in symbol_upper:
                        adx_threshold = 24.0
                    if last_adx_value is not None and last_adx_value < adx_threshold:
                        logging.info(
                            f"[{symbol}] ❌ Blocked trade: weak trend (ADX {last_adx_value:.2f} < {adx_threshold})"
                        )
                        return False
                except Exception as exc:
                    logging.warning(f"[{symbol}] ADX dynamic filter error: {exc}")
            if (
                'MACD' in signals_map
                and 'Candle' in signals_map
                and macd_signal in {'CALL', 'PUT'}
                and candle_signal in {'CALL', 'PUT'}
                and macd_signal != candle_signal
            ):
                logging.info(
                    f"[{symbol}] ⚠️ Skipped trade: MACD–Candle conflict ({macd_signal} vs {candle_signal})"
                )
                return False
        except Exception as exc:
            logging.warning(f"[{symbol}] Confluence validation error: {exc}")
        return True
    
    def _extract_pattern_context(
        self,
        entry: Dict[str, Any],
        fallback_rsi: float,
        fallback_ema: float,
        fallback_volatility: float,
    ) -> Dict[str, str]:
        rsi_direction = entry.get('rsi_direction')
        ema_direction = entry.get('ema_direction')
        volatility_level = entry.get('volatility_level')
        if not rsi_direction:
            rsi_direction = self._classify_rsi_direction(fallback_rsi)
        if not ema_direction:
            ema_direction = self._classify_ema_direction(fallback_ema)
        if not volatility_level:
            volatility_level = self._classify_volatility_level(fallback_volatility)
        return {
            'RSI_direction': str(rsi_direction).upper(),
            'EMA_direction': str(ema_direction).upper(),
            'volatility_level': str(volatility_level).upper(),
        }

    @staticmethod
    def _context_match_score(
        pattern_context: Dict[str, str],
        current_context: Dict[str, str],
    ) -> int:
        matches = 0
        for key in ('RSI_direction', 'EMA_direction', 'volatility_level'):
            pattern_value = str(pattern_context.get(key, '')).upper()
            current_value = str(current_context.get(key, '')).upper()
            if pattern_value and current_value and pattern_value == current_value:
                matches += 1
        return matches

    def _passes_high_confidence_checks(
        self,
        evaluation: Dict[str, Any],
        direction: str,
    ) -> bool:
        try:
            confidence_value = float(evaluation.get('final_confidence', 0.0))
        except (TypeError, ValueError):
            confidence_value = 0.0
        if confidence_value < MIN_CONFIDENCE:
            return True
        try:
            latest_rsi = float(evaluation.get('latest_rsi', 0.0))
        except (TypeError, ValueError):
            latest_rsi = 0.0
        breakout_signal = str(evaluation.get('breakout_signal', 'NONE')).upper()
        divergence_signal = str(evaluation.get('divergence_signal', 'NONE')).upper()
        rsi_outside_neutral = (
            latest_rsi < NEUTRAL_RSI_BAND[0] or latest_rsi > NEUTRAL_RSI_BAND[1]
        )
        breakout_confirmed = breakout_signal == direction
        divergence_confirmed = divergence_signal == direction
        score = int(rsi_outside_neutral) + int(breakout_confirmed) + int(divergence_confirmed)
        if score >= 2:
            return True
        symbol = evaluation.get('symbol', 'UNKNOWN')
        logging.info(
            "🚫 High-confidence trade skipped for %s: insufficient confirmation (%d/%d).",
            symbol,
            score,
            MIN_ALIGNED_STRATEGIES,
        )
        return False

    def _is_macro_trend_aligned(
        self,
        evaluation: Dict[str, Any],
        direction: str,
    ) -> bool:
        try:
            ema_current = float(evaluation.get('ema50_current'))
            ema_previous = float(evaluation.get('ema50_previous'))
        except (TypeError, ValueError):
            return True
        slope = ema_current - ema_previous
        evaluation['ema50_slope'] = slope
        if direction == 'CALL' and slope <= 0:
            symbol = evaluation.get('symbol', 'UNKNOWN')
            logging.info(
                "🚫 Trade skipped for %s: EMA50 slope %.6f opposes CALL direction.",
                symbol,
                slope,
            )
            return False
        if direction == 'PUT' and slope >= 0:
            symbol = evaluation.get('symbol', 'UNKNOWN')
            logging.info(
                "🚫 Trade skipped for %s: EMA50 slope %.6f opposes PUT direction.",
                symbol,
                slope,
            )
            return False
        return True

    def scan_market(self) -> None:
        if not self.running.is_set():
            return
        if not BOT_ACTIVE:
            logging.info("⏸ Bot paused — waiting for Telegram resume command.")
            time.sleep(5)
            return
        now = time.time()
        if now < self._next_cycle_time:
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
            return
        self._cycle_id_counter += 1
        self._current_cycle_id = self._cycle_id_counter
        trade_executed = False
        symbols = list(SYMBOLS)
        def _cycle_pause(delay: float = 0.5) -> None:
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
            self._next_cycle_time = time.time() + delay

        evaluations_found = False
        min_required = auto_learn.get_min_confidence()

        for symbol in symbols:
            if not self.running.is_set():
                break
            evaluation = self._evaluate_symbol(symbol)
            if evaluation is None:
                continue
            evaluations_found = True
            probability = float(evaluation.get('predicted_probability', 0.5))
            evaluation['stake'] = self._calculate_kelly_stake(probability)
            if evaluation.get('confluence_confirmed'):
                if self.confirm_and_execute(evaluation):
                    trade_executed = True
                    break

            signal = evaluation.get('signal')
            if signal not in {'CALL', 'PUT'}:
                continue

            is_strong = int(evaluation.get('strong', 0)) == 1
            confidence_raw = evaluation.get('final_confidence')
            confidence_value = float(confidence_raw) if confidence_raw is not None else 0.0
            if not is_strong and confidence_value < min_required:
                continue

            volatility_value_raw = evaluation.get('volatility')
            volatility_value: Optional[float] = None
            if volatility_value_raw is not None:
                try:
                    volatility_value = float(volatility_value_raw)
                except (TypeError, ValueError):
                    volatility_value = None
            volatility_display = f"{volatility_value:.4f}" if volatility_value is not None else "N/A"
            logging.info(
                f"📊 Final confidence {symbol}: {confidence_value:.2f} | Volatility: {volatility_display} | Action: {signal}"
            )

            aligned_total = int(
                evaluation.get('aligned', evaluation.get('consensus', {}).get('aligned', 0))
            )
            min_confidence = CONFIDENCE_MIN
            # === GLOBAL CONFLUENCE RULE ===
            min_confluence = MIN_ALIGNED_STRATEGIES  # fixed at 3 for all symbols

            if confidence_value < min_confidence:
                logging.info(
                    f"🚫 Trade skipped due to low confidence ({confidence_value:.2f} < {min_confidence:.2f})"
                )
                continue
            if aligned_total < min_confluence:
                logging.info(
                    f"🚫 Trade skipped due to low confluence ({aligned_total}/{min_confluence})"
                )
                continue
            if volatility_value is not None and volatility_value < MIN_VOLATILITY:
                logging.info(
                    f"🚫 Skipping trade on {symbol} due to low volatility ({volatility_value:.4f})"
                )
                continue

            rsi_signal = str(evaluation.get('rsi_signal', 'NONE')).upper()
            ema_signal = str(evaluation.get('ema_signal', 'NONE')).upper()
            if (rsi_signal != 'NONE' and ema_signal != 'NONE' and rsi_signal != ema_signal):
                logging.info(
                    f"🚫 Skipping trade on {symbol} due to RSI/EMA conflict → RSI={rsi_signal}, EMA={ema_signal}"
                )
                continue

            if not self._passes_confluence_validation(evaluation):
                continue

            volatility_output = f"{volatility_value:.6f}" if volatility_value is not None else "N/A"
            logging.info(
                f"🚀 Executing trade on {symbol} | Confidence={confidence_value:.2f} | Confluence={aligned_total}/{min_confluence} | Volatility={volatility_output}"
            )
            self._execute_selected_trade(evaluation)
            trade_executed = True
            break

        if not self.running.is_set():
            _cycle_pause()
            return
        if trade_executed:
            _cycle_pause(0.75)
            return
        if not evaluations_found:
            _cycle_pause()
            return
        _cycle_pause()
        return

    def confirm_and_execute(self, evaluation: Dict[str, Any]) -> bool:
        direction = evaluation.get('confluence_direction')
        if direction not in {'CALL', 'PUT'}:
            return False
        symbol = evaluation.get('symbol', 'UNKNOWN')
        logging.info(f"✅ Confluence confirmed for {symbol}: {direction}")
        skip_trade = False
        try:
            confidence_raw = evaluation.get('final_confidence')
            if confidence_raw is None:
                confidence_raw = evaluation.get('confidence')
            confidence_value: Optional[float] = None
            if confidence_raw is not None:
                try:
                    confidence_value = float(confidence_raw)
                except (TypeError, ValueError):
                    confidence_value = None
            volatility_raw = evaluation.get('volatility')
            volatility_value: Optional[float] = None
            if volatility_raw is not None:
                try:
                    volatility_value = float(volatility_raw)
                except (TypeError, ValueError):
                    volatility_value = None
            aprendizaje_ref = getattr(self, 'aprendizaje', None)
            dynamic_min_conf = max(0.70, MIN_CONFIDENCE)
            latest_rsi_value = 0.0
            ema_value_for_threshold = 0.0
            try:
                latest_rsi_value = float(evaluation.get('latest_rsi', 0.0))
            except (TypeError, ValueError):
                latest_rsi_value = 0.0
            try:
                ema_value_for_threshold = float(evaluation.get('ema_spread', 0.0))
            except (TypeError, ValueError):
                ema_value_for_threshold = 0.0
            if aprendizaje_ref is not None:
                try:
                    dynamic_min_conf = max(
                        dynamic_min_conf,
                        aprendizaje_ref.get_dynamic_threshold(
                            latest_rsi_value,
                            ema_value_for_threshold,
                        ),
                    )
                except Exception as exc:
                    logging.debug(f"No se pudo obtener umbral dinámico: {exc}")
            confidence_for_log = confidence_value if confidence_value is not None else 0.0
            volatility_display = f"{volatility_value:.4f}" if volatility_value is not None else "N/A"
            logging.info(
                f"📊 Final confidence {symbol}: {confidence_for_log:.2f} | Volatility: {volatility_display} | Action: {direction}"
            )
            if confidence_value is None or confidence_value < dynamic_min_conf:
                logging.info(
                    f"🚫 Skipping trade on {symbol} due to low confidence ({confidence_for_log:.2f})"
                )
                skip_trade = True
                pass
            elif volatility_value is None or volatility_value < MIN_VOLATILITY:
                volatility_for_log = volatility_value if volatility_value is not None else 0.0
                logging.info(
                    f"🚫 Skipping trade on {symbol} due to low volatility ({volatility_for_log:.4f})"
                )
                skip_trade = True
                pass
            if not skip_trade:
                if not self._passes_high_confidence_checks(evaluation, direction):
                    skip_trade = True
                elif not self._is_macro_trend_aligned(evaluation, direction):
                    skip_trade = True
        except Exception as exc:
            logging.error(f"⚠️ Error in pre-trade validation for {symbol}: {exc}")
            skip_trade = True
            pass
        if skip_trade:
            return False
        if not self._passes_confluence_validation(evaluation):
            return False
        enriched = dict(evaluation)
        enriched['signal'] = direction
        reasons = list(enriched.get('reasons', []))
        reasons.append('Multi-indicator confirmation')
        enriched['reasons'] = reasons
        consensus_snapshot = dict(enriched.get('consensus', {}))
        final_snapshot = confidence_value if confidence_value is not None else dynamic_min_conf
        perfect_alignment = bool(enriched.get('perfect_alignment'))
        if perfect_alignment:
            final_snapshot = 1.0
        else:
            final_snapshot = float(np.clip(final_snapshot, 0.0, 0.95))
        consensus_snapshot['confidence_label'] = 'Alta' if final_snapshot >= MIN_CONFIDENCE else 'Media'
        consensus_snapshot['confidence'] = final_snapshot
        consensus_snapshot['main_reason'] = 'multi-confirmation'
        consensus_snapshot['override'] = True
        consensus_snapshot['override_reason'] = 'multi-confirmation'
        enriched['consensus'] = consensus_snapshot
        enriched['ai_confidence'] = max(float(enriched.get('ai_confidence', 0.0)), final_snapshot)
        enriched['final_confidence'] = final_snapshot
        enriched['eligible'] = True
        probability = float(enriched.get('predicted_probability', 0.6))
        enriched['stake'] = self._calculate_kelly_stake(probability)
        return self._execute_selected_trade(enriched)

    def _fetch_exit_price(self, symbol: str, fallback: float) -> float:
        try:
            candles = fetch_axi_candle_objects(symbol)
            if candles:
                return float(candles[-1].close)
        except Exception as exc:
            logging.debug(f"No se pudo obtener precio de salida para {symbol}: {exc}")
        return fallback

    def _execute_selected_trade(self, evaluation: Dict[str, Any]) -> bool:
        global operation_active
        symbol = evaluation['symbol']
        symbol_upper = str(symbol).upper()
        signal = evaluation['signal']
        ai_confidence = float(evaluation['ai_confidence'])
        combined_confidence = float(evaluation.get('final_confidence', ai_confidence))
        reasons = evaluation['reasons']
        ai_notes = evaluation['ai_notes']
        consensus = evaluation['consensus']
        features = evaluation['features']
        entry_price = float(evaluation['entry_price'])
        signal_source = evaluation['signal_source']
        results: List[Tuple[str, StrategyResult]] = evaluation.get('results', [])
        strategy_signals_map: Dict[str, str] = {
            name: str(res.signal or "NONE").upper()
            for name, res in results
        }
        if 'stake' not in evaluation:
            evaluation['stake'] = self._calculate_kelly_stake(evaluation.get('predicted_probability', 0.5))
        stake_amount = float(LOT_SIZE)
        evaluation['stake'] = stake_amount
        ml_probability = float(evaluation.get('predicted_probability', 0.5))
        trade_initiated = False
        volatility_raw = evaluation.get('volatility')
        volatility_value: Optional[float] = None
        if volatility_raw is not None:
            try:
                volatility_value = float(volatility_raw)
            except (TypeError, ValueError):
                volatility_value = None
        if volatility_value is not None:
            self.current_volatility = volatility_value

        symbol_upper = str(symbol).upper()
        evaluated_volatility = float(getattr(self, "current_volatility", 0.0) or 0.0)
        recent_closes = evaluation.get('recent_closes') or []
        recent_highs = evaluation.get('recent_highs') or []
        recent_lows = evaluation.get('recent_lows') or []
        confidence_value = float(combined_confidence) if combined_confidence is not None else 0.0
        latest_rsi_value = float(evaluation.get('latest_rsi', 0.0))
        ema_spread_value = float(evaluation.get('ema_spread', 0.0))
        current_context = evaluation.get('current_context')
        if not isinstance(current_context, dict):
            current_context = self._build_context_snapshot(
                latest_rsi_value,
                ema_spread_value,
                evaluated_volatility,
            )
            evaluation['current_context'] = current_context
        aligned_strategies = int(
            evaluation.get('aligned', evaluation.get('consensus', {}).get('aligned', 0))
        )
        base_action = evaluation.get('base_action', signal)
        final_action = signal
        latest_prices_seq = evaluation.get('latest_prices')
        if (
            final_action in {'CALL', 'PUT'}
            and base_action in {'CALL', 'PUT'}
            and final_action != base_action
            and confidence_value < MIN_CONFIDENCE
        ):
            logging.info(
                f"⚠️ Señal inconsistente: {base_action} vs {final_action} (conf={confidence_value:.2f}) → Cancelada"
            )
            return False
        rsi_signal_eval = str(evaluation.get('rsi_signal', 'NONE')).upper()
        ema_signal_eval = str(evaluation.get('ema_signal', 'NONE')).upper()
        if (
            rsi_signal_eval != 'NONE'
            and ema_signal_eval != 'NONE'
            and rsi_signal_eval != ema_signal_eval
        ):
            logging.info(
                f"⚠️ RSI/EMA conflict detected for {symbol} → penalizing confidence."
            )
            confidence_value *= 0.80
            combined_confidence = confidence_value
            evaluation['final_confidence'] = confidence_value
        adx_value = float(evaluation.get('adx', 0.0) or 0.0)
        bb_width_value = float(evaluation.get('bb_width', evaluation.get('boll_width', 0.0)) or 0.0)
        regime_value = evaluation.get('regime') or detect_regime(adx_value, bb_width_value)
        context_key = f"{symbol}|{final_action}|RSI:{round(latest_rsi_value, 1)}|EMA:{round(ema_spread_value, 1)}"
        if final_action in {'CALL', 'PUT'}:
            current_rsi_rounded = round(latest_rsi_value, 1)
            current_ema_rounded = round(ema_spread_value, 1)
            bias_pattern_key = build_learning_pattern_key(
                symbol, final_action, current_rsi_rounded, current_ema_rounded
            )
            bias_entry = self.learning_memory.get(bias_pattern_key)
            if (
                bias_entry
                and str(bias_entry.get('result', '')).upper() == 'LOSS'
                and int(bias_entry.get('loss_streak', 0) or 0) >= 2
            ):
                logging.info(
                    f"🧱 Pattern blocked due to repeated losses (streak={bias_entry['loss_streak']})"
                )
                return False
            adjusted_confidence = confidence_value
            for memory_entry in self.learning_memory.values():
                if str(memory_entry.get('symbol', '')) != symbol:
                    continue
                if str(memory_entry.get('action', 'NONE')).upper() != final_action:
                    continue
                if str(memory_entry.get('pattern_key', '')) == bias_pattern_key:
                    continue
                try:
                    stored_rsi = float(memory_entry.get('rsi', 0.0))
                    stored_ema = float(memory_entry.get('ema', 0.0))
                except (TypeError, ValueError):
                    continue
                rsi_diff = abs(stored_rsi - current_rsi_rounded)
                ema_diff = abs(stored_ema - current_ema_rounded)
                if (
                    rsi_diff <= 0.5
                    and ema_diff <= 3
                    and str(memory_entry.get('result', '')).upper() == 'LOSS'
                ):
                    penalty = 0.20 * adjusted_confidence
                    adjusted_confidence = max(0.0, adjusted_confidence - penalty)
                    logging.info(
                        f"⚠️ Confianza reducida por patrón perdedor muy similar ({symbol}, {final_action}, RSI diff={rsi_diff:.2f}, EMA diff={ema_diff:.2f}) → Nueva confianza={adjusted_confidence:.2f}"
                    )
                elif (
                    rsi_diff <= 2
                    and ema_diff <= 10
                    and str(memory_entry.get('result', '')).upper() == 'WIN'
                ):
                    stored_volatility = float(
                        memory_entry.get('volatility', evaluated_volatility)
                        or evaluated_volatility
                    )
                    pattern_context = self._extract_pattern_context(
                        memory_entry,
                        stored_rsi,
                        stored_ema,
                        stored_volatility,
                    )
                    context_matches = self._context_match_score(
                        pattern_context,
                        current_context,
                    )
                    if context_matches < 2:
                        logging.info(
                            f"❌ Context mismatch ({context_matches}/3) → pattern boost skipped"
                        )
                        continue
                    match_count, considered = is_similar_pattern(
                        strategy_signals_map, memory_entry.get('signals')
                    )
                    total_reference = considered
                    if total_reference <= 0:
                        total_reference = sum(
                            1 for value in strategy_signals_map.values() if value != 'NONE'
                        )
                    if total_reference <= 0:
                        total_reference = max(
                            1,
                            len(
                                {
                                    key: value
                                    for key, value in (memory_entry.get('signals') or {}).items()
                                }
                            ),
                        )
                    if match_count >= 5:
                        bonus = 0.20 * (1 - adjusted_confidence)
                        adjusted_confidence = min(1.0, adjusted_confidence + bonus)
                        logging.info(
                            f"✅ Strong match ({match_count}/{total_reference}) → full boost applied"
                        )
                    elif match_count >= 3:
                        bonus = 0.10 * (1 - adjusted_confidence)
                        adjusted_confidence = min(0.80, adjusted_confidence + bonus)
                        logging.info(
                            f"⚠️ Partial match ({match_count}/{total_reference}) → confidence capped at 0.80"
                        )
                    else:
                        logging.info(
                            f"❌ Weak match ({match_count}/{total_reference}) → pattern ignored"
                        )
            confidence_value = float(adjusted_confidence)
            evaluation['final_confidence'] = confidence_value
        pattern_key = build_learning_pattern_key(
            symbol,
            final_action,
            latest_rsi_value,
            ema_spread_value,
        )
        current_vol_reference = float(getattr(self, "current_volatility", 0.0) or 0.0)
        bias_entry = self.learning_memory.get(pattern_key)
        if isinstance(bias_entry, dict):
            timestamp_value = float(bias_entry.get("timestamp", 0.0) or 0.0)
            age_seconds = time.time() - timestamp_value if timestamp_value else float("inf")
            result_flag = str(bias_entry.get("result", "")).upper()
            if result_flag == "LOSS":
                logging.info("❌ LOSS pattern ignored.")
            elif result_flag == "WIN":
                stored_signals = bias_entry.get("signals")
                stored_rsi_bias = float(
                    bias_entry.get('rsi', latest_rsi_value) or latest_rsi_value
                )
                stored_ema_bias = float(
                    bias_entry.get('ema', ema_spread_value) or ema_spread_value
                )
                stored_volatility_bias = float(
                    bias_entry.get('volatility', current_vol_reference)
                    or current_vol_reference
                )
                pattern_context = self._extract_pattern_context(
                    bias_entry,
                    stored_rsi_bias,
                    stored_ema_bias,
                    stored_volatility_bias,
                )
                context_matches = self._context_match_score(
                    pattern_context,
                    current_context,
                )
                if context_matches < 2:
                    logging.info(
                        f"❌ Context mismatch ({context_matches}/3) → bias boost skipped"
                    )
                else:
                    match_count, considered = is_similar_pattern(
                        strategy_signals_map, stored_signals
                    )
                    total_reference = considered
                    if total_reference <= 0:
                        total_reference = sum(
                            1 for value in strategy_signals_map.values() if value != 'NONE'
                        )
                    if total_reference <= 0:
                        total_reference = max(
                            1,
                            len({key: value for key, value in (stored_signals or {}).items()}),
                        )
                    if match_count >= 5:
                        if age_seconds > 3600:
                            logging.info(f"🟡 Old WIN bias applied ({age_seconds:.0f}s)")
                        else:
                            logging.info(f"✅ Recent WIN bias applied ({age_seconds:.0f}s)")
                        streak = int(bias_entry.get("wins_in_a_row", 0) or 0)
                        confidence_boost = 0.2 if streak >= 2 else 0.1
                        confidence_boost = min(confidence_boost, 0.2)
                        confidence_value = min(1.0, confidence_value + confidence_boost)
                        evaluation['final_confidence'] = confidence_value
                        logging.info(
                            f"✅ Strong match ({match_count}/{total_reference}) → full boost applied"
                        )
                    elif match_count >= 3:
                        streak = int(bias_entry.get("wins_in_a_row", 0) or 0)
                        confidence_boost = 0.1 if streak >= 2 else 0.05
                        confidence_value = min(0.80, confidence_value + confidence_boost)
                        evaluation['final_confidence'] = confidence_value
                        logging.info(
                            f"⚠️ Partial match ({match_count}/{total_reference}) → confidence capped at 0.80"
                        )
                    else:
                        logging.info(
                            f"❌ Weak match ({match_count}/{total_reference}) → pattern ignored"
                        )
            else:
                logging.info("❌ LOSS pattern ignored.")
        if not self._passes_high_confidence_checks(evaluation, final_action):
            return False
        if not self._is_macro_trend_aligned(evaluation, final_action):
            return False
        self.last_volatility = current_vol_reference
        confluence_value = aligned_strategies
        required_confluence = MIN_CONFLUENCE
        regime_value = detect_regime(adx_value, bb_width_value)
        evaluation['regime'] = regime_value
        boost_settings = WEIGHT_BOOST.get(symbol_upper, {})
        if regime_value == 'TREND':
            confidence_value += boost_settings.get('trend', 0.0)
        else:
            confidence_value += boost_settings.get('range', 0.0)
        confidence_value = float(np.clip(confidence_value, 0.0, 1.0))
        combined_confidence = confidence_value
        evaluation['final_confidence'] = confidence_value
        atr_value = float(evaluation.get('atr', 0.0) or 0.0)
        if atr_value <= 0.0:
            fallback_atr = compute_atr_for_symbol(symbol)
            if fallback_atr > 0.0:
                atr_value = fallback_atr
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"❌ Cannot get symbol info for {symbol_upper}")
            return False
        point_value = float(getattr(symbol_info, 'point', 0.0) or 0.0)
        atr_pips_value = atr_value / point_value if point_value else 0.0
        decision_snapshot = (
            f"[{symbol_upper}] 🧮 Conf={confidence_value:.2f} Regime={regime_value} "
            f"ATRp={atr_pips_value:.1f}p Confluence={confluence_value} Dir={final_action}"
        )
        if confluence_value < required_confluence:
            logger.info(decision_snapshot)
            logger.info(f"❌ Skip: only {aligned_strategies}/{required_confluence} strategies aligned")
            return False
        required_confidence = 0.65
        if confidence_value < required_confidence:
            logger.info(decision_snapshot)
            logger.info(f"❌ Skip: confidence {confidence_value:.2f} < 0.65 required")
            return False
        ATR_LIMITS = {
            "XAUUSD": (30, 350),
        }
        min_atr, max_atr = ATR_LIMITS.get(symbol_upper, (2, 40))
        if not (min_atr <= atr_pips_value <= max_atr):
            logger.info(decision_snapshot)
            logger.info(f"❌ Skip {symbol} - ATR {atr_pips_value:.1f} pips not in [{min_atr}-{max_atr}]")
            return False
        logger.info(decision_snapshot)

        if STRICT_MODE_ENABLED:
            if (
                NEUTRAL_RSI_BAND[0]
                <= latest_rsi_value
                <= NEUTRAL_RSI_BAND[1]
                and confidence_value < NEUTRAL_RSI_CONF
            ):
                logging.info(
                    "🚫 Skipping trade — neutral RSI & insufficient confidence (modo estricto)."
                )
                return False
            now_ts = time.time()
            if self._post_loss_cooldown_until and now_ts < self._post_loss_cooldown_until:
                remaining = self._post_loss_cooldown_until - now_ts
                logging.info(
                    f"⏳ Skipping trade — cooldown activo tras pérdida ({remaining:.0f}s restantes, modo estricto)."
                )
                return False
            while self._trade_timestamps and now_ts - self._trade_timestamps[0] > 3600:
                self._trade_timestamps.popleft()
            if len(self._trade_timestamps) >= MAX_TRADES_PER_HOUR:
                logging.info("⛔ Máximo de operaciones por hora alcanzado (modo estricto).")
                return False
        if combined_confidence is None:
            logging.info("🚫 Trade skipped because final confidence was unavailable.")
            return False
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
        logging.info(
            f"🚀 Executing trade on {symbol} | Confidence={confidence_value:.2f} | Confluence={confluence_value}/{required_confluence} | Volatility={evaluated_volatility:.6f}"
        )
        if atr_value <= 0.0:
            atr_value = 1.0
        order_result = None
        try:
            order_result = execute_market_order(symbol, final_action, atr_value, LOT_SIZE)
            logger.info(
                f"🚀 EXECUTED MARKET ORDER {symbol} → {final_action} | Confidence={confidence_value:.2f}"
            )
        except Exception as exc:
            logging.warning(f"No se pudo enviar la orden MT5: {exc}")
            self._notify_trade_state("ready")
            return False
        if order_result is None:
            logging.warning('No se obtuvo respuesta de la orden, se reanuda el análisis.')
            self._notify_trade_state("ready")
            return False
        retcode = getattr(order_result, "retcode", None)
        if retcode != mt5.TRADE_RETCODE_DONE:
            logging.warning(f"Orden rechazada por MT5 (retcode={retcode})")
            self._notify_trade_state("ready")
            return False
        contract_id = order_result.order or order_result.deal or int(time.time())
        operation_active = True
        self.active_trade_symbol = symbol
        self._notify_trade_state('active')
        logging.info(f"🟢 Operación abierta — Ticket #{contract_id} | Volumen: {stake_amount}")
        self._trade_timestamps.append(time.time())
        trade_initiated = True
        try:
            trade_result, pnl = self._resolve_trade_result(order_result, stake_amount, signal)
            if contract_id is not None and not self._register_processed_contract(contract_id):
                logging.debug(f"Ticket #{contract_id} ya procesado, omitiendo duplicado de resultados")
                return trade_initiated
            self.risk.register_trade(pnl)
            if features is not None:
                self.ai.log_trade(features, 1 if trade_result == 'WIN' else 0)
            strategy_details = {
                name: {
                    'signal': res.signal,
                    'score': float(res.score),
                    'weight': float(STRATEGY_WEIGHTS.get(name, 1.0)),
                    'confidence': float(consensus.get('strategy_confidences', {}).get(name, 0.0)),
                }
                for name, res in results
            }
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
                'strategy_confidences': consensus.get('strategy_confidences', {}),
                'strategy_details': strategy_details,
                'contract_id': contract_id,
                'stake': stake_amount,
                'ml_probability': ml_probability,
            }
            record = TradeRecord(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                decision=signal,
                confidence=confidence_value,
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
            confidence_value = float(evaluation.get('final_confidence', combined_confidence))
            win_flag = trade_result == 'WIN'
            self.total_operations += 1
            if win_flag:
                self.win_operations += 1
            self.session_total += 1
            if win_flag:
                self.session_wins += 1
            else:
                self.session_losses += 1
            session_accuracy = (
                (self.session_wins / self.session_total) * 100.0
                if self.session_total > 0
                else 0.0
            )
            update_stats_persist(win_flag)
            if MAINTENANCE_EVERY > 0 and self.total_operations % MAINTENANCE_EVERY == 0:
                # Continuous learning active — no reset after 50 trades
                pass
            self.last_contract_id = contract_id
            self.last_symbol = symbol
            self.last_result = trade_result
            self.last_confidence = confidence_value
            if STRICT_MODE_ENABLED:
                if win_flag:
                    self._post_loss_cooldown_until = 0.0
                else:
                    self._post_loss_cooldown_until = time.time() + POST_LOSS_COOLDOWN_SEC
            bias_state = self.bias_memory.get(
                symbol,
                {"RSI": 0.0, "EMA": 0.0, "last_result": "NONE", "confidence": 0.0},
            )
            bias_state["last_result"] = trade_result
            bias_state["confidence"] = confidence_value
            self.bias_memory[symbol] = bias_state
            save_biases(self.bias_memory)
            profit_value = float(pnl)
            main_reason = str(consensus.get('main_reason', 'Sin motivo disponible'))
            reason_summary = '; '.join(reasons)
            notes = (
                f"{symbol} | RSI: {latest_rsi:.2f} | EMA spread: {ema_spread:.5f} | "
                f"Volatilidad: {volatilidad_actual:.5f} | Reason: {main_reason}"
            )
            if reason_summary:
                notes = f"{notes} | Motivos: {reason_summary}"
            ema_slope_value = float(evaluation.get('ema_diff', 0.0))
            notes = f"{notes} | EMA slope: {ema_slope_value:.5f}"
            self._append_learning_memory_entry(
                symbol,
                signal,
                latest_rsi,
                ema_spread,
                volatilidad_actual,
                trade_result,
                confidence_value,
                strategy_signals_map,
            )
            self._update_learning_weights(
                symbol,
                signal,
                latest_rsi,
                ema_spread,
                volatilidad_actual,
                trade_result,
                strategy_signals_map,
            )
            result_info = {
                "hora": datetime.now().strftime("%H:%M:%S"),
                "simbolo": symbol,
                "decision": signal,
                "confianza": confidence_value,
                "resultado": "GANADA" if win_flag else "PERDIDA",
                "pnl": profit_value,
                "nota": notes,
                "ticket": contract_id,
            }
            if win_flag:
                self._store_winner_bias(
                    symbol,
                    signal,
                    latest_rsi,
                    ema_spread,
                    volatilidad_actual,
                    evaluation.get('regime'),
                    confidence_value,
                )
            should_notify = True
            if contract_id is not None and not self._register_contract_closure(contract_id):
                logging.debug(f"Ticket {contract_id} already closed. Skipping duplicate log.")
                should_notify = False
            if should_notify:
                emoji = "✅" if win_flag else "❌"
                resultado_texto = "GANADA" if win_flag else "PERDIDA"
                logging.info(f"✅ Dynamic confidence applied: {confidence_value:.2f}")
                memory_total = len(self.learning_memory)
                message = (
                    f"{emoji} Activo: {symbol}\n"
                    f"📈 Resultado: {resultado_texto}\n"
                    f"📊 Confianza: {confidence_value:.2f}\n"
                    f"⚙️ Operaciones: {self.session_total}\n"
                    f"🎯 Precisión actual: {session_accuracy:.2f}%\n"
                    f"🧠 Sesgos totales: {memory_total}"
                )
                try:
                    send_telegram_message(''.join(message))
                except Exception:
                    logging.debug('No se pudo enviar actualización de operación por Telegram')
                self._notify_trade_result(result_info)
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
            if not self.learning_enabled:
                logging.info("Learning temporarily disabled.")
            else:
                signals_list = [outcome.signal for _, outcome in results]
                feedback = self.aprendizaje.apply_learning_feedback(
                    trade_result,
                    symbol,
                    latest_rsi,
                    ema_spread,
                    volatilidad_actual,
                    signals_list,
                    signal,
                    confidence_value,
                )
                self.last_learning_feedback = feedback
                if feedback is not None:
                    self.learning_confidences.append(feedback)
            logging.info(
                f"{record.timestamp:%Y-%m-%d %H:%M:%S} INFO: [{symbol}] {signal} @{ai_confidence:.2f} | Stake:{stake_amount:.2f} | EMA:{evaluation['ema_diff']:.2f} RSI:{latest_rsi:.2f} | Motivos: {'; '.join(reasons)}"
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
            logging.warning(f"Error al gestionar el ticket #{contract_id}: {exc}")
        finally:
            operation_active = False
            self.active_trade_symbol = None
            logging.info(RESUME_MESSAGE)
            self._notify_trade_state("ready")
        return trade_initiated

    def run(self) -> None:
        self.start_engine()
        try:
            while self.running.is_set():
                cycle_start = time.time()
                self.scan_market()
                if not self.running.is_set():
                    break
                elapsed = time.time() - cycle_start
                if elapsed > 3.0:
                    logging.warning(f"⚠️ Cycle timeout ({elapsed:.2f}s) — forcing next iteration")
                QtCore.QThread.msleep(100)
        finally:
            self.stop()

    def stop(self) -> None:
        global operation_active
        self.running.clear()
        operation_active = False
        self.active_trade_symbol = None
        self._notify_trade_state("ready")
        self.ai.shutdown()
        try:
            mt5.shutdown()
        except Exception:
            pass
        with self._processed_lock:
            self._processed_contracts.clear()
        with self._closed_lock:
            self._closed_contracts.clear()
        CSV_LOGGED_CONTRACTS.clear()


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


class BotThread(QThread):
    log_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)

    def __init__(self, engine: TradingEngine) -> None:
        super().__init__()
        self.engine = engine
        self._active = True
        self._closed_contracts: Set[int] = set()
        self.logged_contracts: Set[int] = set()
        self._result_callback = self._handle_trade_result
        self.engine.add_result_listener(self._result_callback)

    def _normalize_contract_id(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _handle_trade_result(self, data: Dict[str, Any]) -> None:
        contract_id = self._normalize_contract_id(data.get("ticket") or data.get("contract_id"))
        if contract_id is not None:
            self.logged_contracts = getattr(self, "logged_contracts", set())
            logged_contracts = self.logged_contracts
            if contract_id in logged_contracts:
                logging.debug(f"Duplicate ticket {contract_id} ignored.")
                return
            logged_contracts.add(contract_id)
            self._closed_contracts.add(contract_id)
        ticket = data.get("ticket")
        resultado = data.get("resultado", "-")
        if ticket is not None:
            mensaje = f"✅ Ticket #{ticket} {resultado}"
        else:
            mensaje = f"✅ Ticket {resultado}"
        self.log_signal.emit(mensaje)
        self.result_signal.emit(dict(data))

    def run(self) -> None:  # type: ignore[override]
        logging.info("🚀 BotThread started successfully.")
        self._active = True
        self._closed_contracts.clear()
        self.logged_contracts.clear()
        try:
            self.engine.start_engine()
        except Exception as exc:
            logging.error(f"Thread error: {exc}")
            self._active = False
            return
        try:
            while self._active and self.engine.is_running():
                start_time = time.time()
                try:
                    self.engine.scan_market()
                except Exception as exc:
                    logging.error(f"Thread error: {exc}")
                    break
                elapsed = time.time() - start_time
                if elapsed > 3.0:
                    logging.warning(f"⚠️ Cycle timeout ({elapsed:.2f}s) — forcing next iteration")
                if not self._active or not self.engine.is_running():
                    break
                QThread.msleep(100)
        finally:
            self.engine.stop()
            self.engine.remove_result_listener(self._result_callback)
            logging.info("🧩 BotThread stopped.")
            self._active = False

    def stop(self) -> None:
        self._active = False
        self.engine.stop()
        self.engine.remove_result_listener(self._result_callback)
        self._closed_contracts.clear()
        self.logged_contracts.clear()


class BotWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Bot Axi Pro Trader")
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

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)

        self.bridge = EngineBridge()
        self.bridge.trade.connect(self._on_trade)
        self.bridge.status.connect(self._on_status)
        self.bridge.summary.connect(self._on_summary)
        self.bridge.trade_state.connect(self._on_trade_state)

        logger = logging.getLogger()
        existing_handler = next((h for h in logger.handlers if isinstance(h, QtLogHandler)), None)
        if existing_handler is None:
            self.log_handler = QtLogHandler()
            self.log_handler.emitter.message.connect(self._append_log)
            logger.addHandler(self.log_handler)
        else:
            self.log_handler = existing_handler
            self.log_handler.emitter.message.connect(self._append_log)

        self.engine = TradingEngine()
        global global_engine
        global_engine = self.engine
        self.engine.add_trade_listener(lambda record, stats: self.bridge.trade.emit(record, stats))
        self.engine.add_status_listener(lambda status: self.bridge.status.emit(status))
        self.engine.add_summary_listener(lambda symbol, data: self.bridge.summary.emit(symbol, data))
        self.engine.add_trade_state_listener(lambda state: self.bridge.trade_state.emit(state))
        self.strategy_initial_state = self._load_strategy_config()
        for name, enabled in self.strategy_initial_state.items():
            self.engine.set_strategy_state(name, enabled)

        self.bot_thread: Optional[BotThread] = None
        self.latest_stats: Dict[str, float] = {
            "operations": 0.0,
            "wins": 0.0,
            "losses": 0.0,
            "pnl": 0.0,
            "daily_pnl": 0.0,
            "accuracy": 0.0,
        }
        self.strategy_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
        self.strategy_states: Dict[str, QtWidgets.QCheckBox] = {}
        self._primary_strategy_aliases: Dict[str, str] = {
            'RSI': 'RSI',
            'EMA': 'EMA Trend',
            'Bollinger': 'Bollinger Rebound',
            'ADX': 'ADX',
            'MACD': 'MACD',
            'Candle': 'Candle Momentum',
        }
        self.asset_summary_labels: Dict[str, Dict[str, QtWidgets.QLabel]] = {}
        self.auto_shutdown_active = False
        self.history_accuracy_labels: Dict[str, QtWidgets.QLabel] = {}
        self.history_global_label: Optional[QtWidgets.QLabel] = None
        self.history_status_label: Optional[QtWidgets.QLabel] = None
        self.history_list: Optional[QtWidgets.QListWidget] = None
        self.history_bias_labels: Dict[str, QtWidgets.QLabel] = {}
        self.history_prediction_label: Optional[QtWidgets.QLabel] = None

        self.logged_contracts: Set[int] = set()
        self.pending_contracts: Set[int] = set()
        self.learning_enabled = True
        self._background_initialized = False

        self._build_ui()

        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setInterval(1500)
        self.refresh_timer.timeout.connect(self._refresh_phase)
        self.refresh_timer.start()

        self.learning_timer = QtCore.QTimer(self)
        self.learning_timer.setInterval(2000)
        self.learning_timer.timeout.connect(self.update_learning_tab)
        self.learning_timer.start()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)
        self._build_general_tab()
        self._build_strategies_tab()
        self._build_asset_summary_tab()
        self._build_settings_tab()
        self._build_learning_tab()
        self._initialize_asset_summary()

    def initialize_background_services(self) -> None:
        if self._background_initialized:
            return
        self.engine.start_background_services()
        self._background_initialized = True

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
            'ADX': 'ADX → evaluar fortaleza direccional mediante +DI y -DI',
            'MACD': 'MACD → seguir momentum por cruce de línea y señal',
            'Candle Momentum': 'Candle Momentum → detectar velas de impulso dominante',
            'Pullback': 'Pullback → retroceso controlado con recuperación del RSI',
            'Range Breakout': 'Ruptura de rango → validar cierres sobre resistencia o bajo soporte',
            'Divergence': 'Divergencia → alerta cuando el RSI contradice al precio',
            'Volatility Filter': 'Volatilidad → exigir movimiento mínimo para operar',
        }
        strategy_labels = {
            'RSI': 'RSI',
            'EMA Trend': 'Tendencia EMA',
            'Bollinger Rebound': 'Rebote Bollinger',
            'ADX': 'ADX',
            'MACD': 'MACD',
            'Candle Momentum': 'Candle Momentum',
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
            alineadas = QtWidgets.QLabel(f"Estrategias alineadas: 0/{MIN_ALIGNED_STRATEGIES}")
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

    def _build_learning_tab(self) -> None:
        self.tab_learning = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_learning, "Aprendizaje")
        layout = QtWidgets.QVBoxLayout(self.tab_learning)

        self.label_operations = QtWidgets.QLabel("Operaciones: 0")
        self.label_precision = QtWidgets.QLabel("Precisión: 0.00%")
        self.label_confidence = QtWidgets.QLabel("Confianza media: 0.00")
        self.label_progress = QtWidgets.QLabel("Progreso adaptativo: 0.00%")
        self.label_mode = QtWidgets.QLabel("Modo: Pasivo")

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        layout.addWidget(self.label_operations)
        layout.addWidget(self.label_precision)
        layout.addWidget(self.label_confidence)
        layout.addWidget(self.label_progress)
        layout.addWidget(self.label_mode)
        layout.addWidget(self.progress_bar)
        self.learning_toggle = QtWidgets.QCheckBox("Habilitar aprendizaje adaptativo")
        self.learning_toggle.setChecked(True)
        self.learning_toggle.stateChanged.connect(self._on_learning_toggled)
        layout.addWidget(self.learning_toggle)
        layout.addStretch(1)

    def update_learning_tab(self) -> None:
        try:
            engine = getattr(self, "engine", None)
            if engine is None:
                return

            session_total = int(getattr(engine, "session_total", 0))
            session_wins = int(getattr(engine, "session_wins", 0))
            session_losses = int(getattr(engine, "session_losses", 0))
            precision = (
                (session_wins / session_total) * 100.0 if session_total > 0 else 0.0
            )

            recent_confidences = [
                float(record.confidence)
                for record in list(engine.trade_history)[-session_total:]
                if isinstance(record.confidence, (int, float))
            ]
            avg_conf = float(np.mean(recent_confidences)) if recent_confidences else 0.0

            progress = 0.0
            if session_total > 0:
                progress = min(100.0, (session_total / 100.0) * max(avg_conf, 0.0) * 100.0)

            if not getattr(self, "learning_enabled", True):
                mode = "Pausado"
            elif session_total > 70:
                mode = "Estable"
            elif session_total > 30:
                mode = "Aprendizaje"
            else:
                mode = "Pasivo"

            self.label_operations.setText(f"Operaciones: {session_total}")
            self.label_precision.setText(f"Precisión: {precision:.2f}%")
            self.label_confidence.setText(f"Confianza media: {avg_conf:.2f}")
            self.label_progress.setText(f"Progreso adaptativo: {progress:.2f}%")
            self.label_mode.setText(f"Modo: {mode}")
            self.progress_bar.setValue(int(progress))

            desired_state = getattr(self, "learning_enabled", True)
            if self.learning_toggle.isChecked() != desired_state:
                self.learning_toggle.blockSignals(True)
                self.learning_toggle.setChecked(desired_state)
                self.learning_toggle.blockSignals(False)
        except Exception as exc:
            print(f"[LearningTab] Update error: {exc}")
    def _on_learning_toggled(self, state: int) -> None:
        enabled = state == QtCore.Qt.Checked
        self.learning_enabled = enabled
        engine = getattr(self, "engine", None)
        if engine is not None:
            engine.set_learning_enabled(enabled)

    def start_trading(self) -> None:
        if self.bot_thread is not None and self.bot_thread.isRunning():
            return
        self.auto_shutdown_active = False
        self.engine.auto_shutdown_triggered = False
        self.logged_contracts.clear()
        self.pending_contracts.clear()
        self.bot_thread = BotThread(self.engine)
        self.bot_thread.finished.connect(self._on_thread_finished)
        self.bot_thread.log_signal.connect(self._append_log)
        self.bot_thread.result_signal.connect(self.update_result_table)
        self.bot_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Estado: Iniciando...")

    def stop_trading(self) -> None:
        if self.bot_thread is None:
            if self.auto_shutdown_active:
                self.auto_shutdown_active = False
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.status_label.setText("Estado: Detenido")
                self._on_trade_state("ready")
            return
        self.bot_thread.stop()
        self.bot_thread.wait(2000)
        self.bot_thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Estado: Detenido")
        self.auto_shutdown_active = False
        self._on_trade_state("ready")
        self.logged_contracts.clear()
        self.pending_contracts.clear()

    def _on_trade_amount_changed(self, value: float) -> None:
        self.engine.set_trade_amount(value)

    def _handle_strategy_toggle(self, name: str, state: int) -> None:
        enabled = state == QtCore.Qt.Checked
        self.engine.set_strategy_state(name, enabled)
        alias = next(
            (alias for alias, strategy_name in self._primary_strategy_aliases.items() if strategy_name == name),
            None,
        )
        if alias is not None:
            checkbox = self.strategy_states.get(alias)
            if checkbox is not None and checkbox.isChecked() != enabled:
                blocker = QtCore.QSignalBlocker(checkbox)
                checkbox.setChecked(enabled)
        checkbox = self.strategy_checkboxes.get(name)
        if checkbox is not None and checkbox.isChecked() != enabled:
            blocker = QtCore.QSignalBlocker(checkbox)
            checkbox.setChecked(enabled)
        self._save_strategy_config()

    def _update_auto_shutdown(self) -> None:
        habilitado = self.auto_shutdown_checkbox.isChecked()
        limite = self.auto_shutdown_spin.value()
        self.engine.configure_auto_shutdown(habilitado, limite)
        if not habilitado and self.auto_shutdown_active:
            self.auto_shutdown_active = False
            if self.bot_thread is None:
                self.start_button.setEnabled(True)

    def _on_kelly_toggled(self, checked: bool) -> None:
        self.engine.set_kelly_enabled(bool(checked))

    def _reset_history_data(self) -> None:
        auto_learn.reset_history()
        QtCore.QTimer.singleShot(350, self._update_history_tab)

    def _normalize_contract_id(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _find_contract_row(self, contract_id: Optional[int]) -> Optional[int]:
        if contract_id is None:
            return None
        table = getattr(self, "trade_table", None)
        if table is None:
            return None
        for index in range(table.rowCount()):
            item = table.item(index, 0)
            if item is not None:
                stored_id = item.data(QtCore.Qt.UserRole)
                if stored_id == contract_id:
                    return index
        return None

    def _trim_trade_table(self) -> None:
        table = getattr(self, "trade_table", None)
        if table is None:
            return
        while table.rowCount() > 250:
            last_row = table.rowCount() - 1
            item = table.item(last_row, 0)
            if item is not None:
                stored_id = item.data(QtCore.Qt.UserRole)
                if stored_id is not None:
                    self.logged_contracts.discard(stored_id)
                    self.pending_contracts.discard(stored_id)
            table.removeRow(last_row)

    def update_result_table(self, data: Dict[str, Any]) -> None:
        table = getattr(self, "trade_table", None)
        if table is None:
            return
        contract_id = self._normalize_contract_id(data.get("ticket") or data.get("contract_id"))
        if contract_id is not None:
            self.logged_contracts = getattr(self, "logged_contracts", set())
            logged_contracts = self.logged_contracts
            if contract_id in logged_contracts:
                logging.debug(f"Duplicate ticket {contract_id} ignored.")
                return
            logged_contracts.add(contract_id)
        target_row = self._find_contract_row(contract_id)
        if target_row is None:
            target_row = 0
            table.insertRow(target_row)
        confidence_value = float(data.get("confianza", 0.0))
        pnl_value = float(data.get("pnl", 0.0))
        entries = [
            str(data.get("hora", "")),
            str(data.get("simbolo", "")),
            str(data.get("decision", "")),
            f"{confidence_value:.2f}",
            str(data.get("resultado", "")),
            f"{pnl_value:.2f}",
        ]
        for column, text_value in enumerate(entries):
            item = QtWidgets.QTableWidgetItem(text_value)
            if column == 0 and contract_id is not None:
                item.setData(QtCore.Qt.UserRole, contract_id)
            table.setItem(target_row, column, item)
        nota_text = str(data.get("nota", ""))
        table.setItem(target_row, 6, QtWidgets.QTableWidgetItem(nota_text))
        if contract_id is not None:
            self.pending_contracts.discard(contract_id)
        if table.rowCount() > 250:
            self._trim_trade_table()


    def _on_trade(self, record: TradeRecord, stats: Dict[str, float]) -> None:
        self.latest_stats = stats
        table = self.trade_table
        contract_id = self._normalize_contract_id(record.metadata.get('contract_id') if record.metadata else None)
        if contract_id is not None:
            self.logged_contracts = getattr(self, "logged_contracts", set())
            logged_contracts = self.logged_contracts
            if contract_id in logged_contracts:
                logging.debug(f"Duplicate contract {contract_id} ignored.")
                self._update_stats_labels(stats)
                metadata = record.metadata or {}
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
                return
            logged_contracts.add(contract_id)
        target_row = self._find_contract_row(contract_id)
        if target_row is None:
            target_row = 0
            table.insertRow(target_row)
        entries = [
            record.timestamp.strftime("%H:%M:%S"),
            record.symbol,
            record.decision,
            f"{record.confidence:.2f}",
            record.result or "-",
            f"{record.pnl:.2f}",
            '; '.join(record.reasons),
        ]
        for column, text_value in enumerate(entries):
            item = QtWidgets.QTableWidgetItem(text_value)
            if column == 0 and contract_id is not None:
                item.setData(QtCore.Qt.UserRole, contract_id)
            table.setItem(target_row, column, item)
        if table.rowCount() > 250:
            self._trim_trade_table()
        self._update_stats_labels(stats)
        metadata = record.metadata or {}
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
        if contract_id is not None:
            self.pending_contracts.add(contract_id)
            self.logged_contracts.discard(contract_id)


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
        self.bot_thread = None
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
        try:
            if hasattr(self, "log_view") and self.log_view:
                self.log_view.appendPlainText(message)
                self.log_view.verticalScrollBar().setValue(
                    self.log_view.verticalScrollBar().maximum()
                )
            else:
                print(message)
        except Exception as exc:
            print(f"[LogError] {exc}: {message}")

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
        alineadas = int(data.get('aligned', 0))
        etiquetas['aligned'].setText(
            f"Estrategias alineadas: {alineadas}/{MIN_ALIGNED_STRATEGIES}"
        )
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
        for alias, strategy_name in self._primary_strategy_aliases.items():
            checkbox = self.strategy_states.get(alias)
            if checkbox is None:
                continue
            desired = states.get(strategy_name, True)
            if checkbox.isChecked() != desired:
                blocker = QtCore.QSignalBlocker(checkbox)
                checkbox.setChecked(desired)
        self._update_history_tab()
        self.update_learning_tab()

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
        if self.bot_thread is not None and self.bot_thread.isRunning():
            self.bot_thread.stop()
            self.bot_thread.wait(2000)
        logging.getLogger().removeHandler(self.log_handler)
        super().closeEvent(event)


# ===============================================================
# ENTRY POINT
# ===============================================================
def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = BotWindow()
    window.show()
    window.initialize_background_services()
    app.exec_()


if __name__ == "__main__":
    main()
