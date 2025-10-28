import sys
import time
import threading
import os
import datetime as dt
import json
import csv
import re
from queue import Queue
from threading import Lock

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QTabWidget,
    QTextEdit,
    QCheckBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHBoxLayout,
    QHeaderView,
)
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal, QMetaObject, Q_ARG, pyqtSlot

from visual_overlay import OverlayWindow


SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "EURJPY",
    "GBPJPY",
    "USDCHF",
    "EURGBP",
]
CANDLE_INTERVAL = 60
CANDLE_COUNT = 200
SLEEP_TIME = 60
TRADE_DURATION = 60
MIN_CONFIDENCE = 0.55
CSV_FILE = "signals_iq.csv"
IQ_EMAIL = "fornerinoalejandro031@gmail.com"
IQ_PASSWORD = "484572ale"
open_positions = {}
WIN_COUNT = 0
LOSS_COUNT = 0
MOCK_MODE = False
results_memory = []
visual_queue = Queue()
analysis_lock = Lock()


def fmt(value):
    return "NONE" if value is None or (isinstance(value, float) and np.isnan(value)) else round(float(value), 2)


def align_to_next_minute():
    now = dt.datetime.now()
    delay = 60 - now.second
    if delay <= 0:
        delay = 60
    print(f"[SYNC] Waiting {delay}s for next candle close...")
    time.sleep(delay)


def connect_iq_dual(email, password, retries=2):
    try:
        from iqoptionapi.stable_api import IQ_Option as IQOff
        for attempt in range(retries):
            print(f"🔌 Attempt {attempt + 1}: connecting via iqoptionapi...")
            iq_instance = IQOff(email, password)
            iq_instance.connect()
            time.sleep(2)
            if hasattr(iq_instance, "check_connect") and iq_instance.check_connect():
                try:
                    iq_instance.change_balance("PRACTICE")
                except Exception:
                    pass
                print("✅ Connected using iqoptionapi.")
                return iq_instance
            print("⚠️ iqoptionapi connect failed, retrying...")
    except Exception as error:
        print(f"❌ iqoptionapi error: {error}")

    print("🔁 Fallback: connecting via api-iqoption-faria...")
    try:
        from api_iqoption_faria.client import IQ_Option as IQFaria
        iq_instance = IQFaria(email, password)
        if hasattr(iq_instance, "check_connect") and iq_instance.check_connect():
            try:
                iq_instance.change_balance("PRACTICE")
            except Exception:
                pass
            print("✅ Connected using api-iqoption-faria.")
            return iq_instance
        print("⚠️ api-iqoption-faria connect failed.")
    except Exception as error:
        print(f"❌ api-iqoption-faria error: {error}")

    print("🚫 Could not connect with any API.")
    return None


def ensure_csv_header():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp",
                "symbol",
                "signal",
                "confidence",
                "rsi_val",
                "ema9",
                "ema21",
                "RSI",
                "EMA",
                "MACD",
                "BOLL",
                "STOCH",
                "MOM",
                "VOL",
            ])


print("Connecting to IQ Option...")
Iq = connect_iq_dual(IQ_EMAIL, IQ_PASSWORD)
if not Iq:
    print("⚠️ Offline mode (no data). You can enable MOCK_MODE=True to test the loop.")
else:
    print("📡 Connected to IQ Option (Practice Mode).")


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill().ffill()


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal


def compute_bbands(series, period=20):
    middle = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + (std * 2)
    lower = middle - (std * 2)
    return upper.bfill().ffill(), middle.bfill().ffill(), lower.bfill().ffill()


def compute_stochastic(df, k_period=14, d_period=3):
    lowest_low = df["min"].rolling(window=k_period, min_periods=k_period).min()
    highest_high = df["max"].rolling(window=k_period, min_periods=k_period).max()
    k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
    k = k.replace([np.inf, -np.inf], np.nan)
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k.bfill().ffill(), d.bfill().ffill()


def compute_momentum(series, period=5):
    momentum = series.diff(period)
    return momentum.bfill().ffill()


def compute_volatility(series, period=20):
    returns = series.pct_change()
    volatility = returns.rolling(window=period, min_periods=period).std()
    return volatility.bfill().ffill()


def get_signal(df):
    """Compute all strategies and return unified signal + confidence like botDeriv."""
    try:
        df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()
        df["rsi"] = compute_rsi(df["close"])
        df["macd"], df["macd_signal"] = compute_macd(df["close"])
        df["upper"], df["middle"], df["lower"] = compute_bbands(df["close"])
        df["stoch_k"], df["stoch_d"] = compute_stochastic(df)
        df["momentum"] = compute_momentum(df["close"])
        df["volatility"] = compute_volatility(df["close"])

        rsi = df["rsi"].iloc[-1]
        ema9 = df["ema_fast"].iloc[-1]
        ema21 = df["ema_slow"].iloc[-1]
        macd = df["macd"].iloc[-1]
        macd_signal = df["macd_signal"].iloc[-1]
        upper = df["upper"].iloc[-1]
        lower = df["lower"].iloc[-1]
        stoch_k = df["stoch_k"].iloc[-1]
        stoch_d = df["stoch_d"].iloc[-1]
        momentum = df["momentum"].iloc[-1]
        volatility = df["volatility"].iloc[-1]
        close = df["close"].iloc[-1]

        votes = []
        weights = {
            "RSI": 1.0,
            "EMA": 1.0,
            "MACD": 1.0,
            "BOLL": 0.8,
            "STOCH": 0.8,
            "MOM": 0.6,
            "VOL": 0.6,
        }

        rsi_sig = "NONE"
        if rsi < 45:
            rsi_sig = "CALL"
            votes.append(("CALL", weights["RSI"]))
        elif rsi > 55:
            rsi_sig = "PUT"
            votes.append(("PUT", weights["RSI"]))

        ema_sig = "NONE"
        if ema9 > ema21 * 0.999:
            ema_sig = "CALL"
            votes.append(("CALL", weights["EMA"]))
        elif ema9 < ema21 * 1.001:
            ema_sig = "PUT"
            votes.append(("PUT", weights["EMA"]))

        macd_sig = "NONE"
        if macd > macd_signal * 0.995:
            macd_sig = "CALL"
            votes.append(("CALL", weights["MACD"]))
        elif macd < macd_signal * 1.005:
            macd_sig = "PUT"
            votes.append(("PUT", weights["MACD"]))

        boll_sig = "NONE"
        if close <= lower * 1.001:
            boll_sig = "CALL"
            votes.append(("CALL", weights["BOLL"]))
        elif close >= upper * 0.999:
            boll_sig = "PUT"
            votes.append(("PUT", weights["BOLL"]))

        stoch_sig = "NONE"
        if stoch_k < 20 and stoch_d < 20:
            stoch_sig = "CALL"
            votes.append(("CALL", weights["STOCH"]))
        elif stoch_k > 80 and stoch_d > 80:
            stoch_sig = "PUT"
            votes.append(("PUT", weights["STOCH"]))

        mom_sig = "NONE"
        if momentum > 0:
            mom_sig = "CALL"
            votes.append(("CALL", weights["MOM"]))
        elif momentum < 0:
            mom_sig = "PUT"
            votes.append(("PUT", weights["MOM"]))

        vol_sig = "LOW" if volatility < 0.00005 else "OK"
        info = {
            "RSI": f"{rsi_sig} ({fmt(rsi)})",
            "EMA": f"{ema_sig} (9:{fmt(ema9)} / 21:{fmt(ema21)})",
            "MACD": macd_sig,
            "BOLL": boll_sig,
            "STOCH": stoch_sig,
            "MOM": mom_sig,
            "VOL": vol_sig,
            "rsi_val": rsi,
            "ema9": ema9,
            "ema21": ema21,
            "macd_val": macd,
            "macd_signal_val": macd_signal,
            "boll_upper": upper,
            "boll_lower": lower,
            "stoch_k_val": stoch_k,
            "stoch_d_val": stoch_d,
            "momentum_val": momentum,
            "volatility_val": volatility,
            "close": close,
        }

        if not votes:
            return "NONE", 0.0, info

        call_strength = sum(w for s, w in votes if s == "CALL")
        put_strength = sum(w for s, w in votes if s == "PUT")
        total_strength = call_strength + put_strength

        if total_strength > 0:
            confidence = max(call_strength, put_strength) / total_strength
            direction = "CALL" if call_strength > put_strength else "PUT"
        else:
            direction = "NONE"
            confidence = 0.0

        if confidence < 0.45:
            direction = "NONE"

        print(f"[DEBUG] Votes={votes} | CALL={call_strength:.2f} | PUT={put_strength:.2f} | CONF={confidence:.2f}")

        return direction, round(confidence, 2), info
    except Exception as error:
        print(f"Error in get_signal: {error}")
        return "NONE", 0.0, {}


class Worker(QThread):
    log_signal = pyqtSignal(str)
    table_signal = pyqtSignal(object)
    result_signal = pyqtSignal(str, str)
    stats_signal = pyqtSignal(int, int)

    def __init__(self, gui, iq_client=None):
        super().__init__()
        self.gui = gui
        self.iq = iq_client
        self.running = False
        self.backup_running = False
        self.stopped_by_request = False
        self.last_candle_time = {symbol: None for symbol in SYMBOLS}
        self.open_trades = {}
        ensure_csv_header()

    def run(self):
        print("[INFO] Continuous analysis loop started.")
        self.running = True
        self.stopped_by_request = False
        align_to_next_minute()
        while self.running:
            try:
                if analysis_lock.locked():
                    time.sleep(1)
                    continue
                with analysis_lock:
                    if not MOCK_MODE:
                        if not self.iq or not hasattr(self.iq, "check_connect") or not self.iq.check_connect():
                            print("🔁 Reconnecting broker...")
                            self.iq = connect_iq_dual(IQ_EMAIL, IQ_PASSWORD)
                            if not self.iq:
                                self.log_signal.emit("⚠️ Offline. Retrying in 10s")
                                time.sleep(10)
                                continue
                    self.analyze_all_symbols()
                    self.resolve_trades()
                print("[HB] Full cycle complete ✓")
                align_to_next_minute()
            except Exception as error:
                print(f"[ERROR] Worker loop failed: {error}")
                time.sleep(5)
        print("[INFO] Worker thread stopped.")
        # Disabled backup thread to prevent duplicate analysis cycles
        # if not self.stopped_by_request:
        #     self._start_backup_thread()

    def stop(self):
        self.stopped_by_request = True
        self.running = False
        self.backup_running = False

    def _start_backup_thread(self):
        if self.backup_running:
            return
        self.backup_running = True

        def backup_loop():
            print("[⚙️] Backup thread engaged (threading mode).")
            while self.backup_running:
                try:
                    self.analyze_all_symbols()
                    self.resolve_trades()
                    print("[HB] Backup thread heartbeat ✓")
                    align_to_next_minute()
                except Exception as error:
                    print(f"⚠️ Backup loop error: {error}")
                    time.sleep(5)

        threading.Thread(target=backup_loop, daemon=True).start()

    def analyze_all_symbols(self):
        now_text = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now_text}] Analysis started.")
        self.log_signal.emit(f"[{now_text}] Analysis started.")
        for symbol in SYMBOLS:
            self.analyze_symbol(symbol, now_text)

    def analyze_symbol(self, symbol, now_text):
        try:
            global results_memory
            df = self._mock_df() if MOCK_MODE else self._fetch_closed_candles(symbol)
            if df is None or df.empty:
                return
            last_timestamp = df.index[-1]
            if self.last_candle_time[symbol] == last_timestamp:
                return
            self.last_candle_time[symbol] = last_timestamp
            signal, confidence, info = get_signal(df)
            if not info:
                return
            payload = {
                "symbol": symbol,
                "signal": signal,
                "confidence": confidence,
                "timestamp": now_text,
                "info": info,
            }
            conf = confidence
            summary = (
                f"{now_text} | {symbol} | {signal} | Conf: {conf:.2f} | "
                f"RSI:{info['RSI']}, EMA:{info['EMA']}, "
                f"MACD:{info['MACD']}, Bollinger:{info['BOLL']}, "
                f"Stochastic:{info['STOCH']}, Momentum:{info['MOM']}, Volatility:{info['VOL']}"
            )
            self.log_signal.emit(summary)
            overlay = getattr(self.gui, "overlay", None)
            if overlay:
                try:
                    if signal in {"CALL", "PUT"}:
                        rel_x = 0.95
                        rel_y = 0.3 if signal == "CALL" else 0.7
                        cache_key = f"{symbol}-{now_text}-{signal}"
                        overlay.draw_arrow(rel_x, rel_y, signal, cache_key=cache_key)
                    else:
                        overlay.clear()
                except Exception as overlay_error:
                    print(f"[OVERLAY ERROR] {overlay_error}")
            self.table_signal.emit(payload)
            visual_queue.put({
                "symbol": symbol,
                "signal": signal,
                "confidence": conf,
                "timestamp": now_text,
                "rsi": info["rsi_val"],
                "ema9": info["ema9"],
                "ema21": info["ema21"],
            })
            with open(CSV_FILE, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    symbol, signal, confidence,
                    fmt(info["rsi_val"]), fmt(info["ema9"]), fmt(info["ema21"]),
                    info["RSI"], info["EMA"], info["MACD"],
                    info["BOLL"], info["STOCH"], info["MOM"], info["VOL"]
                ])

            if len(df) >= 3:
                last_candle = df.iloc[-2]
                prev_candle = df.iloc[-3]

                if signal == "CALL" and last_candle["close"] > prev_candle["close"]:
                    result = "WIN"
                elif signal == "PUT" and last_candle["close"] < prev_candle["close"]:
                    result = "WIN"
                elif signal == "NONE":
                    result = "-"
                else:
                    result = "LOSS"

                results_memory.append(result)
                if len(results_memory) > 50:
                    results_memory.pop(0)

                wins = results_memory.count("WIN")
                losses = results_memory.count("LOSS")
                total = wins + losses
                precision = (wins / total * 100) if total > 0 else 0.0

                print(f"📊 Precision: {precision:.2f}% | Wins: {wins} | Losses: {losses}")

            if confidence >= MIN_CONFIDENCE and signal in {"CALL", "PUT"} and symbol not in self.open_trades:
                self.open_trades[symbol] = {
                    "direction": signal,
                    "entry": info.get("close", df["close"].iloc[-1]),
                    "time": dt.datetime.now(),
                }
                self.log_trade(symbol, signal, confidence, "OPEN", info.get("close", 0.0), now_text)
        except Exception as error:
            self.log_signal.emit(f"⚠️ {symbol} error: {error}")

    def resolve_trades(self):
        global WIN_COUNT, LOSS_COUNT
        symbols_to_clear = []
        for symbol, trade in list(self.open_trades.items()):
            elapsed = (dt.datetime.now() - trade["time"]).total_seconds()
            if elapsed < TRADE_DURATION:
                continue
            dataframe = self._mock_df() if MOCK_MODE else self._fetch_closed_candles(symbol)
            if dataframe is None or dataframe.empty:
                continue
            last_close = float(dataframe["close"].iloc[-1])
            won = (trade["direction"] == "CALL" and last_close > trade["entry"]) or (
                trade["direction"] == "PUT" and last_close < trade["entry"]
            )
            result = "WIN" if won else "LOSS"
            if won:
                WIN_COUNT += 1
            else:
                LOSS_COUNT += 1
            now_text = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.result_signal.emit(symbol, result)
            self.stats_signal.emit(WIN_COUNT, LOSS_COUNT)
            self.log_trade(symbol, trade["direction"], 1.0 if won else 0.0, result, last_close, now_text)
            symbols_to_clear.append(symbol)
        for symbol in symbols_to_clear:
            del self.open_trades[symbol]

    def log_trade(self, symbol, signal, confidence, status, price, timestamp):
        message = f"{timestamp} | {symbol} | {signal} | {status} | {price:.5f}"
        self.log_signal.emit(message)

    def _fetch_closed_candles(self, symbol):
        if MOCK_MODE or not self.iq:
            return None
        end = int(time.time() // CANDLE_INTERVAL) * CANDLE_INTERVAL - 1
        try:
            candles = self.iq.get_candles(symbol, CANDLE_INTERVAL, CANDLE_COUNT, end)
        except Exception as error:
            print(f"get_candles error on {symbol}: {error}")
            return None
        if not candles:
            return None
        dataframe = pd.DataFrame(candles)[["from", "open", "close", "min", "max"]]
        dataframe["time"] = pd.to_datetime(dataframe["from"], unit="s")
        dataframe.set_index("time", inplace=True)
        return dataframe

    def _mock_df(self, length=200):
        base_price = 1.10000
        timestamps = [
            dt.datetime.fromtimestamp(int(time.time() // CANDLE_INTERVAL) * CANDLE_INTERVAL - (length - index) * CANDLE_INTERVAL)
            for index in range(length)
        ]
        values = np.cumsum(np.random.randn(length) * 0.0002) + base_price
        opens = np.roll(values, 1)
        opens[0] = values[0]
        mins = np.minimum(opens, values) - 0.0001
        maxs = np.maximum(opens, values) + 0.0001
        dataframe = pd.DataFrame(
            {"open": opens, "close": values, "min": mins, "max": maxs}, index=pd.to_datetime(timestamps)
        )
        dataframe.index.name = "time"
        return dataframe


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("botIQ Ultimate v2.1 – Demo Mode")
        self.resize(1280, 760)
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #101419;
                color: #f0f0f0;
            }
            QLabel {
                color: #f0f0f0;
                font-size: 14px;
            }
            QPushButton {
                background-color: #1f2630;
                color: #f0f0f0;
                padding: 8px 16px;
                border: 1px solid #2c3440;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2c3440;
            }
            QPushButton:disabled {
                background-color: #1a1f27;
                color: #777777;
            }
            QTabWidget::pane {
                border: 1px solid #1f2630;
            }
            QTabBar::tab {
                background: #1a1f27;
                color: #f0f0f0;
                padding: 10px 20px;
            }
            QTabBar::tab:selected {
                background: #2c3440;
            }
            QTextEdit {
                background-color: #0e1116;
                color: #b9d1f0;
                border: 1x solid #1f2630;
                font-family: Consolas, monospace;
            }
            QTableWidget {
                background-color: #0e1116;
                color: #f0f0f0;
                gridline-color: #1f2630;
                selection-background-color: #2c3440;
            }
            QHeaderView::section {
                background-color: #1a1f27;
                color: #f0f0f0;
                border: none;
                padding: 6px;
            }
            QCheckBox {
                color: #f0f0f0;
            }
            """
        )
        self.worker = None
        self._last_log_message = None
        self.strategy_lock = threading.Lock()
        self.active_strategies = {
            "RSI": True,
            "EMA": True,
            "MACD": True,
            "Bollinger": True,
            "Stochastic": True,
            "Momentum": True,
            "Volatility": True,
        }
        self.tabs = QTabWidget()
        self.dashboard_tab = QWidget()
        self.strategies_tab = QWidget()
        self.log_tab = QWidget()
        self.settings_tab = QWidget()
        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.strategies_tab, "Strategies")
        self.tabs.addTab(self.log_tab, "Log")
        self.tabs.addTab(self.settings_tab, "Settings")
        container = QWidget()
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self._build_dashboard_tab()
        self._build_strategies_tab()
        self._build_log_tab()
        self._build_settings_tab()
        self.overlay = None
        self.overlay_timer = None
        self._init_overlay()
        self.start_heartbeat()

    def _build_dashboard_tab(self):
        layout = QVBoxLayout()
        self.stats_label = QLabel("Wins: 0 | Losses: 0")
        layout.addWidget(self.stats_label)
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.setup_table()
        layout.addWidget(self.table)
        self.dashboard_tab.setLayout(layout)

    def setup_table(self):
        columns = [
            "Symbol",
            "Signal",
            "Conf.",
            "RSI",
            "EMA",
            "MACD",
            "Bollinger",
            "Stoch",
            "Momentum",
            "Volatility",
            "Time",
        ]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(SYMBOLS))
        for index, symbol in enumerate(SYMBOLS):
            for column, header in enumerate(columns):
                item_text = symbol if column == 0 else "-"
                item = QTableWidgetItem(item_text)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(index, column, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

    def _build_strategies_tab(self):
        layout = QVBoxLayout()
        info_label = QLabel("Enable or disable the analysis strategies:")
        layout.addWidget(info_label)
        for name in self.active_strategies:
            checkbox = QCheckBox(name)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, strategy=name: self.toggle_strategy(strategy, state))
            layout.addWidget(checkbox)
        layout.addStretch(1)
        self.strategies_tab.setLayout(layout)

    def _build_log_tab(self):
        layout = QVBoxLayout()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_output = self.log_box
        layout.addWidget(self.log_box)
        self.log_tab.setLayout(layout)

    def _build_settings_tab(self):
        layout = QVBoxLayout()
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Idle")
        mode_text = "Mode: Mock" if MOCK_MODE else ("Mode: Practice" if Iq else "Mode: Offline")
        self.mode_label = QLabel(mode_text)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch(1)
        status_layout.addWidget(self.mode_label)
        layout.addLayout(status_layout)
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Analysis")
        self.start_button.clicked.connect(lambda: self.toggle_start(True))
        button_layout.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(lambda: self.toggle_start(False))
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)
        layout.addStretch(1)
        self.settings_tab.setLayout(layout)
    def _init_overlay(self):
        try:
            self.overlay = OverlayWindow()
            self.overlay_timer = QTimer(self)
            self.overlay_timer.setInterval(100)
            self.overlay_timer.timeout.connect(self._service_overlay)
            self.overlay_timer.start()
        except Exception as overlay_error:
            self.overlay = None
            self.overlay_timer = None
            print(f"[OVERLAY INIT ERROR] {overlay_error}")

    def _service_overlay(self):
        if not self.overlay:
            return
        try:
            self.overlay.pump()
            if self.overlay_timer and not self.overlay.running:
                self.overlay_timer.stop()
                self.overlay = None
                self.overlay_timer = None
        except Exception as overlay_error:
            print(f"[OVERLAY ERROR] {overlay_error}")
            if self.overlay_timer:
                self.overlay_timer.stop()
            self.overlay = None
            self.overlay_timer = None


    def toggle_strategy(self, name, state):
        with self.strategy_lock:
            self.active_strategies[name] = state == Qt.Checked

    def get_active_strategies(self):
        with self.strategy_lock:
            return dict(self.active_strategies)

    def toggle_start(self, start):
        global Iq, open_positions, WIN_COUNT, LOSS_COUNT
        if start:
            if self.worker and self.worker.isRunning():
                return
            if not MOCK_MODE:
                if Iq is None:
                    self.append_log("Attempting to connect to IQ Option...")
                    Iq = connect_iq_dual(IQ_EMAIL, IQ_PASSWORD)
                    if Iq:
                        self.append_log("IQ Option connection established.")
                    else:
                        self.append_log("Running without live IQ Option data.")
                else:
                    try:
                        connected = hasattr(Iq, "check_connect") and Iq.check_connect()
                    except Exception as error:
                        connected = False
                        self.append_log(f"Connection check failed: {error}")
                    if not connected:
                        self.append_log("Connection lost. Reconnecting...")
                        Iq = connect_iq_dual(IQ_EMAIL, IQ_PASSWORD)
                        if Iq:
                            self.append_log("Reconnected to IQ Option.")
                        else:
                            self.append_log("Unable to reconnect. Running offline.")
            open_positions = {}
            WIN_COUNT = 0
            LOSS_COUNT = 0
            self.update_stats(WIN_COUNT, LOSS_COUNT)
            mode_text = "Mode: Mock" if MOCK_MODE else ("Mode: Practice" if Iq else "Mode: Offline")
            self.mode_label.setText(mode_text)
            self.worker = Worker(self, Iq)
            for signal, slot in [
                (self.worker.log_signal, self.append_log),
                (self.worker.table_signal, self.update_table),
                (self.worker.result_signal, self.update_result),
                (self.worker.stats_signal, self.update_stats),
            ]:
                try:
                    signal.disconnect(slot)
                except Exception:
                    pass
                signal.connect(slot)
            self.worker.running = True
            self.worker.start()
            self.status_label.setText("Status: Running")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        else:
            if self.worker:
                self.worker.stop()
                self.worker.wait()
                self.worker = None
            self.status_label.setText("Status: Idle")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            if self.overlay:
                self.overlay.clear()
            self.append_log("Analysis stopped.")

    def safe_log_emit(self, message):
        try:
            if hasattr(self.log_output, "appendPlainText"):
                method_name = "appendPlainText"
            elif hasattr(self.log_output, "append"):
                method_name = "append"
            else:
                print("[LOG WARNING] log_output has no valid append method.")
                return

            QMetaObject.invokeMethod(
                self.log_output,
                method_name,
                Qt.QueuedConnection,
                Q_ARG(str, message)
            )
            QMetaObject.invokeMethod(
                self,
                "_scroll_log_to_bottom",
                Qt.QueuedConnection
            )
        except Exception as error:
            print(f"[LOG ERROR] {error}")

    @pyqtSlot()
    def _scroll_log_to_bottom(self):
        """
        Smoothly scrolls the log_output widget to the bottom after each new message.
        This prevents the QMetaObject::invokeMethod error if the method was missing.
        """
        try:
            if hasattr(self, "log_output") and self.log_output:
                scrollbar = self.log_output.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
        except Exception as error:
            print(f"[LOG SCROLL ERROR] {error}")

    def append_log(self, message):
        if message == getattr(self, "_last_log_message", None):
            return
        self._last_log_message = message
        has_ts = bool(re.match(r"^(\[\d{4}-\d{2}-\d{2}|\d{4}-\d{2}-\d{2})", message))
        formatted = message if has_ts else f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(formatted)
        try:
            self.safe_log_emit(formatted)
        except Exception as error:
            print(f"[LOG ERROR] {error}")

    def update_table(self, payload):
        symbol = payload.get("symbol")
        info = payload.get("info", {})
        for row in range(self.table.rowCount()):
            cell = self.table.item(row, 0)
            if cell and cell.text() == symbol:
                updates = [
                    payload.get("signal", "-"),
                    f"{payload.get('confidence', 0.0):.2f}",
                    info.get("RSI", "NONE"),
                    info.get("EMA", "NONE"),
                    f"{fmt(info.get('macd_val'))}/{fmt(info.get('macd_signal_val'))} ({info.get('MACD', 'NONE')})",
                    info.get("BOLL", "NONE"),
                    f"{fmt(info.get('stoch_k_val'))}/{fmt(info.get('stoch_d_val'))} ({info.get('STOCH', 'NONE')})",
                    f"{fmt(info.get('momentum_val'))} ({info.get('MOM', 'NONE')})",
                    f"{fmt(info.get('volatility_val'))} ({info.get('VOL', 'NONE')})",
                    payload.get("timestamp", "-")
                ]
                for column, text in enumerate(updates, start=1):
                    item = self.table.item(row, column)
                    if not item:
                        item = QTableWidgetItem()
                        item.setTextAlignment(Qt.AlignCenter)
                        self.table.setItem(row, column, item)
                    item.setText(text)
                break

    def update_result(self, symbol, result):
        for row in range(self.table.rowCount()):
            cell = self.table.item(row, 0)
            if cell and cell.text() == symbol:
                result_item = self.table.item(row, 2)
                if result_item:
                    result_item.setText(result)
                    result_item.setForeground(Qt.green if result == "WIN" else Qt.red)
                break

    def update_stats(self, wins, losses):
        self.stats_label.setText(f"Wins: {wins} | Losses: {losses}")

    def start_heartbeat(self):
        def ping():
            if self.worker:
                print(f"[🫀] GUI alive, Worker running = {self.worker.isRunning()}")
            else:
                print("[🫀] GUI alive, Worker idle")
            QTimer.singleShot(60000, ping)

        ping()

    def closeEvent(self, event):
        self.toggle_start(False)
        if hasattr(self, "overlay_timer") and self.overlay_timer:
            self.overlay_timer.stop()
            self.overlay_timer = None
        if self.overlay:
            self.overlay.close()
            self.overlay = None
        event.accept()


if __name__ == "__main__":
    print("🟢 Launching GUI...")
    ensure_csv_header()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
