import sys
import time
import json
import threading
import os
import datetime as dt

import pandas as pd
import numpy as np
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
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor


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
INTERVAL = 60
CANDLE_COUNT = 200
SLEEP_TIME = 60
CSV_FILE = "signals_iq.csv"
JSON_MEMORY = "learning_memory.json"
IQ_EMAIL = "fornerinoalejandro031@gmail.com"
IQ_PASSWORD = "484572ale"
open_positions = {}
WIN_COUNT = 0
LOSS_COUNT = 0
MOCK_MODE = False


def connect_iq_dual(email, password, retries=2):
    """Try official iqoptionapi first, then fallback to api-iqoption-faria."""
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


print("Connecting to IQ Option...")
Iq = connect_iq_dual(IQ_EMAIL, IQ_PASSWORD)
if not Iq:
    print("⚠️ Offline mode (no data). You can enable MOCK_MODE=True to test the loop.")
else:
    print("📡 Connected to IQ Option (Practice Mode).")


def calculate_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill().ffill()


def calculate_ema(df, period):
    return df["close"].ewm(span=period, adjust=False).mean()


def calculate_macd(df):
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def calculate_bollinger(df, window=20):
    mid = df["close"].rolling(window, min_periods=window).mean()
    std = df["close"].rolling(window, min_periods=window).std()
    upper = mid + (2 * std)
    lower = mid - (2 * std)
    return upper, mid, lower


def calculate_stochastic(df, k_period=14, d_period=3):
    low_min = df["min"].rolling(k_period, min_periods=k_period).min()
    high_max = df["max"].rolling(k_period, min_periods=k_period).max()
    range_span = high_max - low_min
    stoch_k = 100 * (df["close"] - low_min) / range_span.replace(0, np.nan)
    stoch_k = stoch_k.replace([np.inf, -np.inf], np.nan)
    stoch_d = stoch_k.rolling(d_period, min_periods=d_period).mean()
    return stoch_k.bfill().ffill(), stoch_d.bfill().ffill()


def calculate_momentum(df, period=10):
    momentum = (df["close"] / df["close"].shift(period)) * 100
    return momentum.bfill().ffill()


def calculate_volatility(df, period=10):
    volatility = df["close"].pct_change().rolling(period, min_periods=period).std()
    return volatility.bfill().ffill()


def get_signal(df, active_strategies=None):
    if active_strategies is None:
        active_strategies = {
            "RSI": True,
            "EMA": True,
            "MACD": True,
            "Bollinger": True,
            "Stochastic": True,
            "Momentum": True,
            "Volatility": True,
        }
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    votes_call = 0
    votes_put = 0
    votes_detail = {}
    enabled = 0

    def register_vote(name, decision):
        nonlocal votes_call, votes_put, enabled
        if not active_strategies.get(name, True):
            votes_detail[name] = "OFF"
            return
        enabled += 1
        votes_detail[name] = decision
        if decision == "CALL":
            votes_call += 1
        elif decision == "PUT":
            votes_put += 1

    rsi_decision = "NONE"
    if latest["rsi"] < 30:
        rsi_decision = "CALL"
    elif latest["rsi"] > 70:
        rsi_decision = "PUT"
    register_vote("RSI", rsi_decision)

    ema_decision = "NONE"
    if latest["ema_fast"] > latest["ema_slow"] and previous["ema_fast"] <= previous["ema_slow"]:
        ema_decision = "CALL"
    elif latest["ema_fast"] < latest["ema_slow"] and previous["ema_fast"] >= previous["ema_slow"]:
        ema_decision = "PUT"
    register_vote("EMA", ema_decision)

    macd_decision = "NONE"
    if latest["macd"] > latest["signal_macd"]:
        macd_decision = "CALL"
    elif latest["macd"] < latest["signal_macd"]:
        macd_decision = "PUT"
    register_vote("MACD", macd_decision)

    bollinger_decision = "NONE"
    if latest["close"] <= latest["boll_lower"]:
        bollinger_decision = "CALL"
    elif latest["close"] >= latest["boll_upper"]:
        bollinger_decision = "PUT"
    register_vote("Bollinger", bollinger_decision)

    stochastic_decision = "NONE"
    if latest["stoch_k"] < 20 and latest["stoch_d"] < 20:
        stochastic_decision = "CALL"
    elif latest["stoch_k"] > 80 and latest["stoch_d"] > 80:
        stochastic_decision = "PUT"
    register_vote("Stochastic", stochastic_decision)

    momentum_decision = "NONE"
    if latest["momentum"] > 100:
        momentum_decision = "CALL"
    elif latest["momentum"] < 100:
        momentum_decision = "PUT"
    register_vote("Momentum", momentum_decision)

    volatility_decision = "NONE"
    volatility_threshold = df["volatility"].mean() * 1.5
    if latest["volatility"] > volatility_threshold:
        volatility_decision = "PUT"
    else:
        volatility_decision = "CALL"
    register_vote("Volatility", volatility_decision)

    total = max(1, enabled)
    confidence = max(votes_call, votes_put) / total
    signal = "NONE"
    if votes_call >= 3 and votes_call > votes_put:
        signal = "CALL"
    elif votes_put >= 3 and votes_put > votes_call:
        signal = "PUT"

    return signal, round(confidence, 2), votes_detail


class SignalLogger:
    def __init__(self, csv_path, json_path):
        self.csv_path = csv_path
        self.json_path = json_path
        self.lock = threading.Lock()
        self.memory = {"signals": []}
        self._prepare_files()

    def _prepare_files(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", encoding="utf-8") as file:
                file.write(
                    "datetime,symbol,signal,confidence,rsi,ema_fast,ema_slow,macd,signal_macd,boll_upper,boll_lower,stoch_k,stoch_d,momentum,volatility,result\n"
                )
        if os.path.exists(self.json_path):
            with open(self.json_path, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = {"signals": []}
        else:
            data = {"signals": []}
        if not isinstance(data, dict) or "signals" not in data:
            data = {"signals": []}
        self.memory = data
        with open(self.json_path, "w", encoding="utf-8") as file:
            json.dump(self.memory, file, indent=2)

    def append(self, entry):
        indicators = entry.get("indicators", {})
        with self.lock:
            with open(self.csv_path, "a", encoding="utf-8") as file:
                file.write(
                    ",".join(
                        [
                            entry["datetime"],
                            entry["symbol"],
                            entry["signal"],
                            f"{entry['confidence']:.2f}",
                            f"{indicators.get('rsi', 0.0):.2f}",
                            f"{indicators.get('ema_fast', 0.0):.5f}",
                            f"{indicators.get('ema_slow', 0.0):.5f}",
                            f"{indicators.get('macd', 0.0):.5f}",
                            f"{indicators.get('signal_macd', 0.0):.5f}",
                            f"{indicators.get('boll_upper', 0.0):.5f}",
                            f"{indicators.get('boll_lower', 0.0):.5f}",
                            f"{indicators.get('stoch_k', 0.0):.2f}",
                            f"{indicators.get('stoch_d', 0.0):.2f}",
                            f"{indicators.get('momentum', 0.0):.2f}",
                            f"{indicators.get('volatility', 0.0):.5f}",
                            entry.get("trade_result", "PENDING"),
                        ]
                    )
                    + "\n"
                )
            self.memory.setdefault("signals", []).append(entry)
            with open(self.json_path, "w", encoding="utf-8") as file:
                json.dump(self.memory, file, indent=2)


class Worker(threading.Thread):
    def __init__(self, iq_api, gui):
        super().__init__(daemon=True)
        self.iq = iq_api
        self.gui = gui
        self.logger = gui.logger
        self.running = True

    def run(self):
        print("[INFO] Continuous analysis loop started.")
        self.last_candle_time = {symbol: None for symbol in SYMBOLS}
        global open_positions, WIN_COUNT, LOSS_COUNT

        while self.running:
            try:
                if not MOCK_MODE:
                    if not self.iq or not hasattr(self.iq, "check_connect") or not self.iq.check_connect():
                        print("🔁 Reconnecting broker...")
                        self.iq = connect_iq_dual(IQ_EMAIL, IQ_PASSWORD)
                        if not self.iq:
                            QTimer.singleShot(0, lambda: self.gui.append_log("⚠️ Offline. Retrying in 10s"))
                            time.sleep(10)
                            continue

                for symbol in SYMBOLS:
                    if not self.running:
                        break
                    try:
                        dataframe = self._mock_df() if MOCK_MODE else self._fetch_closed_candles(symbol)
                        if dataframe is None or dataframe.empty:
                            continue

                        last_timestamp = dataframe.index[-1]
                        if self.last_candle_time[symbol] == last_timestamp:
                            continue
                        self.last_candle_time[symbol] = last_timestamp

                        dataframe["rsi"] = calculate_rsi(dataframe)
                        dataframe["ema_fast"] = calculate_ema(dataframe, 9)
                        dataframe["ema_slow"] = calculate_ema(dataframe, 21)
                        macd_line, signal_line = calculate_macd(dataframe)
                        dataframe["macd"] = macd_line
                        dataframe["signal_macd"] = signal_line
                        boll_upper, boll_mid, boll_lower = calculate_bollinger(dataframe)
                        dataframe["boll_upper"] = boll_upper
                        dataframe["boll_mid"] = boll_mid
                        dataframe["boll_lower"] = boll_lower
                        stoch_k, stoch_d = calculate_stochastic(dataframe)
                        dataframe["stoch_k"] = stoch_k
                        dataframe["stoch_d"] = stoch_d
                        dataframe["momentum"] = calculate_momentum(dataframe)
                        dataframe["volatility"] = calculate_volatility(dataframe)

                        strategies = self.gui.get_active_strategies()
                        signal_data = get_signal(dataframe, strategies)
                        if len(signal_data) == 3:
                            signal, confidence, votes = signal_data
                        else:
                            signal, confidence = signal_data
                            votes = {}
                        latest = dataframe.iloc[-1]
                        now_display = dt.datetime.now().strftime("%H:%M:%S")
                        timestamp_display = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        rsi_value = float(latest["rsi"])
                        ema_fast_value = float(latest["ema_fast"])
                        ema_slow_value = float(latest["ema_slow"])
                        close_price = float(latest["close"])

                        QTimer.singleShot(
                            0,
                            lambda s=symbol,
                            sig=signal,
                            conf=confidence,
                            r=rsi_value,
                            ef=ema_fast_value,
                            es=ema_slow_value,
                            stamp=timestamp_display: self.gui.update_table(s, sig, conf, r, ef, es, stamp),
                        )
                        QTimer.singleShot(
                            0,
                            lambda s=symbol, sig=signal, conf=confidence, display_time=now_display: self.gui.append_log(
                                f"[{display_time}] {s} → {sig} (Conf: {conf:.2f})"
                            ),
                        )

                        trade_result = "PENDING"
                        if confidence >= 0.55 and signal != "NONE" and symbol not in open_positions:
                            open_positions[symbol] = {"signal": signal, "open": close_price, "timestamp": last_timestamp}
                            QTimer.singleShot(
                                0,
                                lambda s=symbol, sig=signal, price=close_price: self.gui.append_log(
                                    f"🚀 {s} OPEN {sig} @ {price:.5f}"
                                ),
                            )

                        if symbol in open_positions:
                            position = open_positions[symbol]
                            if last_timestamp > position["timestamp"]:
                                won = (position["signal"] == "CALL" and close_price > position["open"]) or (
                                    position["signal"] == "PUT" and close_price < position["open"]
                                )
                                trade_result = "WIN" if won else "LOSS"
                                if won:
                                    WIN_COUNT += 1
                                else:
                                    LOSS_COUNT += 1
                                QTimer.singleShot(
                                    0,
                                    lambda s=symbol,
                                    result=trade_result,
                                    open_price=position["open"],
                                    final_price=close_price: self.gui.append_log(
                                        f"🎯 {s} {result} → {open_price:.5f} → {final_price:.5f}"
                                    ),
                                )
                                QTimer.singleShot(0, lambda: self.gui.update_stats(WIN_COUNT, LOSS_COUNT))
                                del open_positions[symbol]

                        entry = {
                            "datetime": timestamp_display,
                            "symbol": symbol,
                            "signal": signal,
                            "confidence": confidence,
                            "indicators": {
                                "rsi": rsi_value,
                                "ema_fast": ema_fast_value,
                                "ema_slow": ema_slow_value,
                                "macd": float(latest["macd"]),
                                "signal_macd": float(latest["signal_macd"]),
                                "boll_upper": float(latest["boll_upper"]),
                                "boll_lower": float(latest["boll_lower"]),
                                "stoch_k": float(latest["stoch_k"]),
                                "stoch_d": float(latest["stoch_d"]),
                                "momentum": float(latest["momentum"]),
                                "volatility": float(latest["volatility"]),
                                "close": close_price,
                            },
                            "votes": votes,
                            "trade_result": trade_result,
                        }
                        self.logger.append(entry)
                    except Exception as symbol_error:
                        print(f"⚠️ Step error in {symbol}: {symbol_error}")
                        QTimer.singleShot(
                            0,
                            lambda s=symbol, e=symbol_error: self.gui.append_log(f"⚠️ Step error in {s}: {e}"),
                        )
                        continue

                print("[HB] Full cycle complete ✓")
                time.sleep(SLEEP_TIME)
            except Exception as loop_error:
                print(f"💥 Loop error: {loop_error}")
                QTimer.singleShot(0, lambda e=loop_error: self.gui.append_log(f"💥 Loop error: {e}"))
                time.sleep(5)

    def _fetch_closed_candles(self, symbol):
        end = int(time.time() // INTERVAL) * INTERVAL - 1
        try:
            candles = self.iq.get_candles(symbol, INTERVAL, CANDLE_COUNT, end)
        except Exception as error:
            print(f"get_candles error on {symbol}: {error}")
            return None
        if not candles:
            return None
        dataframe = pd.DataFrame(candles)[["from", "open", "close", "min", "max"]]
        dataframe["time"] = pd.to_datetime(dataframe["from"], unit="s")
        dataframe.set_index("time", inplace=True)
        return dataframe

    def _mock_df(self, n=200):
        base_price = 1.10000
        timestamps = [
            dt.datetime.fromtimestamp(int(time.time() // INTERVAL) * INTERVAL - (n - index) * INTERVAL)
            for index in range(n)
        ]
        values = np.cumsum(np.random.randn(n) * 0.0002) + base_price
        opens = np.roll(values, 1)
        opens[0] = values[0]
        mins = np.minimum(opens, values) - 0.0001
        maxs = np.maximum(opens, values) + 0.0001
        dataframe = pd.DataFrame(
            {"open": opens, "close": values, "min": mins, "max": maxs}, index=pd.to_datetime(timestamps)
        )
        dataframe.index.name = "time"
        return dataframe

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("botIQ Ultimate v1 – Demo Mode")
        self.resize(1180, 720)
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
                border: 1px solid #1f2630;
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

        self.logger = SignalLogger(CSV_FILE, JSON_MEMORY)
        self.worker = None
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

    def _build_dashboard_tab(self):
        layout = QVBoxLayout()
        self.stats_label = QLabel("Wins: 0 | Losses: 0")
        layout.addWidget(self.stats_label)
        self.signal_table = QTableWidget(len(SYMBOLS), 6)
        self.signal_table.setHorizontalHeaderLabels([
            "Symbol",
            "Signal",
            "Confidence",
            "RSI",
            "EMA (9/21)",
            "Last Update",
        ])
        self.signal_table.verticalHeader().setVisible(False)
        self.signal_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.signal_table.setAlternatingRowColors(True)
        header = self.signal_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        for index, symbol in enumerate(SYMBOLS):
            symbol_item = QTableWidgetItem(symbol)
            symbol_item.setTextAlignment(Qt.AlignCenter)
            self.signal_table.setItem(index, 0, symbol_item)
            for column in range(1, 6):
                placeholder = QTableWidgetItem("--")
                placeholder.setTextAlignment(Qt.AlignCenter)
                self.signal_table.setItem(index, column, placeholder)
        layout.addWidget(self.signal_table)
        self.dashboard_tab.setLayout(layout)

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
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)
        self.log_tab.setLayout(layout)

    def _build_settings_tab(self):
        layout = QVBoxLayout()
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Idle")
        self.mode_label = QLabel("Mode: Mock" if MOCK_MODE else ("Mode: Practice" if Iq else "Mode: Offline"))
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

    def toggle_strategy(self, name, state):
        with self.strategy_lock:
            self.active_strategies[name] = state == Qt.Checked

    def get_active_strategies(self):
        with self.strategy_lock:
            return dict(self.active_strategies)

    def toggle_start(self, start):
        global Iq, open_positions, WIN_COUNT, LOSS_COUNT
        if start:
            if self.worker and self.worker.is_alive():
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
            self.worker = Worker(Iq, self)
            self.worker.start()
            self.status_label.setText("Status: Running")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.append_log("Analysis started.")
        else:
            if self.worker:
                self.worker.stop()
                self.worker.join()
                self.worker = None
            self.status_label.setText("Status: Idle")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.append_log("Analysis stopped.")

    def append_log(self, message):
        timestamp = dt.datetime.now().strftime("%H:%M:%S")
        formatted = message if message.startswith("[") else f"[{timestamp}] {message}"
        print(formatted)
        self.log_view.append(formatted)
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_table(self, symbol, signal, confidence, rsi, ema_fast, ema_slow, timestamp):
        if symbol not in SYMBOLS:
            return
        row = SYMBOLS.index(symbol)

        symbol_item = self.signal_table.item(row, 0)
        if symbol_item is None:
            symbol_item = QTableWidgetItem(symbol)
            symbol_item.setTextAlignment(Qt.AlignCenter)
            self.signal_table.setItem(row, 0, symbol_item)

        signal_item = QTableWidgetItem(signal)
        signal_item.setTextAlignment(Qt.AlignCenter)
        confidence_item = QTableWidgetItem(f"{confidence:.2f}")
        confidence_item.setTextAlignment(Qt.AlignCenter)
        rsi_item = QTableWidgetItem(f"{rsi:.2f}")
        rsi_item.setTextAlignment(Qt.AlignCenter)
        ema_item = QTableWidgetItem(f"{ema_fast:.5f} / {ema_slow:.5f}")
        ema_item.setTextAlignment(Qt.AlignCenter)
        time_item = QTableWidgetItem(timestamp)
        time_item.setTextAlignment(Qt.AlignCenter)

        neutral_color = QColor(14, 17, 22)
        call_color = QColor(34, 139, 34)
        put_color = QColor(178, 34, 34)
        if signal == "CALL":
            background = call_color
        elif signal == "PUT":
            background = put_color
        else:
            background = neutral_color

        for column, item in enumerate([signal_item, confidence_item, rsi_item, ema_item, time_item], start=1):
            item.setBackground(background)
            self.signal_table.setItem(row, column, item)
        symbol_item.setBackground(background)
        self.signal_table.resizeColumnsToContents()

    def update_stats(self, wins, losses):
        self.stats_label.setText(f"Wins: {wins} | Losses: {losses}")

    def closeEvent(self, event):
        self.toggle_start(False)
        event.accept()


if __name__ == "__main__":
    print("🟢 Launching GUI...")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
