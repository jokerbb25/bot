import sys
import os
import time
import json
import threading
import datetime as dt
from queue import Queue

import pandas as pd
import numpy as np
from iqoptionapi.stable_api import IQ_Option
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
from dotenv import load_dotenv


load_dotenv()

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
IQ_EMAIL = os.environ.get("IQ_EMAIL", "YOUR_EMAIL")
IQ_PASSWORD = os.environ.get("IQ_PASSWORD", "YOUR_PASSWORD")


def connect_iq(email, password, retries=3):
    from iqoptionapi.stable_api import IQ_Option as IQClient

    for attempt in range(retries):
        try:
            print(f"🔌 Attempt {attempt + 1}: Connecting to IQ Option...")
            iq_instance = IQClient(email, password)
            iq_instance.connect()
            time.sleep(2)
            if iq_instance.check_connect():
                iq_instance.change_balance("PRACTICE")
                print("✅ Connected successfully.")
                return iq_instance
            print("⚠️ Connection failed, retrying...")
        except Exception as error:
            print(f"❌ Error: {error}")
        time.sleep(3)
    print("🚫 Could not connect after retries.")
    return None


Iq = connect_iq(IQ_EMAIL, IQ_PASSWORD)
if not Iq:
    print("❌ IQ Option not responding. GUI will still open in offline mode.")
    print("⚠️ Running in offline/demo mode (no data).")
else:
    print("✅ IQ Option connected (Practice).")
    print("📡 Connection verified, fetching candles.")


def calculate_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill").fillna(method="ffill")


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
    return stoch_k.fillna(method="bfill").fillna(method="ffill"), stoch_d.fillna(method="bfill").fillna(method="ffill")


def calculate_momentum(df, period=10):
    momentum = (df["close"] / df["close"].shift(period)) * 100
    return momentum.fillna(method="bfill").fillna(method="ffill")


def calculate_volatility(df, period=10):
    volatility = df["close"].pct_change().rolling(period, min_periods=period).std()
    return volatility.fillna(method="bfill").fillna(method="ffill")


def get_signal(df, active_strategies):
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
                    "datetime,symbol,signal,confidence,rsi,ema_fast,ema_slow,macd,signal_macd,boll_upper,boll_lower,stoch_k,stoch_d,momentum,volatility\n"
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
        self.iq_api = iq_api
        self.gui = gui
        self.signal_queue = gui.signal_queue
        self.logger = gui.logger
        self.stop_event = threading.Event()
        self.last_processed = {symbol: None for symbol in SYMBOLS}
        self.offline_logged = False

    def run(self):
        while not self.stop_event.is_set():
            if self.iq_api is None:
                if not self.offline_logged:
                    message = "IQ Option connection unavailable. Running offline."
                    print(message)
                    self.signal_queue.put(
                        {
                            "datetime": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "symbol": "SYSTEM",
                            "signal": "OFFLINE",
                            "confidence": 0.0,
                            "indicators": {},
                            "votes": {},
                            "log_only": message,
                        }
                    )
                    self.offline_logged = True
                if self.stop_event.wait(SLEEP_TIME):
                    break
                continue

            try:
                connected = self.iq_api.check_connect()
            except Exception as error:
                print(f"⚠️ Connection check failed: {error}")
                connected = False

            if not connected:
                if not self.offline_logged:
                    warning = "IQ Option connection lost. Waiting to reconnect..."
                    print(warning)
                    self.signal_queue.put(
                        {
                            "datetime": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "symbol": "SYSTEM",
                            "signal": "DISCONNECTED",
                            "confidence": 0.0,
                            "indicators": {},
                            "votes": {},
                            "log_only": warning,
                        }
                    )
                    self.offline_logged = True
                if self.stop_event.wait(5):
                    break
                continue

            self.offline_logged = False
            cycle_start = time.time()
            strategies = self.gui.get_active_strategies()
            for symbol in SYMBOLS:
                if self.stop_event.is_set():
                    break
                try:
                    end_timestamp = time.time()
                    candles = self.iq_api.get_candles(symbol, INTERVAL, CANDLE_COUNT, end_timestamp)
                    if not candles:
                        continue
                    last_candle_time = int(candles[-1]["from"])
                    if self.last_processed.get(symbol) == last_candle_time:
                        continue
                    df = pd.DataFrame(candles)
                    df["time"] = pd.to_datetime(df["from"], unit="s")
                    df.set_index("time", inplace=True)
                    df.sort_index(inplace=True)

                    df["rsi"] = calculate_rsi(df)
                    df["ema_fast"] = calculate_ema(df, 9)
                    df["ema_slow"] = calculate_ema(df, 21)
                    macd_line, signal_line = calculate_macd(df)
                    df["macd"] = macd_line
                    df["signal_macd"] = signal_line
                    boll_upper, boll_mid, boll_lower = calculate_bollinger(df)
                    df["boll_upper"] = boll_upper
                    df["boll_mid"] = boll_mid
                    df["boll_lower"] = boll_lower
                    stoch_k, stoch_d = calculate_stochastic(df)
                    df["stoch_k"] = stoch_k
                    df["stoch_d"] = stoch_d
                    df["momentum"] = calculate_momentum(df)
                    df["volatility"] = calculate_volatility(df)

                    analysis_df = df.dropna().copy()
                    if len(analysis_df) < 2:
                        continue

                    signal, confidence, votes = get_signal(analysis_df, strategies)
                    latest = analysis_df.iloc[-1]
                    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    entry = {
                        "datetime": timestamp,
                        "symbol": symbol,
                        "signal": signal,
                        "confidence": confidence,
                        "indicators": {
                            "rsi": float(latest["rsi"]),
                            "ema_fast": float(latest["ema_fast"]),
                            "ema_slow": float(latest["ema_slow"]),
                            "macd": float(latest["macd"]),
                            "signal_macd": float(latest["signal_macd"]),
                            "boll_upper": float(latest["boll_upper"]),
                            "boll_lower": float(latest["boll_lower"]),
                            "stoch_k": float(latest["stoch_k"]),
                            "stoch_d": float(latest["stoch_d"]),
                            "momentum": float(latest["momentum"]),
                            "volatility": float(latest["volatility"]),
                            "close": float(latest["close"]),
                        },
                        "votes": votes,
                    }

                    log_message = f"[{timestamp}] {symbol} → {signal} (Conf: {confidence:.2f})"
                    print(log_message)
                    self.logger.append(entry)
                    self.signal_queue.put(entry)
                    self.last_processed[symbol] = last_candle_time
                except Exception as exc:
                    error_message = f"⚠️ {symbol} error: {exc}"
                    print(error_message)
                    self.signal_queue.put(
                        {
                            "datetime": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "symbol": symbol,
                            "signal": "ERROR",
                            "confidence": 0.0,
                            "indicators": {},
                            "votes": {"Error": str(exc)},
                            "log_only": error_message,
                        }
                    )
                    continue
            elapsed = time.time() - cycle_start
            remaining = max(0, SLEEP_TIME - elapsed)
            if self.stop_event.wait(remaining):
                break

    def stop(self):
        self.stop_event.set()


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

        self.signal_queue = Queue()
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

        self.update_timer = QTimer()
        self.update_timer.setInterval(500)
        self.update_timer.timeout.connect(self.process_queue)
        self.update_timer.start()

    def _build_dashboard_tab(self):
        layout = QVBoxLayout()
        self.signal_table = QTableWidget(len(SYMBOLS), 6)
        self.signal_table.setHorizontalHeaderLabels(
            ["Symbol", "Signal", "Confidence", "RSI", "EMA (9/21)", "Last Update"]
        )
        self.signal_table.verticalHeader().setVisible(False)
        self.signal_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.signal_table.setAlternatingRowColors(True)
        header = self.signal_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        for row, symbol in enumerate(SYMBOLS):
            symbol_item = QTableWidgetItem(symbol)
            symbol_item.setTextAlignment(Qt.AlignCenter)
            self.signal_table.setItem(row, 0, symbol_item)
            for col in range(1, 6):
                placeholder = QTableWidgetItem("--")
                placeholder.setTextAlignment(Qt.AlignCenter)
                self.signal_table.setItem(row, col, placeholder)
        layout.addWidget(self.signal_table)
        self.dashboard_tab.setLayout(layout)

    def _build_strategies_tab(self):
        layout = QVBoxLayout()
        info = QLabel("Enable or disable the analysis strategies:")
        layout.addWidget(info)
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
        self.mode_label = QLabel("Mode: Practice")
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
        global Iq
        if start:
            if self.worker and self.worker.is_alive():
                return
            if Iq is None:
                self.append_log("Attempting to connect to IQ Option...")
                Iq = connect_iq(IQ_EMAIL, IQ_PASSWORD)
                if Iq:
                    self.append_log("IQ Option connection restored.")
                else:
                    self.append_log("Running without live IQ Option data.")
            else:
                try:
                    connected = Iq.check_connect()
                except Exception as error:
                    connected = False
                    self.append_log(f"Connection check failed: {error}")
                if not connected:
                    self.append_log("Connection lost. Reconnecting...")
                    reconnected = connect_iq(IQ_EMAIL, IQ_PASSWORD)
                    if reconnected:
                        Iq = reconnected
                        self.append_log("Reconnected to IQ Option.")
                    else:
                        self.append_log("Unable to reconnect. Running offline.")
            mode_text = "Mode: Practice" if Iq else "Mode: Offline"
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
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        self.log_view.append(formatted)
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def process_queue(self):
        while not self.signal_queue.empty():
            entry = self.signal_queue.get()
            if entry.get("log_only"):
                self.append_log(entry["log_only"])
                continue
            self.update_table(entry)
            votes_summary = ", ".join(f"{key}:{value}" for key, value in entry.get("votes", {}).items())
            log_text = f"{entry['datetime']} | {entry['symbol']} | {entry['signal']} | Conf: {entry['confidence']:.2f}"
            if votes_summary:
                log_text += f" | {votes_summary}"
            self.log_view.append(log_text)
            scrollbar = self.log_view.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def update_table(self, entry):
        symbol = entry["symbol"]
        if symbol not in SYMBOLS:
            return
        row = SYMBOLS.index(symbol)
        indicators = entry.get("indicators", {})

        symbol_item = self.signal_table.item(row, 0)
        if symbol_item is None:
            symbol_item = QTableWidgetItem(symbol)
            symbol_item.setTextAlignment(Qt.AlignCenter)
            self.signal_table.setItem(row, 0, symbol_item)

        signal_item = QTableWidgetItem(entry["signal"])
        signal_item.setTextAlignment(Qt.AlignCenter)
        confidence_item = QTableWidgetItem(f"{entry['confidence']:.2f}")
        confidence_item.setTextAlignment(Qt.AlignCenter)
        rsi_item = QTableWidgetItem(f"{indicators.get('rsi', 0.0):.2f}")
        rsi_item.setTextAlignment(Qt.AlignCenter)
        ema_item = QTableWidgetItem(
            f"{indicators.get('ema_fast', 0.0):.5f} / {indicators.get('ema_slow', 0.0):.5f}"
        )
        ema_item.setTextAlignment(Qt.AlignCenter)
        time_item = QTableWidgetItem(entry["datetime"])
        time_item.setTextAlignment(Qt.AlignCenter)

        neutral_color = QColor(14, 17, 22)
        call_color = QColor(34, 139, 34)
        put_color = QColor(178, 34, 34)
        if entry["signal"] == "CALL":
            background = call_color
        elif entry["signal"] == "PUT":
            background = put_color
        else:
            background = neutral_color

        for col, item in enumerate(
            [signal_item, confidence_item, rsi_item, ema_item, time_item],
            start=1,
        ):
            item.setBackground(background)
            self.signal_table.setItem(row, col, item)
        symbol_item.setBackground(background)
        self.signal_table.resizeColumnsToContents()

    def closeEvent(self, event):
        self.toggle_start(False)
        event.accept()


if __name__ == "__main__":
    print("🟢 Launching GUI...")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
