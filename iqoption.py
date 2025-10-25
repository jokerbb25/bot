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
)
from PyQt5.QtCore import QTimer, Qt


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

Iq = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
Iq.connect()
Iq.change_balance("PRACTICE")


def calculate_rsi(data, period=14):
    close = data["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(to_replace=0, value=np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill")


def calculate_ema(data, period):
    return data["close"].ewm(span=period, adjust=False).mean()


def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = data["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger(data, period=20, std_dev=2):
    middle_band = data["close"].rolling(window=period, min_periods=period).mean()
    rolling_std = data["close"].rolling(window=period, min_periods=period).std()
    upper_band = middle_band + std_dev * rolling_std
    lower_band = middle_band - std_dev * rolling_std
    return middle_band, upper_band, lower_band


def calculate_stochastic(data, k_period=14, d_period=3):
    lowest_low = data["low"].rolling(window=k_period, min_periods=k_period).min()
    highest_high = data["high"].rolling(window=k_period, min_periods=k_period).max()
    stoch_k = ((data["close"] - lowest_low) / (highest_high - lowest_low)) * 100
    stoch_k = stoch_k.clip(lower=0, upper=100)
    stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()
    return stoch_k.fillna(method="bfill"), stoch_d.fillna(method="bfill")


def calculate_momentum(data, period=10):
    return data["close"].diff(period)


def calculate_volatility(data, period=14):
    returns = data["close"].pct_change()
    volatility = returns.rolling(window=period, min_periods=period).std() * np.sqrt(period)
    return volatility.fillna(method="bfill")


def get_signal(df, active_strategies):
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    strategy_votes = {}

    if active_strategies.get("RSI", True):
        if latest["rsi"] < 30:
            strategy_votes["RSI"] = "CALL"
        elif latest["rsi"] > 70:
            strategy_votes["RSI"] = "PUT"
        else:
            strategy_votes["RSI"] = "NONE"

    if active_strategies.get("EMA", True):
        if latest["ema_fast"] > latest["ema_slow"] and previous["ema_fast"] <= previous["ema_slow"]:
            strategy_votes["EMA"] = "CALL"
        elif latest["ema_fast"] < latest["ema_slow"] and previous["ema_fast"] >= previous["ema_slow"]:
            strategy_votes["EMA"] = "PUT"
        else:
            strategy_votes["EMA"] = "NONE"

    if active_strategies.get("MACD", True):
        if latest["macd"] > latest["macd_signal"] and previous["macd"] <= previous["macd_signal"]:
            strategy_votes["MACD"] = "CALL"
        elif latest["macd"] < latest["macd_signal"] and previous["macd"] >= previous["macd_signal"]:
            strategy_votes["MACD"] = "PUT"
        else:
            strategy_votes["MACD"] = "NONE"

    if active_strategies.get("Bollinger", True):
        if latest["close"] <= latest["bb_lower"]:
            strategy_votes["Bollinger"] = "CALL"
        elif latest["close"] >= latest["bb_upper"]:
            strategy_votes["Bollinger"] = "PUT"
        else:
            strategy_votes["Bollinger"] = "NONE"

    if active_strategies.get("Stochastic", True):
        if latest["stoch_k"] > latest["stoch_d"] and previous["stoch_k"] <= previous["stoch_d"]:
            strategy_votes["Stochastic"] = "CALL"
        elif latest["stoch_k"] < latest["stoch_d"] and previous["stoch_k"] >= previous["stoch_d"]:
            strategy_votes["Stochastic"] = "PUT"
        else:
            strategy_votes["Stochastic"] = "NONE"

    if active_strategies.get("Momentum", True):
        if latest["momentum"] > 0:
            strategy_votes["Momentum"] = "CALL"
        elif latest["momentum"] < 0:
            strategy_votes["Momentum"] = "PUT"
        else:
            strategy_votes["Momentum"] = "NONE"

    if active_strategies.get("Volatility", True):
        volatility_threshold = df["volatility"].rolling(window=20, min_periods=20).mean().iloc[-1]
        if latest["volatility"] >= volatility_threshold and latest["close"] > latest["ema_slow"]:
            strategy_votes["Volatility"] = "CALL"
        elif latest["volatility"] >= volatility_threshold and latest["close"] < latest["ema_slow"]:
            strategy_votes["Volatility"] = "PUT"
        else:
            strategy_votes["Volatility"] = "NONE"

    active_count = sum(1 for key, enabled in active_strategies.items() if enabled)
    call_votes = sum(1 for vote in strategy_votes.values() if vote == "CALL")
    put_votes = sum(1 for vote in strategy_votes.values() if vote == "PUT")

    signal = "NONE"
    aligned = 0
    if call_votes >= 3:
        signal = "CALL"
        aligned = call_votes
    elif put_votes >= 3:
        signal = "PUT"
        aligned = put_votes

    confidence = round(aligned / active_count, 2) if signal != "NONE" and active_count else 0.0

    return signal, confidence, strategy_votes


class SignalLogger:
    def __init__(self, csv_path, json_path):
        self.csv_path = csv_path
        self.json_path = json_path
        self.lock = threading.Lock()
        self.memory = []
        self._prepare_files()

    def _prepare_files(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", encoding="utf-8") as file:
                file.write(
                    "datetime,symbol,signal,confidence,rsi,ema_fast,ema_slow,macd,macd_signal,bb_upper,bb_lower,stoch_k,stoch_d,momentum,volatility\n"
                )
        if os.path.exists(self.json_path):
            with open(self.json_path, "r", encoding="utf-8") as file:
                try:
                    self.memory = json.load(file)
                except json.JSONDecodeError:
                    self.memory = []
        else:
            with open(self.json_path, "w", encoding="utf-8") as file:
                json.dump(self.memory, file, indent=2)

    def append(self, entry):
        with self.lock:
            with open(self.csv_path, "a", encoding="utf-8") as file:
                file.write(
                    ",".join(
                        [
                            entry["datetime"],
                            entry["symbol"],
                            entry["signal"],
                            f"{entry['confidence']:.2f}",
                            f"{entry['indicators']['rsi']:.2f}",
                            f"{entry['indicators']['ema_fast']:.5f}",
                            f"{entry['indicators']['ema_slow']:.5f}",
                            f"{entry['indicators']['macd']:.5f}",
                            f"{entry['indicators']['macd_signal']:.5f}",
                            f"{entry['indicators']['bb_upper']:.5f}",
                            f"{entry['indicators']['bb_lower']:.5f}",
                            f"{entry['indicators']['stoch_k']:.2f}",
                            f"{entry['indicators']['stoch_d']:.2f}",
                            f"{entry['indicators']['momentum']:.5f}",
                            f"{entry['indicators']['volatility']:.5f}",
                        ]
                    )
                    + "\n"
                )
            self.memory.append(entry)
            with open(self.json_path, "w", encoding="utf-8") as file:
                json.dump(self.memory, file, indent=2)


class Worker(threading.Thread):
    def __init__(self, iq_api, signal_queue, logger, strategy_provider):
        super().__init__(daemon=True)
        self.iq_api = iq_api
        self.signal_queue = signal_queue
        self.logger = logger
        self.strategy_provider = strategy_provider
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                strategies = self.strategy_provider()
                end_time = time.time()
                for symbol in SYMBOLS:
                    candles = self.iq_api.get_candles(symbol, INTERVAL, CANDLE_COUNT, end_time)
                    if not candles:
                        continue
                    frame = pd.DataFrame(candles)
                    frame.rename(columns={"min": "low", "max": "high"}, inplace=True)
                    frame["time"] = pd.to_datetime(frame["from"], unit="s")
                    frame.set_index("time", inplace=True)
                    frame.sort_index(inplace=True)

                    frame["rsi"] = calculate_rsi(frame)
                    frame["ema_fast"] = calculate_ema(frame, 9)
                    frame["ema_slow"] = calculate_ema(frame, 21)
                    macd_line, signal_line, histogram = calculate_macd(frame)
                    frame["macd"] = macd_line
                    frame["macd_signal"] = signal_line
                    frame["macd_hist"] = histogram
                    bb_middle, bb_upper, bb_lower = calculate_bollinger(frame)
                    frame["bb_middle"] = bb_middle
                    frame["bb_upper"] = bb_upper
                    frame["bb_lower"] = bb_lower
                    stoch_k, stoch_d = calculate_stochastic(frame)
                    frame["stoch_k"] = stoch_k
                    frame["stoch_d"] = stoch_d
                    frame["momentum"] = calculate_momentum(frame)
                    frame["volatility"] = calculate_volatility(frame)

                    clean_frame = frame.dropna()
                    if len(clean_frame) < 5:
                        continue

                    signal, confidence, votes = get_signal(clean_frame, strategies)
                    latest = clean_frame.iloc[-1]
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
                            "macd_signal": float(latest["macd_signal"]),
                            "bb_upper": float(latest["bb_upper"]),
                            "bb_lower": float(latest["bb_lower"]),
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
                    self.signal_queue.put({"type": "update", "data": entry})

                if self.stop_event.wait(SLEEP_TIME):
                    break
            except Exception as error:
                print(f"Worker error: {error}")
                if self.stop_event.wait(5):
                    break

    def stop(self):
        self.stop_event.set()


class BotIQWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("botIQ Ultimate v1 - Analysis Mode")
        self.resize(1100, 700)

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
        self.balance_mode = "PRACTICE"

        self.tabs = QTabWidget()
        self.dashboard_tab = QWidget()
        self.strategies_tab = QWidget()
        self.log_tab = QWidget()
        self.settings_tab = QWidget()

        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.strategies_tab, "Strategies")
        self.tabs.addTab(self.log_tab, "Log")
        self.tabs.addTab(self.settings_tab, "Settings")

        self._build_dashboard_tab()
        self._build_strategies_tab()
        self._build_log_tab()
        self._build_settings_tab()

        container = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.update_timer = QTimer()
        self.update_timer.setInterval(1000)
        self.update_timer.timeout.connect(self.process_queue)
        self.update_timer.start()

    def _build_dashboard_tab(self):
        layout = QVBoxLayout()
        self.signal_table = QTableWidget(len(SYMBOLS), 7)
        self.signal_table.setHorizontalHeaderLabels(
            ["Symbol", "RSI", "EMA", "MACD", "Signal", "Confidence", "Time"]
        )
        self.signal_table.verticalHeader().setVisible(False)
        self.signal_table.setEditTriggers(QTableWidget.NoEditTriggers)
        for row, symbol in enumerate(SYMBOLS):
            self.signal_table.setItem(row, 0, QTableWidgetItem(symbol))
            for column in range(1, 7):
                self.signal_table.setItem(row, column, QTableWidgetItem("--"))
        layout.addWidget(self.signal_table)
        self.dashboard_tab.setLayout(layout)

    def _build_strategies_tab(self):
        layout = QVBoxLayout()
        info_label = QLabel("Enable or disable analysis strategies:")
        layout.addWidget(info_label)
        for strategy in self.active_strategies:
            checkbox = QCheckBox(strategy)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, name=strategy: self.toggle_strategy(name, state))
            layout.addWidget(checkbox)
        layout.addStretch()
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
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Analysis")
        self.start_button.clicked.connect(self.start_analysis)
        buttons_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_analysis)
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)

        layout.addLayout(buttons_layout)

        self.balance_button = QPushButton("Mode: PRACTICE")
        self.balance_button.clicked.connect(self.toggle_balance_mode)
        layout.addWidget(self.balance_button)

        layout.addStretch()
        self.settings_tab.setLayout(layout)

    def toggle_strategy(self, name, state):
        with self.strategy_lock:
            self.active_strategies[name] = state == Qt.Checked

    def get_active_strategies(self):
        with self.strategy_lock:
            return dict(self.active_strategies)

    def start_analysis(self):
        if self.worker and self.worker.is_alive():
            return
        if not Iq.check_connect():
            self.log_message("Connection lost. Trying to reconnect...")
            Iq.connect()
            if not Iq.check_connect():
                self.log_message("Unable to connect to IQ Option.")
                return
            Iq.change_balance(self.balance_mode)
        self.worker = Worker(Iq, self.signal_queue, self.logger, self.get_active_strategies)
        self.worker.start()
        self.status_label.setText("Status: Running")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log_message("Analysis started.")

    def stop_analysis(self):
        if self.worker:
            self.worker.stop()
            self.worker.join()
            self.worker = None
        self.status_label.setText("Status: Idle")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_message("Analysis stopped.")

    def toggle_balance_mode(self):
        self.balance_mode = "REAL" if self.balance_mode == "PRACTICE" else "PRACTICE"
        try:
            Iq.change_balance(self.balance_mode)
        except Exception as error:
            self.log_message(f"Unable to switch balance: {error}")
            self.balance_mode = "PRACTICE"
            Iq.change_balance(self.balance_mode)
        self.balance_button.setText(f"Mode: {self.balance_mode}")
        self.log_message(f"Balance mode set to {self.balance_mode}.")

    def log_message(self, message):
        timestamp = dt.datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        self.log_view.append(formatted)

    def process_queue(self):
        while not self.signal_queue.empty():
            update = self.signal_queue.get()
            if update.get("type") == "update":
                self.apply_update(update["data"])

    def apply_update(self, entry):
        symbol = entry["symbol"]
        if symbol not in SYMBOLS:
            return
        row = SYMBOLS.index(symbol)
        indicators = entry["indicators"]
        votes = entry.get("votes", {})

        self.signal_table.setItem(row, 1, QTableWidgetItem(f"{indicators['rsi']:.2f}"))
        ema_text = f"{indicators['ema_fast']:.5f} / {indicators['ema_slow']:.5f}"
        self.signal_table.setItem(row, 2, QTableWidgetItem(ema_text))
        macd_text = f"{indicators['macd']:.5f} / {indicators['macd_signal']:.5f}"
        self.signal_table.setItem(row, 3, QTableWidgetItem(macd_text))
        self.signal_table.setItem(row, 4, QTableWidgetItem(entry["signal"]))
        self.signal_table.setItem(row, 5, QTableWidgetItem(f"{entry['confidence']:.2f}"))
        self.signal_table.setItem(row, 6, QTableWidgetItem(entry["datetime"]))

        vote_summary = ", ".join(f"{key}:{value}" for key, value in votes.items())
        self.log_view.append(
            f"{entry['datetime']} | {symbol} | {entry['signal']} | Conf: {entry['confidence']:.2f} | {vote_summary}"
        )

    def closeEvent(self, event):
        self.stop_analysis()
        event.accept()


def main():
    if not Iq.check_connect():
        print("Connection to IQ Option failed. Please verify your credentials.")
        return
    app = QApplication(sys.argv)
    window = BotIQWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
