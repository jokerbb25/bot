import sys
import time
import threading
import os
import datetime as dt
import csv

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
from PyQt5.QtCore import QThread, QTimer, Qt
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
TRADE_DURATION = 60
MIN_CONFIDENCE = 0.65
CSV_FILE = "signals_iq.csv"
IQ_EMAIL = "fornerinoalejandro031@gmail.com"
IQ_PASSWORD = "484572ale"
open_positions = {}
WIN_COUNT = 0
LOSS_COUNT = 0
MOCK_MODE = False


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
            writer.writerow(["timestamp", "symbol", "signal", "confidence", "status", "price"])


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
    enabled = 0

    def register_vote(name, decision):
        nonlocal votes_call, votes_put, enabled
        if not active_strategies.get(name, True):
            return
        enabled += 1
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
    threshold = df["volatility"].mean() * 1.5
    if latest["volatility"] > threshold:
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

    return signal, round(confidence, 2)


class Worker(QThread):
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
        self.stopped_by_request = False
        self.running = True
        align_to_next_minute()
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
                self.analyze_all_symbols()
                self.resolve_trades()
                print("[HB] Full cycle complete ✓")
                align_to_next_minute()
            except Exception as error:
                print(f"⚠️ Loop error: {error}")
                time.sleep(5)
        print("[INFO] Worker thread stopped.")
        if not self.stopped_by_request:
            self._start_backup_thread()

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
        QTimer.singleShot(0, lambda text=now_text: self.gui.append_log(f"[{text}] Analysis started."))
        strategies = self.gui.get_active_strategies()
        for symbol in SYMBOLS:
            try:
                dataframe = self._mock_df() if MOCK_MODE else self._fetch_closed_candles(symbol)
                if dataframe is None or dataframe.empty:
                    continue
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
                clean_df = dataframe.dropna()
                if clean_df.empty:
                    continue
                signal, confidence = get_signal(clean_df, strategies)
                latest = clean_df.iloc[-1]
                last_close = float(latest["close"])
                rsi_value = float(latest["rsi"])
                ema_fast_value = float(latest["ema_fast"])
                ema_slow_value = float(latest["ema_slow"])
                QTimer.singleShot(
                    0,
                    lambda s=symbol,
                    sig=signal,
                    conf=confidence,
                    r=rsi_value,
                    ef=ema_fast_value,
                    es=ema_slow_value,
                    timestamp=now_text: self.gui.update_table(s, sig, conf, r, ef, es, timestamp),
                )
                if confidence >= MIN_CONFIDENCE and signal in {"CALL", "PUT"} and symbol not in self.open_trades:
                    self.open_trades[symbol] = {
                        "direction": signal,
                        "entry": last_close,
                        "time": dt.datetime.now(),
                    }
                    self.log_trade(symbol, signal, confidence, "OPEN", last_close, now_text)
            except Exception as error:
                QTimer.singleShot(0, lambda s=symbol, e=error: self.gui.append_log(f"⚠️ {s} error: {e}"))

    def resolve_trades(self):
        global WIN_COUNT, LOSS_COUNT
        symbols_to_clear = []
        for symbol, trade in self.open_trades.items():
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
            QTimer.singleShot(0, lambda s=symbol, res=result: self.gui.update_result(s, res))
            QTimer.singleShot(0, lambda: self.gui.update_stats(WIN_COUNT, LOSS_COUNT))
            self.log_trade(symbol, trade["direction"], 1.0 if won else 0.0, result, last_close, now_text)
            symbols_to_clear.append(symbol)
        for symbol in symbols_to_clear:
            del self.open_trades[symbol]

    def log_trade(self, symbol, signal, confidence, status, price, timestamp):
        row = [timestamp, symbol, signal, round(confidence, 2), status, round(price, 5)]
        with open(CSV_FILE, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)
        QTimer.singleShot(0, lambda s=symbol, sig=signal, st=status, pr=price, ts=timestamp: self.gui.append_log(
            f"{ts} | {s} | {sig} | {st} | {pr:.5f}"
        ))

    def _fetch_closed_candles(self, symbol):
        if MOCK_MODE or not self.iq:
            return None
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

    def _mock_df(self, length=200):
        base_price = 1.10000
        timestamps = [
            dt.datetime.fromtimestamp(int(time.time() // INTERVAL) * INTERVAL - (length - index) * INTERVAL)
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
        self.setWindowTitle("botIQ Ultimate v2.0 – Demo Mode")
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
        self.start_heartbeat()

    def _build_dashboard_tab(self):
        layout = QVBoxLayout()
        self.stats_label = QLabel("Wins: 0 | Losses: 0")
        layout.addWidget(self.stats_label)
        self.signal_table = QTableWidget(len(SYMBOLS), 6)
        self.signal_table.setHorizontalHeaderLabels([
            "Symbol",
            "Signal",
            "Status",
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

    def toggle_strategy(self, name, state):
        with self.strategy_lock:
            self.active_strategies[name] = state == Qt.Checked

    def get_active_strategies(self):
        with self.strategy_lock:
            return dict(self.active_strategies)

    def toggle_start(self, start):
        global Iq, WIN_COUNT, LOSS_COUNT
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
            WIN_COUNT = 0
            LOSS_COUNT = 0
            self.update_stats(WIN_COUNT, LOSS_COUNT)
            mode_text = "Mode: Mock" if MOCK_MODE else ("Mode: Practice" if Iq else "Mode: Offline")
            self.mode_label.setText(mode_text)
            self.worker = Worker(self, Iq)
            self.worker.start()
            self.status_label.setText("Status: Running")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.append_log("Analysis started.")
        else:
            if self.worker:
                self.worker.stop()
                self.worker.wait()
                self.worker = None
            self.status_label.setText("Status: Idle")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.append_log("Analysis stopped.")

    def append_log(self, message):
        timestamp = dt.datetime.now().strftime("%H:%M:%S")
        text = message if message.startswith("[") else f"[{timestamp}] {message}"
        print(text)
        self.log_view.append(text)
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_table(self, symbol, signal, confidence, rsi, ema_fast, ema_slow, timestamp):
        if symbol not in SYMBOLS:
            return
        row = SYMBOLS.index(symbol)
        signal_item = QTableWidgetItem(signal)
        signal_item.setTextAlignment(Qt.AlignCenter)
        status_item = QTableWidgetItem(f"{confidence:.2f}")
        status_item.setTextAlignment(Qt.AlignCenter)
        rsi_item = QTableWidgetItem(f"{rsi:.2f}")
        rsi_item.setTextAlignment(Qt.AlignCenter)
        ema_item = QTableWidgetItem(f"{ema_fast:.5f} / {ema_slow:.5f}")
        ema_item.setTextAlignment(Qt.AlignCenter)
        time_item = QTableWidgetItem(timestamp)
        time_item.setTextAlignment(Qt.AlignCenter)
        base_color = QColor(14, 17, 22)
        call_color = QColor(34, 139, 34)
        put_color = QColor(178, 34, 34)
        if signal == "CALL":
            background = call_color
        elif signal == "PUT":
            background = put_color
        else:
            background = base_color
        self.signal_table.item(row, 0).setBackground(background)
        self.signal_table.setItem(row, 1, signal_item)
        self.signal_table.setItem(row, 2, status_item)
        self.signal_table.setItem(row, 3, rsi_item)
        self.signal_table.setItem(row, 4, ema_item)
        self.signal_table.setItem(row, 5, time_item)
        for column in range(1, 6):
            self.signal_table.item(row, column).setBackground(background)
        self.signal_table.resizeColumnsToContents()

    def update_result(self, symbol, result):
        if symbol not in SYMBOLS:
            return
        row = SYMBOLS.index(symbol)
        item = self.signal_table.item(row, 2)
        if not item:
            item = QTableWidgetItem(result)
            item.setTextAlignment(Qt.AlignCenter)
            self.signal_table.setItem(row, 2, item)
        item.setText(result)
        if result == "WIN":
            item.setForeground(Qt.green)
        else:
            item.setForeground(Qt.red)

    def update_stats(self, wins, losses):
        self.stats_label.setText(f"Wins: {wins} | Losses: {losses}")

    def start_heartbeat(self):
        def ping():
            running = self.worker.isRunning() if self.worker else False
            print(f"[🫀] GUI alive, Worker running = {running}")
            QTimer.singleShot(60000, ping)
        QTimer.singleShot(0, ping)

    def closeEvent(self, event):
        self.toggle_start(False)
        event.accept()


if __name__ == "__main__":
    print("🟢 Launching GUI...")
    ensure_csv_header()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
