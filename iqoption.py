import sys
import time
import threading
import os
import datetime as dt
import csv

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
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal


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
    sma = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return upper.bfill().ffill(), lower.bfill().ffill()


def compute_momentum(series, period=5):
    momentum = series.diff(period)
    return momentum.bfill().ffill()


def compute_volatility(series, period=20):
    returns = series.pct_change()
    volatility = returns.rolling(window=period, min_periods=period).std()
    return volatility.bfill().ffill()


def get_signal(df):
    try:
        df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()
        df["rsi"] = compute_rsi(df["close"])
        df["macd"], df["macd_signal"] = compute_macd(df["close"])
        df["upper"], df["lower"] = compute_bbands(df["close"])
        df["momentum"] = compute_momentum(df["close"])
        df["volatility"] = compute_volatility(df["close"])

        signals = []

        rsi_value = df["rsi"].iloc[-1]
        if rsi_value > 70:
            signals.append("PUT")
        elif rsi_value < 30:
            signals.append("CALL")

        ema_fast_value = df["ema_fast"].iloc[-1]
        ema_slow_value = df["ema_slow"].iloc[-1]
        if ema_fast_value > ema_slow_value:
            signals.append("CALL")
        elif ema_fast_value < ema_slow_value:
            signals.append("PUT")

        macd_value = df["macd"].iloc[-1]
        macd_signal_value = df["macd_signal"].iloc[-1]
        if macd_value > macd_signal_value:
            signals.append("CALL")
        elif macd_value < macd_signal_value:
            signals.append("PUT")

        price = df["close"].iloc[-1]
        upper_band = df["upper"].iloc[-1]
        lower_band = df["lower"].iloc[-1]
        if price >= upper_band:
            signals.append("PUT")
        elif price <= lower_band:
            signals.append("CALL")

        momentum_value = df["momentum"].iloc[-1]
        if momentum_value > 0:
            signals.append("CALL")
        elif momentum_value < 0:
            signals.append("PUT")

        volatility_value = df["volatility"].iloc[-1]
        if volatility_value < 0.0003:
            return "NONE", 0.0

        if not signals:
            return "NONE", 0.0

        call_votes = signals.count("CALL")
        put_votes = signals.count("PUT")
        total_votes = len(signals)

        if call_votes > put_votes:
            direction = "CALL"
            confidence = call_votes / total_votes
        elif put_votes > call_votes:
            direction = "PUT"
            confidence = put_votes / total_votes
        else:
            direction = "NONE"
            confidence = 0.0

        return direction, round(confidence, 2)
    except Exception as error:
        print(f"Error in get_signal: {error}")
        return "NONE", 0.0


class Worker(QThread):
    log_signal = pyqtSignal(str)
    table_signal = pyqtSignal(str, str, float, float, float, str)
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
        print(f"[{now_text}] Analysis started.")
        self.log_signal.emit(f"[{now_text}] Analysis started.")
        for symbol in SYMBOLS:
            try:
                dataframe = self._mock_df() if MOCK_MODE else self._fetch_closed_candles(symbol)
                if dataframe is None or dataframe.empty:
                    continue
                last_timestamp = dataframe.index[-1]
                if self.last_candle_time[symbol] == last_timestamp:
                    continue
                self.last_candle_time[symbol] = last_timestamp
                signal, confidence = get_signal(dataframe)
                if "rsi" not in dataframe or "ema_fast" not in dataframe:
                    continue
                rsi_value = float(dataframe["rsi"].iloc[-1])
                ema_value = float(dataframe["ema_fast"].iloc[-1])
                last_close = float(dataframe["close"].iloc[-1])
                self.table_signal.emit(symbol, signal, confidence, rsi_value, ema_value, now_text)
                self.log_signal.emit(f"[{now_text}] {symbol} → {signal} ({confidence:.2f})")
                if confidence >= MIN_CONFIDENCE and signal in {"CALL", "PUT"} and symbol not in self.open_trades:
                    self.open_trades[symbol] = {
                        "direction": signal,
                        "entry": last_close,
                        "time": dt.datetime.now(),
                    }
                    self.log_trade(symbol, signal, confidence, "OPEN", last_close, now_text)
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
        row = [timestamp, symbol, signal, round(confidence, 2), status, round(price, 5)]
        with open(CSV_FILE, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)
        self.log_signal.emit(f"{timestamp} | {symbol} | {signal} | {status} | {price:.5f}")

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
        self.setWindowTitle("botIQ Ultimate v2.1 – Demo Mode")
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
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.setup_table()
        layout.addWidget(self.table)
        self.dashboard_tab.setLayout(layout)

    def setup_table(self):
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Symbol", "Signal", "Conf.", "RSI", "EMA", "Time"])
        self.table.setRowCount(len(SYMBOLS))
        for index, symbol in enumerate(SYMBOLS):
            self.table.setItem(index, 0, QTableWidgetItem(symbol))
            self.table.setItem(index, 1, QTableWidgetItem("-"))
            self.table.setItem(index, 2, QTableWidgetItem("-"))
            self.table.setItem(index, 3, QTableWidgetItem("-"))
            self.table.setItem(index, 4, QTableWidgetItem("-"))
            self.table.setItem(index, 5, QTableWidgetItem("-"))
            for column in range(6):
                item = self.table.item(index, column)
                if item:
                    item.setTextAlignment(Qt.AlignCenter)
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
            self.worker.log_signal.connect(self.append_log)
            self.worker.table_signal.connect(self.update_table)
            self.worker.result_signal.connect(self.update_result)
            self.worker.stats_signal.connect(self.update_stats)
            self.worker.running = True
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
        formatted = message if message.startswith("[") else f"[{timestamp}] {message}"
        print(formatted)
        self.log_box.append(formatted)
        scrollbar = self.log_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_table(self, symbol, signal, confidence, rsi, ema, timestamp):
        for row in range(self.table.rowCount()):
            cell = self.table.item(row, 0)
            if cell and cell.text() == symbol:
                signal_item = self.table.item(row, 1)
                confidence_item = self.table.item(row, 2)
                rsi_item = self.table.item(row, 3)
                ema_item = self.table.item(row, 4)
                time_item = self.table.item(row, 5)
                if signal_item:
                    signal_item.setText(signal)
                if confidence_item:
                    confidence_item.setText(f"{confidence:.2f}")
                if rsi_item:
                    rsi_item.setText(f"{rsi:.2f}")
                if ema_item:
                    ema_item.setText(f"{ema:.5f}")
                if time_item:
                    time_item.setText(timestamp)
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
        event.accept()


if __name__ == "__main__":
    print("🟢 Launching GUI...")
    ensure_csv_header()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
