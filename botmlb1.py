import sys
import time
import logging
import threading
import pandas as pd
import numpy as np
import ta
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QLabel
)
from PyQt5.QtCore import Qt, QTimer
from iqoptionapi.stable_api import IQ_Option

# ---------------- CONFIG ----------------
EMAIL = "fornerinoalejandro031@gmail.com"
PASSWORD = "484572ale"
MONTO = 1.0
EXPIRACION = 1
ESPERA_ENTRE_CICLOS = 3
CICLOS = 50
MODO = "PRACTICE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------------- CLASE GUI ----------------
class BotGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IQ Option Bot - Panel de Monitoreo")
        self.resize(1100, 550)

        layout = QVBoxLayout(self)
        self.label_status = QLabel("⏳ Iniciando conexión...", self)
        layout.addWidget(self.label_status)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([
            "Par", "RSI", "EMA Fast", "EMA Slow", "MACD", "Signal", "STK %K", "STK %D", "Señal"
        ])
        layout.addWidget(self.table)

        self.label_footer = QLabel("", self)
        layout.addWidget(self.label_footer)

        self.iq = None
        self.pares_validos = []
        self.thread = threading.Thread(target=self.start_bot, daemon=True)
        self.thread.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_footer)
        self.timer.start(1000)

    def update_footer(self):
        hora = datetime.now().strftime("%H:%M:%S")
        self.label_footer.setText(f"🕓 Última actualización: {hora}")

    def ensure_row(self, par):
        rows = self.table.rowCount()
        for r in range(rows):
            if self.table.item(r, 0) and self.table.item(r, 0).text() == par:
                return r
        self.table.insertRow(rows)
        self.table.setItem(rows, 0, QTableWidgetItem(par))
        return rows

    # 🔧 FIX QTableWidgetItem error
    def on_row_update(self, par, payload):
        r = self.ensure_row(par)
        vals = [
            f"{payload.get('rsi', 0):.1f}",
            f"{payload.get('emaf', 0):.5f}",
            f"{payload.get('emas', 0):.5f}",
            f"{payload.get('macd', 0):.5f}",
            f"{payload.get('macds', 0):.5f}",
            f"{payload.get('stk', 0):.1f}",
            f"{payload.get('std', 0):.1f}",
            f"{payload.get('signal', '-')}"
        ]
        for i, v in enumerate(vals, start=1):
            item = QTableWidgetItem(v)
            if i == 8:  # Señal
                sig = payload.get("signal", "-")
                if sig == "call":
                    item.setForeground(Qt.green)
                elif sig == "put":
                    item.setForeground(Qt.red)
            self.table.setItem(r, i, item)

    def start_bot(self):
        self.label_status.setText("🔌 Conectando a IQ Option...")
        iq = IQ_Option(EMAIL, PASSWORD)
        check, reason = iq.connect()
        if not check:
            self.label_status.setText(f"❌ Error conexión: {reason}")
            return

        iq.change_balance(MODO)
        saldo = iq.get_balance()
        self.label_status.setText(f"✅ Conectado a {MODO} | Saldo: {saldo:.2f}")

        logging.info("♻️ Escaneando pares OTC disponibles...")
        activos = iq.get_all_open_time()
        pares_disponibles = [k for k, v in activos["turbo"].items() if v["open"]]
        pares_otc = [p for p in pares_disponibles if "-OTC" in p]
        self.pares_validos = pares_otc[:20]  # analiza los primeros 20
        logging.info(f"✅ Pares OTC detectados: {self.pares_validos}")

        for ciclo in range(1, CICLOS + 1):
            logging.info(f"=== Ciclo {ciclo}/{CICLOS} ===")
            for par in self.pares_validos:
                df = self.obtener_velas(iq, par)
                if df.empty:
                    continue
                senal, data = self.obtener_senal(df)
                self.on_row_update(par, data)

                if senal:
                    ok, id = iq.buy(MONTO, par, senal, EXPIRACION)
                    if ok:
                        logging.info(f"[OK] {senal.upper()} en {par}")
                time.sleep(0.6)
            time.sleep(ESPERA_ENTRE_CICLOS)
        logging.info("✅ Bot finalizado correctamente.")

    def obtener_velas(self, iq, par, n=60):
        try:
            velas = iq.get_candles(par, 60, n, time.time())
            df = pd.DataFrame(velas)[["open", "max", "min", "close"]]
            df.columns = ["open", "high", "low", "close"]
            return df
        except Exception:
            return pd.DataFrame()

    def obtener_senal(self, df):
        close = df["close"]
        high = df["high"]
        low = df["low"]

        rsi = ta.momentum.RSIIndicator(close).rsi().iloc[-1]
        emaf = ta.trend.EMAIndicator(close, 9).ema_indicator().iloc[-1]
        emas = ta.trend.EMAIndicator(close, 21).ema_indicator().iloc[-1]
        macd = ta.trend.MACD(close).macd().iloc[-1]
        macds = ta.trend.MACD(close).macd_signal().iloc[-1]
        stoch = ta.momentum.StochasticOscillator(high, low, close)
        stk = stoch.stoch().iloc[-1]
        std = stoch.stoch_signal().iloc[-1]

        up, down = 0, 0
        if rsi < 35: up += 1
        if rsi > 65: down += 1
        if emaf > emas: up += 1
        if emaf < emas: down += 1
        if macd > macds: up += 1
        if macd < macds: down += 1
        if stk > std: up += 1
        if stk < std: down += 1

        signal = None
        if up >= 3: signal = "call"
        elif down >= 3: signal = "put"

        return signal, {
            "rsi": rsi,
            "emaf": emaf,
            "emas": emas,
            "macd": macd,
            "macds": macds,
            "stk": stk,
            "std": std,
            "signal": signal or "-"
        }


# ---------------- MAIN ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BotGUI()
    window.show()
    sys.exit(app.exec_())
