import os
import time
import logging
import csv
import requests
import pandas as pd
import numpy as np
from binance.spot import Spot
from datetime import datetime, timedelta

# ---------------- CONFIG ----------------
API_KEY = "3h9AOpQiHiaDU0QUaHOOZmAQo0D6JUz7TJ3AKcqiGD3JBTztLIjd8S1MkI6G5bCn"
API_SECRET = "0uE2fjYRHUEao44Wmfz1v70xnec3Yv1DnMiHCFftz0ejWzmPoJKqbibrkv4afZEM"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
QTY_PERCENT = 1.0
STOP_LOSS_PERCENT = 1.0
TAKE_PROFIT_PERCENT = 2.0
CSV_FILE = "operaciones_multicripto.csv"

# Telegram
TELEGRAM_TOKEN = "8300367826:AAGzaMCJRY6pzZEqzjqgzAaRUXC_19KcB60"
TELEGRAM_CHAT_ID = 8364256476

# Indicadores
RSI_PERIOD = 14
EMA_SHORT = 10
EMA_LONG = 50
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# Binance client
client = Spot(api_key=API_KEY, api_secret=API_SECRET,
              base_url="https://testnet.binance.vision")

# Variables globales
balance_virtual = 1000.0
positions = {}
historicos = {s: pd.Series(dtype=float) for s in SYMBOLS}
last_signal = {s: "HOLD" for s in SYMBOLS}
daily_operations = []

# ---------------- FUNCIONES ----------------
def enviar_telegram(mensaje):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": mensaje})
    except Exception as e:
        logging.error(f"No se pudo enviar mensaje a Telegram: {e}")

def guardar_csv(tipo, precio, cantidad, order_id, symbol):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Tipo", "Symbol", "Precio", "Cantidad", "OrderID", "Timestamp"])
        writer.writerow([tipo, symbol, precio, cantidad, order_id, time.strftime("%Y-%m-%d %H:%M:%S")])
    daily_operations.append((tipo, symbol, precio, cantidad))

def test_connection():
    try:
        client.account()
        logging.info("✅ Conexión a Binance Testnet OK")
        enviar_telegram("✅ Bot conectado a Binance Testnet")
        return True
    except Exception as e:
        logging.error(f"❌ Error de conexión: {e}")
        enviar_telegram(f"❌ Error de conexión: {e}")
        return False

# ---------------- INDICADORES ----------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta>0,0)
    loss = -delta.where(delta<0,0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0,np.nan)
    return 100 - (100 / (1 + rs.fillna(0)))

def generate_signal(series):
    if len(series) < EMA_LONG:
        return "HOLD"
    df = pd.DataFrame({"close": series})
    df["EMA_short"] = ema(df["close"], EMA_SHORT)
    df["EMA_long"] = ema(df["close"], EMA_LONG)
    df["RSI"] = rsi(df["close"], RSI_PERIOD)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last["EMA_short"]>last["EMA_long"] and prev["EMA_short"]<=prev["EMA_long"] and last["RSI"]<RSI_OVERBOUGHT:
        return "BUY"
    elif last["EMA_short"]<last["EMA_long"] and prev["EMA_short"]>=prev["EMA_long"] and last["RSI"]>RSI_OVERSOLD:
        return "SELL"
    return "HOLD"

# ---------------- RESUMEN DIARIO ----------------
def resumen_diario():
    global positions, balance_virtual, daily_operations
    mensaje = f"📊 Resumen Diario Nano Bot\nBalance virtual: {balance_virtual:.2f} USDT\n\n"
    mensaje += "Operaciones abiertas:\n"
    if positions:
        for sym, pos in positions.items():
            mensaje += f"- {sym}: {pos['side']} {pos['qty']:.6f} @ {pos['price']:.2f} | SL {pos['SL']:.2f} TP {pos['TP']:.2f}\n"
    else:
        mensaje += "- Ninguna\n"
    mensaje += "\nOperaciones del día:\n"
    if daily_operations:
        for op in daily_operations:
            mensaje += f"- {op[0]} {op[1]} {op[3]:.6f} @ {op[2]:.2f}\n"
    else:
        mensaje += "- Ninguna\n"
    enviar_telegram(mensaje)
    daily_operations = []  # Reset diario

# ---------------- BOT ----------------
def run_bot():
    global balance_virtual, positions, historicos, last_signal
    next_summary = datetime.now() + timedelta(hours=24)
    while True:
        try:
            for symbol in SYMBOLS:
                klines = client.klines(symbol, "1m", limit=100)
                closes = [float(k[4]) for k in klines]
                historicos[symbol] = pd.Series(closes)
                price = closes[-1]

                signal = generate_signal(historicos[symbol])

                if signal != last_signal[symbol] and signal != "HOLD":
                    last_signal[symbol] = signal
                    qty = (balance_virtual * QTY_PERCENT / 100) / price

                    if signal == "BUY":
                        sl = price * (1 - STOP_LOSS_PERCENT/100)
                        tp = price * (1 + TAKE_PROFIT_PERCENT/100)
                        order = client.new_order(symbol=symbol, side="BUY", type="MARKET", quantity=qty)
                        positions[symbol] = {"side":"BUY","price":price,"qty":qty,"SL":sl,"TP":tp}
                        logging.info(f"✅ COMPRA {symbol}: {qty:.6f} @ {price} | SL {sl:.2f} TP {tp:.2f}")
                        enviar_telegram(f"✅ COMPRA {symbol}: {qty:.6f} @ {price}\nSL {sl:.2f} TP {tp:.2f}")
                        guardar_csv("COMPRA", price, qty, order['orderId'], symbol)

                    elif signal == "SELL" and symbol in positions:
                        order = client.new_order(symbol=symbol, side="SELL", type="MARKET", quantity=positions[symbol]["qty"])
                        logging.info(f"✅ VENTA {symbol}: {positions[symbol]['qty']:.6f} @ {price}")
                        enviar_telegram(f"✅ VENTA {symbol}: {positions[symbol]['qty']:.6f} @ {price}")
                        guardar_csv("VENTA", price, positions[symbol]["qty"], order['orderId'], symbol)
                        del positions[symbol]

                if symbol in positions:
                    pos = positions[symbol]
                    if price <= pos["SL"] or price >= pos["TP"]:
                        order = client.new_order(symbol=symbol, side="SELL", type="MARKET", quantity=pos["qty"])
                        logging.info(f"⚡ SL/TP {symbol}: {pos['qty']:.6f} @ {price}")
                        enviar_telegram(f"⚡ SL/TP {symbol}: {pos['qty']:.6f} @ {price}")
                        guardar_csv("SL/TP", price, pos['qty'], order['orderId'], symbol)
                        del positions[symbol]

            # Enviar resumen diario si corresponde
            if datetime.now() >= next_summary:
                resumen_diario()
                next_summary = datetime.now() + timedelta(hours=24)

            time.sleep(30)

        except Exception as e:
            logging.error(f"Error en loop: {e}")
            enviar_telegram(f"❌ Error en loop: {e}")
            time.sleep(10)

# ---------------- EJECUCIÓN ----------------
if __name__ == "__main__":
    if test_connection():
        run_bot()
