# ============================================================
#  BOT ANALIZADOR IQ OPTION – RSI + EMA + REGISTRO CSV
# ============================================================

from iqoptionapi.stable_api import IQ_Option
import pandas as pd
import numpy as np
import time
import datetime as dt
import os

# === CONFIGURACIÓN ===
EMAIL = "fornerinoalejandro031@gmail.com"
PASSWORD = "484572ale"

PAIR = "EURUSD"           # Activo
INTERVAL = 60             # Duración de vela en segundos (60 = 1 min)
CANDLE_COUNT = 200        # Número de velas a analizar
RSI_PERIOD = 14
EMA_FAST = 9
EMA_SLOW = 21
SLEEP_TIME = 60           # Espera entre análisis (segundos)
CSV_FILE = "signals_iq.csv"

# === CONEXIÓN ===
Iq = IQ_Option(EMAIL, PASSWORD)
Iq.connect()
Iq.change_balance("PRACTICE")

if not Iq.check_connect():
    print("❌ No se pudo conectar a IQ Option.")
    exit()

print(f"✅ Conectado correctamente a IQ Option\n🔹 Analizando: {PAIR}\n")

# === FUNCIONES ===
def calculate_rsi(data, period=14):
    delta = data["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(data, period):
    return data["close"].ewm(span=period, adjust=False).mean()

def get_signal(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    signal = "NONE"
    confidence = 0

    # RSI
    if latest["rsi"] < 30:
        signal = "CALL"
        confidence += 0.5
    elif latest["rsi"] > 70:
        signal = "PUT"
        confidence += 0.5

    # EMA cross
    if latest["ema_fast"] > latest["ema_slow"] and prev["ema_fast"] <= prev["ema_slow"]:
        signal = "CALL"
        confidence += 0.5
    elif latest["ema_fast"] < latest["ema_slow"] and prev["ema_fast"] >= prev["ema_slow"]:
        signal = "PUT"
        confidence += 0.5

    return signal, round(confidence, 2), latest

# === CREAR CSV SI NO EXISTE ===
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w") as f:
        f.write("datetime,pair,signal,confidence,rsi,ema_fast,ema_slow\n")

# === LOOP PRINCIPAL ===
while True:
    try:
        end = time.time()
        candles = Iq.get_candles(PAIR, INTERVAL, CANDLE_COUNT, end)
        df = pd.DataFrame(candles)[["from", "open", "close", "min", "max"]]
        df["time"] = pd.to_datetime(df["from"], unit="s")
        df.set_index("time", inplace=True)

        df["rsi"] = calculate_rsi(df, RSI_PERIOD)
        df["ema_fast"] = calculate_ema(df, EMA_FAST)
        df["ema_slow"] = calculate_ema(df, EMA_SLOW)

        signal, confidence, latest = get_signal(df)

        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] {PAIR} → Señal: {signal} | Confianza: {confidence}")
        print(f"RSI: {latest['rsi']:.2f} | EMA9: {latest['ema_fast']:.5f} | EMA21: {latest['ema_slow']:.5f}")

        # Guardar en CSV
        with open(CSV_FILE, "a") as f:
            f.write(f"{now},{PAIR},{signal},{confidence},{latest['rsi']:.2f},{latest['ema_fast']:.5f},{latest['ema_slow']:.5f}\n")

        time.sleep(SLEEP_TIME)

    except Exception as e:
        print("⚠️ Error:", e)
        time.sleep(5)
