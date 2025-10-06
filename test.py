import time
import logging
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import ta
from iqoptionapi.stable_api import IQ_Option

# ========= CONFIG =========
EMAIL = "fornerinoalejandro031@gmail.com"
PASSWORD = "484572ale"

MODO_INPUT = None  # pide d/r
MONTO = 1.0
EXPIRACION_MIN = 1
CICLOS = 50
ESPERA_ENTRE_CICLOS = 3
PAUSA_ENTRE_TRADES = 1.0  # s

# Risk controls
STOP_LOSS_DIARIO = -30.0     # cierre si PnL <= -30
TAKE_PROFIT_DIARIO = +30.0   # cierre si PnL >= +30
ESPERAR_RESULTADO = True     # marcar WIN/LOSS y PnL por operación

# Indicadores
RSI_PERIOD = 14
EMA_FAST = 9
EMA_SLOW = 21
STOCH_PERIOD = 14
STOCH_SMOOTH = 3

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Universo inicial (FOREX OTC típicos). Luego se filtran por velas reales.
PARES_INICIALES = [
    "EURUSD-OTC", "GBPUSD-OTC", "USDCHF-OTC", "NZDUSD-OTC",
    "USDJPY-OTC", "GBPJPY-OTC", "EURJPY-OTC", "EURGBP-OTC",
    "AUDCAD-OTC", "AUDUSD-OTC", "CADJPY-OTC", "EURAUD-OTC"
]

CSV_PATH = "operaciones.csv"


# ========= UTIL =========
def elegir_modo():
    global MODO_INPUT
    while True:
        x = input("Operar en demo o real? (d/r): ").lower().strip()
        if x in ("d", "r"):
            MODO_INPUT = "PRACTICE" if x == "d" else "REAL"
            return MODO_INPUT

def conectar():
    iq = IQ_Option(EMAIL, PASSWORD)
    ok, reason = iq.connect()
    if not ok:
        logging.error(f"Error de conexión: {reason}")
        raise SystemExit(1)
    iq.change_balance(MODO_INPUT)
    # Desactiva el hilo digital para evitar KeyError: 'underlying'
    try:
        iq.api.close_digital_option_socket()
    except Exception:
        pass
    saldo = iq.get_balance()
    logging.info(f"✅ Conectado a {MODO_INPUT}. Saldo: {saldo:.2f}")
    return iq

def reconectar(iq):
    try:
        iq.connect()
        iq.change_balance(MODO_INPUT)
        iq.api.close_digital_option_socket()
    except Exception:
        pass

def obtener_velas(iq, par, cantidad=100):
    # Devuelve DF con columnas open/high/low/close; vacío si falla
    for _ in range(2):
        try:
            velas = iq.get_candles(par, 60, cantidad, time.time())
            if not velas or "close" not in velas[0]:
                return pd.DataFrame()
            df = pd.DataFrame(velas)[["open", "max", "min", "close"]]
            df.columns = ["open", "high", "low", "close"]
            return df
        except Exception as e:
            logging.warning(f"[{par}] get_candles error: {e}")
            if not iq.check_connect():
                logging.warning("Reconectando socket...")
                reconectar(iq)
            time.sleep(0.8)
    return pd.DataFrame()

def filtrar_pares_validos(iq, pares):
    validos = []
    logging.info("🔎 Filtrando pares con velas reales...")
    for p in pares:
        df = obtener_velas(iq, p, 10)
        if not df.empty:
            validos.append(p)
            logging.info(f"   [✔] {p}")
        else:
            logging.info(f"   [✖] {p} (sin velas)")
        time.sleep(0.2)
    if not validos:
        validos = ["EURUSD-OTC", "GBPUSD-OTC", "USDCHF-OTC"]  # fallback seguro
    logging.info(f"✅ Pares válidos: {validos}")
    return validos

def calcular_indicadores(df):
    close, high, low = df["close"], df["high"], df["low"]
    rsi = ta.momentum.RSIIndicator(close, window=RSI_PERIOD).rsi()
    ema_f = ta.trend.EMAIndicator(close, window=EMA_FAST).ema_indicator()
    ema_s = ta.trend.EMAIndicator(close, window=EMA_SLOW).ema_indicator()
    macd_obj = ta.trend.MACD(close)
    macd_l = macd_obj.macd()
    macd_s = macd_obj.macd_signal()
    stoch = ta.momentum.StochasticOscillator(
        high, low, close, window=STOCH_PERIOD, smooth_window=STOCH_SMOOTH
    )
    st_k = stoch.stoch()
    st_d = stoch.stoch_signal()
    return rsi.iloc[-1], ema_f.iloc[-1], ema_s.iloc[-1], macd_l.iloc[-1], macd_s.iloc[-1], st_k.iloc[-1], st_d.iloc[-1]

def obtener_senal(par, df):
    if df.empty or len(df) < max(EMA_SLOW, RSI_PERIOD, STOCH_PERIOD):
        return None
    rsi, ema_f, ema_s, macd_l, macd_s, st_k, st_d = calcular_indicadores(df)
    logging.info(f"[{par}] RSI={rsi:.1f} EMAf={ema_f:.5f} EMAs={ema_s:.5f} MACD={macd_l:.5f}/{macd_s:.5f} STK={st_k:.1f}/{st_d:.1f}")

    up = down = 0
    if rsi < 35: up += 1
    elif rsi > 65: down += 1
    if ema_f > ema_s: up += 1
    elif ema_f < ema_s: down += 1
    if macd_l > macd_s: up += 1
    elif macd_l < macd_s: down += 1
    if st_k > st_d: up += 1
    elif st_k < st_d: down += 1

    # 3 de 4 señales para entrar
    if up >= 3:
        return "call"
    if down >= 3:
        return "put"
    return None

def abrir_operacion_con_retry(iq, par, direccion, monto, expiracion, reintentos=1, espera_retry=2.0):
    try:
        ok, order_id = iq.buy(monto, par, direccion, expiracion)
        if ok:
            logging.info(f"[OK] {direccion.upper()} en {par} abierta (id={order_id})")
            return True, order_id
        logging.warning(f"[X] Falló {direccion} en {par}. Reintentos pendientes: {reintentos}")
    except Exception as e:
        logging.warning(f"[X] Excepción abriendo {direccion} en {par}: {e}")

    if reintentos > 0:
        time.sleep(espera_retry)
        # Reintenta 1 vez
        try:
            ok2, order_id2 = iq.buy(monto, par, direccion, expiracion)
            if ok2:
                logging.info(f"[OK-RETRY] {direccion.upper()} en {par} (id={order_id2})")
                return True, order_id2
        except Exception as e:
            logging.warning(f"[X-RETRY] Excepción: {e}")

    return False, None

def esperar_resultado(iq, order_id, timeout_s=90):
    # Espera a que la operación cierre y devuelve ("WIN"/"LOSS", pnl_float)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            chk, pnl = iq.check_win_v4(order_id)
            if chk:
                res = "WIN" if pnl > 0 else "LOSS"
                return res, float(pnl)
        except Exception:
            pass
        time.sleep(2)
    return "UNKNOWN", 0.0

def csv_init(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fecha", "par", "tipo", "monto", "order_id", "resultado", "pnl", "saldo_antes", "saldo_despues"])

def csv_log(path, fecha, par, tipo, monto, order_id, resultado, pnl, saldo_antes, saldo_despues):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([fecha, par, tipo, monto, order_id, resultado, f"{pnl:.2f}", f"{saldo_antes:.2f}", f"{saldo_despues:.2f}"])


# ========= MAIN =========
if __name__ == "__main__":
    elegir_modo()
    iq = conectar()
    saldo_inicial = iq.get_balance()
    csv_init(CSV_PATH)

    # Filtra pares con velas reales
    pares = filtrar_pares_validos(iq, PARES_INICIALES)

    for ciclo in range(1, CICLOS + 1):
        # Risk Check (diario)
        saldo_actual = iq.get_balance()
        pnl_dia = saldo_actual - saldo_inicial
        if pnl_dia <= STOP_LOSS_DIARIO:
            logging.error(f"🚨 Stop-loss diario alcanzado ({pnl_dia:.2f}). Cerrando.")
            break
        if pnl_dia >= TAKE_PROFIT_DIARIO:
            logging.info(f"🏁 Take-profit diario alcanzado (+{pnl_dia:.2f}). Cerrando.")
            break

        logging.info(f"=== Ciclo {ciclo}/{CICLOS} | Saldo: {saldo_actual:.2f} | PnL día: {pnl_dia:.2f} ===")

        for par in pares:
            # Revalidar que el par siga con data
            df = obtener_velas(iq, par, 120)
            if df.empty:
                logging.warning(f"[{par}] sin velas ahora. Se omite.")
                continue

            senal = obtener_senal(par, df)
            if not senal:
                time.sleep(0.3)
                continue

            saldo_antes = iq.get_balance()
            ok, oid = abrir_operacion_con_retry(iq, par, senal, MONTO, EXPIRACION_MIN, reintentos=1, espera_retry=2.0)
            if ok and oid:
                if ESPERAR_RESULTADO:
                    resultado, pnl = esperar_resultado(iq, oid, timeout_s=120)
                    saldo_despues = iq.get_balance()
                    csv_log(CSV_PATH, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), par, senal, MONTO, oid, resultado, pnl, saldo_antes, saldo_despues)
                    logging.info(f"🎯 {resultado} {pnl:+.2f} | Saldo: {saldo_despues:.2f}")
                else:
                    csv_log(CSV_PATH, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), par, senal, MONTO, oid, "ABIERTO", 0.0, saldo_antes, saldo_antes)
            else:
                logging.warning(f"[X] No se abrió {senal} en {par}")

            time.sleep(PAUSA_ENTRE_TRADES)

        logging.info(f"⏳ Fin ciclo {ciclo}. Esperando {ESPERA_ENTRE_CICLOS}s...\n")
        time.sleep(ESPERA_ENTRE_CICLOS)

    try:
        iq.close()
    except Exception:
        pass
    logging.info("✅ Bot finalizado correctamente.")
