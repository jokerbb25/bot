"""Módulo principal del bot de trading.

El script original contenía una cantidad considerable de estado global y
credenciales en texto plano.  Se refactoriza para hacer el flujo más claro, se
añaden anotaciones de tipo y se mejora el manejo de errores sin modificar la
lógica general de trading.
"""

from __future__ import annotations

import csv
import logging
import os
from time import sleep, time as epoch_seconds
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import pandas as pd
import ta
from iqoptionapi.stable_api import IQ_Option

# ========= CONFIG =========
EMAIL = os.getenv("IQ_EMAIL", "")
PASSWORD = os.getenv("IQ_PASSWORD", "")

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
    "EURUSD-OTC",
    "GBPUSD-OTC",
    "USDCHF-OTC",
    "NZDUSD-OTC",
    "USDJPY-OTC",
    "GBPJPY-OTC",
    "EURJPY-OTC",
    "EURGBP-OTC",
    "AUDCAD-OTC",
    "AUDUSD-OTC",
    "CADJPY-OTC",
    "EURAUD-OTC",
]

CSV_PATH = Path("operaciones.csv")


@dataclass(frozen=True)
class Credenciales:
    """Representa las credenciales necesarias para iniciar sesión."""

    email: str
    password: str

    @classmethod
    def desde_entorno(cls) -> "Credenciales":
        """Obtiene las credenciales desde variables de entorno.

        Si alguna credencial falta se lanza un ``ValueError`` para evitar que
        el bot intente conectarse con datos incompletos.
        """

        if not EMAIL or not PASSWORD:
            raise ValueError(
                "Las credenciales no están configuradas. Defina IQ_EMAIL e IQ_PASSWORD"
            )
        return cls(email=EMAIL, password=PASSWORD)


# ========= UTIL =========
def elegir_modo() -> str:
    """Solicita al usuario el modo de operación y devuelve PRACTICE o REAL."""

    while True:
        x = input("Operar en demo o real? (d/r): ").lower().strip()
        if x in ("d", "r"):
            return "PRACTICE" if x == "d" else "REAL"


def conectar(modo: str, credenciales: Credenciales) -> IQ_Option:
    """Realiza la conexión inicial con IQ Option."""

    iq = IQ_Option(credenciales.email, credenciales.password)
    ok, reason = iq.connect()
    if not ok:
        logging.error(f"Error de conexión: {reason}")
        raise SystemExit(1)
    iq.change_balance(modo)
    # Desactiva el hilo digital para evitar KeyError: 'underlying'
    try:
        iq.api.close_digital_option_socket()
    except Exception:
        pass
    saldo = iq.get_balance()
    logging.info(f"✅ Conectado a {modo}. Saldo: {saldo:.2f}")
    return iq


def reconectar(iq: IQ_Option, modo: str) -> None:
    try:
        iq.connect()
        iq.change_balance(modo)
        iq.api.close_digital_option_socket()
    except Exception:
        pass

def obtener_velas(iq: IQ_Option, par: str, cantidad: int, modo: str) -> pd.DataFrame:
    # Devuelve DF con columnas open/high/low/close; vacío si falla
    for _ in range(2):
        try:
            velas = iq.get_candles(par, 60, cantidad, epoch_seconds())
            if not velas or "close" not in velas[0]:
                return pd.DataFrame()
            df = pd.DataFrame(velas)[["open", "max", "min", "close"]]
            df.columns = ["open", "high", "low", "close"]
            return df
        except Exception as e:
            logging.warning(f"[{par}] get_candles error: {e}")
            if not iq.check_connect():
                logging.warning("Reconectando socket...")
                reconectar(iq, modo)
            sleep(0.8)
    return pd.DataFrame()


def pares_operables_en_api(iq: IQ_Option, candidatos: Sequence[str]) -> List[str]:
    """Devuelve los pares del listado que la API reconoce como operables."""

    try:
        open_time = iq.get_all_open_time()
    except Exception as exc:
        logging.warning(f"No se pudo consultar la lista de pares abiertos: {exc}")
        return list(candidatos)

    activos: Set[str] = set()
    for mercado in ("binary", "turbo", "digital"):
        mercado_data = open_time.get(mercado, {})
        activos.update({par for par, info in mercado_data.items() if info.get("open")})

    if not activos:
        logging.warning("La API no reportó pares abiertos; se usarán los candidatos originales.")
        return list(candidatos)

    reconocidos = [par for par in candidatos if par in activos]
    excluidos = [par for par in candidatos if par not in activos]
    for par in excluidos:
        logging.info(f"   [✖] {par} no está disponible actualmente en la API")

    if not reconocidos:
        logging.warning("Ningún par candidato está disponible; se usará un subconjunto genérico.")
        return ["EURUSD-OTC", "GBPUSD-OTC", "USDCHF-OTC"]

    return reconocidos


def filtrar_pares_validos(iq: IQ_Option, pares: Sequence[str], modo: str) -> List[str]:
    pares = pares_operables_en_api(iq, pares)
    validos: List[str] = []
    logging.info("🔎 Filtrando pares con velas reales...")
    for p in pares:
        df = obtener_velas(iq, p, 10, modo)
        if not df.empty:
            validos.append(p)
            logging.info(f"   [✔] {p}")
        else:
            logging.info(f"   [✖] {p} (sin velas)")
        sleep(0.2)
    if not validos:
        validos = ["EURUSD-OTC", "GBPUSD-OTC", "USDCHF-OTC"]  # fallback seguro
    logging.info(f"✅ Pares válidos: {validos}")
    return validos

def calcular_indicadores(df: pd.DataFrame) -> Tuple[float, float, float, float, float, float, float]:
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

def obtener_senal(par: str, df: pd.DataFrame) -> Optional[str]:
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

def abrir_operacion_con_retry(
    iq: IQ_Option,
    par: str,
    direccion: str,
    monto: float,
    expiracion: int,
    reintentos: int = 1,
    espera_retry: float = 2.0,
) -> Tuple[bool, Optional[int]]:
    try:
        ok, order_id = iq.buy(monto, par, direccion, expiracion)
        if ok:
            logging.info(f"[OK] {direccion.upper()} en {par} abierta (id={order_id})")
            return True, order_id
        logging.warning(f"[X] Falló {direccion} en {par}. Reintentos pendientes: {reintentos}")
    except Exception as e:
        logging.warning(f"[X] Excepción abriendo {direccion} en {par}: {e}")

    if reintentos > 0:
        sleep(espera_retry)
        # Reintenta 1 vez
        try:
            ok2, order_id2 = iq.buy(monto, par, direccion, expiracion)
            if ok2:
                logging.info(f"[OK-RETRY] {direccion.upper()} en {par} (id={order_id2})")
                return True, order_id2
        except Exception as e:
            logging.warning(f"[X-RETRY] Excepción: {e}")

    return False, None


def esperar_resultado(iq: IQ_Option, order_id: int, timeout_s: int = 90) -> Tuple[str, float]:
    # Espera a que la operación cierre y devuelve ("WIN"/"LOSS", pnl_float)
    t0 = epoch_seconds()
    while epoch_seconds() - t0 < timeout_s:
        try:
            chk, pnl = iq.check_win_v4(order_id)
            if chk:
                res = "WIN" if pnl > 0 else "LOSS"
                return res, float(pnl)
        except Exception:
            pass
        sleep(2)
    return "UNKNOWN", 0.0

def csv_init(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fecha", "par", "tipo", "monto", "order_id", "resultado", "pnl", "saldo_antes", "saldo_despues"])

def csv_log(
    path: Path,
    fecha: str,
    par: str,
    tipo: str,
    monto: float,
    order_id: int,
    resultado: str,
    pnl: float,
    saldo_antes: float,
    saldo_despues: float,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([fecha, par, tipo, monto, order_id, resultado, f"{pnl:.2f}", f"{saldo_antes:.2f}", f"{saldo_despues:.2f}"])


# ========= MAIN =========

def main() -> None:
    try:
        credenciales = Credenciales.desde_entorno()
    except ValueError as exc:
        logging.error(exc)
        raise SystemExit(1) from exc

    modo = elegir_modo()
    iq = conectar(modo, credenciales)
    saldo_inicial = iq.get_balance()
    csv_init(CSV_PATH)

    pares = filtrar_pares_validos(iq, PARES_INICIALES, modo)

    try:
        for ciclo in range(1, CICLOS + 1):
            saldo_actual = iq.get_balance()
            pnl_dia = saldo_actual - saldo_inicial
            if pnl_dia <= STOP_LOSS_DIARIO:
                logging.error(f"🚨 Stop-loss diario alcanzado ({pnl_dia:.2f}). Cerrando.")
                break
            if pnl_dia >= TAKE_PROFIT_DIARIO:
                logging.info(f"🏁 Take-profit diario alcanzado (+{pnl_dia:.2f}). Cerrando.")
                break

            logging.info(
                f"=== Ciclo {ciclo}/{CICLOS} | Saldo: {saldo_actual:.2f} | PnL día: {pnl_dia:.2f} ==="
            )

            for par in pares:
                df = obtener_velas(iq, par, 120, modo)
                if df.empty:
                    logging.warning(f"[{par}] sin velas ahora. Se omite.")
                    continue

                senal = obtener_senal(par, df)
                if not senal:
                    sleep(0.3)
                    continue

                saldo_antes = iq.get_balance()
                ok, oid = abrir_operacion_con_retry(
                    iq, par, senal, MONTO, EXPIRACION_MIN, reintentos=1, espera_retry=2.0
                )
                if ok and oid:
                    if ESPERAR_RESULTADO:
                        resultado, pnl = esperar_resultado(iq, oid, timeout_s=120)
                        saldo_despues = iq.get_balance()
                        csv_log(
                            CSV_PATH,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            par,
                            senal,
                            MONTO,
                            oid,
                            resultado,
                            pnl,
                            saldo_antes,
                            saldo_despues,
                        )
                        logging.info(f"🎯 {resultado} {pnl:+.2f} | Saldo: {saldo_despues:.2f}")
                    else:
                        csv_log(
                            CSV_PATH,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            par,
                            senal,
                            MONTO,
                            oid,
                            "ABIERTO",
                            0.0,
                            saldo_antes,
                            saldo_antes,
                        )
                else:
                    logging.warning(f"[X] No se abrió {senal} en {par}")

                sleep(PAUSA_ENTRE_TRADES)

            logging.info(f"⏳ Fin ciclo {ciclo}. Esperando {ESPERA_ENTRE_CICLOS}s...\n")
            sleep(ESPERA_ENTRE_CICLOS)
    except KeyboardInterrupt:
        logging.info("⏹️ Interrupción manual recibida. Cerrando bot...")
    finally:
        try:
            iq.close()
        except Exception:
            pass
        logging.info("✅ Bot finalizado correctamente.")


if __name__ == "__main__":
    main()
