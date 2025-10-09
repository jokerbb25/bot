import sys
import time
import math
import logging
import threading
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Tuple, Union, Set, Iterable, List, Iterator
from datetime import datetime
from textwrap import dedent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def _ensure_dependency(module_name: str, pip_name: Optional[str] = None) -> None:
    """Ensure a dependency can be imported, otherwise exit with a friendly message."""

    try:
        __import__(module_name)
    except ImportError:
        friendly_name = pip_name or module_name
        message = dedent(
            f"""
            ❌ Falta la dependencia obligatoria "{module_name}".
            Instálala ejecutando: pip install {friendly_name}
            """
        ).strip()
        logging.error(message)
        sys.exit(1)


for _module, _pip in (
    ("pandas", None),
    ("ta", None),
    ("PyQt5", "PyQt5"),
    ("iqoptionapi", "iqoptionapi"),
):
    _ensure_dependency(_module, _pip)

import pandas as pd  # noqa: E402  (after dependency check)
import ta  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
    QPushButton,
    QSplashScreen,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread  # noqa: E402
from PyQt5.QtGui import QFont, QPixmap, QPainter, QColor, QLinearGradient  # noqa: E402
from iqoptionapi.stable_api import IQ_Option  # noqa: E402
from iqoptionapi import constants as iq_constants  # noqa: E402

# ---------------- CONFIG ----------------
EMAIL = "fornerinoalejandro031@gmail.com"
PASSWORD = "484572ale"
MONTO = 1.0
EXPIRACION = 1
ESPERA_ENTRE_CICLOS = 3
CICLOS = 50
MODO = "PRACTICE"


MANUAL_FALLBACK_INSTRUMENTS: Tuple[Tuple[str, str], ...] = (
    # Digital majors
    ("EURUSD", "digital"),
    ("GBPUSD", "digital"),
    ("USDJPY", "digital"),
    ("AUDUSD", "digital"),
    ("USDCHF", "digital"),
    ("USDCAD", "digital"),
    ("EURJPY", "digital"),
    ("EURGBP", "digital"),
    ("GBPJPY", "digital"),
    ("AUDCAD", "digital"),
    ("NZDUSD", "digital"),
    ("CADJPY", "digital"),
    # Binary majors
    ("EURUSD", "binary"),
    ("GBPUSD", "binary"),
    ("USDJPY", "binary"),
    ("AUDUSD", "binary"),
    ("USDCHF", "binary"),
    ("USDCAD", "binary"),
    ("EURJPY", "binary"),
    ("EURGBP", "binary"),
    ("GBPJPY", "binary"),
    ("AUDCAD", "binary"),
)


def _normalize_underlying_payload(data: Optional[Dict]) -> Dict:
    """Ensure the IQ Option response always exposes the ``underlying`` key."""

    if not isinstance(data, dict):
        logging.debug(
            "Respuesta de digitales sin formato dict (%s); se normaliza estructura vacía.",
            type(data).__name__,
        )
        return {"underlying": {}}

    underlying = data.get("underlying")
    if isinstance(underlying, dict):
        return data

    logging.debug(
        "Respuesta de digitales sin clave 'underlying' válida (%s); se fuerza diccionario vacío.",
        type(underlying).__name__,
    )
    data["underlying"] = {}
    return data


def _patch_iqoption_underlying() -> None:
    """Monkey patch IQ Option SDK to avoid KeyError in internal threads."""

    original = IQ_Option.get_digital_underlying_list_data

    # Evita volver a aplicar el parche
    if getattr(original, "__name__", "") == "safe_get_digital_underlying_list_data":
        return

    def safe_get_digital_underlying_list_data(self, *args, **kwargs):  # type: ignore
        try:
            raw = original(self, *args, **kwargs)
        except Exception:
            logging.exception(
                "Fallo al obtener digitales desde el SDK; se devuelve estructura vacía."
            )
            return {"underlying": {}}

        return _normalize_underlying_payload(raw)

    safe_get_digital_underlying_list_data.__name__ = "safe_get_digital_underlying_list_data"
    safe_get_digital_underlying_list_data.__wrapped__ = original  # type: ignore[attr-defined]
    IQ_Option.get_digital_underlying_list_data = safe_get_digital_underlying_list_data  # type: ignore[assignment]


_patch_iqoption_underlying()


def _sanitize_symbol_name(symbol: Optional[str]) -> Optional[str]:
    """Return a broker-friendly symbol stripping IQ Option suffixes."""

    if not symbol or not isinstance(symbol, str):
        return None

    cleaned = symbol.strip().upper()
    for suffix in ("-OP", "-OTC"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return cleaned or None


class SafeIQOption(IQ_Option):
    """Versión protegida del cliente IQ Option que evita fallos del SDK."""

    def get_digital_underlying_list_data(self):  # pragma: no cover - delega en SDK
        try:
            data = super().get_digital_underlying_list_data()
        except Exception:
            logging.exception(
                "Fallo al obtener digitales; se devuelve estructura vacía para evitar KeyError."
            )
            return {"underlying": {}}

        return _normalize_underlying_payload(data)

    def sync_active_catalog(self, open_time_payload: Optional[Dict] = None) -> None:
        """Completa los mapas internos del SDK con los pares disponibles detectados."""

        try:
            containers: Iterable[Dict] = self._collect_active_containers()
        except Exception:
            logging.debug("No fue posible preparar los contenedores de activos.")
            containers = ()

        if open_time_payload is None:
            try:
                open_time_payload = self.get_all_open_time()
            except Exception:
                open_time_payload = {}

        updated_symbols: Set[str] = set()

        def register(symbol: Optional[str], active_id: Optional[Union[int, str]]) -> None:
            if not symbol:
                return
            try:
                normalized_id = int(active_id) if active_id is not None else None
            except (TypeError, ValueError):
                return
            if normalized_id is None:
                return

            symbol_key = symbol.upper()
            changed = False
            for container in containers:
                if self._update_container(container, symbol_key, normalized_id):
                    changed = True
            if changed:
                updated_symbols.add(symbol_key)

        # Horarios de apertura (turbo/digital/binario)
        if isinstance(open_time_payload, dict):
            for category in ("turbo", "digital", "binary"):
                entries = open_time_payload.get(category, {})
                if not isinstance(entries, dict):
                    continue
                for symbol, info in entries.items():
                    if not isinstance(info, dict):
                        continue
                    register(symbol, info.get("id") or info.get("active_id"))

        # Metadatos digitales (incluye instrument_id/asset_id)
        try:
            digital_meta = self.get_digital_underlying_list_data().get("underlying", {})
        except Exception:
            digital_meta = {}

        if isinstance(digital_meta, dict):
            for _, info in digital_meta.items():
                if not isinstance(info, dict):
                    continue
                symbol = (
                    info.get("symbol")
                    or info.get("asset_name")
                    or info.get("underlying")
                    or info.get("active")
                )
                register(
                    symbol,
                    info.get("active_id")
                    or info.get("id")
                    or info.get("asset_id")
                    or info.get("instrument_id"),
                )

        if updated_symbols:
            logging.info(
                "📈 Registrados %s pares adicionales detectados en la API.",
                len(updated_symbols),
            )

    # ---- utilitarios internos ----
    def _collect_active_containers(self) -> Iterable[Dict]:
        containers = []

        for attr in ("active_to_id", "available_leverages", "instruments", "all_underlying_list"):
            data = getattr(self, attr, None)
            if isinstance(data, dict):
                containers.append(data)

        actives = getattr(self, "ACTIVES", None)
        if isinstance(actives, dict):
            containers.append(actives)

        for name in ("ACTIVES", "ACTIVES_ID", "assets", "assets_name"):
            data = getattr(iq_constants, name, None)
            if isinstance(data, dict):
                containers.append(data)

        # Estructuras anidadas usadas por métodos privados del SDK
        for attr in ("api_option_init_all_result", "api_game_getcandles_v2"):
            data = getattr(self, attr, None)
            if isinstance(data, dict):
                containers.append(data)
            elif hasattr(data, "__dict__"):
                nested = getattr(data, "__dict__", {})
                for value in nested.values():
                    if isinstance(value, dict):
                        containers.append(value)

        return containers

    @staticmethod
    def _update_container(container: Dict, symbol: str, active_id: int) -> bool:
        if not isinstance(container, dict):
            return False

        if not container:
            container[symbol] = {"id": active_id, "name": symbol}
            return True

        sample_key = next(iter(container.keys()))

        # Diccionarios {"EURUSD": {...}}
        if isinstance(sample_key, str):
            value = container.get(symbol)
            if isinstance(value, dict):
                changed = value.get("id") != active_id
                value["id"] = active_id
                value.setdefault("name", symbol)
                return changed

            if value != active_id:
                if any(isinstance(v, dict) for v in container.values()):
                    container[symbol] = {"id": active_id, "name": symbol}
                else:
                    container[symbol] = active_id
                return True
            return False

        # Diccionarios {76: "EURUSD"}
        if isinstance(sample_key, int):
            existing = container.get(active_id)
            if existing != symbol:
                container[active_id] = symbol
                return True
            return False

        # Fallback genérico
        container[symbol] = active_id
        return True

    @staticmethod
    def _extract_symbol_from_value(value: Union[str, Dict, None]) -> Optional[str]:
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            for key in ("symbol", "name", "underlying", "active", "asset_name"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate:
                    return candidate
        return None

    def resolve_active_symbol(self, active_id: Optional[Union[int, str]]) -> Optional[str]:
        """Map an ``active_id`` to the canonical symbol when available."""

        try:
            normalized_id = int(active_id) if active_id is not None else None
        except (TypeError, ValueError):
            return None

        if normalized_id is None:
            return None

        for container in self._collect_active_containers():
            if not isinstance(container, dict):
                continue
            for key, value in container.items():
                if isinstance(key, int) and key == normalized_id:
                    symbol = self._extract_symbol_from_value(value)
                    if symbol:
                        return _sanitize_symbol_name(symbol)
                if value == normalized_id and isinstance(key, str):
                    return _sanitize_symbol_name(key)
                if isinstance(value, dict) and value.get("id") == normalized_id:
                    if isinstance(key, str) and key:
                        return _sanitize_symbol_name(key)
                    symbol = self._extract_symbol_from_value(value)
                    if symbol:
                        return _sanitize_symbol_name(symbol)

        return None

    @staticmethod
    def _symbol_variants(symbol: str) -> Set[str]:
        variantes: Set[str] = set()
        if not symbol or not isinstance(symbol, str):
            return variantes

        bases: List[str] = []
        base = symbol.strip()
        if base:
            bases.append(base)
        sanitizado = _sanitize_symbol_name(symbol)
        if sanitizado and sanitizado not in bases:
            bases.append(sanitizado)

        for texto in bases:
            if not texto:
                continue
            variantes.add(texto.upper())
            variantes.add(texto.replace("/", "").upper())
            variantes.add(texto.replace("-", "").upper())
            variantes.add(texto.lower())
            variantes.add(texto.replace("/", "").lower())
            variantes.add(texto.replace("-", "").lower())
            sufijos = ("-OTC", "-otc", "-OP", "-op")
            for sufijo in sufijos:
                variantes.add(f"{texto}{sufijo}".upper())
                variantes.add(f"{texto}{sufijo}".lower())

        return {valor for valor in variantes if valor}

    @staticmethod
    def _extract_id_from_value(value: Union[str, int, float, Dict, None]) -> Optional[int]:
        if isinstance(value, (int, float)):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        if isinstance(value, str):
            texto = value.strip()
            if texto.isdigit():
                return int(texto)
            digitos = "".join(ch for ch in texto if ch.isdigit())
            if digitos:
                try:
                    return int(digitos)
                except ValueError:
                    return None
            return None
        if isinstance(value, dict):
            for clave in ("id", "active_id", "instrument_id", "asset_id", "underlying_id"):
                candidato = value.get(clave)
                if candidato is None:
                    continue
                try:
                    return int(candidato)
                except (TypeError, ValueError):
                    continue
        return None

    def lookup_active_id(self, symbol: str) -> Optional[int]:
        variantes = self._symbol_variants(symbol)
        if not variantes:
            return None

        for container in self._collect_active_containers():
            if not isinstance(container, dict):
                continue
            for key, value in container.items():
                clave_str = key.strip().upper() if isinstance(key, str) else None
                if clave_str and clave_str in variantes:
                    candidato = self._extract_id_from_value(value)
                    if candidato is not None:
                        return candidato
                    if isinstance(value, dict):
                        simbolo = self._extract_symbol_from_value(value)
                        if simbolo and simbolo.strip().upper() in variantes:
                            candidato = self._extract_id_from_value(value)
                            if candidato is not None:
                                return candidato
                clave_id: Optional[int]
                if isinstance(key, (int, float)):
                    try:
                        clave_id = int(key)
                    except (TypeError, ValueError):
                        clave_id = None
                else:
                    clave_id = None

                if isinstance(value, str):
                    valor_str = value.strip().upper()
                    if valor_str in variantes and clave_id is not None:
                        return clave_id

                if isinstance(value, dict):
                    simbolo = self._extract_symbol_from_value(value)
                    if simbolo and simbolo.strip().upper() in variantes:
                        candidato = self._extract_id_from_value(value)
                        if candidato is not None:
                            return candidato
                        if clave_id is not None:
                            return clave_id

                if isinstance(value, (int, float)) and clave_str and clave_str in variantes:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        continue

        return None

# ---------------- CLASE GUI ----------------
@dataclass
class IndicatorSnapshot:
    """Represents the most recent technical indicator values for a pair."""

    rsi: float
    emaf: float
    emas: float
    macd: float
    macds: float
    stk: float
    std: float

    def to_table_payload(self, signal: Optional[str]) -> Dict[str, Union[float, str]]:
        """Prepare the dictionary used to update the GUI table."""

        payload = asdict(self)
        payload["signal"] = signal or "-"
        return payload


@dataclass(frozen=True)
class TradePair:
    """Metadata describing a tradable instrument."""

    display: str
    api_symbol: str
    active_id: int
    instrument_type: str  # "digital" or "binary"
    candle_aliases: Tuple[Union[str, int], ...] = field(default_factory=tuple)
    trade_aliases: Tuple[str, ...] = field(default_factory=tuple)
    digital_instrument_ids: Tuple[str, ...] = field(default_factory=tuple)


class BotWorker(QObject):
    status_changed = pyqtSignal(str)
    row_ready = pyqtSignal(str, dict)
    trade_completed = pyqtSignal(str, str, float, float)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()
        self._pnl_acumulado = 0.0
        self._pares_descartados: Set[str] = set()
        self._cooldowns: Dict[str, float] = {}
        self._fallas_temporales: Dict[str, int] = {}
        self._cooldown_notices: Dict[str, float] = {}
        self._instrument_cache: Dict[Tuple[str, str, int], Optional[str]] = {}

    def stop(self):
        self._stop_event.set()

    def run(self):
        self.status_changed.emit("🔌 Conectando a IQ Option...")
        iq = SafeIQOption(EMAIL, PASSWORD)
        self._stop_event.clear()
        try:
            check, reason = iq.connect()
        except Exception as exc:  # pragma: no cover - network call
            logging.exception("Error inesperado al conectar con IQ Option: %s", exc)
            self.status_changed.emit("❌ Error conexión inesperado. Ver logs.")
            self.finished.emit()
            return

        try:
            if not check:
                self.status_changed.emit(f"❌ Error conexión: {reason}")
                self.finished.emit()
                return

            iq.change_balance(MODO)
            saldo = iq.get_balance()
            self.status_changed.emit(f"✅ Conectado a {MODO} | Saldo: {saldo:.2f}")

            self._pares_descartados.clear()
            logging.info("♻️ Escaneando pares digitales y binarias disponibles...")
            pares_validos = self._descubrir_pares(iq)
            logging.info(
                "✅ Pares detectados: %s",
                [par.display for par in pares_validos],
            )

            pares_validos = self._filtrar_activos_operables(iq, pares_validos)
            if not pares_validos:
                self.status_changed.emit(
                    "⚠️ No se encontraron pares digitales/binarias disponibles."
                )
                self.finished.emit()
                return

            for ciclo in range(1, CICLOS + 1):
                if self._stop_event.is_set():
                    break
                logging.info(f"=== Ciclo {ciclo}/{CICLOS} ===")
                self.status_changed.emit(
                    f"🔁 Ciclo {ciclo}/{CICLOS} | PnL acumulado: {self._pnl_acumulado:.2f}"
                )
                activos_restantes = False
                for par in list(pares_validos):
                    if self._stop_event.is_set():
                        break
                    if par.display in self._pares_descartados:
                        continue
                    en_cooldown, restante = self._cooldown_status(par)
                    if en_cooldown:
                        now = time.time()
                        last_notice = self._cooldown_notices.get(par.display, 0.0)
                        if now - last_notice >= 5.0:
                            logging.info(
                                "⏳ %s en cooldown (%ss restantes, fallos consecutivos: %s)",
                                par.display,
                                math.ceil(restante),
                                self._fallas_temporales.get(par.display, 0),
                            )
                            self._cooldown_notices[par.display] = now
                        continue
                    activos_restantes = True
                    df = self.obtener_velas(iq, par)
                    if df.empty:
                        self._descartar_par(par, "Sin velas devueltas por el broker.")
                        continue
                    senal, data = self.obtener_senal(df)
                    self.row_ready.emit(par.display, data)
                    self._log_snapshot(par, data, senal)

                    if senal:
                        ok, ticket = self._ejecutar_operacion(iq, par, senal)
                        if ok and ticket is not None:
                            logging.info(f"[OK] {senal.upper()} en {par.display}")
                            resultado, pnl = self._esperar_resultado(iq, par, ticket)
                            if resultado:
                                self._pnl_acumulado += pnl
                                texto_resultado = self._normalizar_resultado(resultado)
                                mensaje = (
                                    f"{texto_resultado} en {par.display} | PnL operación: {pnl:.2f} | "
                                    f"PnL acumulado: {self._pnl_acumulado:.2f}"
                                )
                                logging.info(mensaje)
                                self.status_changed.emit(mensaje)
                                self.trade_completed.emit(
                                    par.display, texto_resultado, pnl, self._pnl_acumulado
                                )
                            else:
                                logging.warning(
                                    "No se obtuvo resultado para la operación en %s",
                                    par.display,
                                )
                        else:
                            logging.warning(
                                f"[FAIL] No se pudo ejecutar {senal} en {par.display}"
                            )
                            # El detalle del fallo y el cooldown se gestionan dentro de _ejecutar_operacion
                    time.sleep(0.6)

                pares_validos = [
                    par
                    for par in pares_validos
                    if par.display not in self._pares_descartados
                ]
                if not pares_validos:
                    msg = "⚠️ No quedan pares operables tras los descartes."
                    logging.warning(msg)
                    self.status_changed.emit(msg)
                    break
                if not activos_restantes:
                    msg = "⚠️ Todos los pares fueron descartados por fallos del broker."
                    logging.warning(msg)
                    self.status_changed.emit(msg)
                    break
                time.sleep(ESPERA_ENTRE_CICLOS)

            if self._stop_event.is_set():
                mensaje_final = "⏹️ Bot detenido manualmente."
            else:
                mensaje_final = "✅ Bot finalizado correctamente."
            logging.info(mensaje_final)
            self.status_changed.emit(mensaje_final)
            self.finished.emit()
        finally:
            for cierre in ("close", "close_connection", "api_close"):
                metodo = getattr(iq, cierre, None)
                if callable(metodo):
                    try:
                        metodo()
                    except Exception:
                        continue
                    break

    def _descartar_par(self, par: TradePair, motivo: str) -> None:
        if par.display in self._pares_descartados:
            return
        self._pares_descartados.add(par.display)
        self._cooldowns.pop(par.display, None)
        self._fallas_temporales.pop(par.display, None)
        self._cooldown_notices.pop(par.display, None)
        logging.info("Activo %s descartado: %s", par.display, motivo)

    def _cooldown_status(self, par: TradePair) -> Tuple[bool, float]:
        vencimiento = self._cooldowns.get(par.display)
        if not vencimiento:
            return False, 0.0
        restante = vencimiento - time.time()
        if restante <= 0:
            self._cooldowns.pop(par.display, None)
            self._cooldown_notices.pop(par.display, None)
            return False, 0.0
        return True, restante

    def _marcar_fallo_temporal(self, par: TradePair, motivo: str, cooldown: int = 20) -> None:
        if par.display in self._pares_descartados:
            return
        contador = self._fallas_temporales.get(par.display, 0) + 1
        self._fallas_temporales[par.display] = contador
        self._cooldowns[par.display] = time.time() + max(10, cooldown)
        self._cooldown_notices.pop(par.display, None)
        restantes = max(0, 3 - contador)
        logging.info(
            "Activo %s en cooldown (%s). Nuevos intentos disponibles en %ss. Reintentos restantes antes de descartar: %s",
            par.display,
            motivo,
            cooldown,
            restantes,
        )
        if contador >= 3:
            self._descartar_par(par, "Superó el límite de fallos consecutivos.")

    @staticmethod
    def _to_float(value: Union[float, int, str]) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def _log_snapshot(
        self, par: TradePair, payload: Dict[str, Union[float, str]], signal: Optional[str]
    ) -> None:
        rsi = self._to_float(payload.get("rsi", float("nan")))
        emaf = self._to_float(payload.get("emaf", float("nan")))
        emas = self._to_float(payload.get("emas", float("nan")))
        macd = self._to_float(payload.get("macd", float("nan")))
        macds = self._to_float(payload.get("macds", float("nan")))
        stk = self._to_float(payload.get("stk", float("nan")))
        std = self._to_float(payload.get("std", float("nan")))

        def fmt(valor: float, precision: int) -> str:
            if math.isnan(valor):
                return "-"
            return f"{valor:.{precision}f}"

        resumen = (
            f"[{par.display}] RSI={fmt(rsi, 1)} EMAf={fmt(emaf, 5)} "
            f"EMAs={fmt(emas, 5)} MACD={fmt(macd, 5)}/{fmt(macds, 5)} "
            f"STK={fmt(stk, 1)}/{fmt(std, 1)}"
        )

        if signal:
            logging.info("%s | Señal=%s", resumen, signal.upper())
        else:
            logging.info("%s | Sin señal operativa", resumen)

    @staticmethod
    def _es_info_activo(info: Dict) -> bool:
        if not isinstance(info, dict):
            return False
        if info.get("open") is False or info.get("is_open") is False:
            return False
        claves = ("id", "active_id", "instrument_id", "asset_id", "underlying_id")
        if any(info.get(clave) not in (None, {}, []) for clave in claves):
            return True
        for nested_key in ("active", "instrument", "asset", "option"):
            nested = info.get(nested_key)
            if isinstance(nested, dict) and BotWorker._es_info_activo(nested):
                return True
        return False

    @staticmethod
    def _desempaquetar_id(valor: Union[int, str, Dict, None]) -> Optional[int]:
        if valor is None:
            return None
        if isinstance(valor, dict):
            for clave in ("id", "value", "active_id", "instrument_id"):
                if clave in valor:
                    resultado = BotWorker._desempaquetar_id(valor.get(clave))
                    if resultado is not None:
                        return resultado
            return None
        if isinstance(valor, (int, float)):
            try:
                return int(valor)
            except (TypeError, ValueError):
                return None
        if isinstance(valor, str):
            texto = valor.strip()
            if not texto:
                return None
            try:
                return int(texto)
            except ValueError:
                digitos = "".join(ch for ch in texto if ch.isdigit())
                if digitos:
                    try:
                        return int(digitos)
                    except ValueError:
                        return None
        return None

    @staticmethod
    def _extraer_active_id(info: Dict) -> Optional[int]:
        if not isinstance(info, dict):
            return None
        for clave in ("id", "active_id", "instrument_id", "asset_id", "underlying_id"):
            resultado = BotWorker._desempaquetar_id(info.get(clave))
            if resultado is not None:
                return resultado
        for nested_key in ("active", "instrument", "asset", "option"):
            nested = info.get(nested_key)
            if isinstance(nested, dict):
                nested_id = BotWorker._extraer_active_id(nested)
                if nested_id is not None:
                    return nested_id
        return None

    @staticmethod
    def _inferir_simbolo(clave: Optional[str], info: Dict) -> Optional[str]:
        candidatos: List[str] = []
        if isinstance(clave, str):
            candidatos.append(clave)
        if isinstance(info, dict):
            for key in ("symbol", "active", "underlying", "asset_name", "name", "ticker"):
                valor = info.get(key)
                if isinstance(valor, str):
                    candidatos.append(valor)
        for candidato in candidatos:
            texto = candidato.strip()
            if not texto:
                continue
            texto_lower = texto.lower()
            if texto_lower in {
                "open",
                "close",
                "is_open",
                "actives",
                "list",
                "items",
                "data",
                "schedule",
            }:
                continue
            return texto
        return None

    def _recorrer_catalogo(
        self,
        categoria: str,
        nodo: Union[Dict, List, None],
        clave_actual: Optional[str] = None,
    ) -> Iterator[Tuple[str, str, Dict]]:
        if isinstance(nodo, dict):
            if self._es_info_activo(nodo):
                simbolo = self._inferir_simbolo(clave_actual, nodo)
                if simbolo:
                    yield categoria, simbolo, nodo
            for clave, valor in nodo.items():
                siguiente = clave_actual
                if isinstance(clave, str):
                    clave_limpia = clave.strip()
                    if clave_limpia:
                        siguiente = clave_limpia
                if isinstance(valor, (dict, list)):
                    yield from self._recorrer_catalogo(categoria, valor, siguiente)
        elif isinstance(nodo, list):
            for item in nodo:
                yield from self._recorrer_catalogo(categoria, item, clave_actual)

    def _iterar_activos_abiertos(self, payload: Union[Dict, Tuple, None]) -> Iterator[Tuple[str, str, Dict]]:
        if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[1], dict):
            payload = payload[1]
        if not isinstance(payload, dict):
            return
        etiquetas = ("digital", "binary", "turbo")
        for categoria, contenido in payload.items():
            nombre = str(categoria).lower()
            token_encontrado = None
            for token in etiquetas:
                if token in nombre:
                    token_encontrado = token
                    break
            if not token_encontrado:
                continue
            yield from self._recorrer_catalogo(token_encontrado, contenido)

    def _extraer_digital_meta(self, iq) -> Iterator[Tuple[str, str, Dict]]:
        try:
            payload = iq.get_digital_underlying_list_data()
        except Exception:
            return

        underlying = payload.get("underlying", {}) if isinstance(payload, dict) else {}
        if not isinstance(underlying, dict):
            return

        for info in underlying.values():
            if not isinstance(info, dict):
                continue
            if info.get("enabled") is False or info.get("is_suspended") is True:
                continue
            simbolo = (
                info.get("symbol")
                or info.get("underlying")
                or info.get("active")
                or info.get("asset_name")
                or info.get("name")
            )
            if not simbolo:
                continue
            yield "digital", simbolo, info

    @staticmethod
    def _tipo_desde_categoria(nombre: str) -> str:
        return "digital" if "digital" in nombre else "binary"

    @staticmethod
    def _generar_aliases_candles(
        raw_symbol: Optional[str],
        resolved_symbol: Optional[str],
        info: Optional[Dict],
        instrument_type: str,
        active_id: Optional[int],
    ) -> Tuple[Union[str, int], ...]:
        """Return a tuple of symbol aliases to try when requesting candles."""

        seen: List[Union[str, int]] = []

        def agregar(valor: Union[str, int, None]) -> None:
            if valor is None:
                return
            if isinstance(valor, (int, float)):
                try:
                    cast = int(valor)
                except (TypeError, ValueError):
                    return
                if cast not in seen:
                    seen.append(cast)
                return
            if not isinstance(valor, str):
                return
            texto = valor.strip()
            if not texto:
                return
            if texto not in seen:
                seen.append(texto)

        def agregar_variantes(base: Optional[str]) -> None:
            if not base or not isinstance(base, str):
                return
            base = base.strip()
            if not base:
                return

            candidatos: List[str] = []
            candidatos.append(base)
            candidatos.append(base.upper())

            sanitizado = _sanitize_symbol_name(base)
            if sanitizado:
                candidatos.append(sanitizado)
                candidatos.append(sanitizado.upper())
                if "/" in sanitizado:
                    candidatos.append(sanitizado.replace("/", ""))
                if "-" in sanitizado:
                    candidatos.append(sanitizado.replace("-", ""))

            if "/" in base:
                candidatos.append(base.replace("/", ""))
            if "-" in base:
                candidatos.append(base.replace("-", ""))

            base_sin_sufijo = _sanitize_symbol_name(base)
            if base_sin_sufijo:
                sufijos: List[str] = []
                if instrument_type == "digital":
                    sufijos.extend(["-OP", "-op"])
                sufijos.extend(["-OTC", "-otc"])
                for sufijo in sufijos:
                    candidatos.append(f"{base_sin_sufijo}{sufijo}")

            for candidato in candidatos:
                agregar(candidato)

        agregar_variantes(raw_symbol)
        agregar_variantes(resolved_symbol)

        if isinstance(info, dict):
            for clave in (
                "symbol",
                "underlying",
                "active",
                "asset_name",
                "name",
                "ticker",
                "pair",
            ):
                agregar_variantes(info.get(clave))
            for clave in ("instrument_id", "instrumentId", "symbol_id", "symbolId"):
                valor = info.get(clave)
                if isinstance(valor, str):
                    agregar_variantes(valor)

        if instrument_type == "digital":
            candidatos_do: List[str] = []
            if isinstance(info, dict):
                for clave in ("instrument_id", "instrumentId"):
                    valor = info.get(clave)
                    if isinstance(valor, str):
                        valor_limpio = valor.strip()
                        if valor_limpio and valor_limpio not in candidatos_do:
                            candidatos_do.append(valor_limpio)

            base_principal = _sanitize_symbol_name(resolved_symbol or raw_symbol or "")
            if base_principal:
                variantes_base: List[str] = []
                for variante in (
                    base_principal,
                    base_principal.upper(),
                    base_principal.lower(),
                    base_principal.replace("/", ""),
                    base_principal.replace("/", "").upper(),
                    base_principal.replace("/", "").lower(),
                    base_principal.replace("-", ""),
                    base_principal.replace("-", "").upper(),
                    base_principal.replace("-", "").lower(),
                ):
                    if variante and variante not in variantes_base:
                        variantes_base.append(variante)

                sufijos_extra = ("", "-OTC", "-otc", "-OP", "-op")
                for variante in variantes_base:
                    for sufijo in sufijos_extra:
                        candidato_generado = f"do{variante}{sufijo}"
                        if candidato_generado and candidato_generado not in candidatos_do:
                            candidatos_do.append(candidato_generado)

            for candidato_do in candidatos_do:
                agregar(candidato_do)

        agregar(active_id)

        if seen:
            return tuple(seen)

        fallback = _sanitize_symbol_name(resolved_symbol or raw_symbol or "")
        if fallback:
            return (fallback,)

        return tuple()

    def _generar_aliases_operacion(
        self,
        raw_symbol: Optional[str],
        resolved_symbol: Optional[str],
        info: Optional[Dict],
        instrument_type: str,
        active_id: Optional[int],
    ) -> Tuple[str, ...]:
        candidatos: List[str] = []

        def agregar(valor: Optional[str]) -> None:
            if not isinstance(valor, str):
                return
            texto = valor.strip()
            if not texto or texto in candidatos:
                return
            candidatos.append(texto)

        def agregar_variantes(base: Optional[str]) -> None:
            if not isinstance(base, str):
                return
            texto = base.strip()
            if not texto:
                return
            agregar(texto)
            agregar(texto.upper())
            agregar(texto.lower())

            normalizado = _sanitize_symbol_name(texto)
            if normalizado and normalizado != texto:
                agregar(normalizado)
                agregar(normalizado.upper())
                agregar(normalizado.lower())
            if "/" in texto:
                agregar(texto.replace("/", ""))
            if "-" in texto:
                agregar(texto.replace("-", ""))
            if normalizado:
                if "/" in normalizado:
                    agregar(normalizado.replace("/", ""))
                if "-" in normalizado:
                    agregar(normalizado.replace("-", ""))

        agregar_variantes(resolved_symbol)
        agregar_variantes(raw_symbol)

        if isinstance(info, dict):
            for clave in ("symbol", "underlying", "active", "asset_name", "name", "ticker", "pair"):
                agregar_variantes(info.get(clave))
            for clave in ("instrument_id", "instrumentId", "symbol_id", "symbolId"):
                valor = info.get(clave)
                if isinstance(valor, str):
                    agregar(valor)

        if instrument_type == "digital":
            base = _sanitize_symbol_name(resolved_symbol or raw_symbol or "")
            if base:
                agregar(f"do{base}")
                agregar(f"do{base.upper()}")
                agregar(f"do{base.lower()}")
                if not base.endswith("-OTC"):
                    agregar(f"{base}-OTC")

        simbolo_principal = resolved_symbol or raw_symbol or ""
        if isinstance(simbolo_principal, str):
            simbolo_principal = simbolo_principal.strip()
        else:
            simbolo_principal = ""

        if simbolo_principal:
            agregar(simbolo_principal)
            normalizado_principal = _sanitize_symbol_name(simbolo_principal)
            if normalizado_principal:
                agregar(normalizado_principal)
                if not normalizado_principal.endswith("OTC"):
                    agregar(f"{normalizado_principal}-OTC")
        else:
            normalizado_principal = None

        ordenados: List[str] = []
        for alias in candidatos:
            texto = alias.strip()
            if texto and texto not in ordenados:
                ordenados.append(texto)

        for preferido in (simbolo_principal, normalizado_principal):
            if preferido and preferido in ordenados:
                ordenados.insert(0, ordenados.pop(ordenados.index(preferido)))

        if not ordenados:
            for candidato in (resolved_symbol, raw_symbol):
                if isinstance(candidato, str) and candidato.strip():
                    ordenados.append(candidato.strip())

        return tuple(ordenados)

    @staticmethod
    def _extraer_instrument_ids(
        info: Optional[Dict],
        instrument_type: str,
    ) -> Tuple[str, ...]:
        if instrument_type != "digital" or not isinstance(info, dict):
            return tuple()

        candidatos: List[str] = []

        def agregar(valor: Optional[Union[str, Iterable]]):
            if isinstance(valor, str):
                texto = valor.strip()
                if texto and texto not in candidatos:
                    candidatos.append(texto)
            elif isinstance(valor, (list, tuple, set)):
                for item in valor:
                    agregar(item)
            elif isinstance(valor, dict):
                for clave in ("instrument_id", "instrumentId", "id", "value", "spot_id"):
                    if clave in valor:
                        agregar(valor.get(clave))

        for clave in (
            "instrument_id",
            "instrumentId",
            "id",
            "value",
            "spot_id",
        ):
            agregar(info.get(clave))

        for clave in (
            "instruments",
            "available_instruments",
            "instrument_ids",
            "items",
            "list",
        ):
            agregar(info.get(clave))

        return tuple(candidatos)

    def _descubrir_pares(self, iq) -> List[TradePair]:
        try:
            activos = iq.get_all_open_time()
        except Exception as exc:  # pragma: no cover - network call
            logging.exception("No se pudieron obtener los horarios de activos: %s", exc)
            return []

        iq.sync_active_catalog(activos)

        candidates: Dict[str, TradePair] = {}
        entradas = list(self._iterar_activos_abiertos(activos))
        if not entradas:
            logging.warning("No se encontraron pares en la respuesta de horarios.")
            entradas = list(self._extraer_digital_meta(iq))
            if entradas:
                logging.info(
                    "Se utilizarán metadatos digitales como respaldo para construir la lista de pares."
                )

        vistos: Set[Tuple[int, str]] = set()
        for categoria, simbolo_original, info in entradas:
            if not isinstance(info, dict):
                continue
            if info.get("open") is False or info.get("is_open") is False:
                continue
            active_id = self._extraer_active_id(info)
            if active_id is None:
                continue
            resolved = iq.resolve_active_symbol(active_id)
            if not resolved:
                resolved = _sanitize_symbol_name(simbolo_original)
            if not resolved:
                continue
            tipo = self._tipo_desde_categoria(categoria)
            clave_vista = (active_id, tipo)
            if clave_vista in vistos:
                continue
            vistos.add(clave_vista)
            display_base = resolved
            display = display_base
            if display in candidates:
                sufijo = "digital" if tipo == "digital" else "binary"
                display = f"{display_base} ({sufijo})"
                idx = 2
                while display in candidates:
                    display = f"{display_base} ({sufijo} {idx})"
                    idx += 1
            api_symbol = resolved or _sanitize_symbol_name(simbolo_original) or simbolo_original
            aliases = self._generar_aliases_candles(
                simbolo_original,
                api_symbol,
                info,
                tipo,
                active_id,
            )
            trade_aliases = self._generar_aliases_operacion(
                simbolo_original,
                api_symbol,
                info,
                tipo,
                active_id,
            )
            instrument_ids = self._extraer_instrument_ids(info, tipo)
            candidates[display] = TradePair(
                display=display,
                api_symbol=api_symbol,
                active_id=active_id,
                instrument_type=tipo,
                candle_aliases=aliases,
                trade_aliases=trade_aliases,
                digital_instrument_ids=instrument_ids,
            )

        activos_validos = list(candidates.values())
        if not activos_validos:
            logging.warning(
                "No se encontraron pares operables tras analizar horarios y metadatos digitales."
            )
            manual = self._construir_pares_manuales(iq)
            if manual:
                logging.info(
                    "Se utilizará un catálogo manual de %s pares como respaldo.",
                    len(manual),
                )
                return manual[:20]
            return []

        return activos_validos[:20]

    def _construir_pares_manuales(self, iq) -> List[TradePair]:
        try:
            iq.sync_active_catalog()
        except Exception:
            logging.debug("No fue posible sincronizar catálogo antes del fallback manual.")

        try:
            digital_payload = iq.get_digital_underlying_list_data()
        except Exception:
            digital_payload = {}

        meta_por_simbolo: Dict[str, Dict] = {}
        if isinstance(digital_payload, dict):
            underlying = digital_payload.get("underlying", {})
            if isinstance(underlying, dict):
                for info in underlying.values():
                    if not isinstance(info, dict):
                        continue
                    simbolo = (
                        info.get("symbol")
                        or info.get("underlying")
                        or info.get("active")
                        or info.get("asset_name")
                        or info.get("name")
                    )
                    clave = _sanitize_symbol_name(simbolo)
                    if clave:
                        meta_por_simbolo[clave.upper()] = info

        candidatos: Dict[str, TradePair] = {}
        vistos: Set[Tuple[int, str]] = set()

        for simbolo, tipo in MANUAL_FALLBACK_INSTRUMENTS:
            limpio = _sanitize_symbol_name(simbolo)
            if limpio:
                base = limpio.upper()
            else:
                base = simbolo.strip().upper()
            if not base:
                continue

            active_id = iq.lookup_active_id(base)
            if active_id is None:
                continue

            clave_vista = (active_id, tipo)
            if clave_vista in vistos:
                continue
            vistos.add(clave_vista)

            display = base
            if display in candidatos:
                sufijo = "digital" if tipo == "digital" else "binary"
                display = f"{base} ({sufijo})"
                idx = 2
                while display in candidatos:
                    display = f"{base} ({sufijo} {idx})"
                    idx += 1

            info = meta_por_simbolo.get(base)
            aliases = self._generar_aliases_candles(
                simbolo,
                base,
                info if isinstance(info, dict) else {},
                tipo,
                active_id,
            )
            if not aliases:
                aliases = (base,)

            trade_aliases = self._generar_aliases_operacion(
                simbolo,
                base,
                info if isinstance(info, dict) else {},
                tipo,
                active_id,
            )
            if not trade_aliases:
                trade_aliases = (base,)

            instrument_ids = self._extraer_instrument_ids(
                info if isinstance(info, dict) else {},
                tipo,
            )

            candidatos[display] = TradePair(
                display=display,
                api_symbol=base,
                active_id=active_id,
                instrument_type=tipo,
                candle_aliases=aliases,
                trade_aliases=trade_aliases,
                digital_instrument_ids=instrument_ids,
            )

        return list(candidatos.values())

    def _filtrar_activos_operables(
        self, iq, pares: List[TradePair]
    ) -> List[TradePair]:
        pares_operables = []
        for par in pares:
            if self._stop_event.is_set():
                break
            df = self.obtener_velas(iq, par, n=5)
            if df.empty:
                self._descartar_par(
                    par, "No devolvió velas durante el filtrado inicial."
                )
                continue
            pares_operables.append(par)

        if pares and not pares_operables:
            logging.warning(
                "Ninguno de los %s pares iniciales devolvió velas válidas.", len(pares)
            )
        elif pares_operables:
            logging.info(
                "✅ Pares operables tras filtro: %s",
                [par.display for par in pares_operables],
            )

        return pares_operables

    @staticmethod
    def _normalizar_candles(raw) -> List[Dict]:
        if isinstance(raw, dict):
            for key in ("candles", "data", "list", "items"):
                maybe = raw.get(key)
                if isinstance(maybe, list):
                    return maybe
            return []
        if isinstance(raw, list):
            return raw
        return []

    @staticmethod
    def _construir_dataframe_velas(candles: List[Dict]) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        if df.empty:
            return df

        columnas = {col.lower(): col for col in df.columns if isinstance(col, str)}
        esquemas = [
            (("open", "max", "min", "close"), {"open": "open", "max": "high", "min": "low", "close": "close"}),
            (("open", "high", "low", "close"), {"open": "open", "high": "high", "low": "low", "close": "close"}),
            (("o", "h", "l", "c"), {"o": "open", "h": "high", "l": "low", "c": "close"}),
        ]

        for required, mapping in esquemas:
            if all(name in columnas for name in required):
                selected = [columnas[name] for name in required]
                renombrado = df[selected].rename(
                    columns={columnas[name]: mapping[name] for name in required}
                )
                renombrado = renombrado[["open", "high", "low", "close"]]
                renombrado = renombrado.apply(pd.to_numeric, errors="coerce")
                renombrado = renombrado.dropna().reset_index(drop=True)
                return renombrado

        return pd.DataFrame()

    def obtener_velas(self, iq, par: TradePair, n=60):
        aliases = par.candle_aliases or (par.api_symbol,)
        errores: List[Tuple[Union[str, int], Exception]] = []
        claves_malformadas: Optional[List[str]] = None

        for alias in aliases:
            objetivo = alias if alias not in (None, "") else par.api_symbol
            if objetivo in (None, ""):
                continue

            intento = 0
            while intento < 2:
                try:
                    raw = iq.get_candles(objetivo, 60, n, time.time())
                except Exception as exc:
                    mensaje = str(exc).lower()
                    if "not found on consts" in mensaje and intento == 0:
                        logging.debug(
                            "Alias %s no encontrado en consts. Re-sincronizando catálogo...",
                            objetivo,
                        )
                        try:
                            iq.sync_active_catalog()
                        except Exception:
                            logging.debug(
                                "Sincronización de catálogo falló para %s", objetivo
                            )
                        intento += 1
                        continue
                    errores.append((objetivo, exc))
                    break

                candles = self._normalizar_candles(raw)
                df = self._construir_dataframe_velas(candles)
                if not df.empty:
                    if objetivo != par.api_symbol:
                        logging.debug(
                            "Velas para %s obtenidas usando alias %s.", par.display, objetivo
                        )
                    return df

                if isinstance(raw, dict):
                    claves_malformadas = list(raw.keys())
                elif candles:
                    claves_malformadas = list(candles[0].keys())
                else:
                    claves_malformadas = []
                logging.debug(
                    "Alias %s devolvió estructura de velas sin OHLC: %s",
                    objetivo,
                    claves_malformadas,
                )
                break

        if errores:
            for alias, exc in errores:
                logging.debug("Error solicitando velas para %s: %s", alias, exc)

        if claves_malformadas is not None:
            logging.error(
                "Estructura de velas inválida para %s tras probar alias. Claves: %s",
                par.display,
                claves_malformadas,
            )

        self._descartar_par(
            par, "El broker no entregó velas válidas tras probar alias."
        )
        return pd.DataFrame()

    def obtener_senal(self, df) -> Tuple[Optional[str], Dict[str, Union[float, str]]]:
        datos = df.apply(pd.to_numeric, errors="coerce").dropna()
        if datos.empty:
            snapshot = IndicatorSnapshot(*(float("nan") for _ in range(7)))
            return None, snapshot.to_table_payload(None)

        close = datos["close"]
        high = datos["high"]
        low = datos["low"]

        rsi_indicator = ta.momentum.RSIIndicator(close)
        ema_fast_indicator = ta.trend.EMAIndicator(close, 9)
        ema_slow_indicator = ta.trend.EMAIndicator(close, 21)
        macd_indicator = ta.trend.MACD(close)
        stoch_indicator = ta.momentum.StochasticOscillator(high, low, close)

        rsi_val = float(rsi_indicator.rsi().iloc[-1])
        ema_fast_val = float(ema_fast_indicator.ema_indicator().iloc[-1])
        ema_slow_val = float(ema_slow_indicator.ema_indicator().iloc[-1])
        macd_val = float(macd_indicator.macd().iloc[-1])
        macd_signal_val = float(macd_indicator.macd_signal().iloc[-1])
        stoch_val = float(stoch_indicator.stoch().iloc[-1])
        stoch_signal_val = float(stoch_indicator.stoch_signal().iloc[-1])

        snapshot = IndicatorSnapshot(
            rsi=rsi_val,
            emaf=ema_fast_val,
            emas=ema_slow_val,
            macd=macd_val,
            macds=macd_signal_val,
            stk=stoch_val,
            std=stoch_signal_val,
        )

        up = down = votos = 0

        def voto_umbral(valor: float, minimo: Optional[float], maximo: Optional[float]) -> None:
            nonlocal up, down, votos
            if math.isnan(valor):
                return
            votos += 1
            if minimo is not None and valor < minimo:
                up += 1
            if maximo is not None and valor > maximo:
                down += 1

        def voto_relacional(a: float, b: float) -> None:
            nonlocal up, down, votos
            if math.isnan(a) or math.isnan(b):
                return
            votos += 1
            if a > b:
                up += 1
            elif a < b:
                down += 1

        voto_umbral(rsi_val, 35, 65)
        voto_relacional(ema_fast_val, ema_slow_val)
        voto_relacional(macd_val, macd_signal_val)
        voto_relacional(stoch_val, stoch_signal_val)

        signal: Optional[str] = None
        if votos:
            threshold = max(2, math.ceil(votos * 0.5))
            if up >= threshold:
                signal = "call"
            elif down >= threshold:
                signal = "put"
            elif votos >= 3:
                diferencia = up - down
                if diferencia >= 2:
                    signal = "call"
                elif diferencia <= -2:
                    signal = "put"

        return signal, snapshot.to_table_payload(signal)

    def _resolver_instrumento_digital(
        self,
        iq,
        par: TradePair,
        alias: str,
    ) -> Optional[str]:
        candidato = (alias or "").strip()
        if not candidato:
            return None

        cache_key = (par.display, candidato.upper(), EXPIRACION)
        if cache_key in self._instrument_cache:
            return self._instrument_cache[cache_key]

        posibles: List[str] = []

        def agregar(valor: Optional[str]) -> None:
            if not isinstance(valor, str):
                return
            texto = valor.strip()
            if texto and texto not in posibles:
                posibles.append(texto)

        agregar(candidato)
        if candidato.lower().startswith("do") and len(candidato) > 2:
            agregar(candidato[2:])

        sanitizado = _sanitize_symbol_name(candidato)
        agregar(sanitizado)

        if sanitizado and not sanitizado.endswith("-OTC"):
            agregar(f"{sanitizado}-OTC")

        if candidato.endswith("-OTC"):
            agregar(candidato[:-4])

        for posible in posibles:
            try:
                resultado = iq.get_digital_spot_instrument(posible, EXPIRACION)
            except AttributeError:
                self._instrument_cache[cache_key] = None
                return None
            except Exception as exc:
                logging.debug(
                    "No se pudo obtener instrument_id para %s alias %s: %s",
                    par.display,
                    posible,
                    exc,
                )
                continue

            instrument_id = None
            if isinstance(resultado, dict):
                instrument_id = (
                    resultado.get("instrument_id")
                    or resultado.get("instrumentId")
                    or resultado.get("id")
                    or resultado.get("value")
                )
                if not instrument_id:
                    for clave in ("result", "instrument", "data"):
                        nested = resultado.get(clave)
                        if isinstance(nested, dict):
                            instrument_id = (
                                nested.get("instrument_id")
                                or nested.get("instrumentId")
                                or nested.get("id")
                                or nested.get("value")
                            )
                            if instrument_id:
                                break
                        elif isinstance(nested, (list, tuple)):
                            for item in nested:
                                if isinstance(item, dict):
                                    instrument_id = (
                                        item.get("instrument_id")
                                        or item.get("instrumentId")
                                        or item.get("id")
                                        or item.get("value")
                                    )
                                    if instrument_id:
                                        break
                            if instrument_id:
                                break
            elif isinstance(resultado, (list, tuple)):
                for item in resultado:
                    if isinstance(item, dict):
                        instrument_id = (
                            item.get("instrument_id")
                            or item.get("instrumentId")
                            or item.get("id")
                            or item.get("value")
                        )
                        if instrument_id:
                            break
                    elif isinstance(item, str):
                        instrument_id = item
                        break

            if isinstance(instrument_id, str) and instrument_id.strip():
                instrument_id = instrument_id.strip()
                self._instrument_cache[cache_key] = instrument_id
                logging.debug(
                    "Instrumento digital resuelto para %s usando %s -> %s",
                    par.display,
                    posible,
                    instrument_id,
                )
                return instrument_id

        self._instrument_cache[cache_key] = None
        return None

    def _ejecutar_operacion(
        self, iq, par: TradePair, senal: str
    ) -> Tuple[bool, Optional[Tuple[str, Union[int, str]]]]:
        aliases = par.trade_aliases or (par.api_symbol,)
        ultimo_error: Optional[Exception] = None
        for alias in aliases:
            if not alias:
                continue
            intento = 0
            while intento < 2:
                try:
                    if par.instrument_type == "digital":
                        candidatos_instrumento: List[str] = [
                            instrumento.strip()
                            for instrumento in par.digital_instrument_ids
                            if isinstance(instrumento, str) and instrumento.strip()
                        ]

                        if not candidatos_instrumento:
                            instrument_id = self._resolver_instrumento_digital(iq, par, alias)
                            if instrument_id:
                                candidatos_instrumento.append(instrument_id)

                        ok = False
                        op_id: Optional[Union[int, str]] = None
                        tipo = "digital"

                        for instrument_id in candidatos_instrumento:
                            try:
                                ok, op_id = iq.buy_digital_spot(
                                    instrument_id,
                                    MONTO,
                                    senal,
                                    EXPIRACION,
                                )
                            except Exception as exc:
                                logging.debug(
                                    "Error al ejecutar digital en %s con instrument_id %s: %s",
                                    par.display,
                                    instrument_id,
                                    exc,
                                )
                                ok, op_id = False, None
                                continue

                            if ok and op_id not in (None, ""):
                                break

                        if not ok or op_id in (None, ""):
                            logging.debug(
                                "Digital rechazado en %s con alias %s; se intentará versión binaria.",
                                par.display,
                                alias,
                            )
                            ok, op_id, tipo = self._intentar_operacion_binaria(iq, alias, senal)
                    else:
                        ok, op_id = iq.buy(MONTO, alias, senal, EXPIRACION)
                        tipo = "binary"
                except Exception as exc:
                    ultimo_error = exc
                    mensaje = str(exc).lower()
                    if "not found on consts" in mensaje and intento == 0:
                        logging.debug(
                            "Alias %s no encontrado para operar; se sincroniza catálogo.",
                            alias,
                        )
                        try:
                            iq.sync_active_catalog()
                        except Exception:
                            logging.debug(
                                "Fallo al sincronizar catálogo antes de reintentar operación."
                            )
                        intento += 1
                        continue
                    logging.debug(
                        "Error intentando operar %s con alias %s: %s",
                        par.display,
                        alias,
                        exc,
                    )
                    break

                if ok and op_id not in (None, ""):
                    if alias != par.api_symbol:
                        logging.debug(
                            "Operación en %s ejecutada usando alias %s.",
                            par.display,
                            alias,
                        )
                    self._fallas_temporales.pop(par.display, None)
                    self._cooldowns.pop(par.display, None)
                    self._cooldown_notices.pop(par.display, None)
                    return True, (tipo, op_id)

                logging.debug(
                    "El broker devolvió estado negativo para %s usando alias %s (ok=%s, id=%s).",
                    par.display,
                    alias,
                    ok,
                    op_id,
                )
                logging.info(
                    "[RECHAZADO] %s alias %s (ok=%s id=%s)",
                    par.display,
                    alias,
                    ok,
                    op_id,
                )
                break

        if ultimo_error is not None:
            logging.error(
                "Fallo al ejecutar operacion %s en %s", senal, par.display, exc_info=(
                    type(ultimo_error),
                    ultimo_error,
                    ultimo_error.__traceback__,
                )
            )

        self._marcar_fallo_temporal(
            par,
            "El broker rechazó la operación en todos los alias disponibles.",
        )
        return False, None

    @staticmethod
    def _preparar_alias_binario(alias: str) -> str:
        if not isinstance(alias, str):
            return ""
        candidato = alias.strip()
        if not candidato:
            return ""
        if ":" in candidato:
            candidato = candidato.split(":")[-1].strip()
        if candidato.lower().startswith("do") and len(candidato) > 2:
            candidato = candidato[2:]
        if candidato.lower().startswith("digital-option"):
            partes = candidato.split(":")
            candidato = partes[-1] if partes else candidato
        return candidato

    @staticmethod
    def _intentar_operacion_binaria(
        iq, alias: str, senal: str
    ) -> Tuple[bool, Optional[Union[int, str]], str]:
        alias_binario = BotWorker._preparar_alias_binario(alias)
        candidatos: List[str] = []

        def agregar(valor: Optional[str]) -> None:
            if not isinstance(valor, str):
                return
            texto = valor.strip()
            if not texto or texto in candidatos:
                return
            candidatos.append(texto)

        agregar(alias_binario)
        agregar(alias)
        agregar(_sanitize_symbol_name(alias_binario or alias))

        admite_parametro_option = True

        for objetivo in candidatos:
            if not objetivo:
                continue
            for option_type in ("turbo", "binary"):
                try:
                    if admite_parametro_option:
                        ok_bin, op_id_bin = iq.buy(
                            MONTO,
                            objetivo,
                            senal,
                            EXPIRACION,
                            option=option_type,
                        )
                    else:
                        ok_bin, op_id_bin = iq.buy(MONTO, objetivo, senal, EXPIRACION)
                except TypeError:
                    admite_parametro_option = False
                    try:
                        ok_bin, op_id_bin = iq.buy(MONTO, objetivo, senal, EXPIRACION)
                    except Exception as exc:
                        logging.debug(
                            "Fallo en intento binario para alias %s: %s",
                            objetivo,
                            exc,
                        )
                        break
                except Exception as exc:
                    logging.debug(
                        "Fallo en intento binario para alias %s: %s",
                        objetivo,
                        exc,
                    )
                    break

                if ok_bin and op_id_bin not in (None, ""):
                    opcion = option_type if admite_parametro_option else "turbo"
                    logging.debug(
                        "Operación binaria ejecutada usando alias %s (option=%s).",
                        objetivo,
                        opcion,
                    )
                    return True, op_id_bin, "binary"

                opcion = option_type if admite_parametro_option else "turbo"
                logging.debug(
                    "El broker devolvió estado negativo en fallback binario (alias %s, option=%s, ok=%s, id=%s).",
                    objetivo,
                    opcion,
                    ok_bin,
                    op_id_bin,
                )

                if not admite_parametro_option:
                    break

        return False, None, "binary"

    def _esperar_resultado(
        self, iq, par: TradePair, ticket: Tuple[str, Union[int, str]]
    ) -> Tuple[Optional[str], float]:
        """Espera hasta obtener el resultado de la operación o agotar el timeout."""

        timeout = EXPIRACION * 60 + 120
        waited = 0
        tipo, op_id = ticket
        while waited < timeout and not self._stop_event.is_set():
            try:
                if tipo == "digital":
                    estado, pnl = iq.check_win_digital_v2(op_id)
                else:
                    estado, pnl = iq.check_win_v4(op_id)
            except Exception:
                logging.exception(
                    "Error al verificar resultado de operación %s (%s)",
                    op_id,
                    par.display,
                )
                return None, 0.0

            if estado is not None:
                try:
                    pnl_val = float(pnl)
                except (TypeError, ValueError):
                    pnl_val = 0.0
                return str(estado), pnl_val

            waited += 1
            time.sleep(1)

        logging.warning(
            "Timeout esperando resultado de la operación %s tras %s segundos",
            ticket[1],
            timeout,
        )
        return None, 0.0

    @staticmethod
    def _normalizar_resultado(valor: str) -> str:
        valor_limpio = (valor or "").strip().lower()
        if valor_limpio == "win":
            return "WIN"
        if valor_limpio in {"loss", "loose", "lost"}:
            return "LOST"
        if valor_limpio == "equal":
            return "EQUAL"
        return valor.upper() if valor else "-"


class BotGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IQ Option Bot - Panel de Monitoreo")
        self.resize(1100, 550)

        layout = QVBoxLayout(self)
        self.label_status = QLabel("⏳ Iniciando conexión...", self)
        layout.addWidget(self.label_status)

        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(
            [
                "Par",
                "RSI",
                "EMA Fast",
                "EMA Slow",
                "MACD",
                "MACD Señal",
                "STK %K",
                "STK %D",
                "Señal",
                "Resultado",
                "PnL",
            ]
        )
        layout.addWidget(self.table)

        self.label_pnl = QLabel("💰 PnL acumulado: 0.00", self)
        layout.addWidget(self.label_pnl)

        self.label_footer = QLabel("", self)
        layout.addWidget(self.label_footer)

        footer_controls = QHBoxLayout()
        footer_controls.addItem(
            QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )
        self.button_toggle = QPushButton("Iniciar bot", self)
        self.button_toggle.clicked.connect(self.on_toggle_clicked)
        footer_controls.addWidget(self.button_toggle)
        layout.addLayout(footer_controls)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_footer)
        self.timer.start(1000)

        self._thread: Optional[QThread] = None
        self._worker: Optional[BotWorker] = None

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
    def on_row_update(self, par: str, payload: Dict[str, Union[float, str]]) -> None:
        r = self.ensure_row(par)
        keys = ["rsi", "emaf", "emas", "macd", "macds", "stk", "std"]
        formats = ["{:.1f}", "{:.5f}", "{:.5f}", "{:.5f}", "{:.5f}", "{:.1f}", "{:.1f}"]
        for column, (key, fmt) in enumerate(zip(keys, formats), start=1):
            value = payload.get(key, 0)
            display_text: str
            if isinstance(value, (int, float)):
                numero = float(value)
                if math.isnan(numero):
                    display_text = "-"
                else:
                    display_text = fmt.format(numero)
            else:
                try:
                    numero = float(value)
                except (TypeError, ValueError):
                    display_text = str(value)
                else:
                    if math.isnan(numero):
                        display_text = "-"
                    else:
                        display_text = fmt.format(numero)
            item = QTableWidgetItem(display_text)
            self.table.setItem(r, column, item)

        signal_text = f"{payload.get('signal', '-')}"
        signal_item = QTableWidgetItem(signal_text)
        if signal_text == "call":
            signal_item.setForeground(Qt.green)
        elif signal_text == "put":
            signal_item.setForeground(Qt.red)
        self.table.setItem(r, 8, signal_item)

        if self.table.item(r, 9) is None:
            self.table.setItem(r, 9, QTableWidgetItem("-"))
        if self.table.item(r, 10) is None:
            self.table.setItem(r, 10, QTableWidgetItem("-"))

    def on_trade_completed(self, par: str, resultado: str, pnl: float, pnl_total: float) -> None:
        r = self.ensure_row(par)
        texto = f"{resultado} ({pnl:.2f})"
        item = QTableWidgetItem(texto)
        if resultado == "WIN":
            item.setForeground(Qt.green)
        elif resultado == "LOST":
            item.setForeground(Qt.red)
        self.table.setItem(r, 9, item)

        pnl_item = QTableWidgetItem(f"{pnl:.2f}")
        if pnl > 0:
            pnl_item.setForeground(Qt.green)
        elif pnl < 0:
            pnl_item.setForeground(Qt.red)
        self.table.setItem(r, 10, pnl_item)
        self.label_pnl.setText(f"💰 PnL acumulado: {pnl_total:.2f}")

    def on_toggle_clicked(self) -> None:
        if self._thread and self._thread.isRunning():
            self.stop_bot()
        else:
            self.start_bot()

    def start_bot(self) -> None:
        if self._thread and self._thread.isRunning():
            return

        self.table.setRowCount(0)
        self.label_pnl.setText("💰 PnL acumulado: 0.00")
        self.button_toggle.setEnabled(False)
        self.label_status.setText("⏳ Iniciando bot...")

        self._thread = QThread(self)
        self._worker = BotWorker()
        self._worker.moveToThread(self._thread)
        self._worker.status_changed.connect(self.label_status.setText)
        self._worker.row_ready.connect(self.on_row_update)
        self._worker.trade_completed.connect(self.on_trade_completed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.finished.connect(self.on_worker_finished)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self.on_thread_finished)
        self._thread.started.connect(self._worker.run)
        self._thread.start()
        self.button_toggle.setText("Detener bot")
        self.button_toggle.setEnabled(True)

    def stop_bot(self) -> None:
        if not self._worker or not self._thread:
            return
        self.button_toggle.setEnabled(False)
        self.label_status.setText("⏹️ Deteniendo bot...")
        self._worker.stop()
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        self.button_toggle.setEnabled(True)
        self.button_toggle.setText("Iniciar bot")

    def on_worker_finished(self) -> None:
        self.button_toggle.setText("Iniciar bot")
        self.button_toggle.setEnabled(True)
        self._worker = None

    def on_thread_finished(self) -> None:
        self._thread = None

    def closeEvent(self, event):
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        super().closeEvent(event)


# ---------------- MAIN ----------------
def create_splash_screen() -> QSplashScreen:
    """Construye una pantalla de bienvenida ilustrativa antes de abrir la GUI."""

    width, height = 640, 320
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    gradient = QLinearGradient(0, 0, 0, height)
    gradient.setColorAt(0.0, QColor(17, 46, 89))
    gradient.setColorAt(1.0, QColor(6, 20, 43))
    painter.fillRect(pixmap.rect(), gradient)

    painter.setPen(QColor("#F5F5F5"))
    title_font = QFont("Segoe UI", 26, QFont.Bold)
    painter.setFont(title_font)
    painter.drawText(
        pixmap.rect(),
        Qt.AlignHCenter | Qt.AlignTop,
        "\nIQ Option Bot"
    )

    subtitle_font = QFont("Segoe UI", 14)
    painter.setFont(subtitle_font)
    painter.drawText(
        pixmap.rect().adjusted(0, 90, 0, -120),
        Qt.AlignHCenter | Qt.AlignTop,
        "Escáner digitales/binarias + Panel de PnL"
    )

    painter.setFont(QFont("Consolas", 11))
    painter.drawText(
        pixmap.rect().adjusted(0, 150, 0, -80),
        Qt.AlignHCenter | Qt.AlignTop,
        "Conectando con IQ Option y preparando indicadores..."
    )
    painter.end()

    splash = QSplashScreen(pixmap)
    splash.setFont(QFont("Segoe UI", 9))
    return splash


if __name__ == "__main__":
    app = QApplication(sys.argv)
    splash = create_splash_screen()
    splash.show()
    app.processEvents()

    window = BotGUI()
    window.show()

    QTimer.singleShot(2000, lambda: splash.finish(window))
    sys.exit(app.exec_())
