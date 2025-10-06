import time
 import logging
 import threading
+import sys
-import pandas as pd
-import numpy as np
-import ta
+from dataclasses import dataclass, asdict
+from typing import Dict, Optional, Tuple, Union, Set, Iterable, List
 from datetime import datetime
-from PyQt5.QtWidgets import 
-    QApplication, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QLabel
+from textwrap import dedent
+
+logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
+
+
+def _ensure_dependency(module_name: str, pip_name: Optional[str] = None) -> None:
+    """Ensure a dependency can be imported, otherwise exit with a friendly message."""
+
+    try:
+        __import__(module_name)
+    except ImportError:
+        friendly_name = pip_name or module_name
+        message = dedent(
+            f"""
+            ❌ Falta la dependencia obligatoria "{module_name}".
+            Instálala ejecutando: pip install {friendly_name}
+            """
+        ).strip()
+        logging.error(message)
+        sys.exit(1)
+
+
+for _module, _pip in (
+    ("pandas", None),
+    ("ta", None),
+    ("PyQt5", "PyQt5"),
+    ("iqoptionapi", "iqoptionapi"),
+):
+    _ensure_dependency(_module, _pip)
+
+import pandas as pd  # noqa: E402  (after dependency check)
+import ta  # noqa: E402
+from PyQt5.QtWidgets import (  # noqa: E402
+    QApplication,
+    QWidget,
+    QVBoxLayout,
+    QHBoxLayout,
+    QSpacerItem,
+    QSizePolicy,
+    QTableWidget,
+    QTableWidgetItem,
+    QLabel,
+    QPushButton,
+    QSplashScreen,
 )
-from PyQt5.QtCore import Qt, QTimer
-from iqoptionapi.stable_api import IQ_Option
+from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread  # noqa: E402
+from PyQt5.QtGui import QFont, QPixmap, QPainter, QColor, QLinearGradient  # noqa: E402
+from iqoptionapi.stable_api import IQ_Option  # noqa: E402
+from iqoptionapi import constants as iq_constants  # noqa: E402
 
 # ---------------- CONFIG ----------------
 EMAIL = "fornerinoalejandro031@gmail.com"
 PASSWORD = "484572ale"
 MONTO = 1.0
 EXPIRACION = 1
 ESPERA_ENTRE_CICLOS = 3
 CICLOS = 50
 MODO = "PRACTICE"
 
-logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
+
+def _normalize_underlying_payload(data: Optional[Dict]) -> Dict:
+    """Ensure the IQ Option response always exposes the ``underlying`` key."""
+
+    if not isinstance(data, dict):
+        logging.debug(
+            "Respuesta de digitales sin formato dict (%s); se normaliza estructura vacía.",
+            type(data).__name__,
+        )
+        return {"underlying": {}}
+
+    underlying = data.get("underlying")
+    if isinstance(underlying, dict):
+        return data
+
+    logging.debug(
+        "Respuesta de digitales sin clave 'underlying' válida (%s); se fuerza diccionario vacío.",
+        type(underlying).__name__,
+    )
+    data["underlying"] = {}
+    return data
+
+
+def _patch_iqoption_underlying() -> None:
+    """Monkey patch IQ Option SDK to avoid KeyError in internal threads."""
+
+    original = IQ_Option.get_digital_underlying_list_data
+
+    # Evita volver a aplicar el parche
+    if getattr(original, "__name__", "") == "safe_get_digital_underlying_list_data":
+        return
+
+    def safe_get_digital_underlying_list_data(self, *args, **kwargs):  # type: ignore
+        try:
+            raw = original(self, *args, **kwargs)
+        except Exception:
+            logging.exception(
+                "Fallo al obtener digitales desde el SDK; se devuelve estructura vacía."
+            )
+            return {"underlying": {}}
+
+        return _normalize_underlying_payload(raw)
+
+    safe_get_digital_underlying_list_data.__name__ = "safe_get_digital_underlying_list_data"
+    safe_get_digital_underlying_list_data.__wrapped__ = original  # type: ignore[attr-defined]
+    IQ_Option.get_digital_underlying_list_data = safe_get_digital_underlying_list_data  # type: ignore[assignment]
+
+
+_patch_iqoption_underlying()
+
+
+def _sanitize_symbol_name(symbol: Optional[str]) -> Optional[str]:
+    """Return a broker-friendly symbol stripping IQ Option suffixes."""
+
+    if not symbol or not isinstance(symbol, str):
+        return None
+
+    cleaned = symbol.strip().upper()
+    for suffix in ("-OP", "-OTC"):
+        if cleaned.endswith(suffix):
+            cleaned = cleaned[: -len(suffix)]
+            break
+    return cleaned or None
+
+
+class SafeIQOption(IQ_Option):
+    """Versión protegida del cliente IQ Option que evita fallos del SDK."""
+
+    def get_digital_underlying_list_data(self):  # pragma: no cover - delega en SDK
+        try:
+            data = super().get_digital_underlying_list_data()
+        except Exception:
+            logging.exception(
+                "Fallo al obtener digitales; se devuelve estructura vacía para evitar KeyError."
+            )
+            return {"underlying": {}}
+
+        return _normalize_underlying_payload(data)
+
+    def sync_active_catalog(self, open_time_payload: Optional[Dict] = None) -> None:
+        """Completa los mapas internos del SDK con los pares disponibles detectados."""
+
+        try:
+            containers: Iterable[Dict] = self._collect_active_containers()
+        except Exception:
+            logging.debug("No fue posible preparar los contenedores de activos.")
+            containers = ()
+
+        if open_time_payload is None:
+            try:
+                open_time_payload = self.get_all_open_time()
+            except Exception:
+                open_time_payload = {}
+
+        updated_symbols: Set[str] = set()
+
+        def register(symbol: Optional[str], active_id: Optional[Union[int, str]]) -> None:
+            if not symbol:
+                return
+            try:
+                normalized_id = int(active_id) if active_id is not None else None
+            except (TypeError, ValueError):
+                return
+            if normalized_id is None:
+                return
+
+            symbol_key = symbol.upper()
+            changed = False
+            for container in containers:
+                if self._update_container(container, symbol_key, normalized_id):
+                    changed = True
+            if changed:
+                updated_symbols.add(symbol_key)
+
+        # Horarios de apertura (turbo/digital/binario)
+        if isinstance(open_time_payload, dict):
+            for category in ("turbo", "digital", "binary"):
+                entries = open_time_payload.get(category, {})
+                if not isinstance(entries, dict):
+                    continue
+                for symbol, info in entries.items():
+                    if not isinstance(info, dict):
+                        continue
+                    register(symbol, info.get("id") or info.get("active_id"))
+
+        # Metadatos digitales (incluye instrument_id/asset_id)
+        try:
+            digital_meta = self.get_digital_underlying_list_data().get("underlying", {})
+        except Exception:
+            digital_meta = {}
+
+        if isinstance(digital_meta, dict):
+            for _, info in digital_meta.items():
+                if not isinstance(info, dict):
+                    continue
+                symbol = (
+                    info.get("symbol")
+                    or info.get("asset_name")
+                    or info.get("underlying")
+                    or info.get("active")
+                )
+                register(
+                    symbol,
+                    info.get("active_id")
+                    or info.get("id")
+                    or info.get("asset_id")
+                    or info.get("instrument_id"),
+                )
+
+        if updated_symbols:
+            logging.info(
+                "📈 Registrados %s pares adicionales detectados en la API.",
+                len(updated_symbols),
+            )
+
+    # ---- utilitarios internos ----
+    def _collect_active_containers(self) -> Iterable[Dict]:
+        containers = []
+
+        for attr in ("active_to_id", "available_leverages", "instruments", "all_underlying_list"):
+            data = getattr(self, attr, None)
+            if isinstance(data, dict):
+                containers.append(data)
+
+        actives = getattr(self, "ACTIVES", None)
+        if isinstance(actives, dict):
+            containers.append(actives)
+
+        for name in ("ACTIVES", "ACTIVES_ID", "assets", "assets_name"):
+            data = getattr(iq_constants, name, None)
+            if isinstance(data, dict):
+                containers.append(data)
+
+        # Estructuras anidadas usadas por métodos privados del SDK
+        for attr in ("api_option_init_all_result", "api_game_getcandles_v2"):
+            data = getattr(self, attr, None)
+            if isinstance(data, dict):
+                containers.append(data)
+            elif hasattr(data, "__dict__"):
+                nested = getattr(data, "__dict__", {})
+                for value in nested.values():
+                    if isinstance(value, dict):
+                        containers.append(value)
+
+        return containers
+
+    @staticmethod
+    def _update_container(container: Dict, symbol: str, active_id: int) -> bool:
+        if not isinstance(container, dict):
+            return False
+
+        if not container:
+            container[symbol] = {"id": active_id, "name": symbol}
+            return True
+
+        sample_key = next(iter(container.keys()))
+
+        # Diccionarios {"EURUSD": {...}}
+        if isinstance(sample_key, str):
+            value = container.get(symbol)
+            if isinstance(value, dict):
+                changed = value.get("id") != active_id
+                value["id"] = active_id
+                value.setdefault("name", symbol)
+                return changed
+
+            if value != active_id:
+                if any(isinstance(v, dict) for v in container.values()):
+                    container[symbol] = {"id": active_id, "name": symbol}
+                else:
+                    container[symbol] = active_id
+                return True
+            return False
+
+        # Diccionarios {76: "EURUSD"}
+        if isinstance(sample_key, int):
+            existing = container.get(active_id)
+            if existing != symbol:
+                container[active_id] = symbol
+                return True
+            return False
+
+        # Fallback genérico
+        container[symbol] = active_id
+        return True
+
+    @staticmethod
+    def _extract_symbol_from_value(value: Union[str, Dict, None]) -> Optional[str]:
+        if isinstance(value, str) and value:
+            return value
+        if isinstance(value, dict):
+            for key in ("symbol", "name", "underlying", "active", "asset_name"):
+                candidate = value.get(key)
+                if isinstance(candidate, str) and candidate:
+                    return candidate
+        return None
+
+    def resolve_active_symbol(self, active_id: Optional[Union[int, str]]) -> Optional[str]:
+        """Map an ``active_id`` to the canonical symbol when available."""
+
+        try:
+            normalized_id = int(active_id) if active_id is not None else None
+        except (TypeError, ValueError):
+            return None
+
+        if normalized_id is None:
+            return None
+
+        for container in self._collect_active_containers():
+            if not isinstance(container, dict):
+                continue
+            for key, value in container.items():
+                if isinstance(key, int) and key == normalized_id:
+                    symbol = self._extract_symbol_from_value(value)
+                    if symbol:
+                        return _sanitize_symbol_name(symbol)
+                if value == normalized_id and isinstance(key, str):
+                    return _sanitize_symbol_name(key)
+                if isinstance(value, dict) and value.get("id") == normalized_id:
+                    if isinstance(key, str) and key:
+                        return _sanitize_symbol_name(key)
+                    symbol = self._extract_symbol_from_value(value)
+                    if symbol:
+                        return _sanitize_symbol_name(symbol)
+
+        return None
 
 # ---------------- CLASE GUI ----------------
+@dataclass
+class IndicatorSnapshot:
+    """Represents the most recent technical indicator values for a pair."""
+
+    rsi: float
+    emaf: float
+    emas: float
+    macd: float
+    macds: float
+    stk: float
+    std: float
+
+    def to_table_payload(self, signal: Optional[str]) -> Dict[str, Union[float, str]]:
+        """Prepare the dictionary used to update the GUI table."""
+
+        payload = asdict(self)
+        payload["signal"] = signal or "-"
+        return payload
+
+
+@dataclass(frozen=True)
+class TradePair:
+    """Metadata describing a tradable instrument."""
+
+    display: str
+    api_symbol: str
+    active_id: int
+    instrument_type: str  # "digital" or "binary"
+    candle_aliases: Tuple[Union[str, int], ...]
+
+
+class BotWorker(QObject):
+    status_changed = pyqtSignal(str)
+    row_ready = pyqtSignal(str, dict)
+    trade_completed = pyqtSignal(str, str, float, float)
+    finished = pyqtSignal()
+
+    def __init__(self):
+        super().__init__()
+        self._stop_event = threading.Event()
+        self._pnl_acumulado = 0.0
+        self._pares_descartados: Set[str] = set()
+
+    def stop(self):
+        self._stop_event.set()
+
+    def run(self):
+        self.status_changed.emit("🔌 Conectando a IQ Option...")
+        iq = SafeIQOption(EMAIL, PASSWORD)
+        self._stop_event.clear()
+        try:
+            check, reason = iq.connect()
+        except Exception as exc:  # pragma: no cover - network call
+            logging.exception("Error inesperado al conectar con IQ Option: %s", exc)
+            self.status_changed.emit("❌ Error conexión inesperado. Ver logs.")
+            self.finished.emit()
+            return
+
+        try:
+            if not check:
+                self.status_changed.emit(f"❌ Error conexión: {reason}")
+                self.finished.emit()
+                return
+
+            iq.change_balance(MODO)
+            saldo = iq.get_balance()
+            self.status_changed.emit(f"✅ Conectado a {MODO} | Saldo: {saldo:.2f}")
+
+            self._pares_descartados.clear()
+            logging.info("♻️ Escaneando pares digitales y binarias disponibles...")
+            pares_validos = self._descubrir_pares(iq)
+            logging.info(
+                "✅ Pares detectados: %s",
+                [par.display for par in pares_validos],
+            )
+
+            pares_validos = self._filtrar_activos_operables(iq, pares_validos)
+            if not pares_validos:
+                self.status_changed.emit(
+                    "⚠️ No se encontraron pares digitales/binarias disponibles."
+                )
+                self.finished.emit()
+                return
+
+            for ciclo in range(1, CICLOS + 1):
+                if self._stop_event.is_set():
+                    break
+                logging.info(f"=== Ciclo {ciclo}/{CICLOS} ===")
+                activos_restantes = False
+                for par in list(pares_validos):
+                    if self._stop_event.is_set():
+                        break
+                    if par.display in self._pares_descartados:
+                        continue
+                    activos_restantes = True
+                    df = self.obtener_velas(iq, par)
+                    if df.empty:
+                        self._descartar_par(par, "Sin velas devueltas por el broker.")
+                        continue
+                    senal, data = self.obtener_senal(df)
+                    self.row_ready.emit(par.display, data)
+
+                    if senal:
+                        ok, ticket = self._ejecutar_operacion(iq, par, senal)
+                        if ok and ticket is not None:
+                            logging.info(f"[OK] {senal.upper()} en {par.display}")
+                            resultado, pnl = self._esperar_resultado(iq, par, ticket)
+                            if resultado:
+                                self._pnl_acumulado += pnl
+                                texto_resultado = self._normalizar_resultado(resultado)
+                                mensaje = (
+                                    f"{texto_resultado} en {par.display} | PnL operación: {pnl:.2f} | "
+                                    f"PnL acumulado: {self._pnl_acumulado:.2f}"
+                                )
+                                logging.info(mensaje)
+                                self.status_changed.emit(mensaje)
+                                self.trade_completed.emit(
+                                    par.display, texto_resultado, pnl, self._pnl_acumulado
+                                )
+                            else:
+                                logging.warning(
+                                    "No se obtuvo resultado para la operación en %s",
+                                    par.display,
+                                )
+                        else:
+                            logging.warning(
+                                f"[FAIL] No se pudo ejecutar {senal} en {par.display}"
+                            )
+                            self._descartar_par(
+                                par,
+                                "El broker rechazó la operación, se descarta el par.",
+                            )
+                    time.sleep(0.6)
+
+                pares_validos = [
+                    par
+                    for par in pares_validos
+                    if par.display not in self._pares_descartados
+                ]
+                if not pares_validos:
+                    msg = "⚠️ No quedan pares operables tras los descartes."
+                    logging.warning(msg)
+                    self.status_changed.emit(msg)
+                    break
+                if not activos_restantes:
+                    msg = "⚠️ Todos los pares fueron descartados por fallos del broker."
+                    logging.warning(msg)
+                    self.status_changed.emit(msg)
+                    break
+                time.sleep(ESPERA_ENTRE_CICLOS)
+
+            if self._stop_event.is_set():
+                mensaje_final = "⏹️ Bot detenido manualmente."
+            else:
+                mensaje_final = "✅ Bot finalizado correctamente."
+            logging.info(mensaje_final)
+            self.status_changed.emit(mensaje_final)
+            self.finished.emit()
+        finally:
+            for cierre in ("close", "close_connection", "api_close"):
+                metodo = getattr(iq, cierre, None)
+                if callable(metodo):
+                    try:
+                        metodo()
+                    except Exception:
+                        continue
+                    break
+
+    def _descartar_par(self, par: TradePair, motivo: str) -> None:
+        if par.display in self._pares_descartados:
+            return
+        self._pares_descartados.add(par.display)
+        logging.info("Activo %s descartado: %s", par.display, motivo)
+
+    @staticmethod
+    def _generar_aliases_candles(
+        raw_symbol: Optional[str],
+        resolved_symbol: Optional[str],
+        info: Optional[Dict],
+        instrument_type: str,
+        active_id: Optional[int],
+    ) -> Tuple[Union[str, int], ...]:
+        """Return a tuple of symbol aliases to try when requesting candles."""
+
+        seen: List[Union[str, int]] = []
+
+        def agregar(valor: Union[str, int, None]) -> None:
+            if valor is None:
+                return
+            if isinstance(valor, (int, float)):
+                try:
+                    cast = int(valor)
+                except (TypeError, ValueError):
+                    return
+                if cast not in seen:
+                    seen.append(cast)
+                return
+            if not isinstance(valor, str):
+                return
+            texto = valor.strip()
+            if not texto:
+                return
+            if texto not in seen:
+                seen.append(texto)
+
+        def agregar_variantes(base: Optional[str]) -> None:
+            if not base or not isinstance(base, str):
+                return
+            base = base.strip()
+            if not base:
+                return
+
+            candidatos: List[str] = []
+            candidatos.append(base)
+            candidatos.append(base.upper())
+
+            sanitizado = _sanitize_symbol_name(base)
+            if sanitizado:
+                candidatos.append(sanitizado)
+                candidatos.append(sanitizado.upper())
+                if "/" in sanitizado:
+                    candidatos.append(sanitizado.replace("/", ""))
+                if "-" in sanitizado:
+                    candidatos.append(sanitizado.replace("-", ""))
+
+            if "/" in base:
+                candidatos.append(base.replace("/", ""))
+            if "-" in base:
+                candidatos.append(base.replace("-", ""))
+
+            base_sin_sufijo = _sanitize_symbol_name(base)
+            if base_sin_sufijo:
+                sufijos: List[str] = []
+                if instrument_type == "digital":
+                    sufijos.extend(["-OP", "-op"])
+                sufijos.extend(["-OTC", "-otc"])
+                for sufijo in sufijos:
+                    candidatos.append(f"{base_sin_sufijo}{sufijo}")
+
+            for candidato in candidatos:
+                agregar(candidato)
+
+        agregar_variantes(raw_symbol)
+        agregar_variantes(resolved_symbol)
+
+        if isinstance(info, dict):
+            for clave in ("symbol", "underlying", "active", "asset_name", "name", "ticker"):
+                agregar_variantes(info.get(clave))
+
+        agregar(active_id)
+
+        if seen:
+            return tuple(seen)
+
+        fallback = _sanitize_symbol_name(resolved_symbol or raw_symbol or "")
+        if fallback:
+            return (fallback,)
+
+        return tuple()
+
+    def _descubrir_pares(self, iq) -> List[TradePair]:
+        try:
+            activos = iq.get_all_open_time()
+        except Exception as exc:  # pragma: no cover - network call
+            logging.exception("No se pudieron obtener los horarios de activos: %s", exc)
+            return []
+
+        iq.sync_active_catalog(activos)
+
+        candidates: Dict[str, TradePair] = {}
+        if isinstance(activos, dict):
+            for category in ("digital", "binary", "turbo"):
+                entries = activos.get(category, {})
+                if not isinstance(entries, dict):
+                    continue
+                for par, info in entries.items():
+                    if not isinstance(info, dict):
+                        continue
+                    if info.get("open") is False:
+                        continue
+                    try:
+                        active_id = int(info.get("id") or info.get("active_id"))
+                    except (TypeError, ValueError):
+                        continue
+                    resolved = iq.resolve_active_symbol(active_id)
+                    if not resolved:
+                        resolved = _sanitize_symbol_name(par)
+                    if not resolved:
+                        continue
+                    instrument_type = "digital" if category == "digital" else "binary"
+                    display_base = resolved
+                    display = display_base
+                    if display in candidates:
+                        suffix = "digital" if instrument_type == "digital" else "binary"
+                        display = f"{display_base} ({suffix})"
+                        idx = 2
+                        while display in candidates:
+                            display = f"{display_base} ({suffix} {idx})"
+                            idx += 1
+                    api_symbol = resolved or _sanitize_symbol_name(par) or par
+                    aliases = self._generar_aliases_candles(
+                        par,
+                        api_symbol,
+                        info,
+                        instrument_type,
+                        active_id,
+                    )
+                    candidates[display] = TradePair(
+                        display=display,
+                        api_symbol=api_symbol,
+                        active_id=active_id,
+                        instrument_type=instrument_type,
+                        candle_aliases=aliases,
+                    )
+
+        activos_validos = list(candidates.values())
+        if not activos_validos:
+            logging.warning("No se encontraron pares en la respuesta de horarios.")
+
+        # Solo procesamos los primeros 20 para evitar sobrecargar la GUI
+        return activos_validos[:20]
+
+    def _filtrar_activos_operables(
+        self, iq, pares: List[TradePair]
+    ) -> List[TradePair]:
+        pares_operables = []
+        for par in pares:
+            if self._stop_event.is_set():
+                break
+            df = self.obtener_velas(iq, par, n=5)
+            if df.empty:
+                self._descartar_par(
+                    par, "No devolvió velas durante el filtrado inicial."
+                )
+                continue
+            pares_operables.append(par)
+
+        if pares and not pares_operables:
+            logging.warning(
+                "Ninguno de los %s pares iniciales devolvió velas válidas.", len(pares)
+            )
+        elif pares_operables:
+            logging.info(
+                "✅ Pares operables tras filtro: %s",
+                [par.display for par in pares_operables],
+            )
+
+        return pares_operables
+
+    @staticmethod
+    def _normalizar_candles(raw) -> List[Dict]:
+        if isinstance(raw, dict):
+            for key in ("candles", "data", "list", "items"):
+                maybe = raw.get(key)
+                if isinstance(maybe, list):
+                    return maybe
+            return []
+        if isinstance(raw, list):
+            return raw
+        return []
+
+    @staticmethod
+    def _construir_dataframe_velas(candles: List[Dict]) -> pd.DataFrame:
+        if not candles:
+            return pd.DataFrame()
+
+        df = pd.DataFrame(candles)
+        if df.empty:
+            return df
+
+        columnas = {col.lower(): col for col in df.columns if isinstance(col, str)}
+        esquemas = [
+            (("open", "max", "min", "close"), {"open": "open", "max": "high", "min": "low", "close": "close"}),
+            (("open", "high", "low", "close"), {"open": "open", "high": "high", "low": "low", "close": "close"}),
+            (("o", "h", "l", "c"), {"o": "open", "h": "high", "l": "low", "c": "close"}),
+        ]
+
+        for required, mapping in esquemas:
+            if all(name in columnas for name in required):
+                selected = [columnas[name] for name in required]
+                renombrado = df[selected].rename(
+                    columns={columnas[name]: mapping[name] for name in required}
+                )
+                return renombrado[["open", "high", "low", "close"]]
+
+        return pd.DataFrame()
+
+    def obtener_velas(self, iq, par: TradePair, n=60):
+        aliases = par.candle_aliases or (par.api_symbol,)
+        errores: List[Tuple[Union[str, int], Exception]] = []
+        claves_malformadas: Optional[List[str]] = None
+        sync_intentado = False
+
+        for alias in aliases:
+            objetivo = alias if alias not in (None, "") else par.api_symbol
+            if objetivo in (None, ""):
+                continue
+
+            try:
+                raw = iq.get_candles(objetivo, 60, n, time.time())
+            except Exception as exc:
+                mensaje = str(exc).lower()
+                if "not found on consts" in mensaje and not sync_intentado:
+                    logging.debug(
+                        "Alias %s no encontrado en consts. Re-sincronizando catálogo...",
+                        objetivo,
+                    )
+                    try:
+                        iq.sync_active_catalog()
+                    except Exception:
+                        logging.debug("Sincronización de catálogo falló para %s", objetivo)
+                    sync_intentado = True
+                    try:
+                        raw = iq.get_candles(objetivo, 60, n, time.time())
+                    except Exception as retry_exc:
+                        errores.append((objetivo, retry_exc))
+                        continue
+                else:
+                    errores.append((objetivo, exc))
+                    continue
+
+            candles = self._normalizar_candles(raw)
+            df = self._construir_dataframe_velas(candles)
+            if not df.empty:
+                if objetivo != par.api_symbol:
+                    logging.debug(
+                        "Velas para %s obtenidas usando alias %s.", par.display, objetivo
+                    )
+                return df
+
+            if isinstance(raw, dict):
+                claves_malformadas = list(raw.keys())
+            elif candles:
+                claves_malformadas = list(candles[0].keys())
+            else:
+                claves_malformadas = []
+            logging.debug(
+                "Alias %s devolvió estructura de velas sin OHLC: %s", objetivo, claves_malformadas
+            )
+
+        if errores:
+            for alias, exc in errores:
+                logging.debug("Error solicitando velas para %s: %s", alias, exc)
+
+        if claves_malformadas is not None:
+            logging.error(
+                "Estructura de velas inválida para %s tras probar alias. Claves: %s",
+                par.display,
+                claves_malformadas,
+            )
+
+        self._descartar_par(
+            par, "El broker no entregó velas válidas tras probar alias."
+        )
+        return pd.DataFrame()
+
+    def obtener_senal(self, df) -> Tuple[Optional[str], Dict[str, Union[float, str]]]:
+        close = df["close"]
+        high = df["high"]
+        low = df["low"]
+
+        rsi_indicator = ta.momentum.RSIIndicator(close)
+        ema_fast_indicator = ta.trend.EMAIndicator(close, 9)
+        ema_slow_indicator = ta.trend.EMAIndicator(close, 21)
+        macd_indicator = ta.trend.MACD(close)
+        stoch_indicator = ta.momentum.StochasticOscillator(high, low, close)
+
+        snapshot = IndicatorSnapshot(
+            rsi=rsi_indicator.rsi().iloc[-1],
+            emaf=ema_fast_indicator.ema_indicator().iloc[-1],
+            emas=ema_slow_indicator.ema_indicator().iloc[-1],
+            macd=macd_indicator.macd().iloc[-1],
+            macds=macd_indicator.macd_signal().iloc[-1],
+            stk=stoch_indicator.stoch().iloc[-1],
+            std=stoch_indicator.stoch_signal().iloc[-1],
+        )
+
+        up, down = 0, 0
+        if snapshot.rsi < 35:
+            up += 1
+        if snapshot.rsi > 65:
+            down += 1
+        if snapshot.emaf > snapshot.emas:
+            up += 1
+        if snapshot.emaf < snapshot.emas:
+            down += 1
+        if snapshot.macd > snapshot.macds:
+            up += 1
+        if snapshot.macd < snapshot.macds:
+            down += 1
+        if snapshot.stk > snapshot.std:
+            up += 1
+        if snapshot.stk < snapshot.std:
+            down += 1
+
+        signal: Optional[str] = None
+        if up >= 3:
+            signal = "call"
+        elif down >= 3:
+            signal = "put"
+
+        return signal, snapshot.to_table_payload(signal)
+
+    def _ejecutar_operacion(
+        self, iq, par: TradePair, senal: str
+    ) -> Tuple[bool, Optional[Tuple[str, Union[int, str]]]]:
+        try:
+            if par.instrument_type == "digital":
+                ok, op_id = iq.buy_digital_spot(par.api_symbol, MONTO, senal, EXPIRACION)
+                tipo = "digital"
+            else:
+                ok, op_id = iq.buy(MONTO, par.api_symbol, senal, EXPIRACION)
+                tipo = "binary"
+        except Exception:
+            logging.exception(
+                "Error al ejecutar operacion %s en %s", senal, par.display
+            )
+            self._descartar_par(
+                par,
+                "Error de red al ejecutar operación, se descarta el par.",
+            )
+            return False, None
+
+        if not ok or op_id in (None, ""):
+            return False, None
+
+        return True, (tipo, op_id)
+
+    def _esperar_resultado(
+        self, iq, par: TradePair, ticket: Tuple[str, Union[int, str]]
+    ) -> Tuple[Optional[str], float]:
+        """Espera hasta obtener el resultado de la operación o agotar el timeout."""
+
+        timeout = EXPIRACION * 60 + 120
+        waited = 0
+        tipo, op_id = ticket
+        while waited < timeout and not self._stop_event.is_set():
+            try:
+                if tipo == "digital":
+                    estado, pnl = iq.check_win_digital_v2(op_id)
+                else:
+                    estado, pnl = iq.check_win_v4(op_id)
+            except Exception:
+                logging.exception(
+                    "Error al verificar resultado de operación %s (%s)",
+                    op_id,
+                    par.display,
+                )
+                return None, 0.0
+
+            if estado is not None:
+                try:
+                    pnl_val = float(pnl)
+                except (TypeError, ValueError):
+                    pnl_val = 0.0
+                return str(estado), pnl_val
+
+            waited += 1
+            time.sleep(1)
+
+        logging.warning(
+            "Timeout esperando resultado de la operación %s tras %s segundos",
+            ticket[1],
+            timeout,
+        )
+        return None, 0.0
+
+    @staticmethod
+    def _normalizar_resultado(valor: str) -> str:
+        valor_limpio = (valor or "").strip().lower()
+        if valor_limpio == "win":
+            return "WIN"
+        if valor_limpio in {"loss", "loose", "lost"}:
+            return "LOST"
+        if valor_limpio == "equal":
+            return "EQUAL"
+        return valor.upper() if valor else "-"
+
+
 class BotGUI(QWidget):
     def __init__(self):
         super().__init__()
         self.setWindowTitle("IQ Option Bot - Panel de Monitoreo")
         self.resize(1100, 550)
 
         layout = QVBoxLayout(self)
         self.label_status = QLabel("⏳ Iniciando conexión...", self)
         layout.addWidget(self.label_status)
 
-        self.table = QTableWidget(0, 9)
-        self.table.setHorizontalHeaderLabels([
-            "Par", "RSI", "EMA Fast", "EMA Slow", "MACD", "Signal", "STK %K", "STK %D", "Señal"
-        ])
+        self.table = QTableWidget(0, 11)
+        self.table.setHorizontalHeaderLabels(
+            [
+                "Par",
+                "RSI",
+                "EMA Fast",
+                "EMA Slow",
+                "MACD",
+                "MACD Señal",
+                "STK %K",
+                "STK %D",
+                "Señal",
+                "Resultado",
+                "PnL",
+            ]
+        )
         layout.addWidget(self.table)
 
+        self.label_pnl = QLabel("💰 PnL acumulado: 0.00", self)
+        layout.addWidget(self.label_pnl)
+
         self.label_footer = QLabel("", self)
         layout.addWidget(self.label_footer)
 
-        self.iq = None
-        self.pares_validos = []
-        self.thread = threading.Thread(target=self.start_bot, daemon=True)
-        self.thread.start()
+        footer_controls = QHBoxLayout()
+        footer_controls.addItem(
+            QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
+        )
+        self.button_toggle = QPushButton("Iniciar bot", self)
+        self.button_toggle.clicked.connect(self.on_toggle_clicked)
+        footer_controls.addWidget(self.button_toggle)
+        layout.addLayout(footer_controls)
 
         self.timer = QTimer()
         self.timer.timeout.connect(self.update_footer)
         self.timer.start(1000)
 
+        self._thread: Optional[QThread] = None
+        self._worker: Optional[BotWorker] = None
+
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
-    def on_row_update(self, par, payload):
+    def on_row_update(self, par: str, payload: Dict[str, Union[float, str]]) -> None:
         r = self.ensure_row(par)
-        vals = [
-            f"{payload.get('rsi', 0):.1f}",
-            f"{payload.get('emaf', 0):.5f}",
-            f"{payload.get('emas', 0):.5f}",
-            f"{payload.get('macd', 0):.5f}",
-            f"{payload.get('macds', 0):.5f}",
-            f"{payload.get('stk', 0):.1f}",
-            f"{payload.get('std', 0):.1f}",
-            f"{payload.get('signal', '-')}"
-        ]
-        for i, v in enumerate(vals, start=1):
-            item = QTableWidgetItem(v)
-            if i == 8:  # Señal
-                sig = payload.get("signal", "-")
-                if sig == "call":
-                    item.setForeground(Qt.green)
-                elif sig == "put":
-                    item.setForeground(Qt.red)
-            self.table.setItem(r, i, item)
-
-    def start_bot(self):
-        self.label_status.setText("🔌 Conectando a IQ Option...")
-        iq = IQ_Option(EMAIL, PASSWORD)
-        check, reason = iq.connect()
-        if not check:
-            self.label_status.setText(f"❌ Error conexión: {reason}")
+        keys = ["rsi", "emaf", "emas", "macd", "macds", "stk", "std"]
+        formats = ["{:.1f}", "{:.5f}", "{:.5f}", "{:.5f}", "{:.5f}", "{:.1f}", "{:.1f}"]
+        for column, (key, fmt) in enumerate(zip(keys, formats), start=1):
+            value = payload.get(key, 0)
+            item = QTableWidgetItem(fmt.format(value))
+            self.table.setItem(r, column, item)
+
+        signal_text = f"{payload.get('signal', '-')}"
+        signal_item = QTableWidgetItem(signal_text)
+        if signal_text == "call":
+            signal_item.setForeground(Qt.green)
+        elif signal_text == "put":
+            signal_item.setForeground(Qt.red)
+        self.table.setItem(r, 8, signal_item)
+
+        if self.table.item(r, 9) is None:
+            self.table.setItem(r, 9, QTableWidgetItem("-"))
+        if self.table.item(r, 10) is None:
+            self.table.setItem(r, 10, QTableWidgetItem("-"))
+
+    def on_trade_completed(self, par: str, resultado: str, pnl: float, pnl_total: float) -> None:
+        r = self.ensure_row(par)
+        texto = f"{resultado} ({pnl:.2f})"
+        item = QTableWidgetItem(texto)
+        if resultado == "WIN":
+            item.setForeground(Qt.green)
+        elif resultado == "LOST":
+            item.setForeground(Qt.red)
+        self.table.setItem(r, 9, item)
+
+        pnl_item = QTableWidgetItem(f"{pnl:.2f}")
+        if pnl > 0:
+            pnl_item.setForeground(Qt.green)
+        elif pnl < 0:
+            pnl_item.setForeground(Qt.red)
+        self.table.setItem(r, 10, pnl_item)
+        self.label_pnl.setText(f"💰 PnL acumulado: {pnl_total:.2f}")
+
+    def on_toggle_clicked(self) -> None:
+        if self._thread and self._thread.isRunning():
+            self.stop_bot()
+        else:
+            self.start_bot()
+
+    def start_bot(self) -> None:
+        if self._thread and self._thread.isRunning():
             return
 
-        iq.change_balance(MODO)
-        saldo = iq.get_balance()
-        self.label_status.setText(f"✅ Conectado a {MODO} | Saldo: {saldo:.2f}")
-
-        logging.info("♻️ Escaneando pares OTC disponibles...")
-        activos = iq.get_all_open_time()
-        pares_disponibles = [k for k, v in activos["turbo"].items() if v["open"]]
-        pares_otc = [p for p in pares_disponibles if "-OTC" in p]
-        self.pares_validos = pares_otc[:20]  # analiza los primeros 20
-        logging.info(f"✅ Pares OTC detectados: {self.pares_validos}")
-
-        for ciclo in range(1, CICLOS + 1):
-            logging.info(f"=== Ciclo {ciclo}/{CICLOS} ===")
-            for par in self.pares_validos:
-                df = self.obtener_velas(iq, par)
-                if df.empty:
-                    continue
-                senal, data = self.obtener_senal(df)
-                self.on_row_update(par, data)
-
-                if senal:
-                    ok, id = iq.buy(MONTO, par, senal, EXPIRACION)
-                    if ok:
-                        logging.info(f"[OK] {senal.upper()} en {par}")
-                time.sleep(0.6)
-            time.sleep(ESPERA_ENTRE_CICLOS)
-        logging.info("✅ Bot finalizado correctamente.")
-
-    def obtener_velas(self, iq, par, n=60):
-        try:
-            velas = iq.get_candles(par, 60, n, time.time())
-            df = pd.DataFrame(velas)[["open", "max", "min", "close"]]
-            df.columns = ["open", "high", "low", "close"]
-            return df
-        except Exception:
-            return pd.DataFrame()
+        self.table.setRowCount(0)
+        self.label_pnl.setText("💰 PnL acumulado: 0.00")
+        self.button_toggle.setEnabled(False)
+        self.label_status.setText("⏳ Iniciando bot...")
 
-    def obtener_senal(self, df):
-        close = df["close"]
-        high = df["high"]
-        low = df["low"]
+        self._thread = QThread(self)
+        self._worker = BotWorker()
+        self._worker.moveToThread(self._thread)
+        self._worker.status_changed.connect(self.label_status.setText)
+        self._worker.row_ready.connect(self.on_row_update)
+        self._worker.trade_completed.connect(self.on_trade_completed)
+        self._worker.finished.connect(self._thread.quit)
+        self._worker.finished.connect(self._worker.deleteLater)
+        self._worker.finished.connect(self.on_worker_finished)
+        self._thread.finished.connect(self._thread.deleteLater)
+        self._thread.finished.connect(self.on_thread_finished)
+        self._thread.started.connect(self._worker.run)
+        self._thread.start()
+        self.button_toggle.setText("Detener bot")
+        self.button_toggle.setEnabled(True)
 
-        rsi = ta.momentum.RSIIndicator(close).rsi().iloc[-1]
-        emaf = ta.trend.EMAIndicator(close, 9).ema_indicator().iloc[-1]
-        emas = ta.trend.EMAIndicator(close, 21).ema_indicator().iloc[-1]
-        macd = ta.trend.MACD(close).macd().iloc[-1]
-        macds = ta.trend.MACD(close).macd_signal().iloc[-1]
-        stoch = ta.momentum.StochasticOscillator(high, low, close)
-        stk = stoch.stoch().iloc[-1]
-        std = stoch.stoch_signal().iloc[-1]
+    def stop_bot(self) -> None:
+        if not self._worker or not self._thread:
+            return
+        self.button_toggle.setEnabled(False)
+        self.label_status.setText("⏹️ Deteniendo bot...")
+        self._worker.stop()
+        if self._thread.isRunning():
+            self._thread.quit()
+            self._thread.wait()
+        self.button_toggle.setEnabled(True)
+        self.button_toggle.setText("Iniciar bot")
 
-        up, down = 0, 0
-        if rsi < 35: up += 1
-        if rsi > 65: down += 1
-        if emaf > emas: up += 1
-        if emaf < emas: down += 1
-        if macd > macds: up += 1
-        if macd < macds: down += 1
-        if stk > std: up += 1
-        if stk < std: down += 1
-
-        signal = None
-        if up >= 3: signal = "call"
-        elif down >= 3: signal = "put"
-
-        return signal, {
-            "rsi": rsi,
-            "emaf": emaf,
-            "emas": emas,
-            "macd": macd,
-            "macds": macds,
-            "stk": stk,
-            "std": std,
-            "signal": signal or "-"
-        }
+    def on_worker_finished(self) -> None:
+        self.button_toggle.setText("Iniciar bot")
+        self.button_toggle.setEnabled(True)
+        self._worker = None
+
+    def on_thread_finished(self) -> None:
+        self._thread = None
+
+    def closeEvent(self, event):
+        if self._worker is not None:
+            self._worker.stop()
+        if self._thread is not None and self._thread.isRunning():
+            self._thread.quit()
+            self._thread.wait()
+        super().closeEvent(event)
 
 
 # ---------------- MAIN ----------------
+def create_splash_screen() -> QSplashScreen:
+    """Construye una pantalla de bienvenida ilustrativa antes de abrir la GUI."""
+
+    width, height = 640, 320
+    pixmap = QPixmap(width, height)
+    pixmap.fill(Qt.transparent)
+
+    painter = QPainter(pixmap)
+    gradient = QLinearGradient(0, 0, 0, height)
+    gradient.setColorAt(0.0, QColor(17, 46, 89))
+    gradient.setColorAt(1.0, QColor(6, 20, 43))
+    painter.fillRect(pixmap.rect(), gradient)
+
+    painter.setPen(QColor("#F5F5F5"))
+    title_font = QFont("Segoe UI", 26, QFont.Bold)
+    painter.setFont(title_font)
+    painter.drawText(
+        pixmap.rect(),
+        Qt.AlignHCenter | Qt.AlignTop,
+        "\nIQ Option Bot"
+    )
+
+    subtitle_font = QFont("Segoe UI", 14)
+    painter.setFont(subtitle_font)
+    painter.drawText(
+        pixmap.rect().adjusted(0, 90, 0, -120),
+        Qt.AlignHCenter | Qt.AlignTop,
+        "Escáner digitales/binarias + Panel de PnL"
+    )
+
+    painter.setFont(QFont("Consolas", 11))
+    painter.drawText(
+        pixmap.rect().adjusted(0, 150, 0, -80),
+        Qt.AlignHCenter | Qt.AlignTop,
+        "Conectando con IQ Option y preparando indicadores..."
+    )
+    painter.end()
+
+    splash = QSplashScreen(pixmap)
+    splash.setFont(QFont("Segoe UI", 9))
+    return splash
+
+
 if __name__ == "__main__":
     app = QApplication(sys.argv)
+    splash = create_splash_screen()
+    splash.show()
+    app.processEvents()
+
     window = BotGUI()
     window.show()
+
+    QTimer.singleShot(2000, lambda: splash.finish(window))
     sys.exit(app.exec_())

