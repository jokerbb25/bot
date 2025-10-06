import sys
import time
import logging
import threading
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Union, Set
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
    QTableWidget,
    QTableWidgetItem,
    QLabel,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread  # noqa: E402
from iqoptionapi.stable_api import IQ_Option  # noqa: E402

# ---------------- CONFIG ----------------
EMAIL = "fornerinoalejandro031@gmail.com"
PASSWORD = "484572ale"
MONTO = 1.0
EXPIRACION = 1
ESPERA_ENTRE_CICLOS = 3
CICLOS = 50
MODO = "PRACTICE"

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

        if not isinstance(data, dict):
            logging.debug(
                "Respuesta inesperada %s al solicitar digitales; se fuerza diccionario vacío.",
                type(data).__name__,
            )
            return {"underlying": {}}

        underlying = data.get("underlying")
        if not isinstance(underlying, dict):
            logging.debug(
                "Respuesta sin clave 'underlying' válida al solicitar digitales; se normaliza.",
            )
            data["underlying"] = {}

        return data

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


class BotWorker(QObject):
    status_changed = pyqtSignal(str)
    row_ready = pyqtSignal(str, Dict[str, Union[float, str]])
    trade_completed = pyqtSignal(str, str, float, float)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()
        self._pnl_acumulado = 0.0
        self._pares_descartados: Set[str] = set()

    def stop(self):
        self._stop_event.set()

    def run(self):
        self.status_changed.emit("🔌 Conectando a IQ Option...")
        iq = SafeIQOption(EMAIL, PASSWORD)
        try:
            check, reason = iq.connect()
        except Exception as exc:  # pragma: no cover - network call
            logging.exception("Error inesperado al conectar con IQ Option: %s", exc)
            self.status_changed.emit("❌ Error conexión inesperado. Ver logs.")
            self.finished.emit()
            return

        if not check:
            self.status_changed.emit(f"❌ Error conexión: {reason}")
            self.finished.emit()
            return

        iq.change_balance(MODO)
        saldo = iq.get_balance()
        self.status_changed.emit(f"✅ Conectado a {MODO} | Saldo: {saldo:.2f}")

        self._pares_descartados.clear()
        logging.info("♻️ Escaneando pares OTC disponibles...")
        pares_validos = self._descubrir_pares(iq)
        logging.info(f"✅ Pares OTC detectados: {pares_validos}")

        pares_validos = self._filtrar_activos_operables(iq, pares_validos)
        if not pares_validos:
            self.status_changed.emit("⚠️ No se encontraron pares OTC disponibles.")
            self.finished.emit()
            return

        for ciclo in range(1, CICLOS + 1):
            if self._stop_event.is_set():
                break
            logging.info(f"=== Ciclo {ciclo}/{CICLOS} ===")
            activos_restantes = False
            for par in list(pares_validos):
                if self._stop_event.is_set():
                    break
                if par in self._pares_descartados:
                    continue
                activos_restantes = True
                df = self.obtener_velas(iq, par)
                if df.empty:
                    self._descartar_par(par, "Sin velas devueltas por el broker.")
                    continue
                senal, data = self.obtener_senal(df)
                self.row_ready.emit(par, data)

                if senal:
                    try:
                        ok, op_id = iq.buy(MONTO, par, senal, EXPIRACION)
                    except Exception as exc:  # pragma: no cover - network call
                        logging.exception("Error al ejecutar operacion %s en %s", senal, par)
                        self._descartar_par(
                            par,
                            "Error de red al ejecutar operación, se descarta el par.",
                        )
                    else:
                        if ok:
                            logging.info(f"[OK] {senal.upper()} en {par}")
                            resultado, pnl = self._esperar_resultado(iq, op_id)
                            if resultado:
                                self._pnl_acumulado += pnl
                                texto_resultado = self._normalizar_resultado(resultado)
                                mensaje = (
                                    f"{texto_resultado} en {par} | PnL operación: {pnl:.2f} | "
                                    f"PnL acumulado: {self._pnl_acumulado:.2f}"
                                )
                                logging.info(mensaje)
                                self.status_changed.emit(mensaje)
                                self.trade_completed.emit(par, texto_resultado, pnl, self._pnl_acumulado)
                            else:
                                logging.warning(
                                    "No se obtuvo resultado para la operación en %s", par
                                )
                        else:
                            logging.warning(f"[FAIL] No se pudo ejecutar {senal} en {par}")
                            self._descartar_par(
                                par,
                                "El broker rechazó la operación, se descarta el par.",
                            )
                time.sleep(0.6)

            pares_validos = [par for par in pares_validos if par not in self._pares_descartados]
            if not pares_validos:
                msg = "⚠️ No quedan pares OTC operables tras los descartes."
                logging.warning(msg)
                self.status_changed.emit(msg)
                break
            if not activos_restantes:
                msg = "⚠️ Todos los pares OTC fueron descartados por fallos del broker."
                logging.warning(msg)
                self.status_changed.emit(msg)
                break
            time.sleep(ESPERA_ENTRE_CICLOS)

        logging.info("✅ Bot finalizado correctamente.")
        self.status_changed.emit("✅ Bot finalizado correctamente.")
        self.finished.emit()

    def _descartar_par(self, par: str, motivo: str) -> None:
        if par in self._pares_descartados:
            return
        self._pares_descartados.add(par)
        logging.info("Activo %s descartado: %s", par, motivo)

    def _descubrir_pares(self, iq):
        try:
            activos = iq.get_all_open_time()
        except Exception as exc:  # pragma: no cover - network call
            logging.exception("No se pudieron obtener los horarios de activos: %s", exc)
            return []
        turbo = activos.get("turbo", {}) if isinstance(activos, dict) else {}

        # Algunos activos devueltos por IQ Option no están disponibles para buy().
        activos_validos = []
        activos_disponibles = getattr(iq, "active_to_id", {})
        for par, info in turbo.items():
            if not info.get("open"):
                continue
            if "-OTC" not in par:
                continue
            if activos_disponibles and par not in activos_disponibles:
                logging.debug("Activo %s omitido: no aparece en active_to_id", par)
                continue
            activos_validos.append(par)

        # Solo procesamos los primeros 20 para evitar sobrecargar la GUI
        return activos_validos[:20]

    def _filtrar_activos_operables(self, iq, pares):
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
                "Ninguno de los %s pares OTC iniciales devolvió velas válidas.", len(pares)
            )

        return pares_operables

    def obtener_velas(self, iq, par, n=60):
        try:
            velas = iq.get_candles(par, 60, n, time.time())
            df = pd.DataFrame(velas)[["open", "max", "min", "close"]]
            df.columns = ["open", "high", "low", "close"]
            return df
        except Exception as exc:
            logging.exception("Error al obtener velas para %s: %s", par, exc)
            self._descartar_par(
                par, "Excepción al solicitar velas al broker."
            )
            return pd.DataFrame()

    def obtener_senal(self, df) -> Tuple[Optional[str], Dict[str, Union[float, str]]]:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        rsi_indicator = ta.momentum.RSIIndicator(close)
        ema_fast_indicator = ta.trend.EMAIndicator(close, 9)
        ema_slow_indicator = ta.trend.EMAIndicator(close, 21)
        macd_indicator = ta.trend.MACD(close)
        stoch_indicator = ta.momentum.StochasticOscillator(high, low, close)

        snapshot = IndicatorSnapshot(
            rsi=rsi_indicator.rsi().iloc[-1],
            emaf=ema_fast_indicator.ema_indicator().iloc[-1],
            emas=ema_slow_indicator.ema_indicator().iloc[-1],
            macd=macd_indicator.macd().iloc[-1],
            macds=macd_indicator.macd_signal().iloc[-1],
            stk=stoch_indicator.stoch().iloc[-1],
            std=stoch_indicator.stoch_signal().iloc[-1],
        )

        up, down = 0, 0
        if snapshot.rsi < 35:
            up += 1
        if snapshot.rsi > 65:
            down += 1
        if snapshot.emaf > snapshot.emas:
            up += 1
        if snapshot.emaf < snapshot.emas:
            down += 1
        if snapshot.macd > snapshot.macds:
            up += 1
        if snapshot.macd < snapshot.macds:
            down += 1
        if snapshot.stk > snapshot.std:
            up += 1
        if snapshot.stk < snapshot.std:
            down += 1

        signal: Optional[str] = None
        if up >= 3:
            signal = "call"
        elif down >= 3:
            signal = "put"

        return signal, snapshot.to_table_payload(signal)

    def _esperar_resultado(self, iq, op_id, timeout: int = 180) -> Tuple[Optional[str], float]:
        """Espera hasta obtener el resultado de la operación o agotar el timeout."""

        inicio = time.time()
        while not self._stop_event.is_set():
            try:
                estado, pnl = iq.check_win_v4(op_id)
            except Exception as exc:  # pragma: no cover - network call
                logging.exception("Error al verificar resultado de operación %s", op_id)
                return None, 0.0

            if estado in {"win", "loose", "loss", "equal"}:
                try:
                    pnl_val = float(pnl)
                except (TypeError, ValueError):
                    pnl_val = 0.0
                return estado, pnl_val

            if time.time() - inicio > timeout:
                logging.warning(
                    "Timeout esperando resultado de la operación %s tras %s segundos",
                    op_id,
                    timeout,
                )
                return None, 0.0

            time.sleep(1)

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

        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            [
                "Par",
                "RSI",
                "EMA Fast",
                "EMA Slow",
                "MACD",
                "Signal",
                "STK %K",
                "STK %D",
                "Señal",
                "Resultado",
            ]
        )
        layout.addWidget(self.table)

        self.label_pnl = QLabel("💰 PnL acumulado: 0.00", self)
        layout.addWidget(self.label_pnl)

        self.label_footer = QLabel("", self)
        layout.addWidget(self.label_footer)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_footer)
        self.timer.start(1000)

        self._thread = QThread(self)
        self._worker = BotWorker()
        self._worker.moveToThread(self._thread)
        self._worker.status_changed.connect(self.label_status.setText)
        self._worker.row_ready.connect(self.on_row_update)
        self._worker.trade_completed.connect(self.on_trade_completed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.started.connect(self._worker.run)
        self._thread.start()

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
        vals = [
            f"{payload.get('rsi', 0):.1f}",
            f"{payload.get('emaf', 0):.5f}",
            f"{payload.get('emas', 0):.5f}",
            f"{payload.get('macd', 0):.5f}",
            f"{payload.get('macds', 0):.5f}",
            f"{payload.get('stk', 0):.1f}",
            f"{payload.get('std', 0):.1f}",
            f"{payload.get('signal', '-')}",
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

        if payload.get("result"):
            resultado = payload["result"]
            item_res = QTableWidgetItem(resultado)
            if "WIN" in resultado.upper():
                item_res.setForeground(Qt.green)
            elif any(x in resultado.upper() for x in ("LOST", "LOSS")):
                item_res.setForeground(Qt.red)
            self.table.setItem(r, 9, item_res)
        elif self.table.item(r, 9) is None:
            self.table.setItem(r, 9, QTableWidgetItem("-"))

    def on_trade_completed(self, par: str, resultado: str, pnl: float, pnl_total: float) -> None:
        r = self.ensure_row(par)
        texto = f"{resultado} ({pnl:.2f})"
        item = QTableWidgetItem(texto)
        if resultado == "WIN":
            item.setForeground(Qt.green)
        elif resultado == "LOST":
            item.setForeground(Qt.red)
        self.table.setItem(r, 9, item)
        self.label_pnl.setText(f"💰 PnL acumulado: {pnl_total:.2f}")

    def closeEvent(self, event):
        if hasattr(self, "_worker") and self._worker is not None:
            self._worker.stop()
        if hasattr(self, "_thread") and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        super().closeEvent(event)


# ---------------- MAIN ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BotGUI()
    window.show()
    sys.exit(app.exec_())
