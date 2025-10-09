import sys
import time
import math
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from datetime import datetime
from textwrap import dedent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def _ensure_dependency(module: str, pip_name: Optional[str] = None) -> None:
    try:
        __import__(module)
    except ImportError:  # pragma: no cover - friendly runtime guard
        package = pip_name or module
        message = dedent(
            f"""
            ❌ Falta la dependencia obligatoria "{module}".
            Instálala ejecutando: pip install {package}
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

import pandas as pd  # noqa: E402
import ta  # noqa: E402
from PyQt5.QtCore import QObject, QThread, Qt, QTimer, pyqtSignal  # noqa: E402
from PyQt5.QtGui import QColor, QFont, QLinearGradient, QPainter, QPixmap  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QSplashScreen,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from iqoptionapi import constants as iq_constants  # noqa: E402
from iqoptionapi.stable_api import IQ_Option  # noqa: E402


def _patch_digital_underlying() -> None:
    original = IQ_Option.get_digital_underlying_list_data

    if getattr(original, "__name__", "") == "safe_get_digital_underlying_list_data":
        return

    def safe_get_digital_underlying_list_data(self, *args, **kwargs):  # type: ignore
        try:
            payload = original(self, *args, **kwargs)
        except Exception:
            logging.debug("Fallo al obtener metadatos digitales; se devuelve estructura vacía")
            return {"underlying": {}}

        if not isinstance(payload, dict):
            return {"underlying": {}}

        underlying = payload.get("underlying")
        if not isinstance(underlying, dict):
            payload["underlying"] = {}

        return payload

    safe_get_digital_underlying_list_data.__name__ = "safe_get_digital_underlying_list_data"
    IQ_Option.get_digital_underlying_list_data = safe_get_digital_underlying_list_data  # type: ignore


_patch_digital_underlying()

EMAIL = "fornerinoalejandro031@gmail.com"
PASSWORD = "484572ale"
MODO = "PRACTICE"
MONTO = 1.0
EXPIRACION = 1
ESPERA_ENTRE_CICLOS = 3
CICLOS = 50


OTC_FALLBACK = (
    "EURUSD-OTC",
    "GBPUSD-OTC",
    "USDJPY-OTC",
    "AUDUSD-OTC",
    "USDCHF-OTC",
    "USDCAD-OTC",
    "EURJPY-OTC",
    "EURGBP-OTC",
    "GBPJPY-OTC",
    "AUDCAD-OTC",
    "NZDUSD-OTC",
    "CADJPY-OTC",
)


def _sanitize(symbol: Optional[str]) -> Optional[str]:
    if not symbol or not isinstance(symbol, str):
        return None
    cleaned = symbol.strip().upper()
    if not cleaned or "OTC" not in cleaned:
        return None
    if not cleaned.endswith("-OTC"):
        cleaned = f"{cleaned}-OTC"
    return cleaned


class SafeIQOption(IQ_Option):
    def get_candles(self, *args, **kwargs):  # pragma: no cover - thin wrapper
        return super().get_candles(*args, **kwargs)


@dataclass
class IndicatorSnapshot:
    rsi: float
    ema_fast: float
    ema_slow: float
    macd: float
    macd_signal: float
    stoch_k: float
    stoch_d: float

    def to_row(self, signal: Optional[str]) -> Dict[str, str]:
        return {
            "rsi": _format(self.rsi, 1),
            "ema_fast": _format(self.ema_fast, 5),
            "ema_slow": _format(self.ema_slow, 5),
            "macd": _format(self.macd, 5),
            "macd_signal": _format(self.macd_signal, 5),
            "stoch_k": _format(self.stoch_k, 1),
            "stoch_d": _format(self.stoch_d, 1),
            "signal": signal or "-",
        }


@dataclass
class TradeResult:
    ticket: str
    pnl: float
    outcome: str


@dataclass
class OtcPair:
    name: str
    active_id: int


def _format(value: float, precision: int) -> str:
    if math.isnan(value):
        return "-"
    return f"{value:.{precision}f}"


def _iter_open_otc(payload: Dict) -> Iterable[Tuple[str, int]]:
    for category in ("turbo", "binary", "digital"):
        entries = payload.get(category, {})
        if not isinstance(entries, dict):
            continue
        for symbol, info in entries.items():
            if not isinstance(info, dict):
                continue
            if info.get("open") is False or info.get("enabled") is False:
                continue
            candidate = _sanitize(symbol)
            if not candidate:
                continue
            active_id = info.get("id") or info.get("active_id")
            try:
                active_id = int(active_id)
            except (TypeError, ValueError):
                active_id = None
            if active_id is None:
                continue
            yield candidate, active_id


def _discover_otc_pairs(api: SafeIQOption) -> List[OtcPair]:
    try:
        schedule = api.get_all_open_time()
    except Exception:
        logging.exception("No se pudieron obtener los horarios de activos")
        schedule = {}
    seen: Dict[str, OtcPair] = {}
    for symbol, active_id in _iter_open_otc(schedule):
        seen.setdefault(symbol, OtcPair(symbol, active_id))
    if not seen:
        logging.warning("No se detectaron pares OTC abiertos; se usa respaldo estático")
        for symbol in OTC_FALLBACK:
            active_id = iq_constants.ACTIVES.get(symbol)
            if active_id is None:
                continue
            seen.setdefault(symbol, OtcPair(symbol, int(active_id)))
    return list(seen.values())


def _build_dataframe(candles: Sequence[Dict]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles)
    for columns in (
        ("open", "max", "min", "close"),
        ("open", "high", "low", "close"),
        ("o", "h", "l", "c"),
    ):
        if all(col in df.columns for col in columns):
            mapping = dict(zip(columns, ("open", "high", "low", "close")))
            frame = df[list(columns)].rename(columns=mapping)
            frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
            return frame
    return pd.DataFrame()


def _fetch_candles(api: SafeIQOption, pair: OtcPair, count: int = 120) -> pd.DataFrame:
    for attempt in range(3):
        try:
            payload = api.get_candles(pair.active_id, 60, count, time.time())
        except Exception as exc:
            logging.warning(
                "Error obteniendo velas %s (intento %s/3): %s",
                pair.name,
                attempt + 1,
                exc,
            )
            time.sleep(1)
            continue
        if isinstance(payload, dict):
            payload = payload.get("candles") or payload.get("data") or []
        if not isinstance(payload, list):
            logging.debug("Respuesta de velas inesperada para %s: %s", pair.name, type(payload))
            time.sleep(1)
            continue
        df = _build_dataframe(payload)
        if not df.empty:
            return df
        logging.debug("Velas vacías para %s (intento %s/3)", pair.name, attempt + 1)
        time.sleep(1)
    return pd.DataFrame()


def _compute_signal(df: pd.DataFrame) -> Tuple[Optional[str], IndicatorSnapshot]:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    rsi = float(ta.momentum.RSIIndicator(close).rsi().iloc[-1])
    ema_fast = float(ta.trend.EMAIndicator(close, 9).ema_indicator().iloc[-1])
    ema_slow = float(ta.trend.EMAIndicator(close, 21).ema_indicator().iloc[-1])
    macd_ind = ta.trend.MACD(close)
    macd = float(macd_ind.macd().iloc[-1])
    macd_signal = float(macd_ind.macd_signal().iloc[-1])
    stoch_ind = ta.momentum.StochasticOscillator(high, low, close)
    stoch_k = float(stoch_ind.stoch().iloc[-1])
    stoch_d = float(stoch_ind.stoch_signal().iloc[-1])

    votes_up = votes_down = 0

    if not math.isnan(rsi):
        if rsi < 35:
            votes_up += 1
        elif rsi > 65:
            votes_down += 1

    if not math.isnan(ema_fast) and not math.isnan(ema_slow):
        if ema_fast > ema_slow:
            votes_up += 1
        elif ema_fast < ema_slow:
            votes_down += 1

    if not math.isnan(macd) and not math.isnan(macd_signal):
        if macd > macd_signal:
            votes_up += 1
        elif macd < macd_signal:
            votes_down += 1

    if not math.isnan(stoch_k) and not math.isnan(stoch_d):
        if stoch_k > stoch_d:
            votes_up += 1
        elif stoch_k < stoch_d:
            votes_down += 1

    signal = None
    if votes_up >= 2 and votes_up > votes_down:
        signal = "call"
    elif votes_down >= 2 and votes_down > votes_up:
        signal = "put"

    snapshot = IndicatorSnapshot(rsi, ema_fast, ema_slow, macd, macd_signal, stoch_k, stoch_d)
    return signal, snapshot


def _wait_binary_result(api: SafeIQOption, ticket: str) -> TradeResult:
    timeout = EXPIRACION * 60 + 120
    elapsed = 0
    while elapsed < timeout:
        try:
            outcome, pnl = api.check_win_v3(ticket)
        except Exception as exc:
            logging.warning("Error consultando resultado %s: %s", ticket, exc)
            time.sleep(1)
            elapsed += 1
            continue
        if outcome is not None:
            try:
                pnl_value = float(pnl)
            except (TypeError, ValueError):
                pnl_value = 0.0
            return TradeResult(ticket, pnl_value, outcome)
        time.sleep(1)
        elapsed += 1
    logging.warning("Timeout esperando resultado para %s", ticket)
    return TradeResult(ticket, 0.0, "unknown")


class BotWorker(QObject):
    status_changed = pyqtSignal(str)
    row_ready = pyqtSignal(str, dict)
    trade_completed = pyqtSignal(str, str, float, float)
    finished = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._stop = False
        self._pnl = 0.0
        self._failures: Dict[str, int] = {}

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        self.status_changed.emit("🔌 Conectando a IQ Option...")
        api = SafeIQOption(EMAIL, PASSWORD)
        try:
            connected, reason = api.connect()
        except Exception as exc:
            logging.exception("Error conectando al broker: %s", exc)
            self.status_changed.emit("❌ No se pudo conectar al broker")
            self.finished.emit()
            return
        if not connected:
            self.status_changed.emit(f"❌ Conexión rechazada: {reason}")
            self.finished.emit()
            return

        api.change_balance(MODO)
        balance = api.get_balance()
        self.status_changed.emit(f"✅ Conectado a {MODO} | Saldo: {balance:.2f}")

        pares = _discover_otc_pairs(api)
        if not pares:
            self.status_changed.emit("⚠️ No se detectaron pares OTC disponibles")
            self.finished.emit()
            api.close()
            return

        self.status_changed.emit(
            "📈 Pares OTC cargados: " + ", ".join(par.name for par in pares)
        )

        try:
            for ciclo in range(1, CICLOS + 1):
                if self._stop:
                    break
                logging.info("=== Ciclo %s/%s ===", ciclo, CICLOS)
                self.status_changed.emit(
                    f"🔁 Ciclo {ciclo}/{CICLOS} | PnL acumulado: {self._pnl:.2f}"
                )
                for par in list(pares):
                    if self._stop:
                        break
                    df = _fetch_candles(api, par)
                    if df.empty:
                        logging.warning("%s sin velas válidas", par.name)
                        fails = self._failures.get(par.name, 0) + 1
                        self._failures[par.name] = fails
                        if fails >= 3:
                            logging.warning("%s eliminado tras %s fallos de velas", par.name, fails)
                            pares.remove(par)
                        continue
                    if par.name in self._failures:
                        self._failures.pop(par.name, None)
                    signal, snapshot = _compute_signal(df)
                    self.row_ready.emit(par.name, snapshot.to_row(signal))
                    if not signal:
                        continue
                    ok, ticket = api.buy(MONTO, par.name, signal, EXPIRACION)
                    if not ok or not ticket:
                        logging.warning("[FAIL] Broker rechazó %s en %s", signal.upper(), par.name)
                        continue
                    logging.info("[OK] %s en %s (ticket=%s)", signal.upper(), par.name, ticket)
                    result = _wait_binary_result(api, ticket)
                    outcome = result.outcome.upper() if result.outcome else "UNKNOWN"
                    pnl = result.pnl
                    if outcome == "WIN":
                        self._pnl += pnl
                    elif outcome in {"LOSS", "LOOSE", "LOST"}:
                        self._pnl += pnl
                        outcome = "LOST"
                    elif outcome == "EQUAL":
                        self._pnl += pnl
                    else:
                        logging.warning("Resultado desconocido para %s: %s", par.name, outcome)
                    mensaje = (
                        f"{outcome} en {par.name} | PnL operación: {pnl:.2f} | PnL acumulado: {self._pnl:.2f}"
                    )
                    logging.info(mensaje)
                    self.status_changed.emit(mensaje)
                    self.trade_completed.emit(par.name, outcome, pnl, self._pnl)
                    time.sleep(0.5)
                time.sleep(ESPERA_ENTRE_CICLOS)
        finally:
            try:
                api.close()
            except Exception:
                pass
            self.finished.emit()


class BotGUI(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("IQ Option Bot - OTC Monitor")
        self.resize(1000, 520)

        layout = QVBoxLayout(self)
        self.label_status = QLabel("⏳ Preparando bot...")
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

        self.label_pnl = QLabel("💰 PnL acumulado: 0.00")
        layout.addWidget(self.label_pnl)

        footer_controls = QHBoxLayout()
        footer_controls.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.button_toggle = QPushButton("Iniciar bot")
        self.button_toggle.clicked.connect(self.on_toggle)
        footer_controls.addWidget(self.button_toggle)
        layout.addLayout(footer_controls)

        self.label_footer = QLabel("")
        layout.addWidget(self.label_footer)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_footer)
        self.timer.start(1000)

        self._thread: Optional[QThread] = None
        self._worker: Optional[BotWorker] = None

    def _update_footer(self) -> None:
        self.label_footer.setText(datetime.now().strftime("🕓 %H:%M:%S"))

    def _ensure_row(self, pair: str) -> int:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text() == pair:
                return row
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(pair))
        return row

    def on_row_ready(self, pair: str, data: Dict[str, str]) -> None:
        row = self._ensure_row(pair)
        mapping = [
            ("rsi", 1),
            ("ema_fast", 2),
            ("ema_slow", 3),
            ("macd", 4),
            ("macd_signal", 5),
            ("stoch_k", 6),
            ("stoch_d", 7),
        ]
        for key, column in mapping:
            value = data.get(key, "-")
            self.table.setItem(row, column, QTableWidgetItem(value))
        signal_item = QTableWidgetItem(data.get("signal", "-"))
        if signal_item.text() == "call":
            signal_item.setForeground(Qt.green)
        elif signal_item.text() == "put":
            signal_item.setForeground(Qt.red)
        self.table.setItem(row, 8, signal_item)
        if self.table.item(row, 9) is None:
            self.table.setItem(row, 9, QTableWidgetItem("-"))
        if self.table.item(row, 10) is None:
            self.table.setItem(row, 10, QTableWidgetItem("-"))

    def on_trade_completed(self, pair: str, outcome: str, pnl: float, total: float) -> None:
        row = self._ensure_row(pair)
        result_item = QTableWidgetItem(outcome)
        if outcome == "WIN":
            result_item.setForeground(Qt.green)
        elif outcome == "LOST":
            result_item.setForeground(Qt.red)
        self.table.setItem(row, 9, result_item)
        pnl_item = QTableWidgetItem(f"{pnl:.2f}")
        if pnl > 0:
            pnl_item.setForeground(Qt.green)
        elif pnl < 0:
            pnl_item.setForeground(Qt.red)
        self.table.setItem(row, 10, pnl_item)
        self.label_pnl.setText(f"💰 PnL acumulado: {total:.2f}")

    def on_toggle(self) -> None:
        if self._thread and self._thread.isRunning():
            self._stop_bot()
        else:
            self._start_bot()

    def _start_bot(self) -> None:
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
        self._worker.row_ready.connect(self.on_row_ready)
        self._worker.trade_completed.connect(self.on_trade_completed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.finished.connect(self._on_worker_finished)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.started.connect(self._worker.run)
        self._thread.start()
        self.button_toggle.setText("Detener bot")
        self.button_toggle.setEnabled(True)

    def _stop_bot(self) -> None:
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

    def _on_worker_finished(self) -> None:
        self.button_toggle.setText("Iniciar bot")
        self.button_toggle.setEnabled(True)
        self._worker = None
        self._thread = None

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        super().closeEvent(event)


def create_splash() -> QSplashScreen:
    width, height = 620, 300
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    gradient = QLinearGradient(0, 0, 0, height)
    gradient.setColorAt(0.0, QColor(15, 38, 78))
    gradient.setColorAt(1.0, QColor(4, 14, 32))
    painter.fillRect(pixmap.rect(), gradient)

    painter.setPen(QColor("#F5F5F5"))
    painter.setFont(QFont("Segoe UI", 26, QFont.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignHCenter | Qt.AlignTop, "\nIQ Option OTC Bot")

    painter.setFont(QFont("Segoe UI", 14))
    painter.drawText(
        pixmap.rect().adjusted(0, 90, 0, -140),
        Qt.AlignHCenter | Qt.AlignTop,
        "Escáner y monitor de operaciones OTC",
    )

    painter.setFont(QFont("Consolas", 11))
    painter.drawText(
        pixmap.rect().adjusted(0, 150, 0, -100),
        Qt.AlignHCenter | Qt.AlignTop,
        "Conectando al broker y preparando indicadores...",
    )
    painter.end()

    return QSplashScreen(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    splash = create_splash()
    splash.show()
    app.processEvents()

    window = BotGUI()
    window.show()

    QTimer.singleShot(2000, lambda: splash.finish(window))
    sys.exit(app.exec_())
