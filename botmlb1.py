import sys
import time
import math
import logging
import threading
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Tuple, Union, List, Set
from datetime import datetime
from textwrap import dedent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def _ensure_dependency(module_name: str, pip_name: Optional[str] = None) -> None:
    try:
        __import__(module_name)
    except ImportError:
        friendly_name = pip_name or module_name
        message = dedent(
            f"""
            ❌ Falta la dependencia obligatoria \"{module_name}\".
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

import pandas as pd  # noqa: E402
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

EMAIL = "fornerinoalejandro031@gmail.com"
PASSWORD = "484572ale"
MONTO = 1.0
EXPIRACION = 1
ESPERA_ENTRE_CICLOS = 3
CICLOS = 50
MODO = "PRACTICE"

FALLBACK_DIGITAL = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCHF",
    "USDCAD",
    "EURJPY",
    "EURGBP",
    "GBPJPY",
    "AUDCAD",
    "NZDUSD",
    "CADJPY",
)

FALLBACK_BINARY = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCHF",
    "USDCAD",
    "EURJPY",
    "EURGBP",
    "GBPJPY",
    "AUDCAD",
)


def _sanitize(symbol: Optional[str]) -> Optional[str]:
    if not symbol or not isinstance(symbol, str):
        return None
    cleaned = symbol.strip().upper()
    if not cleaned:
        return None
    if "OTC" in cleaned:
        return None
    for suffix in ("-OP", "-F", "-FX", "-INDEX", "-DIGITAL", "-BINARY"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    cleaned = cleaned.replace(" ", "")
    return cleaned or None


def _ensure_underlying_patch() -> None:
    original = IQ_Option.get_digital_underlying_list_data

    if getattr(original, "__name__", "") == "safe_get_digital_underlying_list_data":
        return

    def safe_get_digital_underlying_list_data(self, *args, **kwargs):  # type: ignore
        try:
            payload = original(self, *args, **kwargs)
        except Exception:
            logging.exception("Fallo al pedir metadatos digitales")
            return {"underlying": {}}
        if not isinstance(payload, dict):
            return {"underlying": {}}
        if not isinstance(payload.get("underlying"), dict):
            payload["underlying"] = {}
        return payload

    safe_get_digital_underlying_list_data.__name__ = "safe_get_digital_underlying_list_data"
    IQ_Option.get_digital_underlying_list_data = safe_get_digital_underlying_list_data  # type: ignore


_ensure_underlying_patch()


class SafeIQOption(IQ_Option):
    def get_all_open_time(self):  # pragma: no cover
        try:
            return super().get_all_open_time()
        except Exception:
            logging.exception("Fallo al pedir horarios")
            return {}

    def get_candles(self, *args, **kwargs):  # pragma: no cover
        try:
            return super().get_candles(*args, **kwargs)
        except Exception:
            raise


@dataclass
class IndicatorSnapshot:
    rsi: float
    ema_fast: float
    ema_slow: float
    macd: float
    macd_signal: float
    stoch_k: float
    stoch_d: float

    def as_payload(self, signal: Optional[str]) -> Dict[str, Union[float, str]]:
        data = asdict(self)
        data["signal"] = signal or "-"
        return data


@dataclass
class TradePair:
    name: str
    instrument_type: str
    candle_aliases: Tuple[str, ...] = field(default_factory=tuple)
    trade_aliases: Tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def build(cls, base_symbol: str, instrument_type: str, raw_symbol: Optional[str] = None) -> "TradePair":
        base = base_symbol.upper()
        aliases: List[str] = []
        trade_aliases: List[str] = []

        candidates: Set[str] = {base}
        if raw_symbol:
            raw_clean = raw_symbol.strip().upper()
            if raw_clean and "OTC" not in raw_clean:
                candidates.add(raw_clean)

        more = set()
        for candidate in list(candidates):
            more.add(candidate.replace("/", ""))
            more.add(candidate.replace("-", ""))
        candidates.update({item for item in more if item})

        if instrument_type == "digital":
            digital_candidates = set()
            for candidate in candidates:
                digital_candidates.add(f"{candidate}-OP")
            candidates.update(digital_candidates)

        for candidate in candidates:
            clean = candidate.strip()
            if not clean:
                continue
            if clean not in aliases:
                aliases.append(clean)
            if clean not in trade_aliases:
                trade_aliases.append(clean)

        candle_aliases = tuple(alias for alias in aliases if alias)
        trade_aliases = tuple(alias for alias in trade_aliases if alias)
        return cls(base, instrument_type, candle_aliases, trade_aliases)


class BotWorker(QObject):
    status_changed = pyqtSignal(str)
    row_ready = pyqtSignal(str, dict)
    trade_finished = pyqtSignal(str, str, float, float)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._stop = threading.Event()
        self._pnl = 0.0

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        self.status_changed.emit("🔌 Conectando a IQ Option...")
        api = SafeIQOption(EMAIL, PASSWORD)
        try:
            ok, reason = api.connect()
        except Exception as exc:
            logging.exception("Fallo conectando")
            self.status_changed.emit("❌ Error conectando al broker")
            self.finished.emit()
            return
        if not ok:
            self.status_changed.emit(f"❌ Conexión rechazada: {reason}")
            self.finished.emit()
            return
        api.change_balance(MODO)
        balance = api.get_balance()
        self.status_changed.emit(f"✅ Conectado a {MODO} | Saldo: {balance:.2f}")

        pares = self._discover_pairs(api)
        if not pares:
            self.status_changed.emit("⚠️ No hay pares digitales/binarias disponibles")
            self.finished.emit()
            api.close()
            return

        self.status_changed.emit(
            f"📈 {len(pares)} pares digitales/binarios listos para operar"
        )

        try:
            for ciclo in range(1, CICLOS + 1):
                if self._stop.is_set():
                    break
                logging.info("=== Ciclo %s/%s ===", ciclo, CICLOS)
                for par in pares:
                    if self._stop.is_set():
                        break
                    df = self._fetch_candles(api, par)
                    if df.empty:
                        logging.warning("%s sin velas válidas", par.name)
                        continue
                    signal, payload = self._analyze(df)
                    self.row_ready.emit(par.name, payload)
                    if not signal:
                        continue
                    ok, ticket = self._execute_trade(api, par, signal)
                    if not ok or ticket is None:
                        continue
                    result, pnl = self._wait_result(api, ticket)
                    if result is None:
                        continue
                    self._pnl += pnl
                    result_text = self._normalize_result(result)
                    mensaje = (
                        f"{result_text} en {par.name} | PnL: {pnl:.2f} | Total: {self._pnl:.2f}"
                    )
                    logging.info(mensaje)
                    self.status_changed.emit(mensaje)
                    self.trade_finished.emit(par.name, result_text, pnl, self._pnl)
                    if self._stop.is_set():
                        break
                    time.sleep(0.5)
                if self._stop.is_set():
                    break
                time.sleep(ESPERA_ENTRE_CICLOS)
        finally:
            try:
                api.close()
            except Exception:
                pass
            self.finished.emit()

    def _discover_pairs(self, api: SafeIQOption) -> List[TradePair]:
        logging.info("♻️ Escaneando pares digitales y binarias disponibles...")
        payload = api.get_all_open_time()
        candidates: List[TradePair] = []
        seen: Set[Tuple[str, str]] = set()
        if isinstance(payload, dict):
            for category in ("digital", "binary", "turbo"):
                entries = payload.get(category, {})
                if not isinstance(entries, dict):
                    continue
                instrument_type = "digital" if category == "digital" else "binary"
                for symbol, info in entries.items():
                    if not isinstance(info, dict):
                        continue
                    if not info.get("open", True):
                        continue
                    if "OTC" in str(symbol).upper():
                        continue
                    normalized = _sanitize(symbol)
                    if not normalized:
                        continue
                    key = (normalized, instrument_type)
                    if key in seen:
                        continue
                    par = TradePair.build(normalized, instrument_type, str(symbol))
                    if not par.candle_aliases or not par.trade_aliases:
                        continue
                    candidates.append(par)
                    seen.add(key)
        if not candidates:
            logging.warning("No se encontraron pares activos; usando respaldo estándar digital/binario")
            for symbol in FALLBACK_DIGITAL:
                par = TradePair.build(symbol, "digital", symbol)
                key = (par.name, par.instrument_type)
                if par.candle_aliases and par.trade_aliases and key not in seen:
                    candidates.append(par)
                    seen.add(key)
            for symbol in FALLBACK_BINARY:
                par = TradePair.build(symbol, "binary", symbol)
                key = (par.name, par.instrument_type)
                if par.candle_aliases and par.trade_aliases and key not in seen:
                    candidates.append(par)
                    seen.add(key)
        pares: List[TradePair] = []
        for par in candidates:
            df = self._fetch_candles(api, par, sample=True)
            if df.empty:
                continue
            pares.append(par)
        return pares

    def _fetch_candles(self, api: SafeIQOption, par: TradePair, sample: bool = False) -> pd.DataFrame:
        history = 30 if sample else 60
        for alias in par.candle_aliases:
            try:
                candles = api.get_candles(alias, 60, history, time.time())
            except Exception as exc:
                logging.debug("Error obteniendo velas %s (%s)", alias, exc)
                continue
            df = self._build_dataframe(candles)
            if not df.empty:
                return df
        return pd.DataFrame()

    @staticmethod
    def _build_dataframe(candles) -> pd.DataFrame:
        if isinstance(candles, dict):
            for key in ("candles", "data", "items", "list"):
                maybe = candles.get(key)
                if isinstance(maybe, list):
                    candles = maybe
                    break
            else:
                return pd.DataFrame()
        if not isinstance(candles, list) or not candles:
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
                if not frame.empty:
                    return frame
        return pd.DataFrame()

    def _analyze(self, df: pd.DataFrame) -> Tuple[Optional[str], Dict[str, Union[float, str]]]:
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

        snapshot = IndicatorSnapshot(rsi, ema_fast, ema_slow, macd, macd_signal, stoch_k, stoch_d)
        up = down = votos = 0

        def threshold(value: float, low_bound: float, high_bound: float) -> None:
            nonlocal up, down, votos
            if math.isnan(value):
                return
            votos += 1
            if value < low_bound:
                up += 1
            elif value > high_bound:
                down += 1

        def compare(a: float, b: float) -> None:
            nonlocal up, down, votos
            if math.isnan(a) or math.isnan(b):
                return
            votos += 1
            if a > b:
                up += 1
            elif a < b:
                down += 1

        threshold(rsi, 35, 65)
        compare(ema_fast, ema_slow)
        compare(macd, macd_signal)
        compare(stoch_k, stoch_d)

        signal: Optional[str] = None
        if votos:
            if up >= 2 and up > down:
                signal = "call"
            elif down >= 2 and down > up:
                signal = "put"
        return signal, snapshot.as_payload(signal)

    def _execute_trade(
        self, api: SafeIQOption, par: TradePair, signal: str
    ) -> Tuple[bool, Optional[Tuple[str, str]]]:
        for alias in par.trade_aliases:
            try:
                if par.instrument_type == "digital":
                    ok, trade_id = api.buy_digital_spot(alias, MONTO, signal, EXPIRACION)
                    trade_kind = "digital"
                else:
                    ok, trade_id = api.buy(MONTO, alias, signal, EXPIRACION)
                    trade_kind = "binary"
            except Exception as exc:
                logging.debug("Error lanzando operación %s (%s)", alias, exc)
                continue
            if ok and trade_id:
                logging.info("[OK] %s en %s (alias %s)", signal.upper(), par.name, alias)
                return True, (trade_kind, str(trade_id))
        logging.warning("[FAIL] Broker rechazó %s en %s", signal.upper(), par.name)
        return False, None

    def _wait_result(
        self, api: SafeIQOption, ticket: Tuple[str, str]
    ) -> Tuple[Optional[str], float]:
        trade_type, trade_id = ticket
        timeout = EXPIRACION * 60 + 120
        waited = 0
        while waited < timeout and not self._stop.is_set():
            try:
                if trade_type == "digital":
                    result, pnl = api.check_win_digital_v2(trade_id)
                else:
                    result, pnl = api.check_win_v4(trade_id)
            except Exception as exc:
                logging.debug("Error consultando resultado %s", exc)
                return None, 0.0
            if result is not None:
                try:
                    pnl_value = float(pnl)
                except (TypeError, ValueError):
                    pnl_value = 0.0
                return result, pnl_value
            time.sleep(1)
            waited += 1
        logging.warning("Timeout esperando resultado %s", trade_id)
        return None, 0.0

    @staticmethod
    def _normalize_result(value: str) -> str:
        text = (value or "").strip().lower()
        if text == "win":
            return "WIN"
        if text in {"loss", "lost", "loose"}:
            return "LOST"
        if text == "equal":
            return "EQUAL"
        return value.upper() if value else "-"


class BotGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IQ Option Digital/Binary Bot")
        self.resize(1000, 520)

        layout = QVBoxLayout(self)
        self.label_status = QLabel("⏳ Iniciando...", self)
        layout.addWidget(self.label_status)

        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels([
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
        ])
        layout.addWidget(self.table)

        self.label_pnl = QLabel("💰 PnL acumulado: 0.00", self)
        layout.addWidget(self.label_pnl)

        footer = QHBoxLayout()
        footer.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.button_toggle = QPushButton("Iniciar bot", self)
        self.button_toggle.clicked.connect(self.on_toggle)
        footer.addWidget(self.button_toggle)
        layout.addLayout(footer)

        self.label_footer = QLabel("", self)
        layout.addWidget(self.label_footer)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_clock)
        self.timer.start(1000)

        self._thread: Optional[QThread] = None
        self._worker: Optional[BotWorker] = None

    def _update_clock(self) -> None:
        self.label_footer.setText(datetime.now().strftime("🕓 %H:%M:%S"))

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
        self.label_status.setText("⏳ Conectando...")

        self._thread = QThread(self)
        self._worker = BotWorker()
        self._worker.moveToThread(self._thread)
        self._worker.status_changed.connect(self.label_status.setText)
        self._worker.row_ready.connect(self._update_row)
        self._worker.trade_finished.connect(self._update_trade)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.finished.connect(self._on_finished)
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

    def _on_finished(self) -> None:
        self.button_toggle.setEnabled(True)
        self.button_toggle.setText("Iniciar bot")
        self._worker = None
        self._thread = None

    def _update_row(self, par: str, payload: Dict[str, Union[float, str]]) -> None:
        row = self._row_for(par)
        keys = [
            ("rsi", "{:.1f}"),
            ("ema_fast", "{:.5f}"),
            ("ema_slow", "{:.5f}"),
            ("macd", "{:.5f}"),
            ("macd_signal", "{:.5f}"),
            ("stoch_k", "{:.1f}"),
            ("stoch_d", "{:.1f}"),
        ]
        for index, (key, fmt) in enumerate(keys, start=1):
            value = payload.get(key, 0)
            text = "-"
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                text = fmt.format(float(value))
            item = QTableWidgetItem(text)
            self.table.setItem(row, index, item)
        signal_text = str(payload.get("signal", "-"))
        item_signal = QTableWidgetItem(signal_text)
        if signal_text == "call":
            item_signal.setForeground(Qt.green)
        elif signal_text == "put":
            item_signal.setForeground(Qt.red)
        self.table.setItem(row, 8, item_signal)
        if self.table.item(row, 9) is None:
            self.table.setItem(row, 9, QTableWidgetItem("-"))
        if self.table.item(row, 10) is None:
            self.table.setItem(row, 10, QTableWidgetItem("-"))

    def _update_trade(self, par: str, result: str, pnl: float, total: float) -> None:
        row = self._row_for(par)
        result_item = QTableWidgetItem(result)
        if result == "WIN":
            result_item.setForeground(Qt.green)
        elif result == "LOST":
            result_item.setForeground(Qt.red)
        self.table.setItem(row, 9, result_item)
        pnl_item = QTableWidgetItem(f"{pnl:.2f}")
        if pnl > 0:
            pnl_item.setForeground(Qt.green)
        elif pnl < 0:
            pnl_item.setForeground(Qt.red)
        self.table.setItem(row, 10, pnl_item)
        self.label_pnl.setText(f"💰 PnL acumulado: {total:.2f}")

    def _row_for(self, par: str) -> int:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text() == par:
                return row
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(par))
        return row

    def closeEvent(self, event):
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        super().closeEvent(event)


def create_splash() -> QSplashScreen:
    width, height = 600, 300
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    gradient = QLinearGradient(0, 0, 0, height)
    gradient.setColorAt(0.0, QColor(10, 32, 68))
    gradient.setColorAt(1.0, QColor(4, 16, 36))
    painter.fillRect(pixmap.rect(), gradient)
    painter.setPen(QColor("#F5F5F5"))
    painter.setFont(QFont("Segoe UI", 26, QFont.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignHCenter | Qt.AlignTop, "\nIQ Option Digital/Binary Bot")
    painter.setFont(QFont("Segoe UI", 14))
    painter.drawText(
        pixmap.rect().adjusted(0, 90, 0, -120),
        Qt.AlignHCenter | Qt.AlignTop,
        "Escáner de pares digitales/binarias con panel de PnL",
    )
    painter.setFont(QFont("Consolas", 11))
    painter.drawText(
        pixmap.rect().adjusted(0, 150, 0, -90),
        Qt.AlignHCenter | Qt.AlignTop,
        "Preparando conexión y analizando indicadores...",
    )
    painter.end()
    splash = QSplashScreen(pixmap)
    splash.setFont(QFont("Segoe UI", 9))
    return splash


if __name__ == "__main__":
    app = QApplication(sys.argv)
    splash = create_splash()
    splash.show()
    app.processEvents()
    window = BotGUI()
    window.show()
    QTimer.singleShot(2000, lambda: splash.finish(window))
    sys.exit(app.exec_())
