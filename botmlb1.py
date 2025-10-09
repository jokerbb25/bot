import sys
import time
import math
import logging
import threading
from dataclasses import dataclass, asdict
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


def _sanitize(symbol: Optional[str]) -> Optional[str]:
    if not symbol or not isinstance(symbol, str):
        return None
    cleaned = symbol.strip().upper()
    if not cleaned or "OTC" in cleaned:
        return None
    for suffix in ("-OP", "-DIGITAL", "-BINARY"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    cleaned = cleaned.replace(" ", "")
    if cleaned.endswith("-"):
        cleaned = cleaned[:-1]
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

    def ensure_connection(self) -> bool:
        try:
            if self.check_connect() is True:
                return True
        except Exception:
            pass
        try:
            ok, reason = self.connect()
        except Exception:
            logging.exception("Error al reconectar")
            return False
        if not ok:
            logging.error("La reconexión fue rechazada: %s", reason)
            return False
        try:
            self.change_balance(MODO)
        except Exception:
            logging.exception("No se pudo restablecer el balance tras la reconexión")
        return True


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
        payload = asdict(self)
        payload["signal"] = signal or "-"
        return payload


def _alias_variants(symbol: str, instrument_type: str) -> Tuple[str, ...]:
    raw = symbol.strip() if symbol else ""
    if not raw:
        return tuple()
    if "OTC" in raw.upper():
        return tuple()
    upper_raw = raw.upper()
    candidates: List[str] = []
    seen: Set[str] = set()

    def add(candidate: Optional[str]) -> None:
        if not candidate:
            return
        cleaned = candidate.strip()
        if not cleaned:
            return
        cleaned = cleaned.replace(" ", "")
        if cleaned not in seen:
            seen.add(cleaned)
            candidates.append(cleaned)

    add(upper_raw)
    add(upper_raw.replace("-", ""))
    add(upper_raw.replace("/", ""))

    base = _sanitize(upper_raw)
    add(base)
    if base:
        add(base.replace("-", ""))
        add(base.replace("/", ""))
    if instrument_type == "digital" and base:
        add(f"{base}-OP")

    return tuple(candidates)


@dataclass(frozen=True)
class TradePair:
    name: str
    instrument_type: str
    candle_aliases: Tuple[str, ...]
    trade_aliases: Tuple[str, ...]

    @classmethod
    def from_symbol(cls, symbol: str, instrument_type: str) -> Optional["TradePair"]:
        if not symbol or "OTC" in symbol.upper():
            return None
        aliases = _alias_variants(symbol, instrument_type)
        if not aliases:
            return None
        display = _sanitize(symbol) or symbol.strip().upper()
        return cls(display, instrument_type, aliases, aliases)


class BotWorker(QObject):
    status_changed = pyqtSignal(str)
    row_ready = pyqtSignal(str, dict)
    trade_finished = pyqtSignal(str, str, float, float)
    finished = pyqtSignal()

    def __init__(self) -> None:
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
        except Exception:
            logging.exception("Error conectando al broker")
            self.status_changed.emit("❌ Error al conectar con IQ Option")
            self.finished.emit()
            return
        if not ok:
            self.status_changed.emit(f"❌ Conexión rechazada: {reason}")
            self.finished.emit()
            return
        api.change_balance(MODO)
        balance = api.get_balance()
        self.status_changed.emit(f"✅ Conectado a {MODO} | Saldo: {balance:.2f}")
        open_time = api.get_all_open_time()
        pairs = self._discover_pairs(api, open_time)
        if not pairs:
            self.status_changed.emit("⚠️ No se encontraron pares digitales/binarios abiertos")
            try:
                api.close()
            except Exception:
                pass
            self.finished.emit()
            return
        self.status_changed.emit(f"📈 {len(pairs)} pares listos para operar")
        try:
            for ciclo in range(1, CICLOS + 1):
                if self._stop.is_set():
                    break
                logging.info("=== Ciclo %s/%s ===", ciclo, CICLOS)
                for pair in pairs:
                    if self._stop.is_set():
                        break
                    df = self._fetch_candles(api, pair)
                    if df.empty:
                        logging.warning("%s sin velas válidas", pair.name)
                        continue
                    signal, payload = self._analyze(df)
                    self.row_ready.emit(pair.name, payload)
                    if not signal:
                        continue
                    ok, ticket = self._execute_trade(api, pair, signal)
                    if not ok or ticket is None:
                        continue
                    result, pnl = self._wait_result(api, ticket)
                    if result is None:
                        continue
                    self._pnl += pnl
                    label = self._normalize_result(result)
                    mensaje = f"{label} en {pair.name} | PnL: {pnl:.2f} | Total: {self._pnl:.2f}"
                    logging.info(mensaje)
                    self.status_changed.emit(mensaje)
                    self.trade_finished.emit(pair.name, label, pnl, self._pnl)
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

    def _discover_pairs(self, api: SafeIQOption, open_time: Dict) -> List[TradePair]:
        logging.info("♻️ Escaneando pares digitales y binarias disponibles...")
        digital_meta = self._load_digital_metadata(api)
        candidates: List[TradePair] = []
        seen: Set[Tuple[str, str]] = set()
        if isinstance(open_time, dict):
            for category in ("digital", "binary", "turbo"):
                entries = open_time.get(category, {})
                if not isinstance(entries, dict):
                    continue
                instrument_type = "digital" if category == "digital" else "binary"
                for symbol, info in entries.items():
                    if not isinstance(info, dict):
                        continue
                    if not info.get("open", True):
                        continue
                    symbol_text = str(symbol)
                    pair = TradePair.from_symbol(symbol_text, instrument_type)
                    if not pair:
                        continue
                    if instrument_type == "digital":
                        meta = self._resolve_digital_meta(digital_meta, symbol_text, pair.name)
                        if meta and (meta.get("enabled") is False or meta.get("is_suspended") is True):
                            logging.info("⏭️ %s omitido (digital cerrado/suspendido)", pair.name)
                            continue
                    key = (pair.name, pair.instrument_type)
                    if key in seen:
                        continue
                    if self._fetch_candles(api, pair, sample=True).empty:
                        continue
                    candidates.append(pair)
                    seen.add(key)
        return candidates

    def _load_digital_metadata(self, api: SafeIQOption) -> Dict[str, Dict]:
        mapping: Dict[str, Dict] = {}
        try:
            payload = api.get_digital_underlying_list_data()
        except Exception:
            logging.exception("No se pudieron obtener metadatos digitales")
            return mapping
        if not isinstance(payload, dict):
            return mapping
        underlying = payload.get("underlying", {})
        if not isinstance(underlying, dict):
            return mapping
        for info in underlying.values():
            if not isinstance(info, dict):
                continue
            symbol = (
                info.get("symbol")
                or info.get("underlying")
                or info.get("active")
                or info.get("asset_name")
                or info.get("name")
            )
            if not symbol:
                continue
            symbol_text = str(symbol).strip().upper()
            if not symbol_text or "OTC" in symbol_text:
                continue
            base = _sanitize(symbol_text)
            keys = [symbol_text]
            if base:
                keys.append(base)
                keys.append(base.replace("/", ""))
                keys.append(base.replace("-", ""))
            for key in keys:
                if key and key not in mapping:
                    mapping[key] = info
        return mapping

    def _resolve_digital_meta(self, metadata: Dict[str, Dict], raw_symbol: str, base_name: str) -> Optional[Dict]:
        candidates = [
            raw_symbol.strip().upper(),
            _sanitize(raw_symbol),
            base_name,
            base_name.replace("/", ""),
            base_name.replace("-", ""),
        ]
        for candidate in candidates:
            if candidate and candidate in metadata:
                return metadata[candidate]
        return None

    def _fetch_candles(self, api: SafeIQOption, pair: TradePair, sample: bool = False) -> pd.DataFrame:
        history = 30 if sample else 60
        for alias in pair.candle_aliases:
            try:
                candles = api.get_candles(alias, 60, history, time.time())
            except Exception as exc:
                message = str(exc).lower()
                logging.debug("Error obteniendo velas para %s (%s)", alias, exc)
                if "need reconnect" in message or "connection" in message:
                    if not api.ensure_connection():
                        break
                    continue
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
        for columns in (("open", "max", "min", "close"), ("open", "high", "low", "close"), ("o", "h", "l", "c")):
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
        macd_indicator = ta.trend.MACD(close)
        macd = float(macd_indicator.macd().iloc[-1])
        macd_signal = float(macd_indicator.macd_signal().iloc[-1])
        stoch_indicator = ta.momentum.StochasticOscillator(high, low, close)
        stoch_k = float(stoch_indicator.stoch().iloc[-1])
        stoch_d = float(stoch_indicator.stoch_signal().iloc[-1])
        snapshot = IndicatorSnapshot(rsi, ema_fast, ema_slow, macd, macd_signal, stoch_k, stoch_d)
        up = down = votos = 0

        def threshold(value: float, lower: float, upper: float) -> None:
            nonlocal up, down, votos
            if math.isnan(value):
                return
            votos += 1
            if value < lower:
                up += 1
            elif value > upper:
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
        self,
        api: SafeIQOption,
        pair: TradePair,
        signal: str,
    ) -> Tuple[bool, Optional[Tuple[str, str]]]:
        for alias in pair.trade_aliases:
            try:
                if pair.instrument_type == "digital":
                    ok, trade_id = api.buy_digital_spot(alias, MONTO, signal, EXPIRACION)
                    trade_kind = "digital"
                else:
                    ok, trade_id = api.buy(MONTO, alias, signal, EXPIRACION)
                    trade_kind = "binary"
            except Exception as exc:
                logging.debug("Error ejecutando operación %s (%s)", alias, exc)
                if "need reconnect" in str(exc).lower():
                    api.ensure_connection()
                continue
            if ok and trade_id:
                logging.info("[OK] %s en %s (alias %s)", signal.upper(), pair.name, alias)
                return True, (trade_kind, str(trade_id))
        logging.warning("[FAIL] Broker rechazó %s en %s", signal.upper(), pair.name)
        return False, None

    def _wait_result(
        self,
        api: SafeIQOption,
        ticket: Tuple[str, str],
    ) -> Tuple[Optional[str], float]:
        trade_type, trade_id = ticket
        timeout = EXPIRACION * 60 + 120
        waited = 0
        while waited < timeout and not self._stop.is_set():
            try:
                if trade_type == "digital":
                    result, pnl = api.check_win_digital_v2(trade_id)
                else:
                    response = api.check_win_v3(trade_id)
                    if response is None:
                        time.sleep(1)
                        waited += 1
                        continue
                    if isinstance(response, (list, tuple)) and len(response) >= 2:
                        result, pnl = response[0], response[1]
                    else:
                        result, pnl = response, 0.0
                        extra = api.check_win_v4(trade_id)
                        if isinstance(extra, (list, tuple)) and len(extra) >= 2:
                            pnl = extra[1]
                if result is None:
                    time.sleep(1)
                    waited += 1
                    continue
                try:
                    pnl_value = float(pnl)
                except (TypeError, ValueError):
                    pnl_value = 0.0
                return result, pnl_value
            except Exception as exc:
                logging.debug("Error consultando resultado (%s)", exc)
                if "need reconnect" in str(exc).lower():
                    api.ensure_connection()
                    time.sleep(1)
                    waited += 1
                    continue
                return None, 0.0
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
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("IQ Option Digital/Binary Bot")
        self.resize(1000, 520)
        layout = QVBoxLayout(self)
        self.label_status = QLabel("⏳ Iniciando...", self)
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

    def _row_for(self, pair: str) -> int:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text() == pair:
                return row
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(pair))
        return row

    def _update_row(self, pair: str, payload: Dict[str, Union[float, str]]) -> None:
        row = self._row_for(pair)
        columns = [
            ("rsi", "{:.1f}"),
            ("ema_fast", "{:.5f}"),
            ("ema_slow", "{:.5f}"),
            ("macd", "{:.5f}"),
            ("macd_signal", "{:.5f}"),
            ("stoch_k", "{:.1f}"),
            ("stoch_d", "{:.1f}"),
        ]
        for index, (key, fmt) in enumerate(columns, start=1):
            value = payload.get(key, "-")
            text = "-"
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                text = fmt.format(float(value))
            item = QTableWidgetItem(text)
            self.table.setItem(row, index, item)
        signal_text = str(payload.get("signal", "-"))
        signal_item = QTableWidgetItem(signal_text)
        if signal_text == "call":
            signal_item.setForeground(Qt.green)
        elif signal_text == "put":
            signal_item.setForeground(Qt.red)
        self.table.setItem(row, 8, signal_item)
        if self.table.item(row, 9) is None:
            self.table.setItem(row, 9, QTableWidgetItem("-"))
        if self.table.item(row, 10) is None:
            self.table.setItem(row, 10, QTableWidgetItem("-"))

    def _update_trade(self, pair: str, result: str, pnl: float, total: float) -> None:
        row = self._row_for(pair)
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

    def closeEvent(self, event):
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        super().closeEvent(event)


def create_splash() -> QSplashScreen:
    width, height = 620, 320
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    gradient = QLinearGradient(0, 0, 0, height)
    gradient.setColorAt(0.0, QColor(12, 38, 76))
    gradient.setColorAt(1.0, QColor(5, 18, 40))
    painter.fillRect(pixmap.rect(), gradient)
    painter.setPen(QColor("#F5F5F5"))
    painter.setFont(QFont("Segoe UI", 26, QFont.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignHCenter | Qt.AlignTop, "\nIQ Option Digital/Binary Bot")
    painter.setFont(QFont("Segoe UI", 14))
    painter.drawText(
        pixmap.rect().adjusted(0, 90, 0, -120),
        Qt.AlignHCenter | Qt.AlignTop,
        "Escáner de pares digitales/binarios con panel de PnL",
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
