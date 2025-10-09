import sys
import time
import math
import logging
from dataclasses import dataclass
from datetime import datetime
from textwrap import dedent
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def _ensure_dependency(module: str, pip_name: Optional[str] = None) -> None:
    try:
        __import__(module)
    except ImportError:  # pragma: no cover - runtime guard
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


EMAIL = "fornerinoalejandro031@gmail.com"
PASSWORD = "484572ale"
MODO = "PRACTICE"
MONTO = 1.0
EXPIRACION = 1
ESPERA_ENTRE_CICLOS = 3
CICLOS = 50

BINARY_FALLBACK = (
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

        if not isinstance(payload.get("underlying"), dict):
            payload["underlying"] = {}

        return payload

    safe_get_digital_underlying_list_data.__name__ = "safe_get_digital_underlying_list_data"
    IQ_Option.get_digital_underlying_list_data = safe_get_digital_underlying_list_data  # type: ignore


_patch_digital_underlying()


class SafeIQOption(IQ_Option):
    def ensure_connection(self) -> bool:
        try:
            status, reason = self.connect()
        except Exception as exc:  # pragma: no cover
            logging.error("Error reconectando con el broker: %s", exc)
            return False
        if not status:
            logging.error("No fue posible reconectar: %s", reason)
            return False
        self.change_balance(MODO)
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

    def to_row(self, signal: Optional[str]) -> Dict[str, str]:
        return {
            "rsi": _fmt(self.rsi, 1),
            "ema_fast": _fmt(self.ema_fast, 5),
            "ema_slow": _fmt(self.ema_slow, 5),
            "macd": _fmt(self.macd, 5),
            "macd_signal": _fmt(self.macd_signal, 5),
            "stoch_k": _fmt(self.stoch_k, 1),
            "stoch_d": _fmt(self.stoch_d, 1),
            "signal": signal or "-",
        }


@dataclass
class TradePair:
    name: str
    active_id: Optional[int]
    aliases: Tuple[str, ...]
    is_otc: bool


@dataclass
class TradeResult:
    ticket: str
    pnl: float
    outcome: str


@dataclass
class PendingTrade:
    pair: str
    signal: str
    ticket: str
    placed_at: float
    expiry_minutes: int

    @property
    def deadline(self) -> float:
        return self.placed_at + self.expiry_minutes * 60 + 120


def _fmt(value: float, precision: int) -> str:
    if math.isnan(value):
        return "-"
    return f"{value:.{precision}f}"


def _sanitize(symbol: Optional[str]) -> Optional[str]:
    if not symbol or not isinstance(symbol, str):
        return None
    cleaned = symbol.strip().upper()
    if not cleaned:
        return None
    return cleaned


def _is_otc(symbol: str) -> bool:
    return "OTC" in symbol.upper()


def _alias_variations(symbol: str) -> List[str]:
    base = symbol.strip()
    if not base:
        return []
    variants = {base, base.upper(), base.lower()}
    variants.update({base.replace("/", ""), base.replace("-", "")})
    upper = base.upper()
    variants.update({upper.replace("/", ""), upper.replace("-", "")})
    lower = base.lower()
    variants.update({lower.replace("/", ""), lower.replace("-", "")})
    return [variant for variant in variants if variant]


def _tables_to_update(api: SafeIQOption) -> List[Dict]:
    tables: List[Dict] = []
    for name in ("ACTIVES", "ACTIVES_ID", "assets", "assets_name"):
        table = getattr(iq_constants, name, None)
        if isinstance(table, dict):
            tables.append(table)
    for attr in ("ACTIVES", "ACTIVES_ID", "assets", "assets_name"):
        table = getattr(api, attr, None)
        if isinstance(table, dict):
            tables.append(table)
    return tables


def _register_symbol(api: SafeIQOption, alias: str, active_id: Optional[int]) -> None:
    if not active_id:
        return
    for candidate in _alias_variations(alias):
        for table in _tables_to_update(api):
            if not table:
                table[candidate.upper()] = {"id": active_id, "name": candidate.upper()}
                continue
            sample_key = next(iter(table.keys()))
            if isinstance(sample_key, str):
                entry = table.get(candidate)
                if isinstance(entry, dict):
                    entry["id"] = active_id
                    entry.setdefault("name", candidate)
                elif isinstance(entry, int):
                    table[candidate] = active_id
                else:
                    table[candidate] = {"id": active_id, "name": candidate}
            elif isinstance(sample_key, int):
                table[active_id] = candidate


def _lookup_active_id(api: SafeIQOption, alias: str) -> Optional[int]:
    for candidate in _alias_variations(alias):
        candidate_upper = candidate.upper()
        for table in _tables_to_update(api):
            if not isinstance(table, dict):
                continue
            value = table.get(candidate_upper) or table.get(candidate)
            if isinstance(value, dict):
                candidate_id = value.get("id")
                if isinstance(candidate_id, int):
                    return candidate_id
            elif isinstance(value, int):
                return value
    return None


def _discover_pairs(api: SafeIQOption) -> List[TradePair]:
    try:
        schedule = api.get_all_open_time()
    except Exception:
        logging.exception("No se pudieron obtener los horarios de activos")
        schedule = {}

    discovered: Dict[str, TradePair] = {}

    def register(symbol: str, info: Optional[Dict], default_otc: Optional[bool] = None) -> None:
        normalized = _sanitize(symbol)
        if not normalized:
            return
        is_otc = _is_otc(normalized) if default_otc is None else default_otc
        aliases = tuple(_alias_variations(normalized))
        active_id = None
        if isinstance(info, dict):
            raw_id = info.get("id") or info.get("active_id")
            try:
                active_id = int(raw_id) if raw_id is not None else None
            except (TypeError, ValueError):
                active_id = None
        if active_id is None:
            active_id = _lookup_active_id(api, normalized)
        discovered.setdefault(normalized, TradePair(normalized, active_id, aliases, is_otc))

    if isinstance(schedule, dict):
        for category in ("turbo", "binary"):
            entries = schedule.get(category, {})
            if not isinstance(entries, dict):
                continue
            for symbol, info in entries.items():
                if not isinstance(info, dict):
                    continue
                if info.get("open") is False or info.get("enabled") is False:
                    continue
                register(symbol, info)

    if isinstance(schedule, dict):
        for symbol, info in schedule.get("digital", {}).items():
            if not isinstance(info, dict):
                continue
            if info.get("open") is False or info.get("enabled") is False:
                continue
            register(symbol, info)

    if not discovered:
        for symbol in BINARY_FALLBACK:
            register(symbol, None, default_otc=False)
        for symbol in OTC_FALLBACK:
            register(symbol, None, default_otc=True)

    return list(discovered.values())


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
            if not frame.empty:
                return frame
    return pd.DataFrame()


def _fetch_candles(api: SafeIQOption, pair: TradePair, count: int = 120) -> pd.DataFrame:
    active_id = pair.active_id
    if active_id is not None:
        try:
            payload = api.get_candles(active_id, 60, count, time.time())
        except Exception as exc:
            message = str(exc).lower()
            if "need reconnect" in message and api.ensure_connection():
                payload = api.get_candles(active_id, 60, count, time.time())
            else:
                payload = None
        if isinstance(payload, dict):
            payload = payload.get("candles") or payload.get("data")
        if isinstance(payload, list):
            df = _build_dataframe(payload)
            if not df.empty:
                return df

    for alias in pair.aliases[:3]:
        _register_symbol(api, alias, pair.active_id)
        try:
            payload = api.get_candles(alias, 60, count, time.time())
        except Exception as exc:
            message = str(exc).lower()
            if "not found on consts" in message:
                candidate_id = _lookup_active_id(api, alias)
                if candidate_id:
                    pair.active_id = candidate_id
                    _register_symbol(api, alias, candidate_id)
            elif "need reconnect" in message and api.ensure_connection():
                try:
                    payload = api.get_candles(alias, 60, count, time.time())
                except Exception:
                    continue
            else:
                continue
        if isinstance(payload, dict):
            payload = payload.get("candles") or payload.get("data")
        if isinstance(payload, list):
            df = _build_dataframe(payload)
            if not df.empty:
                return df
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


def _check_result_once(api: SafeIQOption, ticket: str) -> Optional[TradeResult]:
    try:
        result, pnl = api.check_win_v3(ticket)
    except Exception as exc:
        message = str(exc).lower()
        if "need reconnect" in message and api.ensure_connection():
            return None
        logging.warning("Error consultando resultado %s: %s", ticket, exc)
        return None
    if result is None:
        return None
    try:
        pnl_value = float(pnl)
    except (TypeError, ValueError):
        pnl_value = 0.0
    return TradeResult(ticket, pnl_value, result)


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
        self._pending: Dict[str, PendingTrade] = {}

    def stop(self) -> None:
        self._stop = True

    def _poll_pending(self, api: SafeIQOption) -> None:
        if not self._pending:
            return
        now = time.time()
        for ticket, pending in list(self._pending.items()):
            result = _check_result_once(api, ticket)
            if result is None:
                if now >= pending.deadline:
                    logging.warning("Timeout esperando resultado para %s", ticket)
                    timeout_result = TradeResult(ticket, 0.0, "unknown")
                    self._pending.pop(ticket, None)
                    self._finalize_trade(pending, timeout_result)
                continue
            self._pending.pop(ticket, None)
            self._finalize_trade(pending, result)

    def _finalize_trade(self, pending: PendingTrade, result: TradeResult) -> None:
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
            logging.warning("Resultado desconocido para %s: %s", pending.pair, outcome)

        mensaje = (
            f"{outcome} en {pending.pair} | PnL operación: {pnl:.2f} | PnL acumulado: {self._pnl:.2f}"
        )
        logging.info(mensaje)
        self.status_changed.emit(mensaje)
        self.trade_completed.emit(pending.pair, outcome, pnl, self._pnl)

    def _sleep_with_pending(self, api: SafeIQOption, duration: float) -> None:
        deadline = time.time() + max(duration, 0)
        while not self._stop and time.time() < deadline:
            self._poll_pending(api)
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            time.sleep(min(1.0, remaining))
        self._poll_pending(api)

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

        try:
            for cycle in range(1, CICLOS + 1):
                if self._stop:
                    break
                logging.info("=== Ciclo %s/%s ===", cycle, CICLOS)
                self.status_changed.emit(
                    f"🔁 Ciclo {cycle}/{CICLOS} | PnL acumulado: {self._pnl:.2f}"
                )

                pairs = _discover_pairs(api)
                if not pairs:
                    self.status_changed.emit("⚠️ No se detectaron pares disponibles en este ciclo")
                    self._sleep_with_pending(api, ESPERA_ENTRE_CICLOS)
                    continue

                for name in list(self._failures.keys()):
                    self._failures[name] = max(self._failures[name] - 1, 0)
                    if self._failures[name] == 0:
                        self._failures.pop(name, None)

                for pair in pairs:
                    if self._stop:
                        break

                    self._poll_pending(api)

                    failures = self._failures.get(pair.name, 0)
                    if failures >= 3:
                        continue

                    df = _fetch_candles(api, pair)
                    if df.empty:
                        self._failures[pair.name] = failures + 1
                        continue

                    self._failures[pair.name] = 0
                    signal, snapshot = _compute_signal(df)
                    self.row_ready.emit(pair.name, snapshot.to_row(signal))
                    if not signal:
                        continue

                    active_id = _lookup_active_id(api, pair.name)
                    if active_id is not None:
                        pair.active_id = active_id
                    _register_symbol(api, pair.name, pair.active_id)

                    ok, ticket = api.buy(MONTO, pair.name, signal, EXPIRACION)
                    if not ok or not ticket:
                        self._failures[pair.name] = failures + 1
                        continue

                    logging.info("[OK] %s en %s (ticket=%s)", signal.upper(), pair.name, ticket)
                    self._pending[ticket] = PendingTrade(
                        pair=pair.name,
                        signal=signal,
                        ticket=ticket,
                        placed_at=time.time(),
                        expiry_minutes=EXPIRACION,
                    )
                    time.sleep(1)
                    self._poll_pending(api)

                if self._stop:
                    break

                self._sleep_with_pending(api, ESPERA_ENTRE_CICLOS)
        finally:
            try:
                api.close()
            except Exception:
                pass
            self.finished.emit()


class BotGUI(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("IQ Option Bot - Binary & OTC Monitor")
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
            self.table.setItem(row, column, QTableWidgetItem(data.get(key, "-")))
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

    def closeEvent(self, event) -> None:  # noqa: N802
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
    painter.drawText(pixmap.rect(), Qt.AlignHCenter | Qt.AlignTop, "\nIQ Option Binary/OTC Bot")

    painter.setFont(QFont("Segoe UI", 14))
    painter.drawText(
        pixmap.rect().adjusted(0, 90, 0, -140),
        Qt.AlignHCenter | Qt.AlignTop,
        "Escáner y monitor de operaciones binarias/OTC",
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
