from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QHBoxLayout,
    QComboBox,
    QProgressBar,
    QTabWidget,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGridLayout,
    QDoubleSpinBox,
    QFormLayout,
    QSizePolicy,
)

from engine import BotEngine


class BotAxiGUI(QWidget):
    log_received = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    confidence_changed = pyqtSignal(float, str)
    order_added = pyqtSignal(object)
    stats_changed = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None, gui_log: Optional[Callable[[str], None]] = None):
        super().__init__(parent)
        self.setWindowTitle("Bot AxiNew")
        self.resize(1080, 720)
        self.setStyleSheet(
            """
            QWidget {
                background-color: #0B0F19;
                color: #E2E8F0;
                font-size: 12px;
            }
            QTabBar::tab {
                background: #1E293B;
                padding: 8px;
                color: #E2E8F0;
            }
            QTabBar::tab:selected {
                background: #0EA5E9;
            }
            QPushButton {
                background-color: #0EA5E9;
                padding: 6px;
                border-radius: 6px;
            }
            QPushButton#start {
                background-color: #10B981;
            }
            QPushButton#stop {
                background-color: #EF4444;
            }
            QTableWidget {
                gridline-color: #1E293B;
            }
            QTextEdit {
                background-color: #111827;
                border: 1px solid #1F2937;
            }
            QComboBox, QDoubleSpinBox {
                background-color: #111827;
                border: 1px solid #1F2937;
            }
            QProgressBar {
                border: 1px solid #1F2937;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0EA5E9;
            }
            """
        )

        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignLeft)

        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stop")
        self.stop_button.setEnabled(False)

        self.confidence_label = QLabel("Confianza: 0% (Dirección: NONE)")
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)

        self.trade_table = QTableWidget(0, 7)
        self.trade_table.setHorizontalHeaderLabels([
            "Hora",
            "Símbolo",
            "Decisión",
            "Confianza",
            "Resultado",
            "PnL",
            "Notas",
        ])
        self.trade_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.trade_table.verticalHeader().setVisible(False)
        self.trade_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        self.total_label = QLabel("Operaciones: 0")
        self.win_label = QLabel("Ganadas: 0")
        self.loss_label = QLabel("Perdidas: 0")
        self.pnl_label = QLabel("PnL: 0.00")

        general_tab = self._build_general_tab()
        strategies_tab = self._build_strategies_tab()
        summary_tab = self._build_summary_tab()
        config_tab = self._build_config_tab()
        learning_tab = self._build_learning_tab()

        self.tabs = QTabWidget()
        self.tabs.addTab(general_tab, "General")
        self.tabs.addTab(strategies_tab, "Estrategias")
        self.tabs.addTab(summary_tab, "Resumen")
        self.tabs.addTab(config_tab, "Configuración")
        self.tabs.addTab(learning_tab, "Aprendizaje")

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

        self.log_received.connect(self._append_log)
        self.status_changed.connect(self._update_status)
        self.confidence_changed.connect(self._update_confidence)
        self.order_added.connect(self._append_order)
        self.stats_changed.connect(self._update_stats_labels)

        config_path = Path(__file__).resolve().parent / "config.yaml"
        memory_path = Path(__file__).resolve().parent / "memory.json"
        self.engine = BotEngine(
            gui_log=self.gui_log,
            config_path=config_path,
            memory_path=memory_path,
            status_callback=self._status_from_engine,
            confidence_callback=self._confidence_from_engine,
            order_callback=self._order_from_engine,
            stats_callback=self._stats_from_engine,
        )

        self._populate_config_controls()

        self.start_button.clicked.connect(self._handle_start)
        self.stop_button.clicked.connect(self._handle_stop)

        if gui_log:
            self.log_received.connect(lambda message: gui_log(message))

        self.refresh_memory_view()

    def _build_general_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        header_layout = QHBoxLayout()
        header_layout.addWidget(self.status_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.start_button)
        header_layout.addWidget(self.stop_button)

        stats_layout = QGridLayout()
        stats_layout.addWidget(self.total_label, 0, 0)
        stats_layout.addWidget(self.win_label, 0, 1)
        stats_layout.addWidget(self.loss_label, 1, 0)
        stats_layout.addWidget(self.pnl_label, 1, 1)

        layout.addLayout(header_layout)
        layout.addLayout(stats_layout)
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.confidence_bar)
        layout.addWidget(self.trade_table)
        layout.addWidget(QLabel("Logs"))
        layout.addWidget(self.log_view)
        return widget

    def _build_strategies_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("Activar / desactivar estrategias"))
        self.strategy_checks: Dict[str, QCheckBox] = {}
        for key, label in [
            ("rsi", "RSI"),
            ("ema", "EMA"),
            ("macd", "MACD"),
            ("pullback", "Pullback"),
            ("memory", "Memoria"),
        ]:
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, name=key: self.engine.update_strategy(name, state == Qt.Checked))
            layout.addWidget(checkbox)
            self.strategy_checks[key] = checkbox
        layout.addStretch(1)
        return widget

    def _build_summary_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.summary_trades = QLabel("Operaciones del día: 0")
        self.summary_wins = QLabel("Ganadas: 0")
        self.summary_losses = QLabel("Perdidas: 0")
        self.summary_precision = QLabel("Precisión: 0.0%")
        self.summary_pnl = QLabel("PnL acumulado: 0.00")
        layout.addWidget(self.summary_trades)
        layout.addWidget(self.summary_wins)
        layout.addWidget(self.summary_losses)
        layout.addWidget(self.summary_precision)
        layout.addWidget(self.summary_pnl)
        layout.addStretch(1)
        return widget

    def _build_config_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        form = QFormLayout()
        self.symbol_selector = QComboBox()
        self.symbol_selector.currentTextChanged.connect(self.engine.set_focus_symbol)
        form.addRow("Símbolo preferido", self.symbol_selector)

        self.lot_high_spin = QDoubleSpinBox()
        self.lot_high_spin.setDecimals(2)
        self.lot_high_spin.setRange(0.0, 100.0)
        self.lot_high_spin.valueChanged.connect(self._update_risk_settings)
        form.addRow("Lote alto", self.lot_high_spin)

        self.lot_low_spin = QDoubleSpinBox()
        self.lot_low_spin.setDecimals(2)
        self.lot_low_spin.setRange(0.0, 100.0)
        self.lot_low_spin.valueChanged.connect(self._update_risk_settings)
        form.addRow("Lote bajo", self.lot_low_spin)

        self.base_confidence_spin = QDoubleSpinBox()
        self.base_confidence_spin.setDecimals(2)
        self.base_confidence_spin.setRange(0.0, 1.0)
        self.base_confidence_spin.setSingleStep(0.01)
        self.base_confidence_spin.valueChanged.connect(self._update_confidence_settings)
        form.addRow("Confianza base", self.base_confidence_spin)

        self.lower_confidence_spin = QDoubleSpinBox()
        self.lower_confidence_spin.setDecimals(2)
        self.lower_confidence_spin.setRange(0.0, 1.0)
        self.lower_confidence_spin.setSingleStep(0.01)
        self.lower_confidence_spin.valueChanged.connect(self._update_confidence_settings)
        form.addRow("Confianza baja", self.lower_confidence_spin)

        self.memory_boost_spin = QDoubleSpinBox()
        self.memory_boost_spin.setDecimals(2)
        self.memory_boost_spin.setRange(0.0, 1.0)
        self.memory_boost_spin.setSingleStep(0.01)
        self.memory_boost_spin.valueChanged.connect(self._update_confidence_settings)
        form.addRow("Refuerzo memoria", self.memory_boost_spin)

        layout.addLayout(form)
        layout.addStretch(1)
        return widget

    def _build_learning_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.memory_view = QTextEdit()
        self.memory_view.setReadOnly(True)
        refresh_button = QPushButton("Actualizar memoria")
        refresh_button.clicked.connect(self.refresh_memory_view)
        layout.addWidget(QLabel("Patrones ganadores"))
        layout.addWidget(self.memory_view)
        layout.addWidget(refresh_button)
        layout.addStretch(1)
        return widget

    def _populate_config_controls(self) -> None:
        self.symbol_selector.clear()
        if self.engine.symbols:
            self.symbol_selector.addItems(self.engine.symbols)
            self.symbol_selector.setCurrentIndex(0)
        self.lot_high_spin.setValue(self.engine.lot_high)
        self.lot_low_spin.setValue(self.engine.lot_low)
        self.base_confidence_spin.setValue(self.engine.base_confidence)
        self.lower_confidence_spin.setValue(self.engine.lower_confidence)
        self.memory_boost_spin.setValue(self.engine.memory_boost)

    def _update_risk_settings(self) -> None:
        self.engine.update_risk(self.lot_high_spin.value(), self.lot_low_spin.value())

    def _update_confidence_settings(self) -> None:
        self.engine.update_confidence_levels(
            self.base_confidence_spin.value(),
            self.lower_confidence_spin.value(),
            self.memory_boost_spin.value(),
        )

    def _handle_start(self) -> None:
        started = self.engine.start()
        if started:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self._update_status("Scanning")
        else:
            self.gui_log("Engine already running.")

    def _handle_stop(self) -> None:
        self.engine.stop()
        self._update_status("Idle")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def gui_log(self, message: str) -> None:
        self.log_received.emit(message)

    def _append_log(self, message: str) -> None:
        self.log_view.append(message)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _status_from_engine(self, status: str) -> None:
        self.status_changed.emit(status)

    def _confidence_from_engine(self, confidence: float, direction: str) -> None:
        self.confidence_changed.emit(confidence, direction)

    def _order_from_engine(self, event: Dict[str, Any]) -> None:
        self.order_added.emit(event)
        if str(event.get("notes")) == "Result":
            self.refresh_memory_view()

    def _stats_from_engine(self, stats: Dict[str, Any]) -> None:
        self.stats_changed.emit(stats)

    def _update_status(self, status: str) -> None:
        self.status_label.setText(f"Status: {status}")

    def _update_confidence(self, confidence: float, direction: str) -> None:
        percent = int(max(0.0, min(confidence, 1.0)) * 100)
        self.confidence_bar.setValue(percent)
        self.confidence_label.setText(f"Confianza: {percent}% (Dirección: {direction})")

    def _append_order(self, event: Dict[str, Any]) -> None:
        row_position = self.trade_table.rowCount()
        self.trade_table.insertRow(row_position)
        values = [
            str(event.get("time", "")),
            str(event.get("symbol", "")),
            str(event.get("decision", "")),
            f"{float(event.get('confidence', 0.0)):.2f}",
            str(event.get("result", "")),
            f"{float(event.get('pnl', 0.0)):.2f}",
            str(event.get("notes", "")),
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.trade_table.setItem(row_position, column, item)
        self.trade_table.scrollToBottom()

    def _update_stats_labels(self, stats: Dict[str, Any]) -> None:
        trades = int(stats.get("trades", 0))
        wins = int(stats.get("wins", 0))
        losses = int(stats.get("losses", 0))
        pnl = float(stats.get("pnl", 0.0))
        precision = (wins / trades * 100) if trades else 0.0
        self.total_label.setText(f"Operaciones: {trades}")
        self.win_label.setText(f"Ganadas: {wins}")
        self.loss_label.setText(f"Perdidas: {losses}")
        self.pnl_label.setText(f"PnL: {pnl:.2f}")
        self.summary_trades.setText(f"Operaciones del día: {trades}")
        self.summary_wins.setText(f"Ganadas: {wins}")
        self.summary_losses.setText(f"Perdidas: {losses}")
        self.summary_precision.setText(f"Precisión: {precision:.1f}%")
        self.summary_pnl.setText(f"PnL acumulado: {pnl:.2f}")

    def refresh_memory_view(self) -> None:
        patterns = self.engine.get_memory_snapshot()
        self.memory_view.setPlainText(json.dumps(patterns, indent=2, ensure_ascii=False))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.engine.stop()
        super().closeEvent(event)
