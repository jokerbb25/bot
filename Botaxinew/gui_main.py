from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QComboBox,
    QDoubleSpinBox,
    QProgressBar,
    QTabWidget,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSizePolicy,
    QAbstractItemView,
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
        self.resize(1180, 760)
        self.setStyleSheet(
            """
    QWidget {
        background-color: #0B0F19;   /* BotAxi dark */
        color: #E2E8F0;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        font-size: 10px;             /* BotAxi small text */
    }

    /* Tab bar — BotAxi blue and compact height */
    QTabWidget::pane {
        border: 0px;
        background: #0B0F19;
    }
    QTabBar::tab {
        background: #0F172A;         /* dark slate */
        color: #E2E8F0;
        padding: 6px 16px;
        margin-right: 2px;
        height: 28px;
        border-radius: 6px;
        font-weight: 600;
    }
    QTabBar::tab:selected {
        background: #1599FF;         /* BotAxi light blue */
        color: #FFFFFF;
    }

    /* Buttons */
    QPushButton {
        background-color: #1599FF;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 10px;
        min-width: 80px; min-height: 28px;
    }
    QPushButton#start { background-color: #10B981; }  /* green */
    QPushButton#stop  { background-color: #EF4444; }  /* red   */

    /* Table + headers */
    QTableWidget {
        background-color: #0F172A;
        gridline-color: #1E293B;
        font-size: 10px;
    }
    QHeaderView::section {
        background-color: #1599FF;
        color: white;
        padding: 4px 6px;
        border: 0px;
        font-size: 10px;
    }

    /* Inputs */
    QComboBox, QDoubleSpinBox {
        background-color: #111827;
        border: 1px solid #1F2937;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 10px;
        min-height: 22px;
    }

    /* Logs + progress */
    QTextEdit {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        font-size: 10px;
    }
    QProgressBar {
        border: 1px solid #1F2937;
        text-align: center;
        min-height: 12px;
    }
    QProgressBar::chunk { background-color: #1599FF; }
"""
        )

        self.log_received.connect(self._append_log)
        self.status_changed.connect(self._update_status)
        self.confidence_changed.connect(self._update_confidence)
        self.order_added.connect(self._append_order)
        self.stats_changed.connect(self._update_stats_labels)

        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("start")
        self.start_button.setFixedSize(80, 28)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stop")
        self.stop_button.setFixedSize(80, 28)
        self.stop_button.setEnabled(False)

        self.status_label = QLabel("Estado: Idle")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.ai_label = QLabel("IA Memoria: Activa")
        self.ai_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.mode_label = QLabel("Modo SL/TP: Fixed pips")
        self.mode_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setDecimals(2)
        self.amount_spin.setRange(0.0, 100.0)
        self.amount_spin.setSingleStep(0.01)

        self.sl_tp_mode_combo = QComboBox()
        self.sl_tp_mode_combo.addItems(["Fixed pips", "ATR x Multiplier"])
        self.sl_spin = QDoubleSpinBox()
        self.sl_spin.setDecimals(2)
        self.sl_spin.setRange(0.0, 500.0)
        self.sl_spin.setSingleStep(0.1)
        self.tp_spin = QDoubleSpinBox()
        self.tp_spin.setDecimals(2)
        self.tp_spin.setRange(0.0, 500.0)
        self.tp_spin.setSingleStep(0.1)
        self.apply_pending_check = QCheckBox("Apply to PENDING too")

        self.sl_label = QLabel("SL (pips)")
        self.tp_label = QLabel("TP (pips)")

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
        header = self.trade_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Fixed)
        column_widths = [80, 80, 80, 80, 80, 80, 120]
        for index, width in enumerate(column_widths):
            self.trade_table.setColumnWidth(index, width)
        self.trade_table.verticalHeader().setVisible(False)
        self.trade_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.trade_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.trade_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.trade_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        self.trades_label = QLabel("Operaciones: 0")
        self.wins_label = QLabel("Ganadas: 0")
        self.losses_label = QLabel("Perdidas: 0")
        self.precision_label = QLabel("Precisión: 0.0%")
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
        self._sync_sl_tp_labels()
        self._update_ai_label()

        self.start_button.clicked.connect(self._handle_start)
        self.stop_button.clicked.connect(self._handle_stop)
        self.amount_spin.valueChanged.connect(self._update_operation_amount)
        self.sl_tp_mode_combo.currentTextChanged.connect(self._on_sl_tp_mode_changed)
        self.sl_spin.valueChanged.connect(self._on_sl_tp_values_changed)
        self.tp_spin.valueChanged.connect(self._on_sl_tp_values_changed)
        self.apply_pending_check.stateChanged.connect(self._on_apply_pending_changed)

        if gui_log:
            self.log_received.connect(lambda message: gui_log(message))

        self.refresh_memory_view()

    def _build_general_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)
        header_layout.addWidget(self.start_button)
        header_layout.addWidget(self.stop_button)
        header_layout.addStretch(1)
        header_layout.addWidget(self.status_label)
        header_layout.addWidget(self.ai_label)
        header_layout.addWidget(self.mode_label)

        second_row = QHBoxLayout()
        second_row.setSpacing(16)
        amount_layout = QVBoxLayout()
        amount_label = QLabel("Operation Amount")
        amount_layout.addWidget(amount_label)
        amount_layout.addWidget(self.amount_spin)
        second_row.addLayout(amount_layout)

        sltp_layout = QVBoxLayout()
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("SL/TP Mode"))
        mode_layout.addWidget(self.sl_tp_mode_combo)
        sltp_layout.addLayout(mode_layout)

        values_layout = QGridLayout()
        values_layout.addWidget(self.sl_label, 0, 0)
        values_layout.addWidget(self.sl_spin, 0, 1)
        values_layout.addWidget(self.tp_label, 1, 0)
        values_layout.addWidget(self.tp_spin, 1, 1)
        sltp_layout.addLayout(values_layout)
        sltp_layout.addWidget(self.apply_pending_check)
        second_row.addLayout(sltp_layout)
        second_row.addStretch(1)

        performance_layout = QHBoxLayout()
        performance_layout.setSpacing(16)
        performance_layout.addWidget(self.trades_label)
        performance_layout.addWidget(self.wins_label)
        performance_layout.addWidget(self.losses_label)
        performance_layout.addWidget(self.precision_label)
        performance_layout.addWidget(self.pnl_label)
        performance_layout.addStretch(1)

        layout.addLayout(header_layout)
        layout.addLayout(second_row)
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.confidence_bar)
        layout.addLayout(performance_layout)
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
            checkbox.stateChanged.connect(lambda state, name=key: self._toggle_strategy(name, state))
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
        layout.setSpacing(12)

        symbols_label = QLabel("Símbolos a escanear")
        layout.addWidget(symbols_label)
        self.symbol_checks: Dict[str, QCheckBox] = {}
        symbols_row = QHBoxLayout()
        symbols_row.setSpacing(12)
        layout.addLayout(symbols_row)

        self.symbol_selector = QComboBox()
        layout.addWidget(QLabel("Símbolo preferido"))
        layout.addWidget(self.symbol_selector)
        self.symbol_selector.currentTextChanged.connect(self.engine.set_focus_symbol)

        timeframe_layout = QHBoxLayout()
        timeframe_layout.addWidget(QLabel("Timeframe por defecto"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["M1", "M5", "M15", "M30", "H1"])
        timeframe_layout.addWidget(self.timeframe_combo)
        layout.addLayout(timeframe_layout)
        self.timeframe_combo.currentTextChanged.connect(self._on_timeframe_changed)

        form = QFormLayout()
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
        self.symbol_selector.blockSignals(True)
        self.symbol_selector.clear()
        if self.engine.symbols:
            self.symbol_selector.addItems(self.engine.symbols)
            self.symbol_selector.setCurrentIndex(0)
        self.symbol_selector.blockSignals(False)

        if hasattr(self, "symbol_checks"):
            for checkbox in self.symbol_checks.values():
                checkbox.setParent(None)
            self.symbol_checks.clear()
        if hasattr(self, "tabs"):
            config_tab = self.tabs.widget(3)
            symbol_layout_item = config_tab.layout().itemAt(1)
            symbols_layout = symbol_layout_item.layout() if symbol_layout_item else None
            if isinstance(symbols_layout, QHBoxLayout):
                while symbols_layout.count():
                    item = symbols_layout.takeAt(0)
                    widget = item.widget()
                    if widget is not None:
                        widget.deleteLater()
                for symbol in self.engine.symbols:
                    checkbox = QCheckBox(symbol)
                    checkbox.setChecked(True)
                    checkbox.stateChanged.connect(self._update_symbol_selection)
                    symbols_layout.addWidget(checkbox)
                    self.symbol_checks[symbol] = checkbox
                symbols_layout.addStretch(1)

        self.amount_spin.blockSignals(True)
        self.amount_spin.setValue(self.engine.lot_high)
        self.amount_spin.blockSignals(False)

        self.lot_high_spin.blockSignals(True)
        self.lot_high_spin.setValue(self.engine.lot_high)
        self.lot_high_spin.blockSignals(False)
        self.lot_low_spin.blockSignals(True)
        self.lot_low_spin.setValue(self.engine.lot_low)
        self.lot_low_spin.blockSignals(False)

        self.base_confidence_spin.blockSignals(True)
        self.base_confidence_spin.setValue(self.engine.base_confidence)
        self.base_confidence_spin.blockSignals(False)
        self.lower_confidence_spin.blockSignals(True)
        self.lower_confidence_spin.setValue(self.engine.lower_confidence)
        self.lower_confidence_spin.blockSignals(False)
        self.memory_boost_spin.blockSignals(True)
        self.memory_boost_spin.setValue(self.engine.memory_boost)
        self.memory_boost_spin.blockSignals(False)

        self.sl_tp_mode_combo.blockSignals(True)
        self.sl_tp_mode_combo.setCurrentText(self.engine.sl_tp_mode)
        self.sl_tp_mode_combo.blockSignals(False)
        self.sl_spin.blockSignals(True)
        self.sl_spin.setValue(self.engine.sl_value)
        self.sl_spin.blockSignals(False)
        self.tp_spin.blockSignals(True)
        self.tp_spin.setValue(self.engine.tp_value)
        self.tp_spin.blockSignals(False)
        self.apply_pending_check.blockSignals(True)
        self.apply_pending_check.setChecked(self.engine.apply_sl_tp_on_pending)
        self.apply_pending_check.blockSignals(False)

        self.timeframe_combo.blockSignals(True)
        index = self.timeframe_combo.findText(self.engine.config.get("timeframe", "M1"))
        if index >= 0:
            self.timeframe_combo.setCurrentIndex(index)
        self.timeframe_combo.blockSignals(False)

        self._update_symbol_selection()

    def _sync_sl_tp_labels(self) -> None:
        mode = self.sl_tp_mode_combo.currentText()
        if mode == "Fixed pips":
            self.sl_label.setText("SL (pips)")
            self.tp_label.setText("TP (pips)")
        else:
            self.sl_label.setText("SL (xATR)")
            self.tp_label.setText("TP (xATR)")
        self.mode_label.setText(f"Modo SL/TP: {mode}")

    def _update_ai_label(self) -> None:
        enabled = self.engine.strategy_flags.get("memory", True)
        self.ai_label.setText(f"IA Memoria: {'Activa' if enabled else 'Inactiva'}")

    def _update_symbol_selection(self) -> None:
        selected = [symbol for symbol, checkbox in self.symbol_checks.items() if checkbox.isChecked()]
        self.engine.update_selected_symbols(selected)

    def _toggle_strategy(self, name: str, state: int) -> None:
        enabled = state == Qt.Checked
        self.engine.update_strategy(name, enabled)
        if name == "memory":
            self._update_ai_label()

    def _update_risk_settings(self) -> None:
        self.engine.update_risk(self.lot_high_spin.value(), self.lot_low_spin.value())
        self.amount_spin.blockSignals(True)
        self.amount_spin.setValue(self.engine.lot_high)
        self.amount_spin.blockSignals(False)

    def _update_confidence_settings(self) -> None:
        self.engine.update_confidence_levels(
            self.base_confidence_spin.value(),
            self.lower_confidence_spin.value(),
            self.memory_boost_spin.value(),
        )

    def _update_operation_amount(self) -> None:
        self.engine.update_risk(self.amount_spin.value(), self.engine.lot_low)
        self.lot_high_spin.blockSignals(True)
        self.lot_high_spin.setValue(self.engine.lot_high)
        self.lot_high_spin.blockSignals(False)

    def _on_sl_tp_mode_changed(self, mode: str) -> None:
        self.engine.set_sl_tp_mode(mode)
        self._sync_sl_tp_labels()

    def _on_sl_tp_values_changed(self) -> None:
        self.engine.update_sl_tp_values(self.sl_spin.value(), self.tp_spin.value())

    def _on_apply_pending_changed(self, state: int) -> None:
        self.engine.apply_sl_tp_to_pending(state == Qt.Checked)

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
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

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
        self.status_label.setText(f"Estado: {status}")

    def _update_confidence(self, confidence: float, direction: str) -> None:
        percent = int(max(0.0, min(confidence, 1.0)) * 100)
        self.confidence_bar.setValue(percent)
        self.confidence_label.setText(f"Confianza: {percent}% (Dirección: {direction})")

    def _append_order(self, event: Dict[str, Any]) -> None:
        row = self.trade_table.rowCount()
        self.trade_table.insertRow(row)
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
            item.setTextAlignment(Qt.AlignCenter)
            self.trade_table.setItem(row, column, item)
        self.trade_table.scrollToBottom()

    def _update_stats_labels(self, stats: Dict[str, Any]) -> None:
        trades = int(stats.get("trades", 0))
        wins = int(stats.get("wins", 0))
        losses = int(stats.get("losses", 0))
        pnl = float(stats.get("pnl", 0.0))
        precision = (wins / trades * 100) if trades else 0.0
        self.trades_label.setText(f"Operaciones: {trades}")
        self.wins_label.setText(f"Ganadas: {wins}")
        self.losses_label.setText(f"Perdidas: {losses}")
        self.precision_label.setText(f"Precisión: {precision:.1f}%")
        self.pnl_label.setText(f"PnL: {pnl:.2f}")
        self.summary_trades.setText(f"Operaciones del día: {trades}")
        self.summary_wins.setText(f"Ganadas: {wins}")
        self.summary_losses.setText(f"Perdidas: {losses}")
        self.summary_precision.setText(f"Precisión: {precision:.1f}%")
        self.summary_pnl.setText(f"PnL acumulado: {pnl:.2f}")

    def _on_timeframe_changed(self, timeframe: str) -> None:
        self.engine.config["timeframe"] = timeframe
        self.engine.timeframe = self.engine._resolve_timeframe(timeframe)

    def refresh_memory_view(self) -> None:
        patterns = self.engine.get_memory_snapshot()
        self.memory_view.setPlainText(json.dumps(patterns, indent=2, ensure_ascii=False))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.engine.stop()
        super().closeEvent(event)
