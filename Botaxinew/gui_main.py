from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

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
    QListWidget,
)

from engine import BotEngine


class BotAxiGUI(QWidget):
    log_received = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    confidence_changed = pyqtSignal(float, str)
    order_added = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None, gui_log: Optional[Callable[[str], None]] = None):
        super().__init__(parent)
        self.setWindowTitle("Bot AxiNew")
        self.resize(840, 600)

        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignLeft)

        self.symbol_label = QLabel("Symbol:")
        self.symbol_selector = QComboBox()

        symbol_layout = QHBoxLayout()
        symbol_layout.addWidget(self.symbol_label)
        symbol_layout.addWidget(self.symbol_selector)
        symbol_layout.addStretch(1)

        self.confidence_label = QLabel("Confidence: 0% (Direction: NONE)")
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)

        confidence_layout = QVBoxLayout()
        confidence_layout.addWidget(self.confidence_label)
        confidence_layout.addWidget(self.confidence_bar)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch(1)

        self.last_orders_label = QLabel("Last orders")
        self.order_list = QListWidget()

        layout = QVBoxLayout()
        layout.addLayout(symbol_layout)
        layout.addWidget(self.status_label)
        layout.addLayout(confidence_layout)
        layout.addLayout(button_layout)
        layout.addWidget(QLabel("Logs"))
        layout.addWidget(self.log_view)
        layout.addWidget(self.last_orders_label)
        layout.addWidget(self.order_list)
        self.setLayout(layout)

        self.log_received.connect(self._append_log)
        self.status_changed.connect(self._update_status)
        self.confidence_changed.connect(self._update_confidence)
        self.order_added.connect(self._append_order)

        config_path = Path(__file__).resolve().parent / "config.yaml"
        memory_path = Path(__file__).resolve().parent / "memory.json"
        self.engine = BotEngine(
            gui_log=self.gui_log,
            config_path=config_path,
            memory_path=memory_path,
            status_callback=self._status_from_engine,
            confidence_callback=self._confidence_from_engine,
            order_callback=self._order_from_engine,
        )

        self._recent_orders: List[str] = []

        if self.engine.symbols:
            self.symbol_selector.addItems(self.engine.symbols)
            if self.engine.active_symbol:
                index = self.symbol_selector.findText(self.engine.active_symbol)
                if index >= 0:
                    self.symbol_selector.setCurrentIndex(index)
        self.symbol_selector.currentTextChanged.connect(self.engine.set_active_symbol)

        self.start_button.clicked.connect(self._handle_start)
        self.stop_button.clicked.connect(self._handle_stop)

        if gui_log:
            self.log_received.connect(lambda message: gui_log(message))

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

    def _order_from_engine(self, message: str) -> None:
        self.order_added.emit(message)

    def _update_status(self, status: str) -> None:
        self.status_label.setText(f"Status: {status}")

    def _update_confidence(self, confidence: float, direction: str) -> None:
        percent = int(max(0.0, min(confidence, 1.0)) * 100)
        self.confidence_bar.setValue(percent)
        self.confidence_label.setText(f"Confidence: {percent}% (Direction: {direction})")

    def _append_order(self, message: str) -> None:
        self._recent_orders.append(message)
        if len(self._recent_orders) > 5:
            self._recent_orders = self._recent_orders[-5:]
        self.order_list.clear()
        for entry in self._recent_orders:
            self.order_list.addItem(entry)
        self.order_list.scrollToBottom()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.engine.stop()
        super().closeEvent(event)
