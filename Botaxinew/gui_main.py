from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QHBoxLayout,
)

from engine import BotEngine


class BotAxiGUI(QWidget):
    log_received = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None, gui_log: Optional[Callable[[str], None]] = None):
        super().__init__(parent)
        self.setWindowTitle("Bot AxiNew")
        self.resize(720, 480)

        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignLeft)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addLayout(button_layout)
        layout.addWidget(self.log_view)
        self.setLayout(layout)

        self.log_received.connect(self._append_log)

        config_path = Path(__file__).resolve().parent / "config.yaml"
        self.engine = BotEngine(gui_log=self.gui_log, config_path=config_path)

        self.start_button.clicked.connect(self._handle_start)
        self.stop_button.clicked.connect(self._handle_stop)

        if gui_log:
            self.log_received.connect(lambda message: gui_log(message))

    def _handle_start(self) -> None:
        if not self.engine.start():
            self._update_status("Running")
            return
        self._update_status("Running")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def _handle_stop(self) -> None:
        self.engine.stop()
        self._update_status("Stopped")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def gui_log(self, message: str) -> None:
        self.log_received.emit(message)

    def _append_log(self, message: str) -> None:
        self.log_view.append(message)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _update_status(self, status: str) -> None:
        self.status_label.setText(f"Status: {status}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.engine.stop()
        super().closeEvent(event)
