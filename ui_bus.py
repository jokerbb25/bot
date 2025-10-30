from PyQt5.QtCore import QObject, pyqtSignal


class GuiBridge(QObject):
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()


bridge = GuiBridge()
