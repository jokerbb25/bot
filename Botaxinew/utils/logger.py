import threading
from pathlib import Path
from typing import Callable, Optional


class Logger:
    def __init__(self, gui_log: Optional[Callable[[str], None]] = None, log_to_file: bool = True, log_dir: Optional[Path] = None):
        self._gui_log = gui_log
        self._lock = threading.Lock()
        self._log_to_file = log_to_file
        self._file_handle = None
        if self._log_to_file:
            log_directory = log_dir or Path("logs")
            log_directory.mkdir(parents=True, exist_ok=True)
            self._file_path = log_directory / "axinew.log"
            self._file_handle = self._file_path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        with self._lock:
            formatted_message = message.strip()
            print(formatted_message)
            if self._gui_log:
                self._gui_log(formatted_message)
            if self._file_handle:
                self._file_handle.write(formatted_message + "\n")
                self._file_handle.flush()

    def close(self) -> None:
        with self._lock:
            if self._file_handle and not self._file_handle.closed:
                self._file_handle.close()

    def __del__(self):
        self.close()
