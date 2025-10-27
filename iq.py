import time
from threading import Lock, Thread
from typing import Iterable, List, Optional, Tuple

SYMBOLS: Tuple[str, ...] = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "EURJPY",
    "GBPJPY",
    "USDCHF",
    "EURGBP",
)

analysis_lock = Lock()


class Worker:
    """Simple worker that streams symbol analyses without duplication."""

    def __init__(self, interval: int = 60, symbols: Optional[Iterable[str]] = None) -> None:
        self.interval = interval
        self.symbols: Tuple[str, ...] = tuple(symbols) if symbols else SYMBOLS
        self._last_analysis = 0.0
        self._messages: List[str] = []
        self.running = False
        self.thread: Optional[Thread] = None

    def analyze_all_symbols(self) -> None:
        now_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if not hasattr(self, "_last_analysis") or (time.time() - self._last_analysis) >= 5:
            print(f"[{now_text}] Analysis started.")
            self.log_signal(f"[{now_text}] Analysis started.")
            self._last_analysis = time.time()
        for symbol in self.symbols:
            self.analyze_symbol(symbol, now_text)

    def analyze_symbol(self, symbol: str, timestamp: str) -> None:
        signal, confidence = self.get_signal(symbol)
        message = f"{timestamp} | {symbol} | {signal} | Conf: {confidence:.2f}"
        print(message)
        self.log_signal(message)

    def get_signal(self, symbol: str) -> Tuple[str, float]:
        checksum = sum(ord(char) for char in symbol)
        direction = "CALL" if checksum % 2 else "PUT"
        confidence = 0.45 + (checksum % 20) / 100
        confidence = max(0.0, min(confidence, 1.0))
        return direction, confidence

    def log_signal(self, message: str) -> None:
        self._messages.append(message.replace("\n", ""))

    def analyze_loop(self) -> None:
        while self.running:
            if analysis_lock.locked():
                time.sleep(1)
                continue
            with analysis_lock:
                self.analyze_all_symbols()
            time.sleep(self.interval)

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = Thread(target=self.analyze_loop, name="AnalysisThread", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None


def run_once(interval: int = 60) -> None:
    worker = Worker(interval=interval)
    try:
        worker.start()
        time.sleep(interval * 2)
    finally:
        worker.stop()


if __name__ == "__main__":
    run_once(interval=5)
