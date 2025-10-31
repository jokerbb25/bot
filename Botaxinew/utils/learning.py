from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List


class LearningMemory:
    def __init__(self, memory_path: Path):
        self._path = Path(memory_path)
        self._lock = threading.Lock()
        self._data: Dict[str, List[Dict[str, str]]] = {"patterns": []}
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not self._path.exists():
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
            else:
                try:
                    loaded = json.loads(self._path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict) and "patterns" in loaded:
                        self._data = {"patterns": list(loaded.get("patterns", []))}
                except json.JSONDecodeError:
                    self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def _save(self) -> None:
        with self._lock:
            self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def match_pattern(self, symbol: str, rsi_signal: str, ema_trend: str, macd_signal: str) -> bool:
        with self._lock:
            for pattern in self._data.get("patterns", []):
                if (
                    pattern.get("symbol") == symbol
                    and pattern.get("rsi_signal") == rsi_signal
                    and pattern.get("ema_trend") == ema_trend
                    and pattern.get("macd_signal") == macd_signal
                    and pattern.get("result") == "WIN"
                ):
                    return True
        return False

    def record_result(
        self,
        symbol: str,
        rsi_signal: str,
        ema_trend: str,
        macd_signal: str,
        result: str,
    ) -> None:
        pattern = {
            "symbol": symbol,
            "rsi_signal": rsi_signal,
            "ema_trend": ema_trend,
            "macd_signal": macd_signal,
            "result": result,
        }
        with self._lock:
            existing = [
                p
                for p in self._data.get("patterns", [])
                if p.get("symbol") == symbol
                and p.get("rsi_signal") == rsi_signal
                and p.get("ema_trend") == ema_trend
                and p.get("macd_signal") == macd_signal
            ]
            for item in existing:
                self._data["patterns"].remove(item)
            if result == "WIN":
                self._data.setdefault("patterns", []).append(pattern)
            self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
