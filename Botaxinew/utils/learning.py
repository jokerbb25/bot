from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Tuple


class LearningMemory:
    def __init__(self, memory_path: Path) -> None:
        self._path = Path(memory_path)
        self._lock = threading.Lock()
        self._data: Dict[str, List[Dict[str, object]]] = {"patterns": []}
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not self._path.exists():
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._save_unlocked()
                return
            try:
                loaded = json.loads(self._path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._save_unlocked()
                return
            if isinstance(loaded, dict) and isinstance(loaded.get("patterns"), list):
                self._data = {"patterns": list(loaded.get("patterns", []))}
            else:
                self._save_unlocked()

    def _save_unlocked(self) -> None:
        self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def _save(self) -> None:
        with self._lock:
            self._save_unlocked()

    def get_patterns(self) -> List[Dict[str, object]]:
        with self._lock:
            return list(self._data.get("patterns", []))

    def apply_memory(
        self,
        symbol: str,
        rsi_value: float,
        ema_trend: str,
        macd_signal: str,
        pullback: bool,
        confidence: float,
        boost: float,
    ) -> Tuple[float, bool]:
        with self._lock:
            for pattern in self._data.get("patterns", []):
                if pattern.get("symbol") != symbol:
                    continue
                if pattern.get("ema") != ema_trend or pattern.get("macd") != macd_signal:
                    continue
                stored_pullback = bool(pattern.get("pullback", False))
                if stored_pullback != bool(pullback):
                    continue
                stored_rsi = float(pattern.get("rsi", rsi_value))
                if abs(stored_rsi - rsi_value) > 5.0:
                    continue
                new_confidence = min(1.0, confidence + boost)
                return new_confidence, True
        return confidence, False

    def record_trade(
        self,
        symbol: str,
        rsi_value: float,
        ema_trend: str,
        macd_signal: str,
        pullback: bool,
        confidence: float,
        result: str,
    ) -> None:
        pattern = {
            "symbol": symbol,
            "rsi": round(float(rsi_value), 2),
            "ema": ema_trend,
            "macd": macd_signal,
            "pullback": bool(pullback),
            "confidence": round(float(confidence), 2),
            "result": result.upper(),
        }
        with self._lock:
            existing = [
                p
                for p in self._data.get("patterns", [])
                if p.get("symbol") == symbol
                and p.get("ema") == ema_trend
                and p.get("macd") == macd_signal
                and bool(p.get("pullback", False)) == bool(pullback)
            ]
            for item in existing:
                self._data["patterns"].remove(item)
            if result.upper() == "WIN":
                self._data.setdefault("patterns", []).append(pattern)
            self._save_unlocked()
