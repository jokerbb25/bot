import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import numpy as np


class Aprendizaje:
    def __init__(self, save_path: str = "data/learning_store.json") -> None:
        self.save_path = save_path
        self.weights = {}
        self.rsi_bias = {}
        self.loss_streak = 0
        self.win_streak = 0
        self.cooldown_until = None
        self.total_operations = 0
        self.threshold_offsets = {}
        self.context_history = []
        self.history_limit = 500
        self.base_confidence_threshold = 0.70
        self.load_data()

    def apply_learning_feedback(
        self,
        result,
        symbol,
        rsi_value,
        ema_value,
        volatility,
        signals,
        final_action,
        confidence_value,
    ):
        reward = 1 if result == "WIN" else -1
        alpha = 0.05
        base_weight = self.weights.get(symbol, 1.0)
        updated_weight = base_weight * (1 + alpha * reward)
        self.weights[symbol] = float(np.clip(updated_weight, 0.8, 1.2))
        if reward > 0:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0
        aligned = sum(1 for value in signals if value == final_action)
        conflicts = sum(1 for value in signals if value not in {final_action, "NONE"})
        denominator = aligned + conflicts + 1e-6
        confidence = (aligned - conflicts) / denominator
        confidence *= float(np.clip(volatility / 0.0008 if volatility else 0.0, 0.5, 1.0))
        confidence = min(confidence, 0.85)
        if volatility < 0.0008 or confidence < 0.65 or conflicts > 0:
            logging.info("🚫 Skipping trade: low confidence or volatility")
            self.save_data()
            return None
        if self.loss_streak >= 2:
            self.cooldown_until = datetime.now() + timedelta(minutes=10)
            logging.info("⏸️ Cooldown activated for 10 minutes after 2 consecutive losses")
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            logging.info("⏳ Skipping trade — cooldown active")
            self.save_data()
            return None
        self.total_operations += 1
        self._record_trade_context(rsi_value, ema_value, result, confidence_value, volatility)
        self._adjust_thresholds_if_needed()
        if self.total_operations % 50 == 0:
            self.clean_rsi_biases()
        if self.win_streak >= 3 and symbol in self.rsi_bias:
            self.rsi_bias[symbol] *= 0.9
        self.save_data()
        return confidence

    def clean_rsi_biases(self) -> None:
        for symbol, bias in list(self.rsi_bias.items()):
            if abs(bias) > 8 or np.isnan(bias):
                self.rsi_bias[symbol] = 0
                logging.info(f"🧹 RSI bias reset for {symbol}")

    def save_data(self) -> None:
        data = {
            "weights": self.weights,
            "rsi_bias": self.rsi_bias,
            "total_operations": self.total_operations,
            "threshold_offsets": self.threshold_offsets,
        }
        path = Path(self.save_path)
        backup_path = path.with_name(path.stem + ".bak")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                try:
                    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
                except Exception as exc:
                    logging.debug(f"No se pudo actualizar respaldo de aprendizaje: {exc}")
            with path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
        except Exception as exc:
            logging.debug(f"No se pudo guardar aprendizaje: {exc}")

    def load_data(self) -> None:
        path = Path(self.save_path)
        backup_path = path.with_name(path.stem + ".bak")
        candidates = []
        if path.exists():
            candidates.append(path)
        if backup_path.exists():
            candidates.append(backup_path)
        if not candidates:
            logging.info("ℹ️ No previous learning data found. Starting fresh.")
            return
        for candidate in candidates:
            try:
                with candidate.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                self.weights = data.get("weights", {})
                self.rsi_bias = data.get("rsi_bias", {})
                self.total_operations = data.get("total_operations", 0)
                self.threshold_offsets = {
                    str(key): float(value)
                    for key, value in data.get("threshold_offsets", {}).items()
                }
                logging.info(
                    f"✅ Learning data loaded ({int(self.total_operations)} operations)"
                )
                return
            except Exception as exc:
                logging.debug(f"No se pudo cargar aprendizaje desde {candidate.name}: {exc}")
        logging.info("ℹ️ No previous learning data found. Starting fresh.")

    def _bucket_key(self, rsi_value: float, ema_value: float) -> str:
        rsi_bucket = int(rsi_value // 10) * 10
        ema_bucket = round(float(ema_value), 3)
        return f"rsi_{rsi_bucket}_ema_{ema_bucket}"

    def _record_trade_context(
        self,
        rsi_value: float,
        ema_value: float,
        result: str,
        confidence_value: float,
        volatility: float,
    ) -> None:
        context = {
            "rsi": float(rsi_value),
            "ema": float(ema_value),
            "result": result,
            "confidence": float(confidence_value),
            "volatility": float(volatility),
        }
        self.context_history.append(context)
        if len(self.context_history) > self.history_limit:
            self.context_history = self.context_history[-self.history_limit :]

    def _adjust_thresholds_if_needed(self) -> None:
        if self.total_operations == 0 or self.total_operations % 50 != 0:
            return
        buckets: Dict[str, Dict[str, int]] = {}
        for context in self.context_history:
            confidence_value = float(context.get("confidence", 0.0))
            if confidence_value < 0.55:
                continue
            ema_value = float(context.get("ema", 0.0))
            key = self._bucket_key(float(context.get("rsi", 0.0)), ema_value)
            bucket = buckets.setdefault(key, {"wins": 0, "total": 0})
            bucket["total"] += 1
            if str(context.get("result", "")).upper() == "WIN":
                bucket["wins"] += 1
        for key, stats in buckets.items():
            total = stats["total"]
            if total == 0:
                continue
            success_rate = stats["wins"] / total
            if success_rate > 0.65:
                current = float(self.threshold_offsets.get(key, 0.0))
                self.threshold_offsets[key] = float(np.clip(current + 0.02, 0.0, 0.2))

    def get_dynamic_threshold(self, rsi_value: float, ema_value: float) -> float:
        key = self._bucket_key(rsi_value, ema_value)
        offset = float(self.threshold_offsets.get(key, 0.0))
        return float(np.clip(self.base_confidence_threshold + offset, 0.70, 0.90))
