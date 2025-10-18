import numpy as np
import logging
from datetime import datetime, timedelta


class Aprendizaje:
    def __init__(self) -> None:
        self.weights = {}
        self.rsi_bias = {}
        self.loss_streak = 0
        self.win_streak = 0
        self.cooldown_until = None
        self.total_operations = 0

    def apply_learning_feedback(
        self,
        result,
        symbol,
        rsi_value,
        volatility,
        signals,
        final_action,
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
            return None
        if self.loss_streak >= 2:
            self.cooldown_until = datetime.now() + timedelta(minutes=10)
            logging.info("⏸️ Cooldown activated for 10 minutes after 2 consecutive losses")
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            logging.info("⏳ Skipping trade — cooldown active")
            return None
        self.total_operations += 1
        if self.total_operations % 50 == 0:
            self.clean_rsi_biases()
        if self.win_streak >= 3 and symbol in self.rsi_bias:
            self.rsi_bias[symbol] *= 0.9
        return confidence

    def clean_rsi_biases(self) -> None:
        for symbol, bias in list(self.rsi_bias.items()):
            if abs(bias) > 8 or np.isnan(bias):
                self.rsi_bias[symbol] = 0
                logging.info(f"🧹 RSI bias reset for {symbol}")
