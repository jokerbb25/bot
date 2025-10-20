# =========================================================
# MODULE: ai_regressor.py
# Light learning model for directional context
# =========================================================
from typing import Iterable, Sequence

import numpy as np
from sklearn.linear_model import SGDRegressor


class LightRegressor:
    def __init__(self) -> None:
        self.model = SGDRegressor(max_iter=1000, tol=1e-3)
        self.trained = False

    def train(self, X: Iterable[Sequence[float]], y: Iterable[float]) -> None:
        X_array = np.asarray(list(X), dtype=float)
        y_array = np.asarray(list(y), dtype=float)
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
        if y_array.ndim == 0:
            y_array = y_array.reshape(1)
        if X_array.size == 0 or y_array.size == 0:
            return
        self.model.partial_fit(X_array, y_array)
        self.trained = True

    def predict(self, features: Sequence[float]) -> float:
        if not self.trained:
            return 0.5
        features_array = np.asarray(features, dtype=float).reshape(1, -1)
        prediction = self.model.predict(features_array)[0]
        return float(np.clip(prediction, 0.0, 1.0))
