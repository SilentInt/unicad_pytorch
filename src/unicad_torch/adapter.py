from __future__ import annotations

import numpy as np

from unicad_torch.config import GalaxyConfig
from unicad_torch.galaxy import Galaxy


class GalaxyADBench:
    """ADBench-compatible wrapper for Galaxy.

    Follows the ADBench convention: fit(X_train, y_train) where y_train
    is ignored (unsupervised), and predict_score(X) returns anomaly scores.
    """

    def __init__(self, config: GalaxyConfig | None = None) -> None:
        self.config = config or GalaxyConfig()
        self._galaxy = Galaxy(self.config)

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray | None = None
    ) -> GalaxyADBench:
        self._galaxy.fit(X_train, y_train)
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return self._galaxy.predict_score(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary anomaly labels (0=normal, 1=anomaly) via outlier_ratio threshold."""
        scores = self._galaxy.predict_score(X)
        threshold = np.quantile(scores, 1.0 - self.config.outlier_ratio)
        return (scores > threshold).astype(np.int32)
