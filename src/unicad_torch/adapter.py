from __future__ import annotations

from pathlib import Path

import numpy as np

from unicad_torch.config import GalaxyConfig
from unicad_torch.galaxy import Galaxy


class GalaxyADBench:
    """ADBench-compatible wrapper for Galaxy.

    In ADBench mode, y_train is NOT forwarded to Galaxy.fit() — this ensures
    purely unsupervised evaluation as required by the ADBench benchmark
    protocol. Use Galaxy directly for label-aware training.
    """

    def __init__(self, config: GalaxyConfig | None = None) -> None:
        self.config = config or GalaxyConfig()
        self._galaxy = Galaxy(self.config)

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray | None = None
    ) -> GalaxyADBench:
        self._galaxy.fit(X_train, y_train=None)
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return self._galaxy.predict_score(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary anomaly labels (0=normal, 1=anomaly)."""
        return self._galaxy.predict(X)

    def save(self, path: str | Path) -> None:
        """Save fitted model to disk."""
        self._galaxy.save(path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> GalaxyADBench:
        """Load a fitted model from disk."""
        galaxy = Galaxy.load(path, device=device)
        adapter = cls(galaxy.config)
        adapter._galaxy = galaxy
        return adapter

    @property
    def threshold_(self) -> float | None:
        return self._galaxy.threshold_

    @property
    def fit_info_(self) -> dict[str, object]:
        return self._galaxy.fit_info_
