"""Galaxy anomaly detection model — GPU-native pipeline.

Four-stage pipeline:
  1. Preprocessing (z-score or row-norm)
  2. Autoencoder pretraining (200 epochs, MSE(sum) loss)
  3. Iterative EM (exclude outliers -> update network -> update prototypes)
  4. GOF scoring (scalar or vector)

All computation stays on the configured torch device.
Numpy conversion only at the public API boundary (fit/predict_score).
"""
from __future__ import annotations

import numpy as np
import torch

from unicad_torch.config import GalaxyConfig
from unicad_torch.em import GalaxyEM
from unicad_torch.gof import GOFScorer
from unicad_torch.model import Autoencoder, pretrain_autoencoder
from unicad_torch.preprocessing import RowScaler, StandardScaler


class Galaxy:
    """GPU-native Galaxy anomaly detection model."""

    def __init__(self, config: GalaxyConfig | None = None) -> None:
        self.config = config or GalaxyConfig()
        self.scaler: StandardScaler | RowScaler | None = None
        self.model: Autoencoder | None = None
        self.em: GalaxyEM | None = None
        self.scorer = GOFScorer(device=self.config.device)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray | None = None) -> Galaxy:
        _ = y_train  # unsupervised -- labels ignored

        device = self.config.device
        X_tensor = torch.from_numpy(X_train).to(torch.float32).to(device)

        # Stage 1: Preprocessing (on device)
        if self.config.preprocess == "z-score":
            self.scaler = StandardScaler()
        elif self.config.preprocess == "row-norm":
            self.scaler = RowScaler()
        else:
            self.scaler = None

        if self.scaler is not None:
            X_tensor = self.scaler.fit_transform(X_tensor)

        # Stage 2: Autoencoder pretraining
        input_dim = X_train.shape[1]
        self.model = Autoencoder(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.hidden_dim,
        ).to(device)
        if self.config.pretrain:
            pretrain_autoencoder(self.model, X_tensor, self.config)

        # Stage 3: Iterative EM
        self.em = GalaxyEM(self.model, self.config)
        self.em.fit(X_tensor)

        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        device = self.config.device
        X_tensor = torch.from_numpy(X).to(torch.float32).to(device)

        if self.scaler is not None:
            X_tensor = self.scaler.transform(X_tensor)

        assert self.model is not None
        assert self.em is not None
        assert self.em.means is not None

        with torch.no_grad():
            Z = self.model.encoder(X_tensor)

        score = self.scorer.get_score(
            Z,
            self.em.means,
            covars=self.em.covars,
            weights=self.em.weights,
            score_type=self.config.score_type,
        )
        return score.cpu().detach().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_score(X)
