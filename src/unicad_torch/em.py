"""Iterative EM core for the Galaxy model — pure PyTorch, GPU-native.

Each EM iteration:
  1. Exclude outliers by GOF score threshold
  2. Fine-tune autoencoder with reconstruction + gravity loss
  3. Update prototypes via GPU-native Student-t Mixture Model

Gravity loss computed in log-space via logsumexp to avoid
underflow/overflow from high-dimensional determinant products.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from unicad_torch.config import GalaxyConfig
from unicad_torch.gof import GOFScorer
from unicad_torch.gravity import (
    VAR_FLOOR,
    aggregate_force_scalar,
    aggregate_force_vector,
    compute_log_forces,
)
from unicad_torch.model import Autoencoder
from unicad_torch.smm_torch import SMMTorch


class GalaxyEM:
    """GPU-native iterative EM core."""

    def __init__(self, model: Autoencoder, config: GalaxyConfig) -> None:
        self.model = model
        self.config = config
        self.device = config.device

        self.smm = SMMTorch(
            n_components=config.k,
            n_iter=config.smm_n_iter,
            tol=config.smm_tol,
            random_state=config.seed,
        )

        self.scorer = GOFScorer()
        self.means: torch.Tensor | None = None
        self.weights: torch.Tensor | None = None
        self.covars: torch.Tensor | None = None

    def fit(self, X: torch.Tensor) -> GalaxyEM:
        self.update_prototypes(X)

        n_excluded_per_iter: list[int] = []
        for _iter in range(self.config.em_iters):
            X_filtered, n_excluded = self._exclude_outlier_set(X)
            n_excluded_per_iter.append(n_excluded)
            self.update_network(X_filtered)
            self.update_prototypes(X_filtered)

            if self.config.verbose:
                with torch.no_grad():
                    Z = self.model.encoder(X)
                if self.means is not None and self.covars is not None:
                    score = self.scorer.get_score(
                        Z,
                        self.means,
                        covars=self.covars,
                        weights=self.weights,
                        score_type=self.config.score_type,
                    )
                    print(
                        f"  [EM iter {_iter + 1}/{self.config.em_iters}]  "
                        f"excluded={n_excluded}  "
                        f"score=[{score.min():.2f}, {score.max():.2f}]"
                    )

        self.n_excluded_per_iter = n_excluded_per_iter
        return self

    def _exclude_outlier_set(self, X: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Re-score the FULL X, then filter top outlier_ratio% out.

        Returns (filtered_X, n_excluded).
        """
        with torch.no_grad():
            Z = self.model.encoder(X)
        if self.means is None:
            raise RuntimeError(
                "Prototypes not initialized — call update_prototypes first"
            )
        if self.covars is None:
            raise RuntimeError(
                "Prototypes not initialized — call update_prototypes first"
            )
        score = self.scorer.get_score(
            Z,
            self.means,
            covars=self.covars,
            weights=self.weights,
            score_type=self.config.score_type,
        )

        if torch.isnan(score).any() or torch.isinf(score).any():
            warnings.warn("GOF score contains NaN/Inf — skipping outlier exclusion")
            return X, 0

        threshold = torch.quantile(score, 1.0 - self.config.outlier_ratio)
        mask = score <= threshold
        X_filtered = X[mask]
        n_excluded = int((~mask).sum())

        if X_filtered.shape[0] == 0:
            return X, n_excluded
        return X_filtered, n_excluded

    def update_prototypes(self, X: torch.Tensor) -> None:
        with torch.no_grad():
            Z = self.model.encoder(X)

        if torch.isnan(Z).any() or torch.isinf(Z).any():
            raise RuntimeError(
                "Encoder output contains NaN/Inf — cannot update prototypes"
            )

        self.smm.fit(Z)

        if self.smm.means_ is None:
            raise RuntimeError("SMM fit failed — means_ is None")
        if self.smm.weights_ is None:
            raise RuntimeError("SMM fit failed — weights_ is None")
        if self.smm.covars_ is None:
            raise RuntimeError("SMM fit failed — covars_ is None")

        self.means = self.smm.means_.detach().to(torch.float32)
        self.weights = self.smm.weights_.detach().to(torch.float32)
        self.covars = self.smm.covars_.detach().clamp(min=VAR_FLOOR).to(torch.float32)

    def update_network(self, X: torch.Tensor) -> None:
        """Fine-tune autoencoder with reconstruction + gravity loss (log-space)."""
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.em_finetune_lr
        )

        if self.means is None:
            raise RuntimeError(
                "Prototypes not initialized — call update_prototypes first"
            )
        if self.weights is None:
            raise RuntimeError(
                "Prototypes not initialized — call update_prototypes first"
            )
        if self.covars is None:
            raise RuntimeError(
                "Prototypes not initialized — call update_prototypes first"
            )

        for _step in range(self.config.em_finetune_steps):
            self.model.train()
            embed, x_hat = self.model(X)

            recon_loss = F.mse_loss(x_hat, X, reduction="sum")

            log_forces = compute_log_forces(embed, self.means, self.covars, self.weights)

            if self.config.gravity_version == "scalar":
                gravity_loss = -aggregate_force_scalar(log_forces).sum()

            elif self.config.gravity_version == "vector":
                gravity_loss = -aggregate_force_vector(
                    log_forces, self.means, embed
                ).sum()

            else:
                raise ValueError(
                    f"Unknown gravity_version: {self.config.gravity_version}"
                )

            loss = recon_loss + gravity_loss
            if torch.isnan(loss) or torch.isinf(loss):
                warnings.warn("NaN/Inf loss in update_network — stopping early")
                break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
