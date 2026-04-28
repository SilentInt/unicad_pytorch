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
from typing import Callable

import torch
import torch.nn.functional as F

from unicad_torch.config import GalaxyConfig
from unicad_torch.gof import gof_score
from unicad_torch.gravity import (
    aggregate_force_scalar,
    aggregate_force_vector,
    compute_log_forces,
)
from unicad_torch.model import Autoencoder
from unicad_torch.smm_torch import SMMTorch


class GalaxyEM:
    """GPU-native iterative EM core."""

    def __init__(
        self,
        model: Autoencoder,
        config: GalaxyConfig,
        outlier_ratio: float | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.outlier_ratio = (
            outlier_ratio if outlier_ratio is not None else config.outlier_ratio
        )

        # Resolve gravity aggregation at construction time
        if config.gravity_version == "scalar":

            def _gravity_scalar(
                lf: torch.Tensor,
                means: torch.Tensor,
                embed: torch.Tensor,
            ) -> torch.Tensor:
                return -aggregate_force_scalar(lf).sum()

            self._compute_gravity_loss: Callable[
                [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
            ] = _gravity_scalar
        elif config.gravity_version == "vector":

            def _gravity_vector(
                lf: torch.Tensor,
                means: torch.Tensor,
                embed: torch.Tensor,
            ) -> torch.Tensor:
                return -aggregate_force_vector(lf, means, embed).sum()

            self._compute_gravity_loss = _gravity_vector
        else:
            raise ValueError(f"Unknown gravity_version: {config.gravity_version}")

        self.means: torch.Tensor | None = None
        self.weights: torch.Tensor | None = None
        self.covars: torch.Tensor | None = None
        self.n_excluded_per_iter: list[int] = []

    def load_state(
        self,
        means: torch.Tensor,
        weights: torch.Tensor,
        covars: torch.Tensor,
        n_excluded_per_iter: list[int] | None = None,
    ) -> None:
        """Restore fitted prototype state (used by Galaxy.load)."""
        self.means = means
        self.weights = weights
        self.covars = covars
        self.n_excluded_per_iter = n_excluded_per_iter or []

    def score(self, X_tensor: torch.Tensor) -> torch.Tensor:
        """Score data using current prototypes and model encoder."""
        if self.means is None or self.covars is None or self.weights is None:
            raise RuntimeError("Prototypes not initialized — call fit() first")
        self.model.eval()
        with torch.no_grad():
            Z = self.model.encoder(X_tensor)
        return gof_score(
            Z, self.means, self.covars, self.weights, self.config.score_type
        )

    def fit(self, X: torch.Tensor) -> GalaxyEM:
        # Initial encode of full X
        with torch.no_grad():
            Z = self.model.encoder(X)
        self.update_prototypes(X, Z)

        n_excluded_per_iter: list[int] = []
        for _iter in range(self.config.em_iters):
            X_filtered, Z_filtered, n_excluded = self._exclude_outlier_set(X, Z)
            n_excluded_per_iter.append(n_excluded)
            self.update_network(X_filtered)
            self.update_prototypes(X_filtered, Z_filtered)

            # Re-encode full X for next iteration's outlier exclusion
            # (encoder has changed during update_network)
            with torch.no_grad():
                Z = self.model.encoder(X)

            if self.config.verbose:
                if (
                    self.means is not None
                    and self.covars is not None
                    and self.weights is not None
                ):
                    score = gof_score(
                        Z,
                        self.means,
                        self.covars,
                        self.weights,
                        self.config.score_type,
                    )
                    print(
                        f"  [EM iter {_iter + 1}/{self.config.em_iters}]  "
                        f"excluded={n_excluded}  "
                        f"score=[{score.min():.2f}, {score.max():.2f}]"
                    )

        self.n_excluded_per_iter = n_excluded_per_iter
        return self

    def _exclude_outlier_set(
        self, X: torch.Tensor, Z: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Re-score the FULL X, then filter top outlier_ratio% out.

        Args:
            X: (N, D) raw input tensor.
            Z: (N, D_latent) pre-computed embeddings, if available.

        Returns (filtered_X, filtered_Z, n_excluded).
        """
        if Z is None:
            with torch.no_grad():
                Z = self.model.encoder(X)
        assert Z is not None
        if self.means is None:
            raise RuntimeError(
                "Prototypes not initialized — call update_prototypes first"
            )
        if self.covars is None:
            raise RuntimeError(
                "Prototypes not initialized — call update_prototypes first"
            )
        if self.weights is None:
            raise RuntimeError(
                "Prototypes not initialized — call update_prototypes first"
            )
        score = gof_score(
            Z, self.means, self.covars, self.weights, self.config.score_type
        )

        if torch.isnan(score).any() or torch.isinf(score).any():
            warnings.warn("GOF score contains NaN/Inf — skipping outlier exclusion")
            return X, Z, 0

        threshold = torch.quantile(score, 1.0 - self.outlier_ratio)
        mask = score <= threshold
        X_filtered = X[mask]
        Z_filtered = Z[mask]
        n_excluded = int((~mask).sum())

        if X_filtered.shape[0] == 0:
            return X, Z, n_excluded
        return X_filtered, Z_filtered, n_excluded

    def update_prototypes(
        self, X: torch.Tensor, Z: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Fit SMM on encoded X. Returns Z (encoded embeddings).

        If Z is provided (pre-computed embeddings for X), skips encoding.
        """
        if Z is None:
            with torch.no_grad():
                Z = self.model.encoder(X)
        assert Z is not None

        if torch.isnan(Z).any() or torch.isinf(Z).any():
            raise RuntimeError(
                "Encoder output contains NaN/Inf — cannot update prototypes"
            )

        smm = SMMTorch(
            n_components=self.config.k,
            n_iter=self.config.smm_n_iter,
            tol=self.config.smm_tol,
            random_state=self.config.seed,
        )
        smm.fit(Z)

        if smm.means_ is None:
            raise RuntimeError("SMM fit failed — means_ is None")
        if smm.weights_ is None:
            raise RuntimeError("SMM fit failed — weights_ is None")
        if smm.covars_ is None:
            raise RuntimeError("SMM fit failed — covars_ is None")

        self.means = smm.means_.detach().to(torch.float32)
        self.weights = smm.weights_.detach().to(torch.float32)
        self.covars = smm.covars_.detach().to(torch.float32)

        return Z

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

            log_forces = compute_log_forces(
                embed, self.means, self.covars, self.weights
            )

            gravity_loss = self._compute_gravity_loss(log_forces, self.means, embed)

            loss = recon_loss + gravity_loss
            if torch.isnan(loss) or torch.isinf(loss):
                warnings.warn("NaN/Inf loss in update_network — stopping early")
                break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
