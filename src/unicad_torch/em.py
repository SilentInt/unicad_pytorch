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
from unicad_torch.model import Autoencoder
from unicad_torch.smm_torch import SMMTorch, _mahalanobis_diag

_VAR_FLOOR = 1e-6


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

        self.scorer = GOFScorer(device=self.device)
        self.means: torch.Tensor | None = None
        self.weights: torch.Tensor | None = None
        self.covars: torch.Tensor | None = None

    def fit(self, X: torch.Tensor) -> GalaxyEM:
        self.update_prototypes(X)

        for _iter in range(self.config.em_iters):
            X_filtered = self._exclude_outlier_set(X)
            self.update_network(X_filtered)
            self.update_prototypes(X_filtered)

        return self

    def _exclude_outlier_set(self, X: torch.Tensor) -> torch.Tensor:
        """Re-score the FULL X, then filter top outlier_ratio% out."""
        with torch.no_grad():
            Z = self.model.encoder(X)
        assert self.means is not None
        assert self.covars is not None
        score = self.scorer.get_score(
            Z, self.means, covars=self.covars, score_type=self.config.score_type
        )

        if torch.isnan(score).any() or torch.isinf(score).any():
            warnings.warn("GOF score contains NaN/Inf — skipping outlier exclusion")
            return X

        threshold = torch.quantile(score, 1.0 - self.config.outlier_ratio)
        mask = score <= threshold
        X_filtered = X[mask]

        if X_filtered.shape[0] == 0:
            return X
        return X_filtered

    def update_prototypes(self, X: torch.Tensor) -> None:
        with torch.no_grad():
            Z = self.model.encoder(X)

        if torch.isnan(Z).any() or torch.isinf(Z).any():
            warnings.warn("Encoder output contains NaN/Inf — skipping prototype update")
            return

        self.smm.fit(Z)

        self.means = self.smm.means_.detach().to(torch.float32)
        self.weights = self.smm.weights_.detach().to(torch.float32)
        self.covars = self.smm.covars_.detach().clamp(min=_VAR_FLOOR).to(torch.float32)

    def update_network(self, X: torch.Tensor) -> None:
        """Fine-tune autoencoder with reconstruction + gravity loss (log-space)."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.em_finetune_lr)

        assert self.means is not None
        assert self.weights is not None
        assert self.covars is not None

        # Pre-compute log|Σ_k| = Σ_d log(σ_kd) — stays finite in any dimension
        log_det_covars = torch.log(self.covars.clamp(min=_VAR_FLOOR)).sum(dim=1)  # (K,)

        for _step in range(self.config.em_finetune_steps):
            self.model.train()
            embed, x_hat = self.model(X)

            recon_loss = F.mse_loss(x_hat, X, reduction="sum")

            maha = _mahalanobis_diag(embed, self.means, self.covars)  # (N, K)

            # Log-space force: log F_ik = log(ω_k) - log(π) - 0.5*log|Σ_k| - log(1 + D_M²)
            log_forces = (
                torch.log(self.weights.clamp(min=1e-30)).unsqueeze(0)   # (1, K)
                - torch.log(torch.tensor(torch.pi, dtype=X.dtype, device=X.device))
                - 0.5 * log_det_covars.unsqueeze(0)                     # (1, K)
                - torch.log1p(maha)                                      # (N, K)
            )  # (N, K)

            if self.config.gravity_version == "scalar":
                # log(Σ_k F_k) = logsumexp(log F_k)
                log_total_force = torch.logsumexp(log_forces, dim=1)  # (N,)
                gravity_loss = -log_total_force.sum()

            elif self.config.gravity_version == "vector":
                # Vector force: need ||Σ_k F_k * û_k|| in log-space
                # Scale by max log-force to keep exp() bounded
                log_max, _ = log_forces.max(dim=1, keepdim=True)  # (N, 1)
                scaled = torch.exp(log_forces - log_max)  # (N, K) — in [0, 1]
                unit_vec = F.normalize(
                    self.means.unsqueeze(0) - embed.unsqueeze(1), p=2, dim=-1
                )  # (N, K, D)
                force_vec = (scaled.unsqueeze(2) * unit_vec).sum(dim=1)  # (N, D)
                force_norm = torch.norm(force_vec, dim=-1).clamp(min=1e-30)  # (N,)
                # log(||F||) = log(||scaled_F||) + max_log
                log_force_norm = torch.log(force_norm) + log_max.squeeze(1)
                gravity_loss = -log_force_norm.sum()

            else:
                raise ValueError(f"Unknown gravity_version: {self.config.gravity_version}")

            loss = recon_loss + gravity_loss
            if torch.isnan(loss) or torch.isinf(loss):
                warnings.warn("NaN/Inf loss in update_network — stopping early")
                break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
