"""Iterative EM core for the Galaxy model — pure PyTorch, GPU-native.

Each EM iteration:
  1. Exclude outliers by GOF score threshold
  2. Fine-tune autoencoder with reconstruction + gravity loss
  3. Update prototypes via GPU-native Student-t Mixture Model

Gravity loss computed in log-space via logsumexp to avoid
underflow/overflow from high-dimensional determinant products.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Callable

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

if TYPE_CHECKING:
    from unicad_torch.callbacks import CallbackManager, FitContext

logger = logging.getLogger("unicad_torch.em")


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
        # aggregate_force_{scalar,vector} return -log(force) = -J (per sample)
        # Gravity loss = Σ(-J) so total = recon + gravity = g + (-J) = g - J
        # Minimizing g - J ⟹ minimize g AND maximize J
        if config.gravity_version == "scalar":

            def _gravity_scalar(
                lf: torch.Tensor,
                means: torch.Tensor,
                embed: torch.Tensor,
            ) -> torch.Tensor:
                return aggregate_force_scalar(lf).sum()

            self._compute_gravity_loss: Callable[
                [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
            ] = _gravity_scalar
        elif config.gravity_version == "vector":

            def _gravity_vector(
                lf: torch.Tensor,
                means: torch.Tensor,
                embed: torch.Tensor,
            ) -> torch.Tensor:
                return aggregate_force_vector(lf, means, embed).sum()

            self._compute_gravity_loss = _gravity_vector
        else:
            raise ValueError(f"Unknown gravity_version: {config.gravity_version}")

        # Fitted attributes (sklearn trailing-_ convention)
        self.means_: torch.Tensor | None = None
        self.weights_: torch.Tensor | None = None
        self.covars_: torch.Tensor | None = None
        self.n_excluded_per_iter: list[int] = []

    def _check_prototypes(self) -> None:
        """Raise RuntimeError if prototypes are not initialized."""
        if self.means_ is None:
            raise RuntimeError("Prototypes not initialized — call fit() first")
        if self.covars_ is None:
            raise RuntimeError("Prototypes not initialized — call fit() first")
        if self.weights_ is None:
            raise RuntimeError("Prototypes not initialized — call fit() first")

    def load_state(
        self,
        means: torch.Tensor,
        weights: torch.Tensor,
        covars: torch.Tensor,
        n_excluded_per_iter: list[int] | None = None,
    ) -> None:
        """Restore fitted prototype state (used by Galaxy.load)."""
        self.means_ = means
        self.weights_ = weights
        self.covars_ = covars
        self.n_excluded_per_iter = n_excluded_per_iter or []

    def score(self, X_tensor: torch.Tensor) -> torch.Tensor:
        """Score data using current prototypes and model encoder."""
        self._check_prototypes()
        assert self.means_ is not None
        assert self.covars_ is not None
        assert self.weights_ is not None
        self.model.eval()
        with torch.no_grad():
            Z = self.model.encoder(X_tensor)
        return gof_score(
            Z, self.means_, self.covars_, self.weights_, self.config.score_type
        )

    def fit(
        self,
        X: torch.Tensor,
        callbacks: CallbackManager | None = None,
        ctx: FitContext | None = None,
    ) -> GalaxyEM:
        # Initial encode of full X
        with torch.no_grad():
            Z = self.model.encoder(X)
        self._update_prototypes(X, Z)

        n_excluded_per_iter: list[int] = []
        for _iter in range(self.config.em_iters):
            # --- callback: EM iter begin ---
            if ctx is not None:
                ctx.stage = "em"
                ctx.em_iter = _iter
                ctx.total_em_iters = self.config.em_iters
            if callbacks is not None and ctx is not None:
                callbacks.fire("on_em_iter_begin", ctx)

            X_filtered, _, n_excluded = self._exclude_outlier_set(X, Z)
            n_excluded_per_iter.append(n_excluded)
            self._update_network(X_filtered, callbacks=callbacks, ctx=ctx)
            # Re-encode with updated encoder (encoder changed during _update_network)
            self._update_prototypes(X_filtered, Z=None)

            # Re-encode full X for next iteration's outlier exclusion
            # (encoder has changed during _update_network)
            with torch.no_grad():
                Z = self.model.encoder(X)

            # Compute per-iter metrics for callbacks and logging
            score_min: float | None = None
            score_max: float | None = None
            score_mean: float | None = None
            if (
                self.means_ is not None
                and self.covars_ is not None
                and self.weights_ is not None
            ):
                score = gof_score(
                    Z, self.means_, self.covars_, self.weights_, self.config.score_type
                )
                score_min = score.min().item()
                score_max = score.max().item()
                score_mean = score.mean().item()

            # --- callback: EM iter end ---
            if ctx is not None:
                ctx.n_excluded = n_excluded
                ctx.score_min = score_min
                ctx.score_max = score_max
                ctx.score_mean = score_mean
            if callbacks is not None and ctx is not None:
                callbacks.fire("on_em_iter_end", ctx)

            logger.info(
                "[EM] iter %d/%d  excluded=%d  score=[%.2f, %.2f]  mean=%.2f",
                _iter + 1,
                self.config.em_iters,
                n_excluded,
                score_min or 0.0,
                score_max or 0.0,
                score_mean or 0.0,
            )

            if ctx is not None and ctx.stop_training:
                logger.info("EM stopped early at iter %d", _iter + 1)
                break

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
        self._check_prototypes()
        assert self.means_ is not None
        assert self.covars_ is not None
        assert self.weights_ is not None
        score = gof_score(
            Z, self.means_, self.covars_, self.weights_, self.config.score_type
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

    def _update_prototypes(
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

        self.means_ = smm.means_.detach().to(torch.float32)
        self.weights_ = smm.weights_.detach().to(torch.float32)
        self.covars_ = smm.covars_.detach().to(torch.float32)

        return Z

    def _update_network(
        self,
        X: torch.Tensor,
        callbacks: CallbackManager | None = None,
        ctx: FitContext | None = None,
    ) -> None:
        """Fine-tune autoencoder with reconstruction + gravity loss (log-space)."""
        self._check_prototypes()
        assert self.means_ is not None
        assert self.covars_ is not None
        assert self.weights_ is not None

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.em_finetune_lr
        )

        for step in range(self.config.em_finetune_steps):
            self.model.train()
            embed, x_hat = self.model(X)

            recon_loss = F.mse_loss(x_hat, X, reduction="sum")

            log_forces = compute_log_forces(
                embed, self.means_, self.covars_, self.weights_
            )

            gravity_loss = self._compute_gravity_loss(log_forces, self.means_, embed)

            loss = recon_loss + gravity_loss
            if torch.isnan(loss) or torch.isinf(loss):
                warnings.warn("NaN/Inf loss in _update_network — stopping early")
                break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # --- callback: finetune step end ---
            if ctx is not None:
                ctx.stage = "finetune"
                ctx.finetune_step = step
                ctx.total_finetune_steps = self.config.em_finetune_steps
                ctx.finetune_recon_loss = recon_loss.item()
                ctx.finetune_gravity_loss = gravity_loss.item()
                ctx.finetune_total_loss = loss.item()
            if callbacks is not None and ctx is not None:
                callbacks.fire("on_finetune_step_end", ctx)

            if ctx is not None and ctx.stop_training:
                logger.info("Fine-tuning stopped early at step %d", step + 1)
                break
