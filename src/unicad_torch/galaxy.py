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

import dataclasses
import logging
import warnings
from pathlib import Path

import numpy as np
import torch

from unicad_torch.callbacks import CallbackManager, FitContext
from unicad_torch.config import GalaxyConfig
from unicad_torch.em import GalaxyEM
from unicad_torch.model import Autoencoder, pretrain_autoencoder
from unicad_torch.preprocessing import RowScaler, StandardScaler

logger = logging.getLogger("unicad_torch.galaxy")

_SAVE_FORMAT_VERSION = 2


class Galaxy:
    """GPU-native Galaxy anomaly detection model."""

    def __init__(self, config: GalaxyConfig | None = None) -> None:
        self.config = config or GalaxyConfig()
        self.scaler: StandardScaler | RowScaler | None = None
        self.model: Autoencoder | None = None
        self.em: GalaxyEM | None = None
        self.input_dim: int | None = None
        self.threshold_: float | None = None
        self.effective_outlier_ratio_: float = self.config.outlier_ratio
        # Fit metadata (populated during fit, used by fit_info_ property)
        self._train_score_mean: float = float("nan")
        self._train_score_std: float = float("nan")
        self._outlier_ratio_source: str = "config"
        self._train_scores: torch.Tensor | None = None

    def __repr__(self) -> str:
        if self.model is None:
            return "Galaxy(not fitted)"
        return (
            f"Galaxy(input_dim={self.input_dim}, k={self.config.k}, "
            f"threshold_={self.threshold_:.4f})"
        )

    @property
    def fit_info_(self) -> dict[str, object]:
        """Read-only view of fit metadata assembled from authoritative sources."""
        info: dict[str, object] = {
            "train_score_mean": self._train_score_mean,
            "train_score_std": self._train_score_std,
            "threshold_": self.threshold_,
            "outlier_ratio": self.effective_outlier_ratio_,
            "outlier_ratio_source": self._outlier_ratio_source,
            "n_excluded_per_iter": (
                self.em.n_excluded_per_iter if self.em is not None else []
            ),
        }
        return info

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray | None = None,
        resume_from: str | Path | None = None,
    ) -> Galaxy:
        # Input validation
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            raise ValueError("Input contains NaN or Inf")
        if X_train.shape[0] < self.config.k:
            raise ValueError(
                f"Need at least k={self.config.k} samples, got {X_train.shape[0]}"
            )

        # Set seed for reproducibility (covers model init, DataLoader shuffle)
        torch.manual_seed(self.config.seed)

        # When labels are available, let the true anomaly rate override outlier_ratio
        self._outlier_ratio_source = "config"
        self.effective_outlier_ratio_ = self.config.outlier_ratio
        if y_train is not None and y_train.size > 0:
            label_ratio = float(y_train.mean())
            if label_ratio > 0:
                self.effective_outlier_ratio_ = label_ratio
                self._outlier_ratio_source = "labels"

        # Set up callbacks and context
        cb_list = self.config.callbacks or []
        cb_mgr = CallbackManager(cb_list)
        ctx = FitContext(config=self.config)

        # Wire ModelCheckpoint to this Galaxy instance
        from unicad_torch.callbacks import ModelCheckpoint

        for cb in cb_list:
            if isinstance(cb, ModelCheckpoint):
                cb._galaxy_ref = self  # noqa: SLF001

        # Verbose mode: ensure package logger is at INFO
        if self.config.verbose:
            logging.getLogger("unicad_torch").setLevel(logging.INFO)

        # --- callback: fit begin ---
        cb_mgr.fire("on_fit_begin", ctx)

        # Resume: restore state from checkpoint if provided
        if resume_from is not None:
            self._restore_checkpoint(resume_from)

        device = self.config.device
        if X_train.dtype != np.float32:
            warnings.warn(
                f"Input dtype is {X_train.dtype}, converting to float32 — precision may be lost",
                stacklevel=2,
            )
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
        self.input_dim = X_train.shape[1]
        if self.model is None:
            self.model = Autoencoder(
                input_dim=self.input_dim,
                hidden_dim=self.config.hidden_dim,
                latent_dim=self.config.resolved_latent_dim,
            ).to(device)

        if self.config.pretrain and resume_from is None:
            ctx.model = self.model
            logger.info(
                "[Galaxy] Pretraining autoencoder (%d epochs)...",
                self.config.pretrain_epochs,
            )
            pretrain_autoencoder(self.model, X_tensor, self.config, cb_mgr, ctx)
            if ctx.stop_training:
                logger.info("Training stopped during pretrain phase")
                self.model.eval()
                cb_mgr.fire("on_fit_end", ctx)
                return self

        # Stage 3: Iterative EM
        if self.em is None:
            self.em = GalaxyEM(
                self.model, self.config, outlier_ratio=self.effective_outlier_ratio_
            )
        ctx.model = self.model
        ctx.em = self.em

        logger.info("[Galaxy] Running EM (%d iterations)...", self.config.em_iters)
        self.em.fit(X_tensor, callbacks=cb_mgr, ctx=ctx)

        # Stage 4: Compute absolute threshold from training scores
        ctx.stage = "threshold"
        self._train_scores = self.em.score(X_tensor)
        self.threshold_ = torch.quantile(
            self._train_scores, 1.0 - self.effective_outlier_ratio_
        ).item()

        # Store fit metadata
        self._train_score_mean = float(self._train_scores.mean())
        self._train_score_std = float(self._train_scores.std())

        self.model.eval()

        logger.info(
            "[Galaxy] Fit complete. threshold_=%.4f  outlier_ratio=%.4f "
            "(from %s)  train_score_mean=%.4f",
            self.threshold_ or 0.0,
            self.effective_outlier_ratio_,
            self._outlier_ratio_source,
            self._train_score_mean,
        )

        # --- callback: fit end ---
        cb_mgr.fire("on_fit_end", ctx)

        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Return continuous anomaly scores (higher = more anomalous)."""
        X_tensor = self._prepare_input(X)

        if self.em is None:
            raise RuntimeError("Model not fitted — call fit() first")

        score = self.em.score(X_tensor)
        return score.cpu().detach().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary anomaly labels (0=normal, 1=anomaly) using training threshold."""
        if self.threshold_ is None:
            raise RuntimeError("Model not fitted — call fit() first")
        scores = self.predict_score(X)
        return (scores > self.threshold_).astype(np.int32)

    def fit_predict(
        self, X_train: np.ndarray, y_train: np.ndarray | None = None
    ) -> np.ndarray:
        """Fit the model and return anomaly scores for the training data."""
        self.fit(X_train, y_train)
        if self._train_scores is not None:
            return self._train_scores.cpu().detach().numpy()
        return self.predict_score(X_train)

    def save(self, path: str | Path) -> None:
        """Save all fitted state to disk."""
        if self.model is None or self.em is None:
            raise RuntimeError("Nothing to save — call fit() first")
        if (
            self.em.means_ is None
            or self.em.weights_ is None
            or self.em.covars_ is None
        ):
            raise RuntimeError("Incomplete fit state — nothing to save")

        # Serialize config without callbacks (they contain unpicklable closures)
        config_dict = dataclasses.asdict(self.config)
        config_dict.pop("callbacks", None)

        state: dict[str, object] = {
            "save_format_version": _SAVE_FORMAT_VERSION,
            "fit_stage": "complete",
            "config": config_dict,
            "input_dim": self.input_dim,
            "threshold_": self.threshold_,
            "effective_outlier_ratio_": self.effective_outlier_ratio_,
            "train_score_mean": self._train_score_mean,
            "train_score_std": self._train_score_std,
            "outlier_ratio_source": self._outlier_ratio_source,
            "em_n_excluded_per_iter": self.em.n_excluded_per_iter,
            "model_state": self.model.state_dict(),
            "em_means": self.em.means_.cpu(),
            "em_weights": self.em.weights_.cpu(),
            "em_covars": self.em.covars_.cpu(),
        }

        if (
            isinstance(self.scaler, StandardScaler)
            and self.scaler.mean_ is not None
            and self.scaler.std_ is not None
        ):
            state["scaler_type"] = "StandardScaler"
            state["scaler_mean"] = self.scaler.mean_.cpu()
            state["scaler_std"] = self.scaler.std_.cpu()
        elif isinstance(self.scaler, RowScaler):
            state["scaler_type"] = "RowScaler"
        else:
            state["scaler_type"] = None

        # Attach History callback data if present
        if self.config.callbacks is not None:
            from unicad_torch.callbacks import History

            for cb in self.config.callbacks:
                if isinstance(cb, History):
                    state["history"] = cb.history
                    break

        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> Galaxy:
        """Load a fitted model from disk."""
        state = torch.load(path, map_location=device, weights_only=False)

        config = GalaxyConfig(**state["config"])  # type: ignore[arg-type]
        galaxy = cls(config)
        galaxy.input_dim = state["input_dim"]  # type: ignore[assignment]
        galaxy.threshold_ = state["threshold_"]  # type: ignore[assignment]

        # Restore fit metadata — v2+ flat format
        if "train_score_mean" in state:
            galaxy.effective_outlier_ratio_ = float(state["effective_outlier_ratio_"])
            galaxy._train_score_mean = float(state["train_score_mean"])
            galaxy._train_score_std = float(state["train_score_std"])
            galaxy._outlier_ratio_source = str(
                state.get("outlier_ratio_source", "config")
            )
            n_excl = state.get("em_n_excluded_per_iter")
        # Legacy format (nested fit_info_)
        elif "fit_info_" in state:
            warnings.warn(
                "Nested fit_info_ save format is deprecated and will be "
                "removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
            fi = state["fit_info_"]
            galaxy.effective_outlier_ratio_ = float(
                fi.get("outlier_ratio", config.outlier_ratio)  # type: ignore[arg-type]
            )
            galaxy._train_score_mean = float(
                fi.get("train_score_mean", float("nan"))  # type: ignore[arg-type]
            )
            galaxy._train_score_std = float(
                fi.get("train_score_std", float("nan"))  # type: ignore[arg-type]
            )
            galaxy._outlier_ratio_source = str(
                fi.get("outlier_ratio_source", "config")  # type: ignore[arg-type]
            )
            n_excl = fi.get("n_excluded_per_iter")  # type: ignore[assignment]
        else:
            galaxy.effective_outlier_ratio_ = config.outlier_ratio
            galaxy._outlier_ratio_source = "config"
            n_excl = None

        # Rebuild scaler
        scaler_type = state.get("scaler_type")
        if scaler_type == "StandardScaler":
            scaler = StandardScaler()
            scaler.mean_ = state["scaler_mean"].to(device)  # type: ignore[assignment]
            scaler.std_ = state["scaler_std"].to(device)  # type: ignore[assignment]
            galaxy.scaler = scaler
        elif scaler_type == "RowScaler":
            galaxy.scaler = RowScaler()
        else:
            galaxy.scaler = None

        # Rebuild autoencoder
        galaxy.model = Autoencoder(
            input_dim=galaxy.input_dim,  # type: ignore[arg-type]
            hidden_dim=config.hidden_dim,
            latent_dim=config.resolved_latent_dim,
        ).to(device)
        galaxy.model.load_state_dict(state["model_state"])  # type: ignore[arg-type]
        galaxy.model.eval()

        # Rebuild EM state
        galaxy.em = GalaxyEM(
            galaxy.model, config, outlier_ratio=galaxy.effective_outlier_ratio_
        )
        galaxy.em.load_state(
            means=state["em_means"].to(device),
            weights=state["em_weights"].to(device),
            covars=state["em_covars"].to(device),
            n_excluded_per_iter=n_excl,  # type: ignore[arg-type]
        )

        # Restore History callback data if present
        if "history" in state and config.callbacks is not None:
            from unicad_torch.callbacks import History

            for cb in config.callbacks:
                if isinstance(cb, History):
                    cb.history = state["history"]  # type: ignore[assignment]
                    break

        return galaxy

    def _restore_checkpoint(self, path: str | Path) -> None:
        """Restore model and EM state from a checkpoint for resume."""
        self.load(path, device=self.config.device)

    def _prepare_input(self, X: np.ndarray) -> torch.Tensor:
        """Validate and convert input array to device tensor."""
        if self.input_dim is not None and X.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {X.shape[1]}")

        device = self.config.device
        if X.dtype != np.float32:
            warnings.warn(
                f"Input dtype is {X.dtype}, converting to float32 — precision may be lost",
                stacklevel=3,
            )
        X_tensor = torch.from_numpy(X).to(torch.float32).to(device)

        if self.scaler is not None:
            X_tensor = self.scaler.transform(X_tensor)

        return X_tensor
