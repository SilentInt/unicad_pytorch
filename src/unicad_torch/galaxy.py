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
import warnings
from pathlib import Path

import numpy as np
import torch

from unicad_torch.config import GalaxyConfig
from unicad_torch.em import GalaxyEM
from unicad_torch.model import Autoencoder, pretrain_autoencoder
from unicad_torch.preprocessing import RowScaler, StandardScaler


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

    def fit(self, X_train: np.ndarray, y_train: np.ndarray | None = None) -> Galaxy:
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
        latent_dim = self.config.latent_dim or self.config.hidden_dim
        self.model = Autoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.config.hidden_dim,
            latent_dim=latent_dim,
        ).to(device)
        if self.config.pretrain:
            if self.config.verbose:
                print(
                    f"[Galaxy] Pretraining autoencoder ({self.config.pretrain_epochs} epochs)..."
                )
            pretrain_autoencoder(self.model, X_tensor, self.config)

        # Stage 3: Iterative EM
        if self.config.verbose:
            print(f"[Galaxy] Running EM ({self.config.em_iters} iterations)...")
        self.em = GalaxyEM(
            self.model, self.config, outlier_ratio=self.effective_outlier_ratio_
        )
        self.em.fit(X_tensor)

        # Compute absolute threshold from training scores
        train_scores = self.em.score(X_tensor)
        self.threshold_ = torch.quantile(
            train_scores, 1.0 - self.effective_outlier_ratio_
        ).item()

        # Store fit metadata
        self._train_score_mean = float(train_scores.mean())
        self._train_score_std = float(train_scores.std())

        self.model.eval()

        if self.config.verbose:
            print(
                f"[Galaxy] Fit complete. threshold_={self.threshold_:.4f}  "
                f"outlier_ratio={self.effective_outlier_ratio_:.4f} "
                f"(from {self._outlier_ratio_source})  "
                f"train_score_mean={self._train_score_mean:.4f}"
            )

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
        return self.predict_score(X_train)

    def save(self, path: str | Path) -> None:
        """Save all fitted state to disk."""
        if self.model is None or self.em is None:
            raise RuntimeError("Nothing to save — call fit() first")
        if self.em.means is None or self.em.weights is None or self.em.covars is None:
            raise RuntimeError("Incomplete fit state — nothing to save")

        state: dict[str, object] = {
            "config": dataclasses.asdict(self.config),
            "input_dim": self.input_dim,
            "threshold_": self.threshold_,
            "effective_outlier_ratio_": self.effective_outlier_ratio_,
            "train_score_mean": self._train_score_mean,
            "train_score_std": self._train_score_std,
            "outlier_ratio_source": self._outlier_ratio_source,
            "em_n_excluded_per_iter": self.em.n_excluded_per_iter,
            "model_state": self.model.state_dict(),
            "em_means": self.em.means.cpu(),
            "em_weights": self.em.weights.cpu(),
            "em_covars": self.em.covars.cpu(),
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

        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> Galaxy:
        """Load a fitted model from disk."""
        state = torch.load(path, map_location=device, weights_only=False)

        config = GalaxyConfig(**state["config"])  # type: ignore[arg-type]
        galaxy = cls(config)
        galaxy.input_dim = state["input_dim"]  # type: ignore[assignment]
        galaxy.threshold_ = state["threshold_"]  # type: ignore[assignment]

        # Restore fit metadata — new flat format
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
        latent_dim = config.latent_dim or config.hidden_dim
        galaxy.model = Autoencoder(
            input_dim=galaxy.input_dim,  # type: ignore[arg-type]
            hidden_dim=config.hidden_dim,
            latent_dim=latent_dim,
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

        return galaxy

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
