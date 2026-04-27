"""Preprocessing scalers — pure PyTorch, GPU-native.

StandardScaler : z-score standardisation
RowScaler      : L2 per-row normalisation
"""
from __future__ import annotations

import torch


class StandardScaler:
    """Z-score standardisation in torch. Stores mean/std from fit()."""

    def __init__(self) -> None:
        self.mean_: torch.Tensor | None = None
        self.std_: torch.Tensor | None = None

    def fit(self, X: torch.Tensor) -> StandardScaler:
        self.mean_ = X.mean(dim=0)
        self.std_ = X.std(dim=0).clamp(min=1e-8)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)


class RowScaler:
    """L2 per-row normalisation in torch."""

    def __init__(self, order: int = 2) -> None:
        self.order = order

    def fit(self, X: torch.Tensor) -> RowScaler:
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        row_norms = torch.linalg.norm(X, ord=self.order, dim=1, keepdim=True)
        row_norms = row_norms.clamp(min=1e-8)
        return X / row_norms

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)
