"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unicad_torch.config import GalaxyConfig


@pytest.fixture
def default_config() -> GalaxyConfig:
    return GalaxyConfig(pretrain_epochs=2, em_iters=1, em_finetune_steps=2, verbose=False)


@pytest.fixture
def small_X() -> np.ndarray:
    """100 samples, 6 features, float32, no anomalies by construction."""
    rng = np.random.RandomState(42)
    return rng.randn(100, 6).astype(np.float32)


@pytest.fixture
def small_X_with_labels() -> tuple[np.ndarray, np.ndarray]:
    """100 samples, 6 features, 5% anomaly rate."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 6).astype(np.float32)
    y = np.zeros(100, dtype=np.int32)
    y[:5] = 1  # 5 anomalies
    return X, y


@pytest.fixture
def small_tensor() -> torch.Tensor:
    """50 samples, 4 features on CPU."""
    return torch.randn(50, 4)
