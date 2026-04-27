from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GalaxyConfig:
    """Single source of truth for all Galaxy hyperparameters with paper defaults."""

    seed: int = 42
    k: int = 10
    outlier_ratio: float = 0.01
    hidden_dim: int = 128
    pretrain_epochs: int = 200
    pretrain_lr: float = 3e-3
    pretrain_batch_size: int = 1024
    em_iters: int = 3
    em_finetune_steps: int = 100
    em_finetune_lr: float = 3e-4
    preprocess: str = "z-score"
    gravity_version: str = "scalar"
    score_type: str = "vector"
    pretrain: bool = True
    smm_n_iter: int = 100
    smm_tol: float = 1e-3
    device: str = "cpu"
