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

    def __post_init__(self) -> None:
        if self.preprocess not in {"z-score", "row-norm", "none"}:
            raise ValueError(
                f"preprocess must be 'z-score', 'row-norm', or 'none', got {self.preprocess!r}"
            )
        if self.gravity_version not in {"scalar", "vector"}:
            raise ValueError(
                f"gravity_version must be 'scalar' or 'vector', got {self.gravity_version!r}"
            )
        if self.score_type not in {"scalar", "vector"}:
            raise ValueError(
                f"score_type must be 'scalar' or 'vector', got {self.score_type!r}"
            )
        if not (0 <= self.outlier_ratio < 1):
            raise ValueError(
                f"outlier_ratio must be in [0, 1), got {self.outlier_ratio}"
            )
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {self.hidden_dim}")
        if self.pretrain_batch_size < 1:
            raise ValueError(
                f"pretrain_batch_size must be >= 1, got {self.pretrain_batch_size}"
            )
        if self.em_iters < 1:
            raise ValueError(f"em_iters must be >= 1, got {self.em_iters}")
        if self.smm_n_iter < 1:
            raise ValueError(f"smm_n_iter must be >= 1, got {self.smm_n_iter}")
        if self.pretrain_epochs < 0:
            raise ValueError(
                f"pretrain_epochs must be >= 0, got {self.pretrain_epochs}"
            )
        if self.em_finetune_steps < 0:
            raise ValueError(
                f"em_finetune_steps must be >= 0, got {self.em_finetune_steps}"
            )
        if self.pretrain_lr <= 0:
            raise ValueError(f"pretrain_lr must be > 0, got {self.pretrain_lr}")
        if self.em_finetune_lr <= 0:
            raise ValueError(f"em_finetune_lr must be > 0, got {self.em_finetune_lr}")
        if self.smm_tol <= 0:
            raise ValueError(f"smm_tol must be > 0, got {self.smm_tol}")
