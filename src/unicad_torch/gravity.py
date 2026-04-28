"""Shared gravity computation — log-space force, Mahalanobis distance, aggregation.

All modules that compute gravitational force (SMM E-step, GOF scoring,
EM gravity loss) delegate here to ensure a single source of truth for
the numerical formula.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

VAR_FLOOR = 1e-6
_LOG_PI = torch.log(torch.tensor(torch.pi))


def mahalanobis_diag(
    X: torch.Tensor,
    means: torch.Tensor,
    covars: torch.Tensor,
) -> torch.Tensor:
    """Squared Mahalanobis distance with diagonal covariance.

    Args:
        X: (N, D) data points.
        means: (K, D) cluster means.
        covars: (K, D) diagonal variances.

    Returns:
        (N, K) squared Mahalanobis distances: Σ_d (x_d - μ_kd)² / σ²_kd
    """
    diff = X.unsqueeze(1) - means.unsqueeze(0)  # (N, K, D)
    return (diff**2 / covars.unsqueeze(0).clamp(min=VAR_FLOOR)).sum(dim=-1)  # (N, K)


def compute_log_forces(
    Z: torch.Tensor,
    means: torch.Tensor,
    covars: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute log-space gravitational forces.

    log F_ik = log(ω_k) - log(π) - 0.5·log|Σ_k| - log(1 + D_M²)

    Args:
        Z: (N, D) latent embeddings.
        means: (K, D) cluster means.
        covars: (K, D) diagonal variances (will be clamped to VAR_FLOOR).
        weights: (1, K) mixture weights (caller must ensure shape).

    Returns:
        (N, K) log-forces tensor.
    """
    maha = mahalanobis_diag(Z, means, covars)  # (N, K)

    log_det_covars = torch.log(covars).sum(dim=1)  # (K,)

    return (
        torch.log(weights.clamp(min=1e-30))  # (1, K)
        - _LOG_PI.to(dtype=Z.dtype, device=Z.device)
        - 0.5 * log_det_covars.unsqueeze(0)  # (1, K)
        - torch.log1p(maha)  # (N, K)
    )  # (N, K)


def aggregate_force_scalar(log_forces: torch.Tensor) -> torch.Tensor:
    """Scalar score: -logsumexp(log_forces, dim=1).

    Returns:
        (N,) anomaly scores (higher = more anomalous).
    """
    return -torch.logsumexp(log_forces, dim=1)


def aggregate_force_vector(
    log_forces: torch.Tensor,
    means: torch.Tensor,
    Z: torch.Tensor,
) -> torch.Tensor:
    """Vector score: -(log(||force_vec||) + max_log).

    Scales by max log-force to keep exp() bounded, then computes
    the norm of the weighted unit-direction force sum.

    Args:
        log_forces: (N, K) log-forces.
        means: (K, D) cluster means.
        Z: (N, D) latent embeddings.

    Returns:
        (N,) anomaly scores (higher = more anomalous).
    """
    log_max, _ = log_forces.max(dim=1, keepdim=True)  # (N, 1)
    scaled = torch.exp(log_forces - log_max)  # (N, K)
    delta = means.unsqueeze(0) - Z.unsqueeze(1)  # (N, K, D)
    unit_vec = F.normalize(delta, p=2, dim=-1)
    force_vec = (scaled.unsqueeze(2) * unit_vec).sum(dim=1)  # (N, D)
    force_norm = torch.norm(force_vec, dim=-1).clamp(min=1e-30)  # (N,)
    log_force_norm = torch.log(force_norm) + log_max.squeeze(1)
    return -log_force_norm
