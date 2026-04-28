"""Gravitational Outlier Factor scorer — pure PyTorch, GPU-native, log-space.

Anomaly score is returned as -log(force) so it is always finite regardless
of dimension.  This is a monotonic transform of 1/force:
  -log(force) ↑ ⟺ 1/force ↑ ⟺ more anomalous

For ranking-based metrics (AUC-ROC, AUC-PR), this is equivalent.

  Scalar: -logsumexp(log F_ik)
  Vector: -[log(||scaled_force_vec||) + max_log_force]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from unicad_torch.smm_torch import _mahalanobis_diag


class GOFScorer:
    """GPU-native GOF scorer with log-space numerics."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def get_score(
        self,
        feat: torch.Tensor,
        means: torch.Tensor,
        covars: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        score_type: str = "vector",
    ) -> torch.Tensor:
        """Compute anomaly scores. Returns 1-D torch tensor on feat's device.

        Score = -log(total_force), always finite, monotonically equivalent
        to 1/total_force.
        """
        if covars is None:
            covars = torch.ones_like(means)

        covars = covars.clamp(min=1e-6)

        maha = _mahalanobis_diag(feat, means, covars)  # (N, K)

        if weights is None:
            _, indices = torch.min(maha, dim=1)
            weights = (
                F.one_hot(indices.long(), num_classes=means.shape[0]).float().sum(dim=0)
                / feat.shape[0]
            )
        weights = weights.reshape(1, -1).to(feat.device)

        # Log-space force: log F_ik = log(ω_k) - log(π) - 0.5*log|Σ_k| - log(1 + D_M²)
        log_det_covars = torch.log(covars.clamp(min=1e-6)).sum(dim=1)  # (K,)
        log_forces = (
            torch.log(weights.clamp(min=1e-30))  # (1, K)
            - torch.log(torch.tensor(torch.pi, dtype=feat.dtype, device=feat.device))
            - 0.5 * log_det_covars.unsqueeze(0)  # (1, K)
            - torch.log1p(maha)  # (N, K)
        )  # (N, K)

        if score_type == "scalar":
            score = -torch.logsumexp(log_forces, dim=1)  # (N,)

        elif score_type == "vector":
            # Scale by max log-force to keep exp() bounded
            log_max, _ = log_forces.max(dim=1, keepdim=True)  # (N, 1)
            scaled = torch.exp(log_forces - log_max)  # (N, K)
            delta = means.unsqueeze(0) - feat.unsqueeze(1)  # (N, K, D)
            unit_vec = F.normalize(delta, p=2, dim=-1)
            force_vec = (scaled.unsqueeze(2) * unit_vec).sum(dim=1)  # (N, D)
            force_norm = torch.norm(force_vec, dim=-1).clamp(min=1e-30)  # (N,)
            # log(‖F‖) = log(‖scaled‖) + max_log
            log_force_norm = torch.log(force_norm) + log_max.squeeze(1)  # (N,)
            score = -log_force_norm

        else:
            raise ValueError(f"Unknown score_type: {score_type}")

        return score
