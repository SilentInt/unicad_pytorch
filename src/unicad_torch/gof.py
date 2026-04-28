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

from unicad_torch.gravity import (
    aggregate_force_scalar,
    aggregate_force_vector,
    compute_log_forces,
    mahalanobis_diag,
)


class GOFScorer:
    """GPU-native GOF scorer with log-space numerics."""

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

        if weights is None:
            maha = mahalanobis_diag(feat, means, covars)
            _, indices = torch.min(maha, dim=1)
            weights = (
                F.one_hot(indices.long(), num_classes=means.shape[0]).float().sum(dim=0)
                / feat.shape[0]
            )
        weights = weights.reshape(1, -1).to(feat.device)

        log_forces = compute_log_forces(feat, means, covars, weights)  # (N, K)

        if score_type == "scalar":
            return aggregate_force_scalar(log_forces)

        if score_type == "vector":
            return aggregate_force_vector(log_forces, means, feat)

        raise ValueError(f"Unknown score_type: {score_type}")
