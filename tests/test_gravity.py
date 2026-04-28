"""Tests for gravity module — shared computation building blocks."""

from __future__ import annotations

import torch

from unicad_torch.gravity import (
    VAR_FLOOR,
    aggregate_force_scalar,
    aggregate_force_vector,
    compute_log_forces,
    mahalanobis_diag,
)


class TestMahalanobisDiag:
    def test_shape(self):
        X = torch.randn(20, 4)
        means = torch.randn(3, 4)
        covars = torch.ones(3, 4)
        d = mahalanobis_diag(X, means, covars)
        assert d.shape == (20, 3)

    def test_identity_covars(self):
        X = torch.randn(20, 4)
        means = torch.randn(3, 4)
        covars = torch.ones(3, 4)
        d = mahalanobis_diag(X, means, covars)
        # With identity covars, should equal squared Euclidean distance
        diff = X.unsqueeze(1) - means.unsqueeze(0)
        euclidean_sq = (diff**2).sum(dim=-1)
        assert torch.allclose(d, euclidean_sq, atol=1e-5)

    def test_clamps_small_covars(self):
        X = torch.randn(10, 3)
        means = torch.randn(2, 3)
        covars = torch.full((2, 3), 1e-10)  # very small
        d = mahalanobis_diag(X, means, covars)
        assert d.isfinite().all()

    def test_zero_distance_to_self(self):
        means = torch.randn(5, 4)
        covars = torch.ones(5, 4) * 2
        d = mahalanobis_diag(means, means, covars)
        assert (d.diag().abs() < 1e-5).all()


class TestComputeLogForces:
    def test_shape(self):
        Z = torch.randn(20, 4)
        means = torch.randn(3, 4)
        covars = torch.ones(3, 4)
        weights = torch.full((3,), 1.0 / 3)
        lf = compute_log_forces(Z, means, covars, weights)
        assert lf.shape == (20, 3)

    def test_is_finite(self):
        Z = torch.randn(20, 4)
        means = torch.randn(3, 4)
        covars = torch.ones(3, 4).clamp(min=VAR_FLOOR)
        weights = torch.full((3,), 1.0 / 3)
        lf = compute_log_forces(Z, means, covars, weights)
        assert lf.isfinite().all()


class TestAggregateForceScalar:
    def test_shape(self):
        log_forces = torch.randn(20, 3)
        scores = aggregate_force_scalar(log_forces)
        assert scores.shape == (20,)

    def test_higher_log_force_means_lower_score(self):
        # More force → lower anomaly score
        lf_low = torch.full((1, 3), -10.0)
        lf_high = torch.full((1, 3), -1.0)
        s_low = aggregate_force_scalar(lf_low)
        s_high = aggregate_force_scalar(lf_high)
        assert s_low.item() > s_high.item()


class TestAggregateForceVector:
    def test_shape(self):
        log_forces = torch.randn(20, 3)
        means = torch.randn(3, 4)
        Z = torch.randn(20, 4)
        scores = aggregate_force_vector(log_forces, means, Z)
        assert scores.shape == (20,)

    def test_is_finite(self):
        log_forces = torch.randn(20, 3)
        means = torch.randn(3, 4)
        Z = torch.randn(20, 4)
        scores = aggregate_force_vector(log_forces, means, Z)
        assert scores.isfinite().all()
