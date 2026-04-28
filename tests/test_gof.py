"""Tests for GOFScorer."""

from __future__ import annotations

import pytest
import torch

from unicad_torch.gof import GOFScorer


class TestGOFScorer:
    def test_scalar_score_shape(self):
        scorer = GOFScorer()
        Z = torch.randn(20, 4)
        means = torch.randn(3, 4)
        covars = torch.ones(3, 4)
        weights = torch.full((3,), 1.0 / 3)
        scores = scorer.get_score(
            Z, means, covars=covars, weights=weights, score_type="scalar"
        )
        assert scores.shape == (20,)

    def test_vector_score_shape(self):
        scorer = GOFScorer()
        Z = torch.randn(20, 4)
        means = torch.randn(3, 4)
        covars = torch.ones(3, 4)
        weights = torch.full((3,), 1.0 / 3)
        scores = scorer.get_score(
            Z, means, covars=covars, weights=weights, score_type="vector"
        )
        assert scores.shape == (20,)

    def test_scores_finite(self):
        scorer = GOFScorer()
        Z = torch.randn(20, 4)
        means = torch.randn(3, 4)
        covars = torch.ones(3, 4)
        weights = torch.full((3,), 1.0 / 3)
        for st in ["scalar", "vector"]:
            scores = scorer.get_score(
                Z, means, covars=covars, weights=weights, score_type=st
            )
            assert scores.isfinite().all(), f"Non-finite scores for score_type={st}"

    def test_default_covars(self):
        scorer = GOFScorer()
        Z = torch.randn(20, 4)
        means = torch.randn(3, 4)
        scores = scorer.get_score(Z, means, score_type="scalar")
        assert scores.shape == (20,)

    def test_default_weights(self):
        scorer = GOFScorer()
        Z = torch.randn(20, 4)
        means = torch.randn(3, 4)
        covars = torch.ones(3, 4)
        scores = scorer.get_score(Z, means, covars=covars, score_type="scalar")
        assert scores.shape == (20,)

    def test_anomalous_points_score_higher(self):
        scorer = GOFScorer()
        # Normal points near cluster centers
        means = torch.randn(3, 4)
        normal = means[0].unsqueeze(0).expand(10, -1) + torch.randn(10, 4) * 0.1
        # Anomalous points far from all clusters
        anomaly = means.mean(0).unsqueeze(0).expand(5, -1) + 100

        Z = torch.cat([normal, anomaly])
        covars = torch.ones(3, 4)
        weights = torch.full((3,), 1.0 / 3)
        scores = scorer.get_score(
            Z, means, covars=covars, weights=weights, score_type="scalar"
        )

        normal_mean = scores[:10].mean().item()
        anomaly_mean = scores[10:].mean().item()
        assert anomaly_mean > normal_mean

    def test_invalid_score_type_raises(self):
        scorer = GOFScorer()
        Z = torch.randn(10, 4)
        means = torch.randn(3, 4)
        with pytest.raises(ValueError, match="score_type"):
            scorer.get_score(Z, means, score_type="other")
