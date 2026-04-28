"""Tests for preprocessing scalers."""

from __future__ import annotations

import pytest
import torch

from unicad_torch.preprocessing import RowScaler, StandardScaler


class TestStandardScaler:
    def test_fit_stores_stats(self):
        X = torch.randn(50, 4)
        scaler = StandardScaler()
        scaler.fit(X)
        assert scaler.mean_ is not None
        assert scaler.std_ is not None
        assert scaler.mean_.shape == (4,)
        assert scaler.std_.shape == (4,)

    def test_transform_shapes(self):
        X = torch.randn(50, 4)
        scaler = StandardScaler()
        X_t = scaler.fit_transform(X)
        assert X_t.shape == X.shape

    def test_zero_mean_unit_var(self):
        X = torch.randn(200, 4) * 3 + 5
        scaler = StandardScaler()
        X_t = scaler.fit_transform(X)
        assert X_t.mean(dim=0).abs().max() < 0.1
        assert (X_t.std(dim=0, correction=0) - 1).abs().max() < 0.1

    def test_not_fitted_raises(self):
        scaler = StandardScaler()
        with pytest.raises(RuntimeError, match="not fitted"):
            scaler.transform(torch.randn(10, 4))

    def test_fit_transform_deterministic(self):
        X = torch.randn(50, 4)
        scaler1 = StandardScaler()
        X1 = scaler1.fit_transform(X)
        scaler2 = StandardScaler()
        X2 = scaler2.fit_transform(X)
        assert torch.allclose(X1, X2)


class TestRowScaler:
    def test_fit_is_noop(self):
        X = torch.randn(50, 4)
        scaler = RowScaler()
        result = scaler.fit(X)
        assert result is scaler

    def test_transform_shapes(self):
        X = torch.randn(50, 4)
        scaler = RowScaler()
        X_t = scaler.fit_transform(X)
        assert X_t.shape == X.shape

    def test_unit_row_norm(self):
        X = torch.randn(50, 4)
        scaler = RowScaler()
        X_t = scaler.fit_transform(X)
        norms = torch.linalg.norm(X_t, dim=1)
        assert (norms - 1).abs().max() < 1e-5

    def test_custom_order(self):
        X = torch.randn(50, 4).abs() + 0.1
        scaler_l1 = RowScaler(order=1)
        X_l1 = scaler_l1.fit_transform(X)
        l1_norms = X_l1.abs().sum(dim=1)
        assert (l1_norms - 1).abs().max() < 1e-5
