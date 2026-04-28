"""Tests for SMMTorch."""

from __future__ import annotations

import torch

from unicad_torch.smm_torch import SMMTorch


class TestSMMTorch:
    def test_fit_assigns_attributes(self, small_tensor):
        smm = SMMTorch(n_components=3, n_iter=10, random_state=42)
        smm.fit(small_tensor)
        assert smm.means_ is not None
        assert smm.covars_ is not None
        assert smm.weights_ is not None
        assert smm.means_.shape == (3, 4)
        assert smm.covars_.shape == (3, 4)
        assert smm.weights_.shape == (3,)

    def test_weights_sum_to_one(self, small_tensor):
        smm = SMMTorch(n_components=3, n_iter=20, random_state=42)
        smm.fit(small_tensor)
        assert smm.weights_ is not None
        assert abs(smm.weights_.sum().item() - 1.0) < 1e-4

    def test_covars_positive(self, small_tensor):
        smm = SMMTorch(n_components=3, n_iter=20, random_state=42)
        smm.fit(small_tensor)
        assert smm.covars_ is not None
        assert (smm.covars_ > 0).all()

    def test_means_finite(self, small_tensor):
        smm = SMMTorch(n_components=3, n_iter=20, random_state=42)
        smm.fit(small_tensor)
        assert smm.means_ is not None
        assert smm.means_.isfinite().all()

    def test_reproducibility(self, small_tensor):
        smm1 = SMMTorch(n_components=3, n_iter=20, random_state=42)
        smm1.fit(small_tensor)
        smm2 = SMMTorch(n_components=3, n_iter=20, random_state=42)
        smm2.fit(small_tensor)
        assert smm1.means_ is not None
        assert smm2.means_ is not None
        assert torch.allclose(smm1.means_, smm2.means_, atol=1e-5)

    def test_k_exceeds_n(self):
        X = torch.randn(5, 3)
        smm = SMMTorch(n_components=10, n_iter=10, random_state=42)
        smm.fit(X)
        assert smm.means_ is not None
        assert smm.means_.shape[0] <= 5
