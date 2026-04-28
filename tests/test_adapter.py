"""Tests for GalaxyADBench adapter."""

from __future__ import annotations

import numpy as np

from unicad_torch import GalaxyADBench, GalaxyConfig


class TestGalaxyADBench:
    def test_fit_returns_self(self, small_X):
        adapter = GalaxyADBench(
            GalaxyConfig(pretrain_epochs=2, em_iters=1, em_finetune_steps=2)
        )
        result = adapter.fit(small_X)
        assert result is adapter

    def test_predict_score_shape(self, small_X):
        adapter = GalaxyADBench(
            GalaxyConfig(pretrain_epochs=2, em_iters=1, em_finetune_steps=2)
        )
        adapter.fit(small_X)
        scores = adapter.predict_score(small_X)
        assert scores.shape == (100,)

    def test_predict_labels(self, small_X):
        adapter = GalaxyADBench(
            GalaxyConfig(pretrain_epochs=2, em_iters=1, em_finetune_steps=2)
        )
        adapter.fit(small_X)
        labels = adapter.predict(small_X)
        assert labels.dtype == np.int32

    def test_labels_not_forwarded(self, small_X_with_labels):
        """ADBench adapter should NOT forward y_train to Galaxy.fit()."""
        X, y = small_X_with_labels
        adapter = GalaxyADBench(
            GalaxyConfig(pretrain_epochs=2, em_iters=1, em_finetune_steps=2)
        )
        adapter.fit(X, y)
        # Since y_train is not forwarded, outlier_ratio_source should be "config"
        assert adapter.fit_info_["outlier_ratio_source"] == "config"

    def test_threshold_property(self, small_X):
        adapter = GalaxyADBench(
            GalaxyConfig(pretrain_epochs=2, em_iters=1, em_finetune_steps=2)
        )
        adapter.fit(small_X)
        assert adapter.threshold_ is not None
