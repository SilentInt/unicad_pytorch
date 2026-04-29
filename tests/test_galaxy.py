"""Tests for Galaxy full pipeline — fit, predict, save/load."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from unicad_torch import Galaxy, GalaxyConfig


class TestGalaxyFit:
    def test_fit_returns_self(self, small_X):
        model = Galaxy(GalaxyConfig(pretrain_epochs=2, em_iters=1, em_finetune_steps=2))
        result = model.fit(small_X)
        assert result is model

    def test_fit_assigns_attributes(self, small_X):
        model = Galaxy(GalaxyConfig(pretrain_epochs=2, em_iters=1, em_finetune_steps=2))
        model.fit(small_X)
        assert model.input_dim == 6
        assert model.threshold_ is not None
        assert model.model is not None
        assert model.em is not None
        assert model.em.means_ is not None

    def test_nan_input_raises(self):
        X = np.array([[1, 2], [np.nan, 4]], dtype=np.float32)
        model = Galaxy()
        with pytest.raises(ValueError, match="NaN"):
            model.fit(X)

    def test_inf_input_raises(self):
        X = np.array([[1, 2], [3, np.inf]], dtype=np.float32)
        model = Galaxy()
        with pytest.raises(ValueError, match="Inf"):
            model.fit(X)

    def test_too_few_samples_raises(self):
        X = np.random.randn(5, 6).astype(np.float32)
        model = Galaxy(GalaxyConfig(k=10))
        with pytest.raises(ValueError, match="k="):
            model.fit(X)


class TestGalaxyPredict:
    def test_predict_score_shape(self, small_X):
        model = Galaxy(GalaxyConfig(pretrain_epochs=2, em_iters=1, em_finetune_steps=2))
        model.fit(small_X)
        scores = model.predict_score(small_X)
        assert scores.shape == (100,)
        assert np.isfinite(scores).all()

    def test_predict_binary_labels(self, small_X):
        model = Galaxy(GalaxyConfig(pretrain_epochs=2, em_iters=1, em_finetune_steps=2))
        model.fit(small_X)
        labels = model.predict(small_X)
        assert labels.dtype == np.int32
        assert set(labels.tolist()).issubset({0, 1})

    def test_predict_before_fit_raises(self):
        model = Galaxy()
        with pytest.raises(RuntimeError):
            model.predict_score(np.random.randn(10, 6).astype(np.float32))


class TestGalaxyLabelAware:
    def test_labels_override_outlier_ratio(self, small_X_with_labels):
        X, y = small_X_with_labels
        model = Galaxy(GalaxyConfig(pretrain_epochs=2, em_iters=1, em_finetune_steps=2))
        model.fit(X, y)
        assert model.effective_outlier_ratio_ == 0.05  # 5/100
        assert model.fit_info_["outlier_ratio_source"] == "labels"

    def test_config_not_mutated(self, small_X_with_labels):
        X, y = small_X_with_labels
        config = GalaxyConfig(
            outlier_ratio=0.01, pretrain_epochs=2, em_iters=1, em_finetune_steps=2
        )
        model = Galaxy(config)
        model.fit(X, y)
        assert model.config.outlier_ratio == 0.01  # original unchanged
        assert model.effective_outlier_ratio_ == 0.05


class TestGalaxySaveLoad:
    def test_save_load_roundtrip(self, small_X):
        config = GalaxyConfig(pretrain_epochs=2, em_iters=1, em_finetune_steps=2)
        model = Galaxy(config)
        model.fit(small_X)
        scores_before = model.predict_score(small_X)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
            model.save(path)

            loaded = Galaxy.load(path)
            scores_after = loaded.predict_score(small_X)

        np.testing.assert_allclose(scores_before, scores_after, rtol=1e-4)

    def test_save_preserves_threshold(self, small_X):
        config = GalaxyConfig(pretrain_epochs=2, em_iters=1, em_finetune_steps=2)
        model = Galaxy(config)
        model.fit(small_X)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
            model.save(path)
            loaded = Galaxy.load(path)

        assert loaded.threshold_ is not None
        assert model.threshold_ is not None
        assert abs(loaded.threshold_ - model.threshold_) < 1e-5

    def test_save_before_fit_raises(self):
        model = Galaxy()
        with pytest.raises(RuntimeError, match="save"):
            model.save("/tmp/test.pt")


class TestGalaxyFitPredict:
    def test_fit_predict_shape(self, small_X):
        config = GalaxyConfig(pretrain_epochs=2, em_iters=1, em_finetune_steps=2)
        model = Galaxy(config)
        scores = model.fit_predict(small_X)
        assert scores.shape == (100,)
