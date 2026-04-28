"""Tests for GalaxyConfig validation and replace()."""

from __future__ import annotations

import warnings

import pytest

from unicad_torch.config import GalaxyConfig


class TestGalaxyConfigDefaults:
    def test_default_creation(self):
        c = GalaxyConfig()
        assert c.k == 10
        assert c.outlier_ratio == 0.01
        assert c.hidden_dim == 128
        assert c.latent_dim is None
        assert c.gravity_version == "vector"
        assert c.score_type == "vector"
        assert c.device == "cpu"

    def test_custom_creation(self):
        c = GalaxyConfig(k=5, hidden_dim=64, device="cuda")
        assert c.k == 5
        assert c.hidden_dim == 64
        assert c.device == "cuda"


class TestGalaxyConfigValidation:
    @pytest.mark.parametrize("preprocess", ["z-score", "row-norm", "none"])
    def test_valid_preprocess(self, preprocess):
        GalaxyConfig(preprocess=preprocess)

    def test_invalid_preprocess(self):
        with pytest.raises(ValueError, match="preprocess"):
            GalaxyConfig(preprocess="minmax")

    @pytest.mark.parametrize("version", ["scalar", "vector"])
    def test_valid_gravity_version(self, version):
        GalaxyConfig(gravity_version=version)

    def test_invalid_gravity_version(self):
        with pytest.raises(ValueError, match="gravity_version"):
            GalaxyConfig(gravity_version="other")

    def test_invalid_score_type(self):
        with pytest.raises(ValueError, match="score_type"):
            GalaxyConfig(score_type="other")

    def test_outlier_ratio_bounds(self):
        with pytest.raises(ValueError, match="outlier_ratio"):
            GalaxyConfig(outlier_ratio=-0.1)
        with pytest.raises(ValueError, match="outlier_ratio"):
            GalaxyConfig(outlier_ratio=1.0)

    def test_k_positive(self):
        with pytest.raises(ValueError, match="k"):
            GalaxyConfig(k=0)

    def test_latent_dim_positive(self):
        with pytest.raises(ValueError, match="latent_dim"):
            GalaxyConfig(latent_dim=0)

    def test_latent_dim_none_ok(self):
        c = GalaxyConfig(latent_dim=None)
        assert c.latent_dim is None

    def test_invalid_device(self):
        with pytest.raises(ValueError, match="device"):
            GalaxyConfig(device="cudo")

    def test_negative_lr(self):
        with pytest.raises(ValueError, match="pretrain_lr"):
            GalaxyConfig(pretrain_lr=-1)


class TestGalaxyConfigMismatch:
    def test_scalar_gravity_vector_score_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GalaxyConfig(gravity_version="scalar", score_type="vector")
            assert len(w) == 1
            assert "mismatch" in str(w[0].message).lower()

    def test_matched_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GalaxyConfig(gravity_version="vector", score_type="vector")
            assert len(w) == 0


class TestGalaxyConfigReplace:
    def test_replace_single_field(self):
        c = GalaxyConfig()
        c2 = c.replace(k=5)
        assert c2.k == 5
        assert c.k == 10  # original unchanged

    def test_replace_multiple_fields(self):
        c = GalaxyConfig()
        c2 = c.replace(k=5, hidden_dim=64)
        assert c2.k == 5
        assert c2.hidden_dim == 64
        assert c.hidden_dim == 128

    def test_replace_validates(self):
        c = GalaxyConfig()
        with pytest.raises(ValueError):
            c.replace(k=-1)
