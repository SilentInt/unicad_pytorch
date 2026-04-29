"""Tests for the callback system."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from unicad_torch import Galaxy, GalaxyConfig
from unicad_torch.callbacks import (
    Callback,
    CallbackManager,
    EarlyStopping,
    FitContext,
    History,
    ModelCheckpoint,
    TqdmProgress,
)


def _small_X():
    rng = np.random.RandomState(42)
    return rng.randn(100, 6).astype(np.float32)


class TestFitContext:
    def test_defaults(self):
        ctx = FitContext(config=GalaxyConfig())
        assert ctx.stop_training is False
        assert ctx.pretrain_loss is None
        assert ctx.em_iter is None

    def test_mutable(self):
        ctx = FitContext(config=GalaxyConfig())
        ctx.stop_training = True
        ctx.pretrain_loss = 1.5
        assert ctx.stop_training is True
        assert ctx.pretrain_loss == 1.5


class TestCallbackManager:
    def test_fire_calls_method(self):
        call_log: list[str] = []

        class LogCallback(Callback):
            def on_fit_begin(self, ctx: FitContext) -> None:
                call_log.append("on_fit_begin")

        mgr = CallbackManager([LogCallback()])
        mgr.fire("on_fit_begin", FitContext(config=GalaxyConfig()))
        assert call_log == ["on_fit_begin"]

    def test_fire_multiple_callbacks(self):
        call_log: list[str] = []

        class CB1(Callback):
            def on_fit_end(self, ctx: FitContext) -> None:
                call_log.append("cb1")

        class CB2(Callback):
            def on_fit_end(self, ctx: FitContext) -> None:
                call_log.append("cb2")

        mgr = CallbackManager([CB1(), CB2()])
        mgr.fire("on_fit_end", FitContext(config=GalaxyConfig()))
        assert call_log == ["cb1", "cb2"]

    def test_empty_manager(self):
        mgr = CallbackManager()
        mgr.fire("on_fit_begin", FitContext(config=GalaxyConfig()))


class TestHistory:
    def test_records_pretrain_loss(self):
        history = History()
        model = Galaxy(
            GalaxyConfig(
                pretrain_epochs=3,
                em_iters=1,
                em_finetune_steps=2,
                verbose=False,
                callbacks=[history],
            )
        )
        model.fit(_small_X())
        assert "pretrain_loss" in history.history
        assert len(history.history["pretrain_loss"]) == 3

    def test_records_em_metrics(self):
        history = History()
        model = Galaxy(
            GalaxyConfig(
                pretrain_epochs=2,
                em_iters=2,
                em_finetune_steps=2,
                verbose=False,
                callbacks=[history],
            )
        )
        model.fit(_small_X())
        assert "n_excluded" in history.history
        assert len(history.history["n_excluded"]) == 2
        assert "score_mean" in history.history


class TestEarlyStopping:
    def test_stops_pretrain_early(self):
        es = EarlyStopping(patience=1, monitor="pretrain_loss")
        history = History()
        model = Galaxy(
            GalaxyConfig(
                pretrain_epochs=50,
                em_iters=1,
                em_finetune_steps=2,
                verbose=False,
                callbacks=[es, history],
            )
        )
        model.fit(_small_X())
        # Should have stopped before 50 epochs
        assert len(history.history.get("pretrain_loss", [])) < 50

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            EarlyStopping(monitor="pretrain_loss", mode="invalid")


class TestModelCheckpoint:
    def test_saves_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = ModelCheckpoint(dirpath=tmpdir, save_last=True, save_best=False)
            model = Galaxy(
                GalaxyConfig(
                    pretrain_epochs=2,
                    em_iters=2,
                    em_finetune_steps=2,
                    verbose=False,
                    callbacks=[ckpt],
                )
            )
            model.fit(_small_X())
            assert os.path.exists(os.path.join(tmpdir, "latest.pt"))

    def test_saves_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = ModelCheckpoint(
                dirpath=tmpdir, save_best=True, monitor="score_mean", mode="min"
            )
            history = History()
            model = Galaxy(
                GalaxyConfig(
                    pretrain_epochs=2,
                    em_iters=3,
                    em_finetune_steps=2,
                    verbose=False,
                    callbacks=[ckpt, history],
                )
            )
            model.fit(_small_X())
            # History should have EM metrics confirming EM ran
            assert "n_excluded" in history.history
            # At least one best checkpoint should exist
            files = os.listdir(tmpdir)
            assert any(f.startswith("best_") for f in files)


class TestTqdmProgress:
    def test_no_error_with_tqdm(self):
        """TqdmProgress should not raise even if tqdm is installed."""
        progress = TqdmProgress()
        model = Galaxy(
            GalaxyConfig(
                pretrain_epochs=2,
                em_iters=1,
                em_finetune_steps=2,
                verbose=False,
                callbacks=[progress],
            )
        )
        model.fit(_small_X())


class TestIntegration:
    def test_callback_galaxy_fit(self):
        """Full pipeline with History."""
        history = History()
        model = Galaxy(
            GalaxyConfig(
                pretrain_epochs=5,
                em_iters=2,
                em_finetune_steps=3,
                verbose=False,
                callbacks=[history],
            )
        )
        model.fit(_small_X())
        # Pretrain should have recorded losses
        assert len(history.history.get("pretrain_loss", [])) == 5
        # EM should have recorded metrics
        assert "n_excluded" in history.history
        assert len(history.history["n_excluded"]) == 2

    def test_save_load_with_history(self):
        history = History()
        model = Galaxy(
            GalaxyConfig(
                pretrain_epochs=2,
                em_iters=1,
                em_finetune_steps=2,
                verbose=False,
                callbacks=[history],
            )
        )
        model.fit(_small_X())

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
            model.save(path)
            loaded = Galaxy.load(path)

        scores_before = model.predict_score(_small_X())
        scores_after = loaded.predict_score(_small_X())
        np.testing.assert_allclose(scores_before, scores_after, rtol=1e-4)
