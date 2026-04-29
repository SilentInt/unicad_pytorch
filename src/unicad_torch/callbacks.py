"""Callback system for the Galaxy training pipeline.

Provides hook points at epoch/iteration boundaries during pretraining,
EM, and fine-tuning loops. Built-in callbacks cover common needs:
history logging, early stopping, model checkpointing, progress bars,
and LR scheduler injection.

Usage::

    from unicad_torch.callbacks import (
        EarlyStopping, History, LRScheduler, ModelCheckpoint, TqdmProgress,
    )

    config = GalaxyConfig(callbacks=[
        History(),
        EarlyStopping(patience=10, monitor="pretrain_loss"),
        ModelCheckpoint(dirpath="ckpt/", save_best=True, monitor="em_score_mean"),
        TqdmProgress(),
        LRScheduler(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=200),
    ])
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch

from unicad_torch.config import GalaxyConfig

logger = logging.getLogger("unicad_torch.callbacks")


# ---------------------------------------------------------------------------
# FitContext — event context passed to every callback
# ---------------------------------------------------------------------------


@dataclass
class FitContext:
    """Mutable context object shared between the training loop and callbacks.

    The training loop writes stage/epoch/metric fields; callbacks read them.
    Callbacks write ``stop_training`` to signal early termination.
    """

    config: GalaxyConfig
    stage: str = "pretrain"
    epoch: int | None = None
    total_epochs: int | None = None
    em_iter: int | None = None
    total_em_iters: int | None = None
    finetune_step: int | None = None
    total_finetune_steps: int | None = None
    # Metrics — written by training loop
    pretrain_loss: float | None = None
    finetune_recon_loss: float | None = None
    finetune_gravity_loss: float | None = None
    finetune_total_loss: float | None = None
    n_excluded: int | None = None
    score_min: float | None = None
    score_max: float | None = None
    score_mean: float | None = None
    # Control signal — written by callbacks
    stop_training: bool = False
    # Model references — set by Galaxy.fit()
    model: Any = None  # Autoencoder | None (Any to avoid circular import)
    em: Any = None  # GalaxyEM | None
    # LR scheduler internals (set by LRScheduler callback)
    lr_optimizer: Any = None
    lr_scheduler: Any = None
    # Checkpoint directory
    checkpoint_dir: str | None = None


# ---------------------------------------------------------------------------
# Callback base class
# ---------------------------------------------------------------------------


class Callback:
    """Base callback. Override any ``on_*`` method to hook into training."""

    def on_fit_begin(self, ctx: FitContext) -> None: ...

    def on_fit_end(self, ctx: FitContext) -> None: ...

    def on_pretrain_epoch_begin(self, ctx: FitContext) -> None: ...

    def on_pretrain_epoch_end(self, ctx: FitContext) -> None: ...

    def on_em_iter_begin(self, ctx: FitContext) -> None: ...

    def on_em_iter_end(self, ctx: FitContext) -> None: ...

    def on_finetune_step_end(self, ctx: FitContext) -> None: ...


# ---------------------------------------------------------------------------
# CallbackManager
# ---------------------------------------------------------------------------


class CallbackManager:
    """Dispatches events to a list of callbacks in order."""

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self.callbacks: list[Callback] = callbacks or []

    def fire(self, event: str, ctx: FitContext) -> None:
        for cb in self.callbacks:
            getattr(cb, event)(ctx)


# ---------------------------------------------------------------------------
# Built-in callbacks
# ---------------------------------------------------------------------------


class History(Callback):
    """Records metric values per event into a dict for post-fit inspection."""

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = {}

    def _record(self, ctx: FitContext, attrs: tuple[str, ...]) -> None:
        for attr in attrs:
            val = getattr(ctx, attr)
            if val is not None:
                self.history.setdefault(attr, []).append(float(val))

    _PRETRAIN_ATTRS = ("pretrain_loss",)
    _EM_ATTRS = ("n_excluded", "score_min", "score_max", "score_mean")
    _FINETUNE_ATTRS = (
        "finetune_recon_loss",
        "finetune_gravity_loss",
        "finetune_total_loss",
    )

    def on_pretrain_epoch_end(self, ctx: FitContext) -> None:
        self._record(ctx, self._PRETRAIN_ATTRS)

    def on_em_iter_end(self, ctx: FitContext) -> None:
        self._record(ctx, self._EM_ATTRS)

    def on_finetune_step_end(self, ctx: FitContext) -> None:
        self._record(ctx, self._FINETUNE_ATTRS)


class EarlyStopping(Callback):
    """Stops training when a monitored metric stops improving.

    Args:
        patience: Number of events with no improvement before stopping.
        monitor: Metric name from FitContext (e.g. ``"pretrain_loss"``).
        mode: ``"min"`` (lower is better) or ``"max"`` (higher is better).
        threshold: Minimum change to qualify as improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        monitor: str = "pretrain_loss",
        mode: str = "min",
        threshold: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.threshold = threshold
        self._best: float | None = None
        self._counter: int = 0

    def _check(self, ctx: FitContext) -> None:
        val = getattr(ctx, self.monitor, None)
        if val is None:
            return
        improved = (
            self._best is None
            or (self.mode == "min" and val < self._best - self.threshold)
            or (self.mode == "max" and val > self._best + self.threshold)
        )
        if improved:
            self._best = float(val)
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                logger.info(
                    "EarlyStopping: %s did not improve for %d events, stopping.",
                    self.monitor,
                    self.patience,
                )
                ctx.stop_training = True

    def on_pretrain_epoch_end(self, ctx: FitContext) -> None:
        self._check(ctx)

    def on_em_iter_end(self, ctx: FitContext) -> None:
        self._check(ctx)


class ModelCheckpoint(Callback):
    """Saves model checkpoints at EM iteration boundaries.

    Args:
        dirpath: Directory to write checkpoints into.
        save_best: Only save when monitored metric improves.
        monitor: Metric name from FitContext.
        mode: ``"min"`` or ``"max"``.
        save_last: Always save ``latest.pt`` at each EM iter end.
        save_top_k: Keep only the k best checkpoints (0 = keep all).
    """

    def __init__(
        self,
        dirpath: str = "checkpoints/",
        save_best: bool = False,
        monitor: str = "em_score_mean",
        mode: str = "min",
        save_last: bool = True,
        save_top_k: int = 1,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.dirpath = dirpath
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.save_top_k = save_top_k
        self._best: float | None = None
        self._best_paths: list[str] = []
        self._galaxy_ref: Any = None

    def on_fit_begin(self, ctx: FitContext) -> None:
        os.makedirs(self.dirpath, exist_ok=True)

    def on_em_iter_end(self, ctx: FitContext) -> None:
        if self._galaxy_ref is None:
            return

        if self.save_last:
            path = os.path.join(self.dirpath, "latest.pt")
            self._galaxy_ref.save(path)
            logger.info("Saved latest checkpoint: %s", path)

        if self.save_best:
            val = getattr(ctx, self.monitor, None)
            if val is None:
                return
            improved = (
                self._best is None
                or (self.mode == "min" and val < self._best)
                or (self.mode == "max" and val > self._best)
            )
            if improved:
                self._best = float(val)
                path = os.path.join(self.dirpath, f"best_em{ctx.em_iter}.pt")
                self._galaxy_ref.save(path)
                self._best_paths.append(path)
                logger.info("Saved best checkpoint: %s (monitor=%.6f)", path, val)

                if self.save_top_k > 0 and len(self._best_paths) > self.save_top_k:
                    removed = self._best_paths.pop(0)
                    if os.path.exists(removed) and removed != path:
                        os.remove(removed)


class TqdmProgress(Callback):
    """Displays tqdm progress bars for pretraining and EM loops."""

    def __init__(self, refresh_rate: int = 1) -> None:
        self.refresh_rate = refresh_rate
        self._pretrain_bar: Any = None
        self._em_bar: Any = None
        self._finetune_bar: Any = None

    def on_fit_begin(self, ctx: FitContext) -> None:
        try:
            import tqdm.auto  # noqa: F401 — import to check availability
        except ImportError:
            logger.warning("tqdm not installed — TqdmProgress disabled")
            return

    def on_pretrain_epoch_begin(self, ctx: FitContext) -> None:
        if ctx.epoch == 0:
            try:
                from tqdm.auto import tqdm

                self._pretrain_bar = tqdm(
                    total=ctx.total_epochs,
                    desc="Pretrain",
                    unit="epoch",
                    leave=True,
                )
            except ImportError:
                pass

    def on_pretrain_epoch_end(self, ctx: FitContext) -> None:
        if self._pretrain_bar is not None:
            loss_str = f"loss={ctx.pretrain_loss:.4f}" if ctx.pretrain_loss else ""
            self._pretrain_bar.set_postfix_str(loss_str)
            self._pretrain_bar.update(1)
        if ctx.epoch is not None and ctx.epoch == (ctx.total_epochs or 0) - 1:
            if self._pretrain_bar is not None:
                self._pretrain_bar.close()
                self._pretrain_bar = None

    def on_em_iter_begin(self, ctx: FitContext) -> None:
        if ctx.em_iter == 0:
            try:
                from tqdm.auto import tqdm

                self._em_bar = tqdm(
                    total=ctx.total_em_iters,
                    desc="EM",
                    unit="iter",
                    leave=True,
                )
            except ImportError:
                pass

    def on_em_iter_end(self, ctx: FitContext) -> None:
        if self._em_bar is not None:
            excl = f"excl={ctx.n_excluded}" if ctx.n_excluded is not None else ""
            smean = f"score={ctx.score_mean:.2f}" if ctx.score_mean is not None else ""
            self._em_bar.set_postfix_str(f"{excl} {smean}".strip())
            self._em_bar.update(1)
        if ctx.em_iter is not None and ctx.em_iter == (ctx.total_em_iters or 0) - 1:
            if self._em_bar is not None:
                self._em_bar.close()
                self._em_bar = None

    def on_finetune_step_end(self, ctx: FitContext) -> None:
        pass  # fine-tune step progress omitted to avoid visual noise


class LRScheduler(Callback):
    """Injects a custom LR scheduler that steps at pretrain epoch boundaries.

    Args:
        scheduler_cls: A ``torch.optim.lr.scheduler`` class.
        **scheduler_kwargs: Keyword arguments forwarded to the scheduler constructor.
    """

    def __init__(self, scheduler_cls: type, **scheduler_kwargs: Any) -> None:
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler: Any = None

    def on_fit_begin(self, ctx: FitContext) -> None:
        if ctx.model is not None:
            optimizer = torch.optim.Adam(
                ctx.model.parameters(), lr=ctx.config.pretrain_lr
            )
            self.scheduler = self.scheduler_cls(optimizer, **self.scheduler_kwargs)
            ctx.lr_optimizer = optimizer
            ctx.lr_scheduler = self.scheduler

    def on_pretrain_epoch_end(self, ctx: FitContext) -> None:
        if self.scheduler is not None:
            self.scheduler.step()
