"""CLI entry points and shared utilities for Galaxy commands.

Provides:
- ``galaxy-train``, ``galaxy-predict``, ``galaxy-benchmark``, ``galaxy-runs``
  console scripts
- Shared utilities: :func:`load_data`, :func:`build_callbacks`,
  :func:`add_run_args`, :func:`build_config_overrides`
- Command implementations: :func:`train_main`, :func:`predict_main`,
  :func:`benchmark_main`, :func:`runs_main`

Scripts in ``scripts/`` are thin wrappers that import and call the
``*_main`` functions, so logic lives in one place.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

import numpy as np

from unicad_torch import Galaxy, GalaxyConfig
from unicad_torch.callbacks import (
    EarlyStopping,
    History,
    ModelCheckpoint,
    TqdmProgress,
)


# ---------------------------------------------------------------------------
# Shared: data loading
# ---------------------------------------------------------------------------


def load_data(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Load data from .npz or .csv. Returns (X, y) where y may be None."""
    if path.endswith(".npz"):
        data = np.load(path)
        X = data["X"].astype(np.float32)
        y = data["y"].ravel() if "y" in data else None
        return X, y

    if path.endswith(".csv"):
        import pandas as pd

        df = pd.read_csv(path)
        if "y" in df.columns:
            y = df["y"].to_numpy().ravel()
            X = df.drop(columns=["y"]).to_numpy().astype(np.float32)
        else:
            y = None
            X = df.to_numpy().astype(np.float32)
        return X, y

    raise ValueError(f"Unsupported file format: {path} (expected .npz or .csv)")


# ---------------------------------------------------------------------------
# Shared: config file loading
# ---------------------------------------------------------------------------


def load_config_file(path: str) -> dict[str, Any]:
    """Load a config dict from YAML or JSON file."""
    text = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml

            return yaml.safe_load(text)
        except ImportError:
            print("PyYAML not installed — install with: uv pip install pyyaml")
            sys.exit(1)
    if path.endswith(".json"):
        return json.loads(text)
    raise ValueError(f"Unsupported config format: {path} (expected .yaml/.yml/.json)")


# ---------------------------------------------------------------------------
# Shared: callback building
# ---------------------------------------------------------------------------


def build_callbacks(args: argparse.Namespace, run_dir: Path | None = None) -> list:
    """Build callback list from CLI flags (shared by train and benchmark).

    History, Checkpoint, and TqdmProgress are enabled by default.
    Pass ``--no-history``, ``--no-checkpoint``, or ``--no-tqdm`` to disable.
    If *run_dir* is provided and Checkpoint is used without an explicit
    ``--checkpoint-dir``, the checkpoint directory is auto-wired to
    ``{run_dir}/checkpoints/``.
    """
    callbacks: list = []

    # History: default on, --no-history to disable
    if not getattr(args, "no_history", False):
        callbacks.append(History())

    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                patience=args.early_stopping_patience,
                monitor=args.early_stopping_monitor,
            )
        )

    # Checkpoint: default on when run_dir exists, --no-checkpoint to disable
    if not getattr(args, "no_checkpoint", False) and run_dir is not None:
        dirpath = args.checkpoint_dir
        if dirpath == "checkpoints/":
            dirpath = str(run_dir / "checkpoints")

        callbacks.append(
            ModelCheckpoint(
                dirpath=dirpath,
                save_best=args.checkpoint_best,
                save_last=True,
                monitor=args.checkpoint_monitor,
            )
        )

    # TqdmProgress: default on, --no-tqdm to disable
    if not getattr(args, "no_tqdm", False):
        callbacks.append(TqdmProgress())

    return callbacks


# ---------------------------------------------------------------------------
# Shared: CLI argument groups
# ---------------------------------------------------------------------------


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add GalaxyConfig override arguments to an argument parser."""
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--pretrain-lr", type=float, default=None)
    parser.add_argument("--pretrain-batch-size", type=int, default=None)
    parser.add_argument("--em-iters", type=int, default=None)
    parser.add_argument("--em-finetune-steps", type=int, default=None)
    parser.add_argument("--em-finetune-lr", type=float, default=None)
    parser.add_argument("--outlier-ratio", type=float, default=None)
    parser.add_argument("--smm-n-iter", type=int, default=None)
    parser.add_argument(
        "--preprocess", choices=["z-score", "row-norm", "none"], default=None
    )
    parser.add_argument("--gravity-version", choices=["scalar", "vector"], default=None)
    parser.add_argument("--score-type", choices=["scalar", "vector"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--verbose", action="store_true")


def add_run_args(parser: argparse.ArgumentParser) -> None:
    """Add run directory arguments to an argument parser."""
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Root directory for run output (default: runs/). "
        "Creates a timestamped subdirectory per run.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Custom name/tag for this run (appended to timestamp)",
    )


def add_callback_args(parser: argparse.ArgumentParser) -> None:
    """Add callback-related arguments to an argument parser."""
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable training history recording (enabled by default)",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bars (enabled by default)",
    )
    parser.add_argument(
        "--early-stopping", action="store_true", help="Enable early stopping"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="EarlyStopping patience (default: 10)",
    )
    parser.add_argument(
        "--early-stopping-monitor",
        default="pretrain_loss",
        help="EarlyStopping monitor metric (default: pretrain_loss)",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable model checkpointing (enabled by default when run_dir exists)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/",
        help="Checkpoint directory (default: checkpoints/)",
    )
    parser.add_argument(
        "--checkpoint-best",
        action="store_true",
        help="Save best checkpoint (by monitor metric)",
    )
    parser.add_argument(
        "--checkpoint-monitor",
        default="score_mean",
        help="Checkpoint monitor metric (default: score_mean)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )


# ---------------------------------------------------------------------------
# Shared: config building from args
# ---------------------------------------------------------------------------


def build_config_overrides(
    args: argparse.Namespace, run_dir: Path | None = None
) -> dict[str, Any]:
    """Build GalaxyConfig override dict from parsed CLI args."""
    overrides: dict[str, Any] = {}

    # Config file
    if getattr(args, "config", None):
        file_cfg = load_config_file(args.config)
        file_cfg.pop("callbacks", None)
        overrides.update(file_cfg)

    # CLI overrides
    mapping: list[tuple[str, str]] = [
        ("k", "k"),
        ("hidden_dim", "hidden_dim"),
        ("latent_dim", "latent_dim"),
        ("pretrain_epochs", "pretrain_epochs"),
        ("pretrain_lr", "pretrain_lr"),
        ("pretrain_batch_size", "pretrain_batch_size"),
        ("em_iters", "em_iters"),
        ("em_finetune_steps", "em_finetune_steps"),
        ("em_finetune_lr", "em_finetune_lr"),
        ("outlier_ratio", "outlier_ratio"),
        ("smm_n_iter", "smm_n_iter"),
        ("preprocess", "preprocess"),
        ("gravity_version", "gravity_version"),
        ("score_type", "score_type"),
        ("device", "device"),
        ("seed", "seed"),
    ]
    for attr, key in mapping:
        val = getattr(args, attr, None)
        if val is not None:
            overrides[key] = val
    if getattr(args, "no_pretrain", False):
        overrides["pretrain"] = False
    if getattr(args, "verbose", False):
        overrides["verbose"] = True

    # Auto-detect CUDA when user didn't specify --device
    if "device" not in overrides:
        if torch.cuda.is_available():
            overrides["device"] = "cuda"

    # Callbacks -- pass run_dir for auto-wiring
    callbacks = build_callbacks(args, run_dir=run_dir)
    if callbacks:
        overrides["callbacks"] = callbacks

    return overrides


# ---------------------------------------------------------------------------
# Shared: output helpers
# ---------------------------------------------------------------------------


def save_csv(results: list[dict[str, Any]], path: str) -> None:
    """Save a list of result dicts to CSV."""
    if not results:
        return
    cols = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {path}")


def print_history(history: dict[str, list[float]]) -> None:
    """Print a summary of training history from the History callback."""
    print("\nTraining History:")
    for key, values in history.items():
        if not values:
            continue
        arr = np.array(values)
        print(
            f"  {key}: n={len(arr)}  "
            f"min={arr.min():.6g}  max={arr.max():.6g}  "
            f"last={arr[-1]:.6g}"
        )


# ---------------------------------------------------------------------------
# train command
# ---------------------------------------------------------------------------


def train_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Galaxy anomaly detection model"
    )
    parser.add_argument(
        "--data", required=True, help="Training data file (.npz or .csv)"
    )
    parser.add_argument("--config", default=None, help="Config file (.yaml/.yml/.json)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for trained model (.pt). "
        "If not set, model is saved to {run_dir}/model.pt",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume training from checkpoint (.pt) or run directory",
    )
    add_run_args(parser)
    add_config_args(parser)
    add_callback_args(parser)
    return parser.parse_args(argv)


def train_main(argv: list[str] | None = None) -> None:
    """Main logic for the ``galaxy-train`` command."""
    args = train_parse_args(argv)
    quiet = getattr(args, "quiet", False)

    # --- Run directory setup ---
    from unicad_torch.run import (
        create_run_dir,
        resolve_resume_path,
        save_config,
        save_history,
        save_summary,
    )

    run_root = args.run_dir or "runs/"
    run_dir = create_run_dir(run_root, tag=args.run_name)
    if not quiet:
        print(f"Run directory: {run_dir}")

    # Build config (pass run_dir for auto-wiring ModelCheckpoint)
    overrides = build_config_overrides(args, run_dir=run_dir)
    config = GalaxyConfig().replace(**overrides)

    # Save config at the start of training
    save_config(config, run_dir)

    # Resolve resume path (accepts run directory or .pt file)
    resume_path = None
    if args.resume:
        resume_path = resolve_resume_path(args.resume)

    if not quiet:
        print("Galaxy Training")
        print(
            f"Config: k={config.k}, hidden_dim={config.hidden_dim}, "
            f"latent_dim={config.resolved_latent_dim}, "
            f"pretrain={config.pretrain}({config.pretrain_epochs}ep), "
            f"em_iters={config.em_iters}, gravity={config.gravity_version}, "
            f"score={config.score_type}, device={config.device}"
        )
        if config.callbacks:
            cb_names = [type(cb).__name__ for cb in config.callbacks]
            print(f"Callbacks: {', '.join(cb_names)}")

    X, y = load_data(args.data)
    if not quiet:
        print(f"Data: {X.shape[0]} samples, {X.shape[1]} features", end="")
        if y is not None:
            print(f", anomaly_rate={y.mean():.4f}")
        else:
            print()

    model = Galaxy(config)
    t0 = time.perf_counter()
    model.fit(X, y_train=y, resume_from=resume_path)
    fit_time = time.perf_counter() - t0

    if not quiet:
        print(f"\nFit complete in {fit_time:.2f}s")
        print(f"  threshold_={model.threshold_:.4f}")
        print(f"  effective_outlier_ratio_={model.effective_outlier_ratio_:.4f}")
        print(f"  train_score_mean={model._train_score_mean:.4f}")

    # Evaluate on training data when labels are available
    eval_metrics: dict[str, Any] | None = None
    if y is not None:
        eval_result = _evaluate(model, X, y, model.threshold_, "training")
        eval_metrics = {
            k: v
            for k, v in eval_result.items()
            if k
            in (
                "auc_roc",
                "auc_pr",
                "f1",
                "precision",
                "recall",
                "anomaly_rate",
                "n_predicted_anomaly",
                "pred_rate",
            )
        }
        if not quiet and eval_metrics:
            auc_str = f"{eval_metrics.get('auc_roc', float('nan')):.4f}"
            f1_str = f"{eval_metrics.get('f1', float('nan')):.4f}"
            print(f"  AUC-ROC={auc_str}  F1={f1_str}")

    # Save model
    output_path = args.output or str(run_dir / "model.pt")
    model.save(output_path)
    if not quiet:
        print(f"\nModel saved to {output_path}")

    # Auto-save history
    if config.callbacks:
        for cb in config.callbacks:
            if isinstance(cb, History) and cb.history:
                if not quiet:
                    print_history(cb.history)
                save_history(cb.history, run_dir)

    # Save summary
    save_summary(
        run_dir,
        fit_time=fit_time,
        threshold=model.threshold_,
        train_score_mean=model._train_score_mean,
        train_score_std=model._train_score_std,
        effective_outlier_ratio=model.effective_outlier_ratio_,
        outlier_ratio_source=model._outlier_ratio_source,
        cli_args=sys.argv,
        data_path=args.data,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        extra=eval_metrics,
    )
    if not quiet:
        print(f"Summary saved to {run_dir / 'summary.json'}")


# ---------------------------------------------------------------------------
# predict command
# ---------------------------------------------------------------------------


def predict_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained Galaxy model and perform anomaly detection"
    )
    parser.add_argument("--model", required=True, help="Path to saved model (.pt)")
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data", help="Single data file (.npz or .csv)")
    data_group.add_argument(
        "--data-dir", help="Directory with .npz/.csv files (batch mode)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Filter datasets by name (batch mode)",
    )
    parser.add_argument(
        "--device", default=None, help="Device for inference (default: auto-detect)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None, help="Override anomaly threshold"
    )
    parser.add_argument(
        "--labels", action="store_true", help="Print per-sample labels/scores"
    )
    parser.add_argument(
        "--output-scores", default=None, help="Save per-sample scores to CSV"
    )
    parser.add_argument("--output", default=None, help="Save summary results to CSV")
    return parser.parse_args(argv)


def _evaluate(
    model: Galaxy,
    X: np.ndarray,
    y: np.ndarray | None,
    threshold: float | None,
    threshold_source: str = "training",
) -> dict[str, Any]:
    """Score data and compute metrics. Returns a results dict."""
    scores = model.predict_score(X)
    eff_threshold = threshold if threshold is not None else model.threshold_
    pred_labels = (
        (scores > eff_threshold).astype(np.int32) if eff_threshold is not None else None
    )

    result: dict[str, Any] = {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_p50": float(np.median(scores)),
        "threshold": eff_threshold,
        "threshold_source": threshold_source,
        "n_predicted_anomaly": int(pred_labels.sum())
        if pred_labels is not None
        else None,
        "pred_rate": float(pred_labels.mean()) if pred_labels is not None else None,
    }

    train_mean_raw = model.fit_info_.get("train_score_mean")
    train_std_raw = model.fit_info_.get("train_score_std")
    if train_mean_raw is not None and train_std_raw is not None:
        train_mean_val: float = float(train_mean_raw)  # type: ignore[arg-type]
        train_std_val: float = float(train_std_raw)  # type: ignore[arg-type]
        if train_std_val > 0:
            result["score_shift_z"] = (
                float(np.mean(scores)) - train_mean_val
            ) / train_std_val

    if y is not None:
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        result["anomaly_rate"] = float(y.mean())
        result["n_true_anomaly"] = int(y.sum())

        if not (np.isnan(scores).any() or np.isinf(scores).any()):
            try:
                result["auc_roc"] = float(roc_auc_score(y, scores))
            except ValueError:
                result["auc_roc"] = float("nan")
            try:
                result["auc_pr"] = float(average_precision_score(y, scores))
            except ValueError:
                result["auc_pr"] = float("nan")
        else:
            result["auc_roc"] = float("nan")
            result["auc_pr"] = float("nan")

        if pred_labels is not None:
            try:
                result["f1"] = float(f1_score(y, pred_labels, zero_division=0))  # type: ignore[arg-type]
                result["precision"] = float(
                    precision_score(y, pred_labels, zero_division=0)  # type: ignore[arg-type]
                )
                result["recall"] = float(recall_score(y, pred_labels, zero_division=0))  # type: ignore[arg-type]
            except ValueError:
                result["f1"] = float("nan")
                result["precision"] = float("nan")
                result["recall"] = float("nan")

    return result


def _print_single_report(
    name: str, result: dict[str, Any], y: np.ndarray | None
) -> None:
    """Print a detailed report for a single dataset."""
    n = result["n_samples"]
    d = result["n_features"]
    print(f"Dataset: {name}  N={n}  D={d}")

    if y is not None:
        ar = result.get("anomaly_rate", 0)
        print(f"  Anomaly rate: {ar:.2%} ({result.get('n_true_anomaly', '?')}/{n})")

    print(
        f"  Score: mean={result['score_mean']:.4f}  std={result['score_std']:.4f}  "
        f"min={result['score_min']:.4f}  max={result['score_max']:.4f}  "
        f"p50={result['score_p50']:.4f}"
    )

    thr = result["threshold"]
    thr_source = result.get("threshold_source", "training")
    print(f"  Threshold: {thr:.4f} (from {thr_source})")

    n_pred = result.get("n_predicted_anomaly")
    if n_pred is not None:
        print(f"  Predicted anomaly: {n_pred}/{n} ({result['pred_rate']:.2%})")

    shift = result.get("score_shift_z")
    if shift is not None:
        print(f"  Score shift vs training: {shift:+.2f} z")

    if y is not None:
        auc = result.get("auc_roc")
        auc_pr = result.get("auc_pr")
        f1 = result.get("f1")
        prec = result.get("precision")
        rec = result.get("recall")
        if auc is not None:
            print(f"  AUC-ROC: {auc:.4f}")
        if auc_pr is not None:
            print(f"  AUC-PR:  {auc_pr:.4f}")
        if f1 is not None:
            print(f"  F1@threshold: {f1:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}")


def _print_comparison_table(results: list[dict[str, Any]]) -> None:
    """Print a comparison table for batch mode."""
    if not results:
        return

    cols = [
        "dataset",
        "N",
        "D",
        "anomaly_rate",
        "threshold",
        "AUC-ROC",
        "AUC-PR",
        "F1",
        "Precision",
        "Recall",
        "pred_rate",
    ]

    def _fmt_float(val: object) -> str:
        if val is None:
            return "N/A"
        try:
            fval = float(val)  # type: ignore[arg-type]
            if np.isnan(fval):
                return "N/A"
            return f"{fval:.4f}"
        except (TypeError, ValueError):
            return "N/A"

    rows: list[dict[str, str]] = []
    for r in results:
        rows.append(
            {
                "dataset": str(r.get("dataset", "")),
                "N": str(r.get("n_samples", "")),
                "D": str(r.get("n_features", "")),
                "anomaly_rate": _fmt_float(r.get("anomaly_rate")),
                "threshold": _fmt_float(r.get("threshold")),
                "AUC-ROC": _fmt_float(r.get("auc_roc")),
                "AUC-PR": _fmt_float(r.get("auc_pr")),
                "F1": _fmt_float(r.get("f1")),
                "Precision": _fmt_float(r.get("precision")),
                "Recall": _fmt_float(r.get("recall")),
                "pred_rate": _fmt_float(r.get("pred_rate")),
            }
        )

    widths = {c: max(len(c), *(len(row[c]) for row in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(row[c].ljust(widths[c]) for c in cols))
    print(sep)

    valid_aucs = [
        float(r["auc_roc"])
        for r in results
        if isinstance(r["auc_roc"], float) and not np.isnan(r["auc_roc"])
    ]
    if valid_aucs:
        print(
            f"Mean AUC-ROC: {np.mean(valid_aucs):.4f} "
            f"({len(valid_aucs)}/{len(results)} valid)"
        )


def _print_per_sample(
    scores: np.ndarray, threshold: float | None, y: np.ndarray | None
) -> None:
    """Print per-sample score/label table (--labels mode)."""
    labels = (scores > threshold).astype(np.int32) if threshold is not None else None

    print(
        f"{'index':>6}  {'score':>12}  {'label':>6}"
        + (f"  {'true':>6}" if y is not None else "")
    )
    print("-" * (28 + (9 if y is not None else 0)))
    for i in range(len(scores)):
        line = (
            f"{i:>6}  {scores[i]:>12.4f}  {labels[i]:>6}"
            if labels is not None
            else f"{i:>6}  {scores[i]:>12.4f}  {'?':>6}"
        )
        if y is not None:
            line += f"  {int(y[i]):>6}"
        print(line)


def _save_scores(
    path: str,
    scores: np.ndarray,
    threshold: float | None,
    y: np.ndarray | None,
) -> None:
    """Save per-sample scores to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["index", "score"]
        labels = (
            (scores > threshold).astype(np.int32) if threshold is not None else None
        )
        if labels is not None:
            header.append("label")
        if y is not None:
            header.append("true_label")
        writer.writerow(header)
        for i in range(len(scores)):
            row: list[object] = [i, float(scores[i])]
            if labels is not None:
                row.append(int(labels[i]))
            if y is not None:
                row.append(int(y[i]))
            writer.writerow(row)
    print(f"Scores saved to {path}")


def predict_main(argv: list[str] | None = None) -> None:
    """Main logic for the ``galaxy-predict`` command."""
    args = predict_parse_args(argv)

    print(f"Loading model: {args.model}")
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Galaxy.load(args.model, device=device)
    print(f"  {model}")
    if args.threshold is None:
        print(f"  Training threshold: {model.threshold_:.4f}")
    else:
        print(f"  Custom threshold: {args.threshold:.4f}")
    print()

    threshold = args.threshold if args.threshold is not None else model.threshold_
    threshold_source = "custom" if args.threshold is not None else "training"

    if args.data:
        X, y = load_data(args.data)
        name = os.path.splitext(os.path.basename(args.data))[0]

        if args.labels:
            scores = model.predict_score(X)
            _print_per_sample(scores, threshold, y)
        elif args.output_scores:
            scores = model.predict_score(X)
            _save_scores(args.output_scores, scores, threshold, y)
        else:
            result = _evaluate(model, X, y, threshold, threshold_source)
            result["dataset"] = name
            _print_single_report(name, result, y)

    else:
        from unicad_torch.datasets import find_datasets

        datasets = find_datasets(
            args.data_dir, args.datasets, extensions=(".npz", ".csv")
        )
        if not datasets:
            print("No datasets found.")
            sys.exit(1)

        print(f"Evaluating {len(datasets)} dataset(s)...")
        print()

        results: list[dict[str, Any]] = []
        for i, (name, path) in enumerate(datasets, 1):
            print(f"[{i}/{len(datasets)}] {name} ...", end=" ", flush=True)
            try:
                X, y = load_data(path)
                result = _evaluate(model, X, y, threshold, threshold_source)
                result["dataset"] = name
                results.append(result)
                auc = result.get("auc_roc")
                try:
                    auc_val = float(auc) if auc is not None else None
                    auc_str = (
                        f"{auc_val:.4f}"
                        if auc_val is not None and not np.isnan(auc_val)
                        else "N/A"
                    )
                except (TypeError, ValueError):
                    auc_str = "N/A"
                print(f"AUC-ROC={auc_str}  fit_info=OK")
            except Exception as e:
                print(f"FAILED: {e}")

        print()
        _print_comparison_table(results)

        if args.output:
            save_csv(results, args.output)


# ---------------------------------------------------------------------------
# benchmark command
# ---------------------------------------------------------------------------


def benchmark_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Galaxy on ADBench datasets")
    parser.add_argument(
        "--data-dir",
        default="data/Classical",
        help="Directory containing .npz datasets (default: data/Classical)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Run on specific datasets by name (e.g. 38_thyroid 2_annthyroid)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to CSV file (default: saved to run directory)",
    )
    add_run_args(parser)
    add_config_args(parser)
    add_callback_args(parser)
    return parser.parse_args(argv)


def _benchmark_single(
    name: str,
    path: str,
    config: GalaxyConfig,
) -> dict[str, Any]:
    """Run Galaxy on a single dataset and return metrics dict."""
    data = np.load(path)
    X = data["X"].astype(np.float32)
    y = data["y"].ravel()

    n_samples, n_features = X.shape
    anomaly_rate = y.mean()

    model = Galaxy(config)

    t0 = time.perf_counter()
    model.fit(X)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = model.predict_score(X)
    score_time = time.perf_counter() - t0

    threshold = model.threshold_

    # Extract history if History callback was used
    history_data: dict | None = None
    if config.callbacks:
        for cb in config.callbacks:
            if isinstance(cb, History):
                history_data = cb.history
                break

    has_nan = np.isnan(scores).any()
    has_inf = np.isinf(scores).any()

    if has_nan or has_inf:
        auc = float("nan")
        status = "NaN" if has_nan else "Inf"
    else:
        try:
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(y, scores)
            status = "OK"
        except ValueError as e:
            auc = float("nan")
            status = f"Error: {e}"

    return {
        "dataset": name,
        "n_samples": n_samples,
        "n_features": n_features,
        "anomaly_rate": f"{anomaly_rate:.4f}",
        "threshold": f"{threshold:.4f}" if threshold is not None else "N/A",
        "auc": auc,
        "fit_time": f"{fit_time:.2f}",
        "score_time": f"{score_time:.2f}",
        "status": status,
        "history": history_data,
    }


def _benchmark_print_table(results: list[dict[str, Any]]) -> None:
    """Print results as a formatted table."""
    if not results:
        print("No results.")
        return

    cols = [
        "dataset",
        "n_samples",
        "n_features",
        "anomaly_rate",
        "threshold",
        "auc",
        "fit_time",
        "status",
    ]
    widths = {c: max(len(str(r.get(c, ""))) for r in results) for c in cols}
    widths = {c: max(widths[c], len(c)) for c in cols}

    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)

    for r in results:
        row = " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols)
        print(row)

    print(sep)
    valid_aucs = [
        r["auc"]
        for r in results
        if isinstance(r["auc"], float) and not np.isnan(r["auc"])
    ]
    if valid_aucs:
        print(
            f"Mean AUC: {np.mean(valid_aucs):.4f} ({len(valid_aucs)}/{len(results)} valid)"
        )


def benchmark_main(argv: list[str] | None = None) -> None:
    """Main logic for the ``galaxy-benchmark`` command."""
    args = benchmark_parse_args(argv)
    quiet = getattr(args, "quiet", False)

    # --- Run directory setup ---
    from unicad_torch.run import (
        create_run_dir,
        save_config,
        save_summary,
    )

    run_root = args.run_dir or "runs/"
    run_dir = create_run_dir(run_root, tag=args.run_name or "benchmark")
    if not quiet:
        print(f"Run directory: {run_dir}")

    overrides = build_config_overrides(args)
    config = GalaxyConfig().replace(**overrides)

    # Save config at the start
    save_config(config, run_dir)

    from unicad_torch.datasets import find_datasets

    datasets = find_datasets(args.data_dir, args.datasets)
    if not datasets:
        print(
            "No datasets found. Download with: uv run python scripts/download_data.py"
        )
        sys.exit(1)

    if not quiet:
        print(f"Galaxy Benchmark — {len(datasets)} dataset(s)")
        print(
            f"Config: k={config.k}, hidden_dim={config.hidden_dim}, "
            f"pretrain={config.pretrain}({config.pretrain_epochs}ep), "
            f"em_iters={config.em_iters}, gravity={config.gravity_version}, "
            f"score={config.score_type}, device={config.device}"
        )
        if config.callbacks:
            cb_names = [type(cb).__name__ for cb in config.callbacks]
            print(f"Callbacks: {', '.join(cb_names)}")
        print()

    results: list[dict[str, Any]] = []
    for i, (name, path) in enumerate(datasets, 1):
        if not quiet:
            print(f"[{i}/{len(datasets)}] {name} ...", end=" ", flush=True)
        try:
            r = _benchmark_single(name, path, config)
            results.append(r)
            if not quiet:
                auc_str = (
                    f"{r['auc']:.4f}"
                    if isinstance(r["auc"], float) and not np.isnan(r["auc"])
                    else str(r["auc"])
                )
                print(f"AUC={auc_str}  ({r['status']})  fit={r['fit_time']}s")
        except Exception as e:
            if not quiet:
                print(f"FAILED: {e}")
            results.append(
                {
                    "dataset": name,
                    "n_samples": "?",
                    "n_features": "?",
                    "anomaly_rate": "?",
                    "threshold": "N/A",
                    "auc": float("nan"),
                    "fit_time": "?",
                    "score_time": "?",
                    "status": f"Exception: {e}",
                }
            )

    if not quiet:
        print()
        _benchmark_print_table(results)

    # Save results to run directory (and optional --output path)
    results_path = str(run_dir / "results.csv")
    save_csv(results, results_path)
    if args.output and args.output != results_path:
        save_csv(results, args.output)

    # Save summary
    valid_aucs = [
        r["auc"]
        for r in results
        if isinstance(r["auc"], float) and not np.isnan(r["auc"])
    ]
    save_summary(
        run_dir,
        data_path=args.data_dir,
        n_samples=len(datasets),
        train_score_mean=float(np.mean(valid_aucs)) if valid_aucs else None,
        cli_args=sys.argv,
    )
    if not quiet:
        print(f"Results saved to {results_path}")
        print(f"Summary saved to {run_dir / 'summary.json'}")


# ---------------------------------------------------------------------------
# Console script entry points
# ---------------------------------------------------------------------------


def train() -> None:
    """Entry point for ``galaxy-train``."""
    train_main()


def predict() -> None:
    """Entry point for ``galaxy-predict``."""
    predict_main()


def benchmark() -> None:
    """Entry point for ``galaxy-benchmark``."""
    benchmark_main()


# ---------------------------------------------------------------------------
# runs command
# ---------------------------------------------------------------------------


def runs_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List and inspect Galaxy training runs"
    )
    parser.add_argument(
        "--run-dir",
        default="runs/",
        help="Root directory containing runs (default: runs/)",
    )
    parser.add_argument(
        "--detail",
        default=None,
        help="Show details for a specific run (run name or path)",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        help="Compare specific runs by name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max number of runs to list (default: 20)",
    )
    return parser.parse_args(argv)


def runs_main(argv: list[str] | None = None) -> None:
    """Main logic for the ``galaxy-runs`` command."""
    args = runs_parse_args(argv)

    from unicad_torch.run import find_runs, load_summary

    if args.detail:
        run_path = Path(args.detail)
        if not run_path.is_dir():
            run_path = Path(args.run_dir) / args.detail
        if not run_path.is_dir():
            print(f"Run not found: {args.detail}")
            sys.exit(1)

        summary = load_summary(run_path)
        print(f"Run: {run_path.name}")
        print(f"Path: {run_path}")
        for key, val in summary.items():
            print(f"  {key}: {val}")

        print("\nFiles:")
        for f in sorted(run_path.iterdir()):
            prefix = "  [dir] " if f.is_dir() else "  "
            print(f"{prefix}{f.name}")
        return

    if args.compare:
        root = Path(args.run_dir)
        summaries: list[tuple[str, dict[str, Any]]] = []
        for name in args.compare:
            run_path = root / name
            if run_path.is_dir():
                summaries.append((name, load_summary(run_path)))
            else:
                print(f"Warning: run '{name}' not found")

        if not summaries:
            print("No runs found to compare.")
            return

        keys_to_compare = [
            "threshold",
            "train_score_mean",
            "train_score_std",
            "effective_outlier_ratio",
            "fit_duration_seconds",
        ]
        header = f"{'field':<28}" + "".join(f"{name:<20}" for name, _ in summaries)
        print(header)
        print("-" * len(header))
        for key in keys_to_compare:
            row = f"{key:<28}"
            for _, summary in summaries:
                val = summary.get(key, "N/A")
                row += f"{str(val):<20}"
            print(row)
        return

    # Default: list recent runs
    runs = find_runs(args.run_dir)
    if not runs:
        print(f"No runs found in {args.run_dir}/")
        return

    runs = runs[: args.limit]

    print(
        f"{'run_name':<30} {'threshold':>10} {'score_mean':>12} "
        f"{'duration':>10} {'timestamp':<22}"
    )
    print("-" * 88)

    for run_dir in runs:
        summary = load_summary(run_dir)
        name = summary.get("run_name", run_dir.name)
        threshold = summary.get("threshold", "N/A")
        score_mean = summary.get("train_score_mean", "N/A")
        duration = summary.get("fit_duration_seconds", "N/A")
        ts = summary.get("timestamp", "N/A")

        thr_str = f"{threshold:.4f}" if isinstance(threshold, (int, float)) else "N/A"
        mean_str = (
            f"{score_mean:.4f}" if isinstance(score_mean, (int, float)) else "N/A"
        )
        dur_str = f"{duration:.1f}s" if isinstance(duration, (int, float)) else "N/A"
        ts_str = ts[:19] if isinstance(ts, str) else "N/A"

        print(f"{name:<30} {thr_str:>10} {mean_str:>12} {dur_str:>10} {ts_str:<22}")

    print(f"\n({len(runs)} run(s) shown, use --detail <name> for more info)")


def runs() -> None:
    """Entry point for ``galaxy-runs``."""
    runs_main()
