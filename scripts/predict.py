#!/usr/bin/env python3
"""Load a trained Galaxy model and perform anomaly detection on new data.

Usage
-----
    # Score a single dataset (with labels for evaluation)
    uv run python scripts/predict.py --model galaxy.pt --data data/Classical/38_thyroid.npz

    # Batch evaluation on a directory
    uv run python scripts/predict.py --model galaxy.pt --data-dir data/Classical

    # Filter specific datasets
    uv run python scripts/predict.py --model galaxy.pt --data-dir data/Classical --datasets 38_thyroid 2_annthyroid

    # Output binary labels instead of scores
    uv run python scripts/predict.py --model galaxy.pt --data data/test.csv --labels

    # Save per-sample scores to CSV
    uv run python scripts/predict.py --model galaxy.pt --data data/test.csv --output-scores scores.csv

    # Custom threshold (override training threshold)
    uv run python scripts/predict.py --model galaxy.pt --data data/test.npz --threshold -600.0

    # Use GPU
    uv run python scripts/predict.py --model galaxy.pt --data data/test.csv --device cuda
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from unicad_torch.datasets import find_datasets
from unicad_torch.galaxy import Galaxy


# ---------------------------------------------------------------------------
# Data loading
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
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    model: Galaxy,
    X: np.ndarray,
    y: np.ndarray | None,
    threshold: float | None,
    threshold_source: str = "training",
) -> dict[str, object]:
    """Score data and compute metrics. Returns a results dict."""
    scores = model.predict_score(X)
    eff_threshold = threshold if threshold is not None else model.threshold_
    pred_labels = (
        (scores > eff_threshold).astype(np.int32) if eff_threshold is not None else None
    )

    result: dict[str, object] = {
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

    # Score shift vs training distribution
    train_mean_raw = model.fit_info_.get("train_score_mean")
    train_std_raw = model.fit_info_.get("train_score_std")
    if train_mean_raw is not None and train_std_raw is not None:
        train_mean_val = float(train_mean_raw)  # type: ignore[arg-type]
        train_std_val = float(train_std_raw)  # type: ignore[arg-type]
        if train_std_val > 0:
            result["score_shift_z"] = (
                float(np.mean(scores)) - train_mean_val
            ) / train_std_val

    if y is not None:
        result["anomaly_rate"] = float(y.mean())
        result["n_true_anomaly"] = int(y.sum())

        # AUC-ROC
        if not (np.isnan(scores).any() or np.isinf(scores).any()):
            try:
                result["auc_roc"] = float(roc_auc_score(y, scores))
            except ValueError:
                result["auc_roc"] = float("nan")

            # AUC-PR
            try:
                result["auc_pr"] = float(average_precision_score(y, scores))
            except ValueError:
                result["auc_pr"] = float("nan")
        else:
            result["auc_roc"] = float("nan")
            result["auc_pr"] = float("nan")

        # F1 / Precision / Recall at threshold
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


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def print_single_report(
    name: str, result: dict[str, object], y: np.ndarray | None
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


def print_comparison_table(results: list[dict[str, object]]) -> None:
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

    # Build display values
    def _fmt_float(val: object, key: str) -> str:
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
                "anomaly_rate": _fmt_float(r.get("anomaly_rate"), "anomaly_rate"),
                "threshold": _fmt_float(r.get("threshold"), "threshold"),
                "AUC-ROC": _fmt_float(r.get("auc_roc"), "auc_roc"),
                "AUC-PR": _fmt_float(r.get("auc_pr"), "auc_pr"),
                "F1": _fmt_float(r.get("f1"), "f1"),
                "Precision": _fmt_float(r.get("precision"), "precision"),
                "Recall": _fmt_float(r.get("recall"), "recall"),
                "pred_rate": _fmt_float(r.get("pred_rate"), "pred_rate"),
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

    # Mean AUC
    valid_aucs = [
        float(r["auc_roc"])
        for r in results
        if isinstance(r["auc_roc"], float) and not np.isnan(r["auc_roc"])
    ]
    if valid_aucs:
        print(
            f"Mean AUC-ROC: {np.mean(valid_aucs):.4f} ({len(valid_aucs)}/{len(results)} valid)"
        )


def print_per_sample(
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


def save_scores(
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


def save_results_csv(results: list[dict[str, object]], path: str) -> None:
    """Save summary results to CSV."""
    if not results:
        return
    cols = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained Galaxy model and perform anomaly detection"
    )
    parser.add_argument("--model", required=True, help="Path to saved model (.pt)")
    # Data source (mutually exclusive group)
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
        "--device", default="cpu", help="Device for inference (default: cpu)"
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    model = Galaxy.load(args.model, device=args.device)
    print(f"  {model}")
    if args.threshold is None:
        print(f"  Training threshold: {model.threshold_:.4f}")
    else:
        print(f"  Custom threshold: {args.threshold:.4f}")
    print()

    threshold = args.threshold if args.threshold is not None else model.threshold_
    threshold_source = "custom" if args.threshold is not None else "training"

    # Single file mode
    if args.data:
        X, y = load_data(args.data)
        name = os.path.splitext(os.path.basename(args.data))[0]

        if args.labels:
            scores = model.predict_score(X)
            print_per_sample(scores, threshold, y)
        elif args.output_scores:
            scores = model.predict_score(X)
            save_scores(args.output_scores, scores, threshold, y)
        else:
            result = evaluate(model, X, y, threshold, threshold_source)
            result["dataset"] = name
            print_single_report(name, result, y)

    # Batch mode
    else:
        datasets = find_datasets(
            args.data_dir, args.datasets, extensions=(".npz", ".csv")
        )
        if not datasets:
            print("No datasets found.")
            sys.exit(1)

        print(f"Evaluating {len(datasets)} dataset(s)...")
        print()

        results: list[dict[str, object]] = []
        for i, (name, path) in enumerate(datasets, 1):
            print(f"[{i}/{len(datasets)}] {name} ...", end=" ", flush=True)
            try:
                X, y = load_data(path)
                result = evaluate(model, X, y, threshold, threshold_source)
                result["dataset"] = name
                results.append(result)
                auc = result.get("auc_roc")
                try:
                    auc_val = float(auc) if auc is not None else None  # type: ignore[arg-type]
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
        print_comparison_table(results)

        if args.output:
            save_results_csv(results, args.output)


if __name__ == "__main__":
    main()
