#!/usr/bin/env python3
"""Benchmark Galaxy on ADBench datasets.

Usage
-----
    # Run on all Classical datasets
    uv run python scripts/run_benchmark.py

    # Run on a single dataset
    uv run python scripts/run_benchmark.py --datasets 38_thyroid

    # Run on multiple datasets
    uv run python scripts/run_benchmark.py --datasets 38_thyroid 2_annthyroid 14_glass

    # Custom config overrides
    uv run python scripts/run_benchmark.py --k 5 --em-iters 5 --device cuda

    # Save results to CSV
    uv run python scripts/run_benchmark.py --output results.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from unicad_torch import Galaxy, GalaxyConfig


def find_datasets(
    data_dir: str, datasets: list[str] | None = None
) -> list[tuple[str, str]]:
    """Find .npz files, optionally filtered by name. Returns (name, path) pairs."""
    results: list[tuple[str, str]] = []
    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
        return results

    for root, _dirs, files in os.walk(data_dir):
        for f in sorted(files):
            if not f.endswith(".npz"):
                continue
            name = f[:-4]  # strip .npz
            path = os.path.join(root, f)
            if datasets is None or name in datasets or f in datasets:
                results.append((name, path))

    if datasets is not None:
        requested = set(datasets)
        found = {name for name, _ in results}
        missing = requested - found
        if missing:
            print(f"Warning: datasets not found: {', '.join(sorted(missing))}")

    return results


def run_single(
    name: str,
    path: str,
    config: GalaxyConfig,
) -> dict[str, object]:
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

    has_nan = np.isnan(scores).any()
    has_inf = np.isinf(scores).any()

    if has_nan or has_inf:
        auc = float("nan")
        status = "NaN" if has_nan else "Inf"
    else:
        try:
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
    }


def print_table(results: list[dict[str, object]]) -> None:
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


def save_csv(results: list[dict[str, object]], path: str) -> None:
    """Save results to CSV."""
    import csv

    if not results:
        return
    cols = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {path}")


def parse_args() -> argparse.Namespace:
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
        help="Save results to CSV file",
    )
    # Config overrides
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--em-iters", type=int, default=None)
    parser.add_argument("--em-finetune-steps", type=int, default=None)
    parser.add_argument("--outlier-ratio", type=float, default=None)
    parser.add_argument(
        "--preprocess", choices=["z-score", "row-norm", "none"], default=None
    )
    parser.add_argument("--gravity-version", choices=["scalar", "vector"], default=None)
    parser.add_argument("--score-type", choices=["scalar", "vector"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build config from overrides using replace()
    overrides: dict[str, object] = {}
    if args.k is not None:
        overrides["k"] = args.k
    if args.hidden_dim is not None:
        overrides["hidden_dim"] = args.hidden_dim
    if args.pretrain_epochs is not None:
        overrides["pretrain_epochs"] = args.pretrain_epochs
    if args.em_iters is not None:
        overrides["em_iters"] = args.em_iters
    if args.em_finetune_steps is not None:
        overrides["em_finetune_steps"] = args.em_finetune_steps
    if args.outlier_ratio is not None:
        overrides["outlier_ratio"] = args.outlier_ratio
    if args.preprocess is not None:
        overrides["preprocess"] = args.preprocess
    if args.gravity_version is not None:
        overrides["gravity_version"] = args.gravity_version
    if args.score_type is not None:
        overrides["score_type"] = args.score_type
    if args.device is not None:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.no_pretrain:
        overrides["pretrain"] = False
    if args.verbose:
        overrides["verbose"] = True

    config = GalaxyConfig().replace(**overrides)

    datasets = find_datasets(args.data_dir, args.datasets)
    if not datasets:
        print(
            "No datasets found. Download with: uv run python scripts/download_data.py"
        )
        sys.exit(1)

    print(f"Galaxy Benchmark — {len(datasets)} dataset(s)")
    print(
        f"Config: k={config.k}, hidden_dim={config.hidden_dim}, "
        f"pretrain={config.pretrain}({config.pretrain_epochs}ep), "
        f"em_iters={config.em_iters}, gravity={config.gravity_version}, "
        f"score={config.score_type}, device={config.device}"
    )
    print()

    results: list[dict[str, object]] = []
    for i, (name, path) in enumerate(datasets, 1):
        print(f"[{i}/{len(datasets)}] {name} ...", end=" ", flush=True)
        try:
            r = run_single(name, path, config)
            results.append(r)
            auc_str = (
                f"{r['auc']:.4f}"
                if isinstance(r["auc"], float) and not np.isnan(r["auc"])
                else str(r["auc"])
            )
            print(f"AUC={auc_str}  ({r['status']})  fit={r['fit_time']}s")
        except Exception as e:
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

    print()
    print_table(results)

    if args.output:
        save_csv(results, args.output)


if __name__ == "__main__":
    main()
