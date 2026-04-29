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

from unicad_torch.cli import predict_main

if __name__ == "__main__":
    predict_main()
