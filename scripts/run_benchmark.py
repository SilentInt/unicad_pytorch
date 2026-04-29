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

    # With callbacks
    uv run python scripts/run_benchmark.py --tqdm --history --early-stopping
"""

from unicad_torch.cli import benchmark_main

if __name__ == "__main__":
    benchmark_main()
