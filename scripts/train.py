#!/usr/bin/env python3
"""Train a Galaxy anomaly detection model.

Usage
-----
    # Basic training
    uv run python scripts/train.py --data data/Classical/38_thyroid.npz --output galaxy.pt

    # With all callbacks
    uv run python scripts/train.py --data data/Classical/38_thyroid.npz --output galaxy.pt \\
        --history --early-stopping --tqdm --checkpoint --checkpoint-dir ckpt/

    # Custom hyperparameters
    uv run python scripts/train.py --data data/Classical/38_thyroid.npz --output galaxy.pt \\
        --k 5 --em-iters 5 --device cuda

    # Resume from checkpoint
    uv run python scripts/train.py --data data/Classical/38_thyroid.npz --output galaxy.pt \\
        --resume checkpoints/latest.pt

    # Using a config file (YAML or JSON)
    uv run python scripts/train.py --config train_config.yaml --output galaxy.pt
"""

from unicad_torch.cli import train_main

if __name__ == "__main__":
    train_main()
