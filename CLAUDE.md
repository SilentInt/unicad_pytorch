# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unicad-torch is a PyTorch reimplementation of the Galaxy anomaly detection model ("Towards a Unified Framework of Clustering-based Anomaly Detection"). It uses a gravitational analogy where cluster centers exert "gravity" on data points; anomalies receive less gravitational force and thus score higher.

Standalone reimplementation — no PyTorch Lightning, ADBench, or TensorFlow dependency.

## Quick Start

```python
from unicad_torch import Galaxy, GalaxyConfig

# Train
config = GalaxyConfig(device="cpu", verbose=True)
model = Galaxy(config)
model.fit(X_train)                        # X_train: np.ndarray (N, D), float32

# Score
scores = model.predict_score(X_test)      # continuous scores, higher = more anomalous
labels = model.predict(X_test)            # 0/1 labels via training threshold

# Persist
model.save("galaxy.pt")
model = Galaxy.load("galaxy.pt", device="cpu")

# Inspect
print(model)                              # Galaxy(input_dim=32, k=10, threshold_=1.23)
print(model.threshold_)                   # absolute anomaly threshold from training
print(model.fit_info_)                    # {"train_score_mean": ..., "n_excluded_per_iter": [...]}
```

## Commands

```bash
# Install (with dev tools)
uv sync --extra dev

# Run diagnostic test
uv run python scripts/test_modules.py

# Lint / format / type check
uv run ruff check src/ scripts/
uv run ruff format --check src/ scripts/
uv run pyright src/

# Download ADBench datasets
uv run python scripts/download_data.py --category Classical

# Run benchmarks
uv run python scripts/run_benchmark.py --data-dir data/Classical
uv run python scripts/run_benchmark.py --datasets 38_thyroid --verbose

# Anomaly detection (load model → score → evaluate)
uv run python scripts/predict.py --model galaxy.pt --data data/Classical/38_thyroid.npz
uv run python scripts/predict.py --model galaxy.pt --data-dir data/Classical
uv run python scripts/predict.py --model galaxy.pt --data test.csv --labels
uv run python scripts/predict.py --model galaxy.pt --data test.csv --output-scores scores.csv
```

## Public API

**`Galaxy`** (`galaxy.py`) — main model class:
- `fit(X_train, y_train=None)` — train the four-stage pipeline; when `y_train` is provided, the true anomaly rate overrides `outlier_ratio` for EM exclusion and threshold computation
- `predict_score(X)` — return continuous anomaly scores (`np.float32`)
- `predict(X)` — return binary 0/1 labels using `threshold_` from training (`np.int32`)
- `fit_predict(X_train, y_train=None)` — fit then return training scores
- `save(path)` / `Galaxy.load(path, device="cpu")` — persist/restore all fitted state
- `threshold_` — absolute anomaly threshold (99th percentile of training scores by default)
- `fit_info_` — dict with `train_score_mean`, `train_score_std`, `threshold_`, `outlier_ratio`, `outlier_ratio_source` ("labels" or "config"), `n_excluded_per_iter`
- `input_dim` — number of features from training data
- `__repr__` — shows `Galaxy(not fitted)` or `Galaxy(input_dim=..., k=..., threshold_=...)`

**`GalaxyConfig`** (`config.py`) — dataclass with `__post_init__` validation and `replace(**overrides)` method.

**`GalaxyADBench`** (`adapter.py`) — ADBench-compatible wrapper exposing `fit`, `predict_score`, `predict`, `save`, `load`, `threshold_`, `fit_info_`.

## Architecture

Four-stage pipeline orchestrated by `Galaxy`:

1. **Preprocessing** (`preprocessing.py`): `StandardScaler` (z-score, `correction=0`) or `RowScaler` (L2 per-row norm), selected via `GalaxyConfig.preprocess`.

2. **Autoencoder Pretraining** (`model.py`): `Encoder` -> `Decoder` with MSE(sum) loss, Adam + StepLR. `pretrain_autoencoder()` trains for 200 epochs. Supports `verbose` loss logging.

3. **Iterative EM** (`em.py`): `GalaxyEM.fit()` runs `em_iters` rounds, each:
   - **Exclude outliers**: GOF-score all data, remove top `outlier_ratio`%
   - **Update network**: fine-tune autoencoder with reconstruction + gravity loss
   - **Update prototypes**: encode data -> fit SMM -> update means/weights/covars

4. **GOF Scoring** (`gof.py`): Anomaly score = `-log(force)`, computed in log-space. Scalar mode: `-logsumexp(log F_ik)`. Vector mode: `-[log(||force_vec||) + max_log_force]`.

**Supporting modules**:
- `config.py`: `GalaxyConfig` dataclass — hyperparameters with validation
- `adapter.py`: `GalaxyADBench` — ADBench-compatible wrapper
- `smm_torch.py`: `SMMTorch` — GPU-native Student-t Mixture Model (ν=1); device-aware Generator; relative tolerance convergence

## Critical Implementation Details

- **Log-space scoring**: GOF scores are `-log(force)`, not `1/force`. Avoids division-by-zero, keeps scores finite. Ranking identical for AUC.
- **float64 det_covars**: `em.py` computes determinant in float64 to avoid underflow. `gof.py` uses `torch.log(covars).sum()` instead.
- **Full-dataset re-scoring**: EM exclude step always re-scores the full X, not the filtered subset.
- **Gravity version + score type**: Default is `vector`/`vector` (matched pair). Scalar gravity with vector scoring is a mismatch that wastes training effect.
- **Absolute threshold**: `predict()` uses `threshold_` learned from training scores (quantile `1 - outlier_ratio`), not relative to the test batch.
- **Label-informed outlier ratio**: When `y_train` is provided to `fit()`, the true anomaly rate (`y_train.mean()`) overrides `config.outlier_ratio` for EM exclusion and threshold computation. Stored in `fit_info_["outlier_ratio_source"]` as "labels" or "config".
- **Input requirements**: float32 preferred (float64 triggers warning); no NaN/Inf; minimum `k` samples.

## GalaxyConfig Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `seed` | int | 42 | Random seed |
| `k` | int | 10 | Number of clusters |
| `outlier_ratio` | float | 0.01 | Expected anomaly fraction, in [0, 1); overridden by `y_train.mean()` when labels provided |
| `hidden_dim` | int | 128 | Autoencoder hidden/latent dimension |
| `pretrain_epochs` | int | 200 | AE pretraining epochs |
| `pretrain_lr` | float | 3e-3 | AE learning rate |
| `pretrain_batch_size` | int | 1024 | AE batch size |
| `em_iters` | int | 3 | EM iterations |
| `em_finetune_steps` | int | 100 | Fine-tune steps per EM iteration |
| `em_finetune_lr` | float | 3e-4 | Fine-tune learning rate |
| `preprocess` | str | "z-score" | "z-score", "row-norm", or "none" |
| `gravity_version` | str | "vector" | "scalar" or "vector" |
| `score_type` | str | "vector" | "scalar" or "vector" |
| `pretrain` | bool | True | Skip AE pretraining if False |
| `smm_n_iter` | int | 100 | SMM EM iterations |
| `smm_tol` | float | 1e-3 | SMM convergence tolerance (relative) |
| `device` | str | "cpu" | Torch device |
| `verbose` | bool | False | Print training progress |

## Data

ADBench `.npz` files (with `X` features and `y` 0/1 labels) in `data/`. Also supports `.csv` with a `y` column. Download via `scripts/download_data.py`.
