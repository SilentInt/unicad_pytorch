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

# Run tests
uv run pytest tests/ -v

# Run diagnostic test (module-by-module with real data)
uv run python scripts/test_modules.py

# Lint / format / type check
uv run ruff check src/ scripts/
uv run ruff format --check src/ scripts/
uv run pyright src/ scripts/

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
- `fit(X_train, y_train=None)` — train the four-stage pipeline; when `y_train` is provided, the true anomaly rate is used for EM exclusion and threshold computation (stored as `effective_outlier_ratio_`, config is NOT mutated)
- `predict_score(X)` — return continuous anomaly scores (`np.float32`)
- `predict(X)` — return binary 0/1 labels using `threshold_` from training (`np.int32`)
- `fit_predict(X_train, y_train=None)` — fit then return training scores
- `save(path)` / `Galaxy.load(path, device="cpu")` — persist/restore all fitted state
- `threshold_` — absolute anomaly threshold (quantile `1 - effective_outlier_ratio_` of training scores)
- `effective_outlier_ratio_` — the outlier ratio actually used in fit (from config or labels)
- `fit_info_` — dict with `train_score_mean`, `train_score_std`, `threshold_`, `outlier_ratio`, `outlier_ratio_source` ("labels" or "config"), `n_excluded_per_iter`
- `input_dim` — number of features from training data
- `__repr__` — shows `Galaxy(not fitted)` or `Galaxy(input_dim=..., k=..., threshold_=...)`

**`GalaxyConfig`** (`config.py`) — dataclass with `__post_init__` validation and `replace(**overrides)` method. Validates device string, warns on gravity/score mismatch.

**`GalaxyADBench`** (`adapter.py`) — ADBench-compatible wrapper. Does NOT forward `y_train` to `Galaxy.fit()` to ensure purely unsupervised evaluation per ADBench protocol.

## Architecture

Four-stage pipeline orchestrated by `Galaxy`:

1. **Preprocessing** (`preprocessing.py`): `StandardScaler` (z-score, `correction=0`) or `RowScaler` (L2 per-row norm), selected via `GalaxyConfig.preprocess`.

2. **Autoencoder Pretraining** (`model.py`): `Encoder` -> `Decoder` with MSE(sum) loss, Adam + StepLR. `pretrain_autoencoder()` trains for 200 epochs. Supports `verbose` loss logging.

3. **Iterative EM** (`em.py`): `GalaxyEM.fit()` runs `em_iters` rounds, each:
   - **Exclude outliers**: GOF-score all data, remove top `outlier_ratio`%
   - **Update network**: fine-tune autoencoder with reconstruction + gravity loss
   - **Update prototypes**: encode data -> fit SMM -> update means/weights/covars

4. **GOF Scoring** (`gof.py`): Anomaly score = `-log(force)`, computed in log-space. Scalar mode: `-logsumexp(log F_ik)`. Vector mode: `-[log(||force_vec||) + max_log_force]`.

**Shared computation** (`gravity.py`): Single source of truth for gravitational force computation. Exports `mahalanobis_diag`, `compute_log_forces`, `aggregate_force_scalar`, `aggregate_force_vector`, and `VAR_FLOOR`. Used by `smm_torch.py`, `gof.py`, and `em.py`.

**Supporting modules**:
- `config.py`: `GalaxyConfig` dataclass — hyperparameters with validation, device check, mismatch warning
- `adapter.py`: `GalaxyADBench` — ADBench-compatible wrapper (labels NOT forwarded)
- `smm_torch.py`: `SMMTorch` — GPU-native Student-t Mixture Model (ν=1); device-aware Generator; relative tolerance convergence
- `datasets.py`: `find_datasets()` — shared dataset discovery for scripts

## Critical Implementation Details

- **Log-space scoring**: GOF scores are `-log(force)`, not `1/force`. Avoids division-by-zero, keeps scores finite. Ranking identical for AUC.
- **gravity.py is the single source of truth**: All log-force computation (SMM E-step, GOF scoring, EM gravity loss) shares the same `compute_log_forces` function. The SMM E-step uses `mahalanobis_diag` directly (its `log_resp` lacks the `-log(π)` term that GOF/EM include).
- **VAR_FLOOR constant**: Defined once in `gravity.py` (`1e-6`), imported everywhere. No duplicate definitions.
- **Full-dataset re-scoring**: EM exclude step always re-scores the full X, not the filtered subset.
- **Gravity version + score type**: Default is `vector`/`vector` (matched pair). Scalar gravity with vector scoring triggers a `UserWarning` from `GalaxyConfig`.
- **Absolute threshold**: `predict()` uses `threshold_` learned from training scores (quantile `1 - effective_outlier_ratio_`), not relative to the test batch.
- **Label-informed outlier ratio**: When `y_train` is provided to `fit()`, the true anomaly rate (`y_train.mean()`) is stored as `effective_outlier_ratio_` for EM exclusion and threshold computation. The original `config.outlier_ratio` is NOT mutated. Stored in `fit_info_["outlier_ratio_source"]` as "labels" or "config".
- **ADBench label isolation**: `GalaxyADBench.fit(X, y)` does NOT forward `y_train` to ensure purely unsupervised evaluation.
- **Reproducibility**: `Galaxy.fit()` calls `torch.manual_seed(config.seed)` at the start, covering model initialization and DataLoader shuffle.
- **Input requirements**: float32 preferred (float64 triggers warning); no NaN/Inf; minimum `k` samples.
- **latent_dim**: `GalaxyConfig.latent_dim` defaults to `None` (equals `hidden_dim`), can be set independently for bottleneck control.

## GalaxyConfig Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `seed` | int | 42 | Random seed (used in `Galaxy.fit()` for reproducibility) |
| `k` | int | 10 | Number of clusters |
| `outlier_ratio` | float | 0.01 | Expected anomaly fraction, in [0, 1); may be overridden by labels at fit time |
| `hidden_dim` | int | 128 | Autoencoder hidden dimension |
| `latent_dim` | int\|None | None | Autoencoder latent dimension; None = equals hidden_dim |
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
| `device` | str | "cpu" | Torch device (validated on creation) |
| `verbose` | bool | False | Print training progress |

## Data

ADBench `.npz` files (with `X` features and `y` 0/1 labels) in `data/`. Also supports `.csv` with a `y` column. Download via `scripts/download_data.py`.
