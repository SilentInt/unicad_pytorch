# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unicad-torch is a PyTorch reimplementation of the Galaxy anomaly detection model ("Towards a Unified Framework of Clustering-based Anomaly Detection"). It uses a gravitational analogy where cluster centers exert "gravity" on data points; anomalies receive less gravitational force and thus score higher.

This is a clean standalone reimplementation with no dependency on PyTorch Lightning, ADBench, or TensorFlow.

## Commands

```bash
# Install (with dev tools)
uv sync --extra dev

# Run all tests
uv run pytest tests/ -v

# Run a single test
uv run pytest tests/test_galaxy.py::test_galaxy_fit_predict -v

# Lint
uv run ruff check src/ tests/

# Format check
uv run ruff format --check src/ tests/

# Type check
uv run pyright src/ tests/
uv run mypy src/ tests/

# Download ADBench datasets
uv run python scripts/download_data.py --category Classical

# Run benchmarks
uv run python scripts/run_benchmark.py --data-dir data/Classical
```

## Architecture

The Galaxy model is a four-stage pipeline orchestrated by `Galaxy` in `galaxy.py`:

1. **Preprocessing** (`preprocessing.py`): `StandardScaler` (z-score) or `RowScaler` (L2 per-row normalization), selected via `GalaxyConfig.preprocess`.

2. **Autoencoder Pretraining** (`model.py`): `Encoder` -> `Decoder` with MSE(sum) loss, Adam + StepLR. `pretrain_autoencoder()` trains for 200 epochs. `estimate_alpha()` computes distance thresholds.

3. **Iterative EM** (`em.py`): `GalaxyEM.fit()` runs `em_iters` rounds, each:
   - **Exclude outliers**: GOF-score all data, remove top `outlier_ratio`%
   - **Update network**: fine-tune autoencoder with reconstruction + gravity loss
   - **Update prototypes**: encode data -> fit SMM -> update means/weights/covars

4. **GOF Scoring** (`gof.py`): Anomaly score = 1 / gravitational force. Scalar mode: `1 / sum_c force(x_i, mu_c)`. Vector mode: `1 / ||sum_c vec_f(x_i, mu_c)||`.

**Supporting modules**:
- `config.py`: `GalaxyConfig` dataclass -- single source of truth for all hyperparameters with paper defaults
- `adapter.py`: `GalaxyADBench` -- ADBench-compatible wrapper (y_train ignored, unsupervised)
- `smm.py`: `SMMPyTorch` -- GPU-native t-Student Mixture Model; patches `np.infty` -> `np.inf` for NumPy 2.0+ compatibility; re-exports `SMM`

## Critical Implementation Details

- **float64 det_covars**: Both `em.py` and `gof.py` compute `det_covars` in float64. The product of 128 small variance values underflows in float32 -- this is a numerical stability requirement, not an optimization.
- **Full-dataset re-scoring**: The EM exclude step always re-scores the full X, not the filtered subset.
- **Gravity loss versions**: "scalar" (`-sum_i log(sum_c force_ic)`) and "vector" (`-sum_i log(||sum_c vec_f_ic||)`), controlled by `GalaxyConfig.gravity_version`.
- **No PyTorch Lightning**: Uses vanilla PyTorch only.

## Package Structure

`src/unicad_torch/` layout with `hatchling` build backend. Public API exported from `__init__.py`: `Galaxy`, `GalaxyADBench`, `GalaxyConfig`.

## Data

ADBench `.npz` files (with `X` features and `y` 0/1 labels) in `data/`. Also supports `.csv` with a `y` column. Download via `scripts/download_data.py`.
