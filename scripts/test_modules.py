#!/usr/bin/env python3
"""Step-by-step module test with real data. Exposes intermediate values for diagnosis.

Usage
-----
    uv run python scripts/test_modules.py
    uv run python scripts/test_modules.py --datasets 38_thyroid 2_annthyroid
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from unicad_torch.config import GalaxyConfig
from unicad_torch.datasets import find_datasets
from unicad_torch.preprocessing import StandardScaler, RowScaler
from unicad_torch.model import Autoencoder, pretrain_autoencoder
from unicad_torch.smm_torch import SMMTorch
from unicad_torch.gravity import mahalanobis_diag
from unicad_torch.gof import gof_score
from unicad_torch.em import GalaxyEM
from unicad_torch.galaxy import Galaxy
from unicad_torch.adapter import GalaxyADBench


def _stats(arr: np.ndarray | torch.Tensor, name: str) -> str:
    """Return a one-line stats summary for a tensor/array."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().float().numpy()
    arr = np.asarray(arr, dtype=np.float64)
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    return (
        f"{name}: shape={arr.shape}  "
        f"min={np.nanmin(arr):.6g}  max={np.nanmax(arr):.6g}  "
        f"mean={np.nanmean(arr):.6g}  std={np.nanstd(arr):.6g}  "
        f"NaN={has_nan}  Inf={has_inf}"
    )


def _section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def test_load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    _section("1. Data Loading")
    data = np.load(path)
    X = data["X"].astype(np.float32)
    y = data["y"].ravel()
    print(_stats(X, "X_raw"))
    print(
        f"y: shape={y.shape}  anomaly_rate={y.mean():.4f}  "
        f"normal={int((y == 0).sum())}  anomaly={int((y == 1).sum())}"
    )
    return X, y


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
def test_preprocessing(
    X: np.ndarray, config: GalaxyConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    _section("2. Preprocessing")

    device = config.device
    X_tensor = torch.from_numpy(X).to(torch.float32).to(device)

    # z-score
    scaler_z = StandardScaler()
    X_z = scaler_z.fit_transform(X_tensor)
    print(_stats(X_z, "X_zscore"))

    # row-norm
    scaler_r = RowScaler()
    X_r = scaler_r.fit_transform(X_tensor)
    print(_stats(X_r, "X_rownorm"))

    return X_z, X_r


# ---------------------------------------------------------------------------
# 3. Autoencoder
# ---------------------------------------------------------------------------
def test_autoencoder(
    X_tensor: torch.Tensor, config: GalaxyConfig
) -> tuple[Autoencoder, torch.Tensor]:
    _section("3. Autoencoder Pretraining")

    input_dim = X_tensor.shape[1]
    model = Autoencoder(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.hidden_dim,
    ).to(config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Autoencoder: input_dim={input_dim}, hidden_dim={config.hidden_dim}, params={n_params}"
    )

    t0 = time.perf_counter()
    pretrain_autoencoder(model, X_tensor, config)
    pretrain_time = time.perf_counter() - t0
    print(f"Pretrain: {config.pretrain_epochs} epochs, {pretrain_time:.2f}s")

    model.eval()
    with torch.no_grad():
        Z, X_hat = model(X_tensor)
    recon_err = ((X_hat - X_tensor) ** 2).sum(dim=1)
    print(_stats(Z, "Z (latent)"))
    print(_stats(X_hat, "X_hat (reconstruction)"))
    print(_stats(recon_err, "recon_error_per_sample"))
    print(f"  Mean recon MSE(sum): {recon_err.mean():.4f}")

    return model, X_tensor


# ---------------------------------------------------------------------------
# 4. SMM fitting (GPU-native)
# ---------------------------------------------------------------------------
def test_smm(Z: torch.Tensor, config: GalaxyConfig) -> SMMTorch:
    _section("4. SMM Fitting (GPU-native)")

    smm = SMMTorch(
        n_components=config.k,
        n_iter=config.smm_n_iter,
        tol=config.smm_tol,
        random_state=config.seed,
    )

    print(f"SMM: k={config.k}, n_iter={config.smm_n_iter}, tol={config.smm_tol}")
    print(f"Fitting on Z: {Z.shape} ...")

    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        smm.fit(Z)
        fit_time = time.perf_counter() - t0

        if w:
            print(f"  Warnings ({len(w)}):")
            for wi in w:
                print(f"    - {wi.category.__name__}: {wi.message}")
        else:
            print("  No warnings.")

    print(f"  Fit time: {fit_time:.2f}s")

    assert smm.means_ is not None
    assert smm.covars_ is not None
    assert smm.weights_ is not None

    print(_stats(smm.means_, "smm.means_"))
    print(_stats(smm.covars_, "smm.covars_"))
    print(_stats(smm.weights_, "smm.weights_"))

    # det_covars in float64 vs float32
    det_f32 = torch.prod(smm.covars_.to(torch.float32), dim=1)
    det_f64 = torch.prod(smm.covars_.to(torch.float64), dim=1)
    print(f"  det_covars (float32): {_stats(det_f32, '')}")
    print(f"  det_covars (float64): {_stats(det_f64, '')}")

    return smm


# ---------------------------------------------------------------------------
# 5. GOF Scoring
# ---------------------------------------------------------------------------
def test_gof(Z: torch.Tensor, smm: SMMTorch, config: GalaxyConfig) -> torch.Tensor:
    _section("5. GOF Scoring")

    assert smm.means_ is not None
    assert smm.covars_ is not None
    assert smm.weights_ is not None

    means = smm.means_.detach()
    covars = smm.covars_.detach().clamp(min=1e-6)
    weights = smm.weights_.detach()

    # Mahalanobis distance
    _section("5a. Mahalanobis Distance")
    maha = mahalanobis_diag(Z, means, covars)
    print(_stats(maha, "maha"))
    for c in range(min(5, means.shape[0])):
        print(
            f"  component {c}: min={maha[:, c].min():.4g}  max={maha[:, c].max():.4g}  "
            f"mean={maha[:, c].mean():.4g}"
        )

    # Scalar score
    _section("5b. Scalar Score")
    score_scalar = gof_score(Z, means, covars, weights, score_type="scalar")
    print(_stats(score_scalar, "score_scalar"))

    # Vector score
    _section("5c. Vector Score")
    score_vector = gof_score(Z, means, covars, weights, score_type="vector")
    print(_stats(score_vector, "score_vector"))

    return score_vector


# ---------------------------------------------------------------------------
# 6. Full EM
# ---------------------------------------------------------------------------
def test_em(
    model: Autoencoder, X_tensor: torch.Tensor, config: GalaxyConfig
) -> GalaxyEM:
    _section("6. Full EM")

    em = GalaxyEM(model, config)

    print(
        f"EM: iters={config.em_iters}, finetune_steps={config.em_finetune_steps}, "
        f"finetune_lr={config.em_finetune_lr}, outlier_ratio={config.outlier_ratio}"
    )

    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        em.fit(X_tensor)
        fit_time = time.perf_counter() - t0

        if w:
            print(f"  Warnings ({len(w)}):")
            for wi in w:
                print(f"    - {wi.category.__name__}: {str(wi.message)[:100]}")
        else:
            print("  No warnings.")

    print(f"  EM fit time: {fit_time:.2f}s")

    assert em.means is not None
    assert em.covars is not None
    assert em.weights is not None

    print(_stats(em.means, "em.means"))
    print(_stats(em.covars, "em.covars"))
    print(_stats(em.weights, "em.weights"))

    # Score with EM results
    score = em.score(X_tensor)
    print(_stats(score, "em_score"))

    return em


# ---------------------------------------------------------------------------
# 7. Full Galaxy pipeline
# ---------------------------------------------------------------------------
def test_galaxy(X: np.ndarray, y: np.ndarray, config: GalaxyConfig) -> None:
    _section("7. Full Galaxy Pipeline")

    model = Galaxy(config)

    t0 = time.perf_counter()
    model.fit(X)
    fit_time = time.perf_counter() - t0
    print(f"  Fit time: {fit_time:.2f}s")

    t0 = time.perf_counter()
    scores = model.predict_score(X)
    score_time = time.perf_counter() - t0
    print(f"  Score time: {score_time:.4f}s")
    print(_stats(scores, "scores"))

    has_nan = np.isnan(scores).any()
    has_inf = np.isinf(scores).any()
    if has_nan or has_inf:
        n_nan = np.isnan(scores).sum()
        n_inf = np.isinf(scores).sum()
        print(f"  PROBLEM: {n_nan} NaN, {n_inf} Inf values in scores")
        bad_idx = np.where(np.isnan(scores) | np.isinf(scores))[0]
        print(f"  Bad sample indices (first 20): {bad_idx[:20]}")
    else:
        auc = roc_auc_score(y, scores)
        print(f"  AUC-ROC: {auc:.4f}")

        s_normal = scores[y == 0]
        s_anomaly = scores[y == 1]
        print(
            f"  Normal scores:   mean={s_normal.mean():.6g}  std={s_normal.std():.6g}  "
            f"min={s_normal.min():.6g}  max={s_normal.max():.6g}"
        )
        print(
            f"  Anomaly scores:  mean={s_anomaly.mean():.6g}  std={s_anomaly.std():.6g}  "
            f"min={s_anomaly.min():.6g}  max={s_anomaly.max():.6g}"
        )


# ---------------------------------------------------------------------------
# 8. ADBench adapter
# ---------------------------------------------------------------------------
def test_adapter(X: np.ndarray, y: np.ndarray, config: GalaxyConfig) -> None:
    _section("8. ADBench Adapter")

    model = GalaxyADBench(config)
    model.fit(X, y)
    scores = model.predict_score(X)
    print(_stats(scores, "adapter_scores"))

    if not np.isnan(scores).any() and not np.isinf(scores).any():
        auc = roc_auc_score(y, scores)
        print(f"  AUC-ROC: {auc:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Module-by-module test with real data")
    parser.add_argument("--data-dir", default="data/Classical")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["38_thyroid"],
        help="Dataset names to test (default: 38_thyroid)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--em-iters", type=int, default=3)
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip autoencoder pretrain (fast but meaningless)",
    )
    args = parser.parse_args()

    config = GalaxyConfig(
        k=args.k,
        pretrain_epochs=args.pretrain_epochs,
        em_iters=args.em_iters,
        device=args.device,
        pretrain=not args.skip_pretrain,
    )

    datasets = find_datasets(args.data_dir, args.datasets)
    if not datasets:
        print(f"No datasets found in {args.data_dir}")
        sys.exit(1)

    for name, path in datasets:
        print(f"\n{'#' * 70}")
        print(f"#  Dataset: {name}")
        print(f"{'#' * 70}")

        X, y = test_load_data(path)

        # 2. Preprocessing
        X_z, X_r = test_preprocessing(X, config)

        # 3. Autoencoder
        model, X_tensor = test_autoencoder(X_z, config)

        # 4. SMM
        with torch.no_grad():
            Z = model.encoder(X_tensor)
        smm = test_smm(Z, config)

        # 5. GOF
        test_gof(Z, smm, config)

        # 6. EM
        test_em(model, X_tensor, config)

        # 7. Full Galaxy
        test_galaxy(X, y, config)

        # 8. Adapter
        test_adapter(X, y, config)

    print(f"\n{'#' * 70}")
    print("#  All tests complete.")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
