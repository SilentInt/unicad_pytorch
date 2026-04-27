"""GPU-native Student-t Mixture Model with diagonal covariance (ν=1)."""

from __future__ import annotations

import torch

_VAR_FLOOR = 1e-6


def _mahalanobis_diag(
    X: torch.Tensor,
    means: torch.Tensor,
    covars: torch.Tensor,
) -> torch.Tensor:
    """Squared Mahalanobis distance with diagonal covariance.

    Args:
        X: (N, D) data points.
        means: (K, D) cluster means.
        covars: (K, D) diagonal variances.

    Returns:
        (N, K) squared Mahalanobis distances: Σ_d (x_d - μ_kd)² / σ²_kd
    """
    diff = X.unsqueeze(1) - means.unsqueeze(0)  # (N, K, D)
    return (diff**2 / covars.unsqueeze(0).clamp(min=_VAR_FLOOR)).sum(dim=-1)  # (N, K)


def _kmeans_pp_init(
    Z: torch.Tensor, K: int, generator: torch.Generator
) -> torch.Tensor:
    """K-means++ seeding followed by a few Lloyd iterations.

    Args:
        Z: (N, D) data on any device.
        K: number of centres.
        generator: CPU generator for reproducibility.

    Returns:
        (K, D) initial centres on Z's device.
    """
    N, D = Z.shape
    device = Z.device

    # K-means++ seeding
    idx = torch.randint(N, (1,), generator=generator).item()
    centres = [Z[idx]]

    for _ in range(K - 1):
        dists = torch.cdist(Z, torch.stack(centres))  # (N, current_k)
        min_dists_sq = dists.min(dim=1).values ** 2  # (N,)
        probs = min_dists_sq / min_dists_sq.sum().clamp(min=1e-30)
        idx = torch.multinomial(probs, 1, generator=generator).item()
        centres.append(Z[idx])

    means = torch.stack(centres).to(device)  # (K, D)

    # A few Lloyd iterations to refine
    for _ in range(10):
        dists = torch.cdist(Z, means)  # (N, K)
        labels = dists.argmin(dim=1)  # (N,)
        for k in range(K):
            mask = labels == k
            if mask.sum() > 0:
                means[k] = Z[mask].mean(dim=0)

    return means


class SMMTorch:
    """GPU-native Student-t Mixture Model with diagonal covariance (ν=1).

    Uses the EM algorithm to fit a mixture of Student-t distributions with
    fixed degrees of freedom ν=1 (Cauchy-like heavy tails for robustness).

    Fitted attributes:
        means_: (K, D) cluster means.
        covars_: (K, D) diagonal variances.
        weights_: (K,) mixture weights.
    """

    def __init__(
        self,
        n_components: int = 10,
        n_iter: int = 100,
        tol: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        self.means_: torch.Tensor | None = None
        self.covars_: torch.Tensor | None = None
        self.weights_: torch.Tensor | None = None

    @torch.no_grad()
    def fit(self, Z: torch.Tensor) -> SMMTorch:
        """Fit the Student-t Mixture Model via EM (ν=1, diagonal covariance).

        E-step computes responsibilities τ_ik and scale factors u_ik
        under the paper's density form:

            p(z_i | c_i=k) ∝ |Σ_k|^{-1/2} / (1 + D_M²(z_i, μ_k))

        M-step updates weights, means, and diagonal covariances using
        the standard sufficient statistics weighted by τ_ik · u_ik.

        Args:
            Z: (N, D) latent embeddings on any device.
        """
        N, D = Z.shape
        K = min(self.n_components, N)
        device = Z.device
        dtype = Z.dtype

        generator = torch.Generator()
        generator.manual_seed(self.random_state)

        # --- Initialise with k-means++ ---
        means = _kmeans_pp_init(Z, K, generator)  # (K, D)

        data_var = Z.var(dim=0, unbiased=False, keepdim=True).expand(K, -1)
        covars = data_var.clamp(min=_VAR_FLOOR).clone()  # (K, D)

        weights = torch.full((K,), 1.0 / K, device=device, dtype=dtype)  # (K,)

        nu = 1.0  # degrees of freedom
        prev_ll: float | None = None

        for _ in range(self.n_iter):
            # ---- E-step ----
            maha = _mahalanobis_diag(Z, means, covars)  # (N, K)

            # Log unnormalised responsibilities (paper's density, ν=1):
            # log τ_ik ∝ log ω_k - 0.5·log|Σ_k| - log(1 + D_M²)
            log_det = torch.log(covars.clamp(min=_VAR_FLOOR)).sum(dim=1)  # (K,)
            log_resp = (
                torch.log(weights.clamp(min=1e-30)).unsqueeze(0)  # (1, K)
                - 0.5 * log_det.unsqueeze(0)  # (1, K)
                - torch.log1p(maha)  # (N, K)
            )  # (N, K)

            log_norm = torch.logsumexp(log_resp, dim=1, keepdim=True)  # (N, 1)
            resp = torch.exp(log_resp - log_norm)  # (N, K)

            # Scale factors: u_ik = (ν + D) / (ν + D_M²)
            u = ((nu + D) / (nu + maha)).to(dtype)  # (N, K)

            # ---- M-step ----
            nk = resp.sum(dim=0).clamp(min=1e-30)  # (K,)

            # Weights: ω_k = n_k / N
            weights = nk / N  # (K,)

            # Means: μ_k = Σ_i τ_ik u_ik z_i / Σ_i τ_ik u_ik
            resp_u = (resp * u).unsqueeze(2)  # (N, K, 1)
            resp_u_sum = resp_u.sum(dim=0).clamp(min=1e-30)  # (K, 1)
            means = (resp_u * Z.unsqueeze(1)).sum(dim=0) / resp_u_sum  # (K, D)

            # Covariances (diagonal):
            # σ²_kd = Σ_i τ_ik u_ik (z_id - μ_kd)² / Σ_i τ_ik
            diff = Z.unsqueeze(1) - means.unsqueeze(0)  # (N, K, D)
            covars = (resp_u * diff**2).sum(dim=0) / nk.unsqueeze(1)  # (K, D)
            covars = covars.clamp(min=_VAR_FLOOR)

            # Reinitialise degenerate components
            empty = nk < 1.0
            if empty.any():
                n_empty = empty.sum()
                new_idx = torch.randperm(N, generator=generator)[:n_empty]
                means[empty] = Z[new_idx]
                covars[empty] = data_var[:n_empty].clamp(min=_VAR_FLOOR)
                weights[empty] = 1.0 / K
                weights = weights / weights.sum()

            # Convergence check
            ll = log_norm.sum().item()
            if prev_ll is not None and abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.means_ = means
        self.covars_ = covars
        self.weights_ = weights

        return self
