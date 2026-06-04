"""Basic 数据集异常检测 — 消融实验：窗口统计 vs 残差 GRU。

架构对比:
  Baseline: ConditionalGRU → 残差 → 窗口统计(240D) → Galaxy AE → SMM
  ResidualGRU: ConditionalGRU → 残差 → ResidualGRU编码(64D) → Galaxy AE → SMM

用法:
  uv run python scripts/basic_detect_v2.py --device cuda:1 --mode both
"""

import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from unicad_torch import Galaxy, GalaxyConfig
from unicad_torch.gof import gof_score
from unicad_torch.smm_torch import SMMTorch

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "Basic"
PREPROC_DIR = DATA_DIR / "preprocessed"

# ─── 特征列 ──────────────────────────────────────────────────

RESP_RAW = [
    "IMU_GyrX", "IMU_GyrY", "IMU_GyrZ",
    "IMU_AccX", "IMU_AccY", "IMU_AccZ",
    "ATT_Roll", "ATT_Pitch", "ATT_Yaw",
    "RATE_R", "RATE_P", "RATE_Y", "RATE_A",
    "XKF1_VN", "XKF1_VE", "XKF1_VD",
    "BARO_Alt", "VIBE_VibeZ", "XKF4_SV",
    "GPS_Alt", "GPS_Spd", "GPS_VZ",
    "MAG_MagX", "MAG_MagY", "MAG_MagZ",
    "XKF3_IVN", "XKF3_IVE", "XKF3_IVD",
    "XKF3_IPN", "XKF3_IPE", "XKF3_IPD",
    "XKF3_IMX", "XKF4_SM",
]
CMD_RAW = [
    "ATT_DesRoll", "ATT_DesPitch", "ATT_DesYaw",
    "RATE_RDes", "RATE_PDes", "RATE_YDes", "RATE_ADes",
]
CMD_MEAS_PAIRS = [
    ("ATT_DesRoll", "ATT_Roll"), ("ATT_DesPitch", "ATT_Pitch"),
    ("ATT_DesYaw", "ATT_Yaw"),
    ("RATE_RDes", "RATE_R"), ("RATE_PDes", "RATE_P"),
    ("RATE_YDes", "RATE_Y"), ("RATE_ADes", "RATE_A"),
]
NR = len(RESP_RAW) + len(CMD_MEAS_PAIRS)  # 40
NC = len(CMD_RAW)  # 7
NF = NR + NC  # 47

# ─── 数据加载 ──────────────────────────────────────────────────

def load_basic_data(max_seqs=None):
    seqs = []
    names = sorted(os.listdir(PREPROC_DIR))
    if max_seqs:
        names = names[:max_seqs]
    for fn in names:
        if not fn.endswith(".npz"):
            continue
        data = np.load(os.path.join(PREPROC_DIR, fn))
        seqs.append({
            "name": fn.replace(".npz", ""),
            "features": data["features"].astype(np.float32),
            "labels": data["labels"].astype(np.int32),
            "n_frames": len(data["labels"]),
            "has_fault": data["labels"].sum() > 0,
        })
    return seqs


# ─── ConditionalGRU ────────────────────────────────────────────

class ConditionalGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(NF, 128, 2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(128 + NC, 128), nn.ReLU(), nn.Linear(128, NR))

    def forward(self, x_hist, x_cmd):
        out, _ = self.gru(x_hist)
        return self.head(torch.cat([out[:, -1, :], x_cmd], 1))

    def encode(self, x_hist, x_cmd):
        """返回 [GRU隐状态, 当前指令] 拼接 — 用于下游聚类."""
        out, _ = self.gru(x_hist)
        return torch.cat([out[:, -1, :], x_cmd], 1)  # (B, 128+7=135)


def train_conditional_gru(sequences, hist_len=75, epochs=100,
                           lr=1e-3, device="cpu", max_samples=200000):
    model = ConditionalGRU().to(device)

    all_Xh, all_Yr, all_Yc = [], [], []
    total_available = 0
    for s in sequences:
        f, lbs = s["features"], s["labels"]
        ni = np.where(lbs == 0)[0]
        total_available += max(0, len(ni) - hist_len)

    sample_ratio = min(1.0, max_samples / max(total_available, 1))
    rng = np.random.RandomState(42)

    for s in sequences:
        f, lbs = s["features"], s["labels"]
        ni = np.where(lbs == 0)[0]
        for i in range(hist_len, len(ni)):
            if rng.random() > sample_ratio:
                continue
            idx = ni[i]
            if np.all(lbs[idx - hist_len:idx] == 0):
                all_Xh.append(f[idx - hist_len:idx])
                all_Yr.append(f[idx, :NR])
                all_Yc.append(f[idx, NR:])

    n_train = len(all_Xh)
    Xh_t = torch.tensor(np.array(all_Xh), dtype=torch.float32, device=device)
    Yr_t = torch.tensor(np.array(all_Yr), dtype=torch.float32, device=device)
    Yc_t = torch.tensor(np.array(all_Yc), dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(Xh_t, Yr_t, Yc_t),
                        batch_size=2048, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)

    for ep in range(epochs):
        total_loss = 0.0
        for Xb, Yrb, Ycb in loader:
            loss = F.mse_loss(model(Xb, Ycb), Yrb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * Xb.size(0)
        scheduler.step()
        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"  [CondGRU] epoch {ep+1}/{epochs} "
                  f"avg_loss={total_loss/n_train:.4f}")

    print(f"  Training samples: {n_train:,}")
    return model


# ─── 残差计算 ────────────────────────────────────────────────

def compute_residuals(sequences, model, hist_len=75, device="cpu"):
    """用 ConditionalGRU 计算所有帧的残差 + 正常帧归一化参数."""
    model.eval()

    normal_errors = []
    for s in sequences:
        f, lbs = s["features"], s["labels"]
        n = len(f)
        if n <= hist_len:
            continue
        idx = [i for i in range(hist_len, n)
               if lbs[i] == 0 and np.all(lbs[i - hist_len:i] == 0)]
        for start in range(0, len(idx), 2048):
            ci = idx[start:start + 2048]
            Xh = torch.tensor(np.stack([f[i - hist_len:i] for i in ci]),
                              dtype=torch.float32, device=device)
            Xc = torch.tensor(np.stack([f[i, NR:] for i in ci]),
                              dtype=torch.float32, device=device)
            Yt = torch.tensor(np.stack([f[i, :NR] for i in ci]),
                              dtype=torch.float32, device=device)
            with torch.no_grad():
                pred = model(Xh, Xc)
            normal_errors.append(((Yt - pred).detach() ** 2).cpu().numpy())

    normal_errors = np.vstack(normal_errors)
    mu = np.mean(normal_errors, 0, keepdims=True)
    sd = np.std(normal_errors, 0, keepdims=True) + 1e-8

    results = []
    for s in sequences:
        f, lbs = s["features"], s["labels"]
        n = len(f)
        residual = np.zeros((n, NR), dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        if n > hist_len:
            idx = list(range(hist_len, n))
            for start in range(0, len(idx), 2048):
                ci = idx[start:start + 2048]
                Xh = torch.tensor(np.stack([f[i - hist_len:i] for i in ci]),
                                  dtype=torch.float32, device=device)
                Xc = torch.tensor(np.stack([f[i, NR:] for i in ci]),
                                  dtype=torch.float32, device=device)
                Yt = np.stack([f[i, :NR] for i in ci])
                with torch.no_grad():
                    pred = model(Xh, Xc).cpu().numpy()
                for j, i in enumerate(ci):
                    residual[i] = (Yt[j] - pred[j]) ** 2
                    valid[i] = True
        # z-score 归一化
        residual = (residual - mu) / sd
        residual[:hist_len] = 0
        results.append({
            "name": s["name"], "residual": residual.astype(np.float32),
            "labels": lbs.copy(), "n_frames": n,
            "has_fault": s["has_fault"], "valid": valid,
        })
    return results


# ─── Baseline: 窗口统计 ─────────────────────────────────────

def _window_stats(errors_norm, W, feat_dim):
    n = len(errors_norm)
    out = np.zeros((n, feat_dim * 3), dtype=np.float32)
    if n <= W:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(errors_norm, W, axis=0)
    out[W:, :feat_dim] = windows[:n - W].mean(axis=2)
    out[W:, feat_dim:feat_dim * 2] = windows[:n - W].std(axis=2)
    out[W:, feat_dim * 2:] = windows[:n - W].max(axis=2)
    return out


def features_window_stats(rseqs, win_sizes=(10, 30)):
    """对残差序列做多尺度窗口统计."""
    results = []
    for s in rseqs:
        err = s["residual"]
        ws_parts = [_window_stats(err, W, NR) for W in win_sizes]
        results.append({
            "name": s["name"],
            "features": np.hstack(ws_parts).astype(np.float32),
            "labels": s["labels"], "n_frames": s["n_frames"],
            "has_fault": s["has_fault"], "valid": s["valid"],
        })
    return results


# ─── GRU 隐状态直接用作特征 ─────────────────────────────

def features_gru_hidden(sequences, model, hist_len=75, device="cpu"):
    """直接用 ConditionalGRU 的隐状态 + 指令作为聚类特征.

    隐状态已编码过去 75 帧飞行模式, 跳过残差/窗口统计.
    """
    model.eval()
    results = []
    hdim = 128 + NC  # GRU hidden + command
    for s in sequences:
        f, lbs = s["features"], s["labels"]
        n = len(f)
        feats = np.zeros((n, hdim), dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        if n > hist_len:
            idx = list(range(hist_len, n))
            for start in range(0, len(idx), 2048):
                ci = idx[start:start + 2048]
                Xh = torch.tensor(np.stack([f[i - hist_len:i] for i in ci]),
                                  dtype=torch.float32, device=device)
                Xc = torch.tensor(np.stack([f[i, NR:] for i in ci]),
                                  dtype=torch.float32, device=device)
                with torch.no_grad():
                    h = model.encode(Xh, Xc).cpu().numpy()
                for j, i in enumerate(ci):
                    feats[i] = h[j]
                    valid[i] = True
        results.append({
            "name": s["name"],
            "features": feats.astype(np.float32),
            "labels": lbs.copy(), "n_frames": n,
            "has_fault": s["has_fault"], "valid": valid,
        })
    return results

class ResidualGRU(nn.Module):
    """输入残差序列 → GRU 编码 → 预测下一帧残差.

    训练后在隐状态中编码了残差的时序模式.
    """
    def __init__(self, input_dim=NR, hidden=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers, batch_first=True)
        self.head = nn.Linear(hidden, input_dim)

    def forward(self, x):
        out, h = self.gru(x)
        return self.head(out[:, -1, :])

    def encode(self, x):
        """提取最后一帧的隐状态作为特征."""
        out, h = self.gru(x)
        return out[:, -1, :]  # (B, hidden)


def train_residual_gru(rseqs, hist_len=10, epochs=50, lr=1e-3, device="cpu"):
    """用正常帧的残差序列训练 ResidualGRU.

    hist_len=10 (1s@10Hz): 用过去 1s 残差预测下一帧残差.
    """
    model = ResidualGRU(input_dim=NR, hidden=64).to(device)

    all_X, all_Y = [], []
    for s in rseqs:
        r = s["residual"]  # (n, NR)
        lbs, vd = s["labels"], s["valid"]
        for i in range(hist_len, len(r)):
            if vd[i] and lbs[i] == 0 and np.all(lbs[i - hist_len:i] == 0):
                all_X.append(r[i - hist_len:i])
                all_Y.append(r[i])

    X_t = torch.tensor(np.array(all_X), dtype=torch.float32, device=device)
    Y_t = torch.tensor(np.array(all_Y), dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(X_t, Y_t), batch_size=2048, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        total_loss = 0.0
        for Xb, Yb in loader:
            loss = F.mse_loss(model(Xb), Yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * Xb.size(0)
        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"  [ResGRU] epoch {ep+1}/{epochs} "
                  f"avg_loss={total_loss/len(X_t):.4f}")

    print(f"  ResidualGRU training samples: {len(X_t):,}")
    return model


def features_residual_gru(rseqs, model, hist_len=10, device="cpu"):
    """用 ResidualGRU 提取隐状态特征."""
    model.eval()
    results = []
    for s in rseqs:
        r = s["residual"]
        n = len(r)
        feats = np.zeros((n, 64), dtype=np.float32)
        if n > hist_len:
            idx = list(range(hist_len, n))
            for start in range(0, len(idx), 2048):
                ci = idx[start:start + 2048]
                Xh = torch.tensor(np.stack([r[i - hist_len:i] for i in ci]),
                                  dtype=torch.float32, device=device)
                with torch.no_grad():
                    h = model.encode(Xh).cpu().numpy()
                for j, i in enumerate(ci):
                    feats[i] = h[j]
        results.append({
            "name": s["name"],
            "features": feats.astype(np.float32),
            "labels": s["labels"], "n_frames": n,
            "has_fault": s["has_fault"], "valid": s["valid"],
        })
    return results


# ─── Deep SVDD: 紧凑编码 + 距离检测 ──────────────────────────

class DeepSVDDEncoder(nn.Module):
    """Encoder: input_dim → hidden → latent. 训练目标: 正常帧编码接近中心."""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        return self.encoder(x)

    def reconstruct(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_svdd(X_normal, device, epochs=100, lr=1e-3, batch_size=2048,
               hidden_mult=1.5, latent_mult=0.75, center_weight=0.1):
    """Autoencoder + 中心吸引: 重建约束防坍缩, 同时拉近正常帧编码."""
    input_dim = X_normal.shape[1]
    hd = max(int(input_dim * hidden_mult), 96)
    ld = max(int(input_dim * latent_mult), 48)
    model = DeepSVDDEncoder(input_dim, hd, ld).to(device)

    X_t = torch.tensor(X_normal, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)

    # 初始化中心
    model.eval()
    with torch.no_grad():
        centers = []
        for (Xb,) in loader:
            centers.append(model(Xb.to(device)).mean(0))
    center = torch.stack(centers).mean(0)
    center.requires_grad = False

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        total_loss = 0
        for (Xb,) in loader:
            x = Xb.to(device)
            recon, z = model.reconstruct(x)
            loss_recon = F.mse_loss(recon, x)
            loss_center = F.mse_loss(z, center.expand_as(z))
            loss = loss_recon + center_weight * loss_center
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"  [SVDD] ep {ep+1}/{epochs} "
                  f"loss={total_loss/len(X_t):.4f} "
                  f"(recon={loss_recon.item():.4f} center={loss_center.item():.4f})")

    return model, center


def score_svdd(X, model, center, device):
    """SVDD 异常分数: 编码到中心的距离."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=2048)
    scores = []
    with torch.no_grad():
        for (Xb,) in loader:
            z = model(Xb.to(device))
            scores.append(((z - center) ** 2).sum(1).cpu().numpy())
    return np.concatenate(scores).astype(np.float32)


# ─── Mahalanobis: 直接距离 ─────────────────────────────────

def fit_mahalanobis(X_normal, shrinkage=0.01):
    """在原始特征空间拟合多元正态分布."""
    mu = X_normal.mean(0, keepdims=True)
    # 收缩估计 (避免奇异协方差)
    cov = np.cov(X_normal, rowvar=False)
    cov_shrunk = (1 - shrinkage) * cov + shrinkage * np.eye(len(cov)) * cov.trace() / len(cov)
    # 存储精度矩阵 (协方差的逆)
    prec = np.linalg.inv(cov_shrunk + np.eye(len(cov)) * 1e-6)
    return mu.astype(np.float32), prec.astype(np.float32)


def score_mahalanobis(X, mu, prec):
    """马氏距离: (x-μ)ᵀ Σ⁻¹ (x-μ)."""
    diff = X - mu
    return (diff @ prec * diff).sum(1).astype(np.float32)

def fit_smm_with_em(Z, k=5, em_iters=3, outlier_ratio=0.01):
    Z = Z.clone()
    sm = None
    for i in range(em_iters):
        sm = SMMTorch(n_components=k, n_iter=200, tol=1e-4, random_state=42)
        sm.fit(Z)
        if i < em_iters - 1 and sm.means_ is not None:
            scores = gof_score(Z, sm.means_, sm.covars_,
                               sm.weights_.reshape(1, -1), "vector")
            mask = scores <= torch.quantile(scores, 1.0 - outlier_ratio)
            if mask.sum() > k:
                Z = Z[mask]
    return sm


# ─── 评估 ─────────────────────────────────────────────────────

def evaluate(scores, y_all, valid_all, seqs, title="", device="cpu"):
    yv, sv = y_all[valid_all], scores[valid_all]
    if len(np.unique(yv)) < 2:
        print(f"  [{title}] Skipped: only one class")
        return 0.0, 0.0

    auc = roc_auc_score(yv, sv)
    ap = average_precision_score(yv, sv)
    best_f1, best_t = 0, 0
    for pct in np.arange(90, 100, 0.5):
        t = np.percentile(sv[yv == 0], pct)
        pred = (sv > t).astype(int)
        f1 = f1_score(yv, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    if title:
        print(f"  [{title}]")
    print(f"  AUC={auc:.3f} AP={ap:.3f} F1={best_f1:.3f}")

    off = 0
    sa = []
    for s in seqs:
        n = s["n_frames"]
        svv = scores[off:off + n][s["valid"]]
        yvv = s["labels"][s["valid"]]
        off += n
        if yvv.sum() > 0 and yvv.sum() < len(yvv):
            sa.append(roc_auc_score(yvv, svv))
    mseq = np.mean(sa) if sa else float("nan")
    print(f"  MeanSeq={mseq:.3f}")
    return auc, mseq


# ─── 统一评估 ─────────────────────────────────────────────────

def run_ablation(fseqs, device, seed=42):
    """跑所有检测方法并对比."""
    X_n = np.vstack([s["features"][(s["labels"] == 0) & s["valid"]]
                     for s in fseqs]).astype(np.float32)
    X_a = np.vstack([s["features"] for s in fseqs]).astype(np.float32)
    y_a = np.concatenate([s["labels"] for s in fseqs])
    va = np.concatenate([s["valid"] for s in fseqs])
    zm, zs = X_n.mean(0, keepdims=True), X_n.std(0, keepdims=True) + 1e-8
    X_nz, X_az = (X_n - zm) / zs, (X_a - zm) / zs
    fdim = X_nz.shape[1]
    print(f"  Features: {fdim}D, normal={len(X_nz):,}")

    results = {}

    # ── 马氏距离 (baseline, 无学习) ──
    print(f"\n  {'─'*50}\n  [1] Mahalanobis Distance (矩阵拟合)")
    t0 = time.time()
    mu, prec = fit_mahalanobis(X_nz)
    s_maha = score_mahalanobis(X_az, mu, prec)
    _, ms = evaluate(s_maha, y_a, va, fseqs, f"Mahalanobis")
    results["Mahalanobis"] = ms
    print(f"  Time: {time.time()-t0:.0f}s")

    # ── Deep SVDD (Encoder, 无聚类) ──
    print(f"\n  {'─'*50}\n  [2] Deep SVDD (Encoder → 距中心)")
    t0 = time.time()
    encoder, center = train_svdd(X_nz, device, epochs=100, lr=1e-3)
    s_svdd = score_svdd(X_az, encoder, center, device)
    _, ms = evaluate(s_svdd, y_a, va, fseqs, f"Deep SVDD")
    results["Deep SVDD"] = ms
    print(f"  Time: {time.time()-t0:.0f}s")

    # ── Galaxy AE + SMM (原始方法) ──
    print(f"\n  {'─'*50}\n  [3] Galaxy AE + SMM (baseline)")
    t0 = time.time()
    cfg = GalaxyConfig(
        k=5, hidden_dim=int(fdim * 1.5), latent_dim=int(fdim * 0.75),
        seed=seed, pretrain_epochs=100, em_iters=3, em_finetune_steps=100,
        gravity_version="vector", score_type="vector",
        device=device, verbose=False,
    )
    g = Galaxy(cfg)
    g.fit(X_nz)
    sg = g.predict_score(X_az)
    _, ms = evaluate(sg, y_a, va, fseqs, f"Galaxy AE+SMM")
    results["Galaxy AE+SMM"] = ms
    print(f"  Time: {time.time()-t0:.0f}s")

    # ── 汇总 ──
    print(f"\n  {'='*50}")
    print(f"  Summary ({fdim}D)")
    for name, ms in results.items():
        print(f"  {name:<25}: MeanSeq={ms:.3f}")

    return results


# ─── 主函数 ────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mode", default="both",
                   choices=["window", "resgru", "gruhidden", "both", "all"],
                   help="window=滑动窗口, resgru=残差GRU, both=两者")
    p.add_argument("--max-seqs", type=int, default=0)
    p.add_argument("--max-train-samples", type=int, default=200000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    device = a.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    max_s = a.max_seqs if a.max_seqs > 0 else None

    # [0] Load
    print("=" * 60)
    print(f"Basic Ablation: Window Stats vs ResidualGRU")
    print("=" * 60)
    print("\n[0] Loading...")
    seqs = load_basic_data(max_seqs=max_s)
    n_fault = int(sum(s["labels"].sum() for s in seqs))
    n_total = sum(s["n_frames"] for s in seqs)
    print(f"  {len(seqs)} seq, {n_total:,} frames, "
          f"{n_fault:,} fault ({n_fault/n_total:.1%})")

    # [1] ConditionalGRU
    print(f"\n[1/4] Training ConditionalGRU (hist=75)...")
    t1 = time.time()
    cgru = train_conditional_gru(seqs, hist_len=75, epochs=100, device=device,
                                  max_samples=a.max_train_samples)
    print(f"  Time: {time.time()-t1:.0f}s")

    # [2] Compute residuals
    print("\n[2/4] Computing residuals...")
    rseqs = compute_residuals(seqs, cgru, hist_len=75, device=device)

    # [3a] Baseline: Window Stats
    if a.mode in ("window", "both", "all"):
        print(f"\n[3a/4] {'='*50}")
        print("Baseline: Window Stats (W=10,30)")
        print("=" * 50)
        fseqs_ws = features_window_stats(rseqs, win_sizes=(10, 30))
        run_ablation(fseqs_ws, device, a.seed)

    # [3b] ResidualGRU
    if a.mode in ("resgru", "both", "all"):
        print(f"\n[3b/4] {'='*50}")
        print("Ablation: ResidualGRU (hist=10, hidden=64)")
        print("=" * 50)
        rgru = train_residual_gru(rseqs, hist_len=10, epochs=50, device=device)
        fseqs_rg = features_residual_gru(rseqs, rgru, hist_len=10, device=device)
        run_ablation(fseqs_rg, device, a.seed)

    # [3c] GRU Hidden State (直接聚类)
    if a.mode in ("gruhidden", "all"):
        print(f"\n[3c/4] {'='*50}")
        print("Ablation: GRU Hidden State (128+7=135D)")
        print("=" * 50)
        fseqs_gh = features_gru_hidden(seqs, cgru, hist_len=75, device=device)
        run_ablation(fseqs_gh, device, a.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
