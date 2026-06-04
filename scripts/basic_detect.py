"""Basic 数据集异常检测 — 基于 ALFA V7 架构。

架构:
  ConditionalGRU(hist=75, 7.5s@10Hz): predict(response_t | history_75f, command_t) → 40D
    → 归一化 → 窗口统计 W=10,30 (mean/std/max) → 240D
    → Galaxy AE (240→360→180 扩展) + SMM(k=5) + GOF(vector)

特征设计 (Tier 1+2, 47D GRU 输入):
  Response (33D raw): IMU 6D + ATT 3D + RATE 4D + XKF1_V 3D + BARO_Alt
                      + VIBE_Z + XKF4_SV + GPS 3D + MAG 3D + EKF 创新 8D
  cmd-meas (7D): ATT/RATE Des-Actual
  Command (7D): ATT_Des(3) + RATE_Des(4)

用法: uv run python scripts/basic_detect.py --device cuda:0
"""

import argparse, csv, os, sys, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from unicad_torch import Galaxy, GalaxyConfig
from unicad_torch.gof import gof_score
from unicad_torch.smm_torch import SMMTorch

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "Basic"

# ─── 特征列定义 ──────────────────────────────────────────────

RESP_RAW = [
    # IMU (6D)
    "IMU_GyrX", "IMU_GyrY", "IMU_GyrZ",
    "IMU_AccX", "IMU_AccY", "IMU_AccZ",
    # Attitude actual (3D)
    "ATT_Roll", "ATT_Pitch", "ATT_Yaw",
    # Rate actual (4D)
    "RATE_R", "RATE_P", "RATE_Y", "RATE_A",
    # EKF velocity (3D)
    "XKF1_VN", "XKF1_VE", "XKF1_VD",
    # Barometer (1D)
    "BARO_Alt",
    # Vibration (1D)
    "VIBE_VibeZ",
    # EKF sensor health (1D)
    "XKF4_SV",
    # ── Tier 2: 传感器故障覆盖 (14D) ──
    # GPS (3D)
    "GPS_Alt", "GPS_Spd", "GPS_VZ",
    # Magnetometer (3D)
    "MAG_MagX", "MAG_MagY", "MAG_MagZ",
    # EKF velocity innovation (3D)
    "XKF3_IVN", "XKF3_IVE", "XKF3_IVD",
    # EKF position innovation (3D)
    "XKF3_IPN", "XKF3_IPE", "XKF3_IPD",
    # EKF magnetometer innovation + health (2D)
    "XKF3_IMX", "XKF4_SM",
]

CMD_RAW = [
    "ATT_DesRoll", "ATT_DesPitch", "ATT_DesYaw",
    "RATE_RDes", "RATE_PDes", "RATE_YDes", "RATE_ADes",
]

CMD_MEAS_PAIRS = [
    ("ATT_DesRoll", "ATT_Roll"),
    ("ATT_DesPitch", "ATT_Pitch"),
    ("ATT_DesYaw", "ATT_Yaw"),
    ("RATE_RDes", "RATE_R"),
    ("RATE_PDes", "RATE_P"),
    ("RATE_YDes", "RATE_Y"),
    ("RATE_ADes", "RATE_A"),
]

NR = len(RESP_RAW) + len(CMD_MEAS_PAIRS)  # 33+7=40
NC = len(CMD_RAW)  # 7
NF = NR + NC  # 47

# ─── 数据加载 ──────────────────────────────────────────────────

def _yaw_diff(des, actual):
    """Wrap yaw difference to [-180, 180]."""
    d = des - actual
    return (d + 180.0) % 360.0 - 180.0


def _load_sequence(main_csv):
    """用 pandas 读取主 CSV，去重时间戳，返回 (timestamps, data, labels)。

    data 列顺序: RESP_RAW(19) + CMD_MEAS(7) + CMD_RAW(7) = 33D
    """
    # 只读取需要的列
    usecols = ["Time"] + RESP_RAW + CMD_RAW + ["Status"]
    try:
        df = pd.read_csv(main_csv, usecols=usecols)
    except (ValueError, FileNotFoundError):
        # Some sequences may have missing columns
        return None, None, None

    if len(df) < 100:
        return None, None, None

    # 去重: 同时间戳保留最后一行
    df = df.groupby("Time", as_index=False, sort=True).last()

    timestamps = df["Time"].values.astype(np.float64)
    n = len(df)
    labels = df["Status"].fillna(0).values.astype(np.int32)
    # Convert to 0/1
    labels = (labels > 0).astype(np.int32)

    data = np.zeros((n, NF), dtype=np.float32)

    # Response raw
    for j, col in enumerate(RESP_RAW):
        if col in df.columns:
            data[:, j] = df[col].fillna(0).values.astype(np.float32)

    # cmd-meas errors
    for j, (cmd_col, resp_col) in enumerate(CMD_MEAS_PAIRS):
        cmd_vals = df[cmd_col].fillna(0).values.astype(np.float32) if cmd_col in df.columns else np.zeros(n)
        resp_vals = df[resp_col].fillna(0).values.astype(np.float32) if resp_col in df.columns else np.zeros(n)
        if "Yaw" in cmd_col:
            data[:, len(RESP_RAW) + j] = _yaw_diff(cmd_vals, resp_vals)
        else:
            data[:, len(RESP_RAW) + j] = cmd_vals - resp_vals

    # Command raw
    for j, col in enumerate(CMD_RAW):
        if col in df.columns:
            data[:, len(RESP_RAW) + len(CMD_MEAS_PAIRS) + j] = df[col].fillna(0).values.astype(np.float32)

    # yaw unwrap
    yaw_actual_idx = RESP_RAW.index("ATT_Yaw")
    yaw_des_cmd_idx = len(RESP_RAW) + len(CMD_MEAS_PAIRS) + CMD_RAW.index("ATT_DesYaw")
    data[:, yaw_actual_idx] = np.unwrap(data[:, yaw_actual_idx], period=360.0)
    data[:, yaw_des_cmd_idx] = np.unwrap(data[:, yaw_des_cmd_idx], period=360.0)

    return timestamps, data, labels


def _resample_uniform(timestamps, data, labels, target_hz=10.0):
    """线性插值到均匀时间网格。"""
    t0, t_end = timestamps[0], timestamps[-1]
    dt = 1.0 / target_hz * 1e6  # microseconds
    n_new = max(int((t_end - t0) / dt) + 1, 10)
    t_new = np.linspace(t0, t_end, n_new)

    resampled = np.zeros((n_new, data.shape[1]), dtype=np.float32)
    for d in range(data.shape[1]):
        resampled[:, d] = np.interp(t_new, timestamps, data[:, d])

    # 标签: nearest-neighbor
    resampled_labels = np.zeros(n_new, dtype=np.int32)
    for i, tn in enumerate(t_new):
        idx = np.searchsorted(timestamps, tn)
        if idx >= len(labels):
            idx = len(labels) - 1
        elif idx > 0 and tn - timestamps[idx - 1] < timestamps[idx] - tn:
            idx = idx - 1
        resampled_labels[i] = labels[idx]

    return resampled, resampled_labels.astype(np.int32)


def load_basic_data(data_dir=DATA_DIR, target_hz=10.0, max_seqs=None):
    """加载所有 Basic 序列 — 优先从预处理 .npz 加载。

    Args:
        max_seqs: 限制加载序列数 (None = all)
    Returns:
        seqs: list of dicts
    """
    preproc_dir = os.path.join(data_dir, "preprocessed")
    seqs = []

    if os.path.isdir(preproc_dir):
        # 从预处理 .npz 加载 (快速)
        names = sorted(os.listdir(preproc_dir))
        if max_seqs:
            names = names[:max_seqs]
        t0 = time.time()
        for i, fn in enumerate(names):
            if not fn.endswith(".npz"):
                continue
            path = os.path.join(preproc_dir, fn)
            data = np.load(path)
            features = data["features"]
            labels = data["labels"]
            has_fault = labels.sum() > 0
            print(f"  [{i+1}/{len(names)}] {fn[:55]}... "
                  f"{len(labels)} frames, {int(labels.sum())} fault "
                  f"({labels.mean()*100:.1f}%) [{time.time()-t0:.1f}s]")
            t0 = time.time()
            seqs.append({
                "name": fn.replace(".npz", ""),
                "features": features.astype(np.float32),
                "labels": labels.astype(np.int32),
                "n_frames": len(labels),
                "has_fault": has_fault,
            })
        return seqs

    # Fallback: 从原始 CSV 加载 (慢)
    import pandas as pd
    # ... (保留原有逻辑作为 fallback, 此处省略以节省空间)
    raise RuntimeError(
        "No preprocessed data found. Run: uv run python scripts/basic_preprocess.py"
    )


# ─── ConditionalGRU ────────────────────────────────────────────

class ConditionalGRU(nn.Module):
    def __init__(self, feat_dim=NF, hidden=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + NC, hidden),
            nn.ReLU(),
            nn.Linear(hidden, NR),
        )

    def forward(self, x_hist, x_cmd):
        out, _ = self.gru(x_hist)
        return self.head(torch.cat([out[:, -1, :], x_cmd], 1))


def train_conditional_predictor(sequences, hist_len=75, epochs=100,
                                 lr=1e-3, device="cpu", max_samples=200000):
    """用正常帧训练 ConditionalGRU。

    训练样本: 从所有正常帧构造 (history, response, command) 三元组。
    max_samples 限制最大训练样本数以控制内存。
    """
    model = ConditionalGRU(feat_dim=NF, hidden=128).to(device)

    # 收集正常帧索引 (序列级)
    all_Xh, all_Yr, all_Yc = [], [], []
    total_available = 0
    for s in sequences:
        f = s["features"]
        lbs = s["labels"]
        ni = np.where(lbs == 0)[0]
        total_available += max(0, len(ni) - hist_len)

    # 如果样本过多，按比例采样
    sample_ratio = min(1.0, max_samples / max(total_available, 1))
    rng = np.random.RandomState(42)

    for s in sequences:
        f = s["features"]
        lbs = s["labels"]
        ni = np.where(lbs == 0)[0]
        for i in range(hist_len, len(ni)):
            if rng.random() > sample_ratio:
                continue
            idx = ni[i]
            if np.all(lbs[idx - hist_len:idx] == 0):
                all_Xh.append(f[idx - hist_len:idx])
                all_Yr.append(f[idx, :NR])  # response columns
                all_Yc.append(f[idx, NR:])  # command columns (last 7)

    n_train = len(all_Xh)
    if n_train == 0:
        raise RuntimeError("No valid training samples found!")

    Xh_t = torch.tensor(np.array(all_Xh), dtype=torch.float32, device=device)
    Yr_t = torch.tensor(np.array(all_Yr), dtype=torch.float32, device=device)
    Yc_t = torch.tensor(np.array(all_Yc), dtype=torch.float32, device=device)

    dataset = TensorDataset(Xh_t, Yr_t, Yc_t)
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)

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
                  f"avg_loss={total_loss/n_train:.4f} lr={scheduler.get_last_lr()[0]:.1e}")

    print(f"  Training samples: {n_train:,} (ratio={sample_ratio:.1%})")
    return model


# ─── 特征提取 ─────────────────────────────────────────────────

def _window_stats(errors_norm, W, feat_dim):
    n = len(errors_norm)
    out = np.zeros((n, feat_dim * 3), dtype=np.float32)
    if n <= W:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(errors_norm, W, axis=0)  # (n-W+1, feat_dim, W)
    # windows[j] corresponds to errors_norm[j:j+W]
    # For frame i (i >= W), the window is errors_norm[i-W:i] = windows[i-W]
    # Valid i range: W to n-1, corresponding to windows[0:n-W]
    wm = windows[:n - W].mean(axis=2)
    ws = windows[:n - W].std(axis=2)
    wx = windows[:n - W].max(axis=2)
    out[W:, :feat_dim] = wm
    out[W:, feat_dim:feat_dim * 2] = ws
    out[W:, feat_dim * 2:] = wx
    return out


def compute_features(sequences, model, hist_len=75, win_sizes=(10, 30), device="cpu"):
    """用 ConditionalGRU 批量计算预测残差 + 窗口统计特征。"""
    model.eval()

    # Phase 1: 收集正常帧的残差用于 Z-score 归一化
    normal_errors = []
    for s in sequences:
        f = s["features"]
        lbs = s["labels"]
        n = len(f)
        if n <= hist_len:
            continue
        idx = [i for i in range(hist_len, n)
               if lbs[i] == 0 and np.all(lbs[i - hist_len:i] == 0)]
        if len(idx) == 0:
            continue
        # 分块处理避免 OOM (25Hz 下大序列可达 18K 正常帧)
        chunk = 2048
        for start in range(0, len(idx), chunk):
            ci = idx[start:start + chunk]
            Xh = torch.tensor(np.stack([f[i - hist_len:i] for i in ci]),
                              dtype=torch.float32, device=device)
            Xc = torch.tensor(np.stack([f[i, NR:] for i in ci]),
                              dtype=torch.float32, device=device)
            Yt = torch.tensor(np.stack([f[i, :NR] for i in ci]),
                              dtype=torch.float32, device=device)
            pred = model(Xh, Xc)
            normal_errors.append(((Yt - pred).detach() ** 2).cpu().numpy())

    normal_errors = np.vstack(normal_errors)
    mu = np.mean(normal_errors, 0, keepdims=True)
    sd = np.std(normal_errors, 0, keepdims=True) + 1e-8

    # Phase 2: 对全部帧计算残差 + 窗口统计
    results = []
    for si, s in enumerate(sequences):
        f = s["features"]
        n = len(f)
        err = np.zeros((n, NR), dtype=np.float32)
        vd = np.zeros(n, dtype=bool)

        if n > hist_len:
            idx = list(range(hist_len, n))
            # 分块处理避免 OOM
            chunk = 2048
            for start in range(0, len(idx), chunk):
                ci = idx[start:start + chunk]
                Xh = torch.tensor(np.stack([f[i - hist_len:i] for i in ci]),
                                  dtype=torch.float32, device=device)
                Xc = torch.tensor(np.stack([f[i, NR:] for i in ci]),
                                  dtype=torch.float32, device=device)
                Yt = np.stack([f[i, :NR] for i in ci])
                pred = model(Xh, Xc).detach().cpu().numpy()
                for j, i in enumerate(ci):
                    err[i] = (Yt[j] - pred[j]) ** 2
                    vd[i] = True

        en = (err - mu) / sd
        en[:hist_len] = 0
        ws_parts = []
        for W in win_sizes:
            ws_parts.append(_window_stats(en, W, NR))
        results.append({
            "name": s["name"],
            "features": np.hstack(ws_parts),
            "labels": s["labels"],
            "n_frames": n,
            "has_fault": s["has_fault"],
            "valid": vd,
        })
        if (si + 1) % 10 == 0:
            print(f"  Features: {si+1}/{len(sequences)} sequences done")
    return results


# ─── SMM+GOF ──────────────────────────────────────────────────

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

def temporal_smooth(preds, window=5):
    s = preds.copy()
    h = window // 2
    for i in range(h, len(preds) - h):
        s[i] = int(np.median(preds[i - h:i + h + 1]))
    return s


def evaluate(scores, y_all, valid_all, seqs, title=""):
    yv, sv = y_all[valid_all], scores[valid_all]
    if len(np.unique(yv)) < 2:
        print(f"  [{title}] Skipped: only one class in valid data")
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
    pred = (sv > best_t).astype(int)
    f1s = f1_score(yv, temporal_smooth(pred, 5), zero_division=0)
    if title:
        print(f"  [{title}]")
    print(f"  AUC={auc:.3f} AP={ap:.3f} F1={best_f1:.3f} F1s={f1s:.3f}")
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
    off = 0
    for s in seqs:
        n = s["n_frames"]
        svv = scores[off:off + n][s["valid"]]
        yvv = s["labels"][s["valid"]]
        off += n
        if yvv.sum() <= 0 or yvv.sum() >= len(yvv):
            continue
        a = roc_auc_score(yvv, svv)
        if a < 0.7:
            print(f"    ! {s['name'][-55:]}: {a:.3f}")
    return auc, mseq


# ─── 主函数 ────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--target-hz", type=float, default=10.0,
                   help="Target sampling rate (Hz). 10=ALFA-compatible, 25=native IMU rate)")
    p.add_argument("--hist-sec", type=float, default=7.5,
                   help="History window in seconds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-samples", type=int, default=200000)
    p.add_argument("--max-seqs", type=int, default=0)
    a = p.parse_args()

    device = a.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True

    # Derive params from target_hz
    hist_len = int(a.target_hz * a.hist_sec)
    win_sizes = (int(a.target_hz * 1), int(a.target_hz * 3))  # 1s, 3s
    fdim_expected = NR * 3 * len(win_sizes)

    print("=" * 60)
    print(f"Basic Anomaly Detection: ConditionalGRU(hist={hist_len}, "
          f"{a.hist_sec}s@{a.target_hz}Hz) -> {fdim_expected}D -> Galaxy AE")
    print(f"Features: {NR}D response + {NC}D command = {NF}D input")
    print(f"Window sizes: {win_sizes} frames ({win_sizes[0]/a.target_hz:.0f}s, "
          f"{win_sizes[1]/a.target_hz:.0f}s)")
    print("=" * 60)

    # [0] 加载数据
    print("\n[0] Loading Basic data...")
    t0 = time.time()
    seqs = load_basic_data(DATA_DIR, max_seqs=a.max_seqs if a.max_seqs > 0 else None)
    n_fault = int(sum(s["labels"].sum() for s in seqs))
    n_total = sum(s["n_frames"] for s in seqs)
    n_normal_seqs = sum(1 for s in seqs if not s["has_fault"])
    n_fault_seqs = sum(1 for s in seqs if s["has_fault"])
    print(f"  {len(seqs)} seq ({n_normal_seqs} normal + {n_fault_seqs} fault), "
          f"{n_total:,} frames, {n_fault:,} fault ({n_fault/n_total:.1%})")
    print(f"  Load time: {time.time()-t0:.1f}s")

    # 检查数据
    feat_dim = seqs[0]["features"].shape[1]
    print(f"  Feature dim: {feat_dim} "
          f"(expected {NF}: resp_raw={len(RESP_RAW)} + "
          f"cmd_meas={len(CMD_MEAS_PAIRS)} + cmd_raw={len(CMD_RAW)})")

    # [1] 训练 ConditionalGRU
    print(f"\n[1/3] Training ConditionalGRU (hist={hist_len}, "
          f"max_samples={a.max_train_samples:,})...")
    t1 = time.time()
    model = train_conditional_predictor(
        seqs, hist_len=hist_len, epochs=100, device=device,
        max_samples=a.max_train_samples,
    )
    print(f"  Train time: {time.time()-t1:.1f}s")

    # [2] 特征提取
    print("\n[2/3] Computing features...")
    t2 = time.time()
    fseqs = compute_features(seqs, model, hist_len=hist_len,
                             win_sizes=win_sizes, device=device)
    fdim = fseqs[0]["features"].shape[1]
    print(f"  Feature dim: {fdim} (expected {fdim_expected})")
    print(f"  Feature time: {time.time()-t2:.1f}s")

    # 收集所有数据
    X_n = np.vstack([s["features"][(s["labels"] == 0) & s["valid"]]
                     for s in fseqs]).astype(np.float32)
    X_a = np.vstack([s["features"] for s in fseqs]).astype(np.float32)
    y_a = np.concatenate([s["labels"] for s in fseqs])
    va = np.concatenate([s["valid"] for s in fseqs])
    zm, zs = X_n.mean(0, keepdims=True), X_n.std(0, keepdims=True) + 1e-8
    X_nz = (X_n - zm) / zs
    X_az = (X_a - zm) / zs

    print(f"  Normal frames: {len(X_nz):,}, All frames: {len(X_az):,}")

    # [3] 训练 & 评估 Galaxy
    print(f"\n[3/3] Training & evaluating...")

    # SMM direct
    print(f"\n{'─'*60}")
    print("  SMM direct (baseline, no AE)")
    Z_n = torch.tensor(X_nz, dtype=torch.float32, device=device)
    Z_a = torch.tensor(X_az, dtype=torch.float32, device=device)
    sm = fit_smm_with_em(Z_n, k=5, em_iters=3)
    scores_smm = gof_score(Z_a, sm.means_, sm.covars_,
                           sm.weights_.reshape(1, -1), "vector").cpu().numpy()
    evaluate(scores_smm, y_a, va, fseqs, "k=5 SMM")

    # Galaxy AE expand
    print(f"\n{'─'*60}")
    print(f"  Galaxy AE {fdim}->{int(fdim*1.5)}->{int(fdim*0.75)} (expand, k=5)")
    cfg = GalaxyConfig(
        k=5, hidden_dim=int(fdim * 1.5), latent_dim=int(fdim * 0.75),
        seed=a.seed, pretrain_epochs=100, em_iters=3, em_finetune_steps=100,
        gravity_version="vector", score_type="vector",
        device=device, verbose=False,
    )
    g = Galaxy(cfg)
    g.fit(X_nz)
    sg = g.predict_score(X_az)
    evaluate(sg, y_a, va, fseqs, "k=5 Galaxy")

    # Galaxy AE expand k=10
    print(f"\n{'─'*60}")
    print(f"  Galaxy AE {fdim}->{int(fdim*1.5)}->{int(fdim*0.75)} (expand, k=10)")
    cfg10 = GalaxyConfig(
        k=10, hidden_dim=int(fdim * 1.5), latent_dim=int(fdim * 0.75),
        seed=a.seed, pretrain_epochs=100, em_iters=3, em_finetune_steps=100,
        gravity_version="vector", score_type="vector",
        device=device, verbose=False,
    )
    g10 = Galaxy(cfg10)
    g10.fit(X_nz)
    sg10 = g10.predict_score(X_az)
    evaluate(sg10, y_a, va, fseqs, "k=10 Galaxy")

    # 按故障类型分报告
    print(f"\n{'='*60}")
    print("Per Fault Type Summary")
    print(f"{'='*60}")
    fault_types = {
        "RC Failure": [], "GPS Failure": [], "Accelerometer Failure": [],
        "Gyro Failure": [], "Compass Failure": [], "Barometer Failure": [],
    }
    off = 0
    for s in fseqs:
        n = s["n_frames"]
        svv = sg[off:off + n][s["valid"]]
        yvv = s["labels"][s["valid"]]
        off += n
        if yvv.sum() <= 0 or yvv.sum() >= len(yvv):
            continue
        a = roc_auc_score(yvv, svv)
        for ft in fault_types:
            if ft.lower().replace(" ", "") in s["name"].lower().replace(" ", ""):
                fault_types[ft].append(a)
                break

    for ft, aucs in fault_types.items():
        if aucs:
            print(f"  {ft:<25}: {len(aucs):>2} seq  "
                  f"[{min(aucs):.3f} ... {max(aucs):.3f}]  mean={np.mean(aucs):.3f}")

    all_fault = [a for aucs in fault_types.values() for a in aucs]
    print(f"\n  Total fault seq: {len(all_fault)}  Mean AUC: {np.mean(all_fault):.3f}")

    # 最弱序列
    print(f"\nWeakest sequences (AUC < 0.8):")
    off = 0
    found = False
    for s in fseqs:
        n = s["n_frames"]
        svv = sg[off:off + n][s["valid"]]
        yvv = s["labels"][s["valid"]]
        off += n
        if yvv.sum() <= 0 or yvv.sum() >= len(yvv):
            continue
        a = roc_auc_score(yvv, svv)
        if a < 0.8:
            found = True
            print(f"  {s['name']:<60} AUC={a:.3f}")
    if not found:
        print("  (none)")

    print("\nDone.")


if __name__ == "__main__":
    main()
