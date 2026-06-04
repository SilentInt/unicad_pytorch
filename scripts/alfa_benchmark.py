"""ALFA 异常检测 Benchmark — 多架构多种子对比。

测试架构:
  V3:   ConditionalGRU + GRUPredictor + 309D + Galaxy AE (压缩)
  V7-S: ConditionalGRU + 78D + SMM 直接
  V7-G: ConditionalGRU + 78D + Galaxy AE (扩展 128→64)

用法: uv run python scripts/alfa_benchmark.py --device cuda:1 --seeds 42,123,456
"""

import argparse
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from unicad_torch import Galaxy, GalaxyConfig
from unicad_torch.gof import gof_score
from unicad_torch.smm_torch import SMMTorch

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ALFA"
RESPONSE_IDX = list(range(0, 13))
COMMAND_IDX = list(range(13, 18))
N_RESPONSE = len(RESPONSE_IDX)
N_COMMAND = len(COMMAND_IDX)
IMU_DIM = 6

CMD_RESP_PAIRS = [
    (13, 1), (13, 7), (14, 5), (14, 11), (14, 10),
    (15, 2), (15, 8), (16, 0), (17, 0), (16, 6), (17, 6),
]

# ═══════════════════════ 数据加载 ═══════════════════════

def _load_csv_numeric(path: str):
    import csv
    with open(path) as f:
        reader = csv.reader(f); header = next(reader); rows = list(reader)
    if not rows: return np.empty((0, 0)), []
    mask = []
    for v in rows[0]:
        try: float(v); mask.append(True)
        except ValueError: mask.append(False)
    idx = [i for i, m in enumerate(mask) if m]
    data = np.array([[float(row[i]) for i in idx] for row in rows], dtype=np.float64)
    return data, [header[i] for i in idx]

def load_alfa_data(data_dir=DATA_DIR, target_hz=10.0):
    IMU_COLS = [
        "field.angular_velocity.x", "field.angular_velocity.y", "field.angular_velocity.z",
        "field.linear_acceleration.x", "field.linear_acceleration.y", "field.linear_acceleration.z",
    ]
    NAV_ERR_COLS = ["field.alt_error", "field.aspd_error", "field.xtrack_error"]
    RC_COLS = ["field.channels1", "field.channels2", "field.channels3",
               "field.channels4", "field.channels5"]
    VFR_COLS = ["field.throttle", "field.altitude", "field.climb"]
    sequences = []
    for seq_name in sorted(os.listdir(data_dir)):
        seq_dir = os.path.join(data_dir, seq_name)
        if not os.path.isdir(seq_dir): continue
        imu_files = [f for f in os.listdir(seq_dir) if "imu-data_raw" in f]
        if not imu_files: continue
        imu_data, imu_header = _load_csv_numeric(os.path.join(seq_dir, imu_files[0]))
        if len(imu_data) < 10: continue
        imu_times = imu_data[:, 0] / 1e9
        grid = np.arange(imu_times[0], imu_times[-1], 1.0 / target_hz)
        n_frames = len(grid)
        def _interp(data, header, cols):
            if len(data) < 2: return np.zeros((n_frames, len(cols)), dtype=np.float32)
            times = data[:, 0] / 1e9; idxs = [header.index(c) for c in cols]
            vals = data[:, idxs].astype(np.float32)
            out = np.empty((n_frames, len(cols)), dtype=np.float32)
            for d in range(len(cols)): out[:, d] = np.interp(grid, times, vals[:, d])
            return out
        def _interp_nav(topic):
            fname = [f for f in os.listdir(seq_dir)
                     if topic.replace("-", "_") in f or topic in f]
            if not fname: return np.zeros((n_frames, 1), dtype=np.float32)
            d, h = _load_csv_numeric(os.path.join(seq_dir, fname[0]))
            if len(d) < 2: return np.zeros((n_frames, 1), dtype=np.float32)
            times = d[:, 0] / 1e9
            cmd_i, meas_i = h.index("field.commanded"), h.index("field.measured")
            err = (d[:, cmd_i] - d[:, meas_i]).reshape(-1, 1).astype(np.float32)
            out = np.empty((n_frames, 1), dtype=np.float32)
            out[:, 0] = np.interp(grid, times, err[:, 0])
            return out
        feat_imu = _interp(imu_data, imu_header, IMU_COLS)
        nav_errs = [_interp_nav(t) for t in
                    ["nav_info-roll", "nav_info-pitch", "nav_info-yaw", "nav_info-airspeed"]]
        feat_nav_err = np.hstack(nav_errs)
        err_file = [f for f in os.listdir(seq_dir) if "nav_info-errors" in f]
        if err_file:
            d, h = _load_csv_numeric(os.path.join(seq_dir, err_file[0]))
            feat_nav = _interp(d, h, NAV_ERR_COLS)
        else: feat_nav = np.zeros((n_frames, 3), dtype=np.float32)
        rc_file = [f for f in os.listdir(seq_dir) if "rc-out" in f]
        if rc_file:
            d, h = _load_csv_numeric(os.path.join(seq_dir, rc_file[0]))
            feat_rc = _interp(d, h, RC_COLS)
        else: feat_rc = np.zeros((n_frames, 5), dtype=np.float32)
        vfr_file = [f for f in os.listdir(seq_dir) if "vfr_hud" in f]
        if vfr_file:
            d, h = _load_csv_numeric(os.path.join(seq_dir, vfr_file[0]))
            feat_vfr = _interp(d, h, VFR_COLS)
        else: feat_vfr = np.zeros((n_frames, 3), dtype=np.float32)
        features = np.hstack([feat_imu, feat_nav_err, feat_nav, feat_rc, feat_vfr]).astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        fault_intervals = []
        for fname in os.listdir(seq_dir):
            if "failure_status" not in fname: continue
            d, _ = _load_csv_numeric(os.path.join(seq_dir, fname))
            if len(d) == 0: continue
            ts = d[:, 0] / 1e9; start, prev = ts[0], ts[0]
            for t in ts[1:]:
                if t - prev > 1.0: fault_intervals.append((start, prev)); start = t
                prev = t
            fault_intervals.append((start, prev))
        labels = np.zeros(n_frames, dtype=np.int32)
        for s, e in fault_intervals: labels[(grid >= s) & (grid <= e)] = 1
        sequences.append({
            "name": seq_name, "features": features, "labels": labels,
            "n_frames": n_frames, "has_fault": len(fault_intervals) > 0,
        })
    return sequences

# ═══════════════════════ 预测器 ═══════════════════════

class GRUPredictor(nn.Module):
    def __init__(self, feat_dim, hidden=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, feat_dim)
    def forward(self, x): out, _ = self.gru(x); return self.fc(out[:, -1, :])

class ConditionalGRU(nn.Module):
    def __init__(self, feat_dim=21, hidden=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden+N_COMMAND, hidden), nn.ReLU(), nn.Linear(hidden, N_RESPONSE))
    def forward(self, x_hist, x_cmd):
        out, _ = self.gru(x_hist); return self.head(torch.cat([out[:, -1, :], x_cmd], dim=1))

# ═══════════════════════ V3 特征 ═══════════════════════

def _window_stats(errors_norm, W, feat_dim):
    n = len(errors_norm); out = np.zeros((n, feat_dim*3), dtype=np.float32)
    for i in range(W, n):
        w = errors_norm[i-W:i]
        out[i, :feat_dim] = w.mean(0)
        out[i, feat_dim:feat_dim*2] = w.std(0)
        out[i, feat_dim*2:] = w.max(0)
    return out

def _fft_features(errors_norm, W, imu_dim=IMU_DIM, n_freq=5):
    n = len(errors_norm); out = np.zeros((n, imu_dim*n_freq), dtype=np.float32)
    for i in range(W, n):
        w = errors_norm[i-W:i, :imu_dim]
        spec = np.abs(np.fft.rfft(w, axis=0))
        out[i] = spec[1:n_freq+1, :].T.flatten()
    return out

def _cross_corr_features(feat, W=10):
    n, n_pairs = len(feat), len(CMD_RESP_PAIRS)
    out = np.zeros((n, n_pairs), dtype=np.float32)
    for i in range(W, n):
        w = feat[i-W:i]
        for p, (ci, ri) in enumerate(CMD_RESP_PAIRS):
            c, r = w[:, ci], w[:, ri]
            c_s, r_s = c.std()+1e-8, r.std()+1e-8
            out[i, p] = np.mean((c-c.mean())*(r-r.mean()))/(c_s*r_s)
    return out

def _cov_deviation_features(feat, normal_cov, W=10):
    n = len(feat); out = np.zeros((n, 1), dtype=np.float32)
    for i in range(W, n):
        cov = np.cov(feat[i-W:i, RESPONSE_IDX].T)
        out[i, 0] = np.sqrt(np.sum((cov-normal_cov)**2))
    return out

# ═══════════════════════ 评估 ═══════════════════════

def fit_smm_with_em(Z, k=5, em_iters=3, outlier_ratio=0.01):
    Z = Z.clone(); smm = None
    for i in range(em_iters):
        smm = SMMTorch(n_components=k, n_iter=200, tol=1e-4)
        smm.fit(Z)
        if i < em_iters-1 and smm.means_ is not None:
            scores = gof_score(Z, smm.means_, smm.covars_, smm.weights_.reshape(1,-1), "vector")
            mask = scores <= torch.quantile(scores, 1.0-outlier_ratio)
            if mask.sum() > k: Z = Z[mask]
    return smm

def compute_metrics(scores, y_all, valid_all, seqs):
    yv, sv = y_all[valid_all], scores[valid_all]
    auc = roc_auc_score(yv, sv); ap = average_precision_score(yv, sv)
    # best F1 by percentile sweep
    best_f1 = max(f1_score(yv, (sv > np.percentile(sv[yv==0], pct)).astype(int))
                  for pct in np.arange(90, 100, 0.5))
    # per-seq mean AUC
    offset = 0; seq_aucs = []
    for s in seqs:
        n = s["n_frames"]; svv = scores[offset:offset+n][s["valid"]]; yvv = s["labels"][s["valid"]]; offset += n
        if yvv.sum() > 0 and yvv.sum() < len(yvv): seq_aucs.append(roc_auc_score(yvv, svv))
    return auc, ap, best_f1, np.mean(seq_aucs) if seq_aucs else float("nan")

# ═══════════════════════ 架构 ═══════════════════════

class V3Pipeline:
    """ConditionalGRU + GRUPredictor + 309D + Galaxy AE"""
    def __init__(self, device="cpu", seed=42):
        self.device = device; self.seed = seed
    def train(self, sequences):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        feat_dim = sequences[0]["features"].shape[1]
        # 训练两个预测器
        self.temporal_pred = GRUPredictor(feat_dim).to(self.device)
        self.cond_pred = ConditionalGRU(feat_dim).to(self.device)
        X_h, Y_f, Y_r, Y_c = [], [], [], []
        for seq in sequences:
            feat, labels = seq["features"], seq["labels"]
            ni = np.where(labels==0)[0]
            for i in range(10, len(ni)):
                idx = ni[i]
                if np.all(labels[idx-10:idx]==0):
                    X_h.append(feat[idx-10:idx]); Y_f.append(feat[idx])
                    Y_r.append(feat[idx, RESPONSE_IDX]); Y_c.append(feat[idx, COMMAND_IDX])
        X_h = torch.tensor(np.array(X_h), dtype=torch.float32, device=self.device)
        Y_f_t = torch.tensor(np.array(Y_f), dtype=torch.float32, device=self.device)
        Y_r_t = torch.tensor(np.array(Y_r), dtype=torch.float32, device=self.device)
        Y_c_t = torch.tensor(np.array(Y_c), dtype=torch.float32, device=self.device)
        opt_t = torch.optim.Adam(self.temporal_pred.parameters(), lr=1e-3)
        opt_c = torch.optim.Adam(self.cond_pred.parameters(), lr=1e-3)
        for _ in range(100):
            loss_t = F.mse_loss(self.temporal_pred(X_h), Y_f_t)
            loss_c = F.mse_loss(self.cond_pred(X_h, Y_c_t), Y_r_t)
            opt_t.zero_grad(); loss_t.backward(); opt_t.step()
            opt_c.zero_grad(); loss_c.backward(); opt_c.step()
        # 提取 V3 309D 特征
        normal_cov = None
        normal_resp = [seq["features"][seq["labels"]==0][:, RESPONSE_IDX] for seq in sequences]
        normal_cov = np.cov(np.vstack(normal_resp).T)
        self.feature_seqs = []
        for seq in sequences:
            feat = seq["features"]; n = len(feat); nm = seq["labels"]==0
            te = np.zeros((n, feat_dim), dtype=np.float32)
            ce = np.zeros((n, N_RESPONSE), dtype=np.float32)
            with torch.no_grad():
                for i in range(10, n):
                    xh = torch.tensor(feat[i-10:i], dtype=torch.float32, device=self.device).unsqueeze(0)
                    te[i] = (feat[i]-self.temporal_pred(xh).squeeze(0).cpu().numpy())**2
                    xc = torch.tensor(feat[i, COMMAND_IDX], dtype=torch.float32, device=self.device).unsqueeze(0)
                    ce[i] = (feat[i, RESPONSE_IDX]-self.cond_pred(xh, xc).squeeze(0).cpu().numpy())**2
            t_mu = te[nm].mean(0, keepdims=True); t_sd = te[nm].std(0, keepdims=True)+1e-8
            c_mu = ce[nm].mean(0, keepdims=True); c_sd = ce[nm].std(0, keepdims=True)+1e-8
            ten = (te-t_mu)/t_sd; ten[:10]=0; cen = (ce-c_mu)/c_sd; cen[:10]=0
            ws5_t  = _window_stats(ten, 5, feat_dim)
            ws10_t = _window_stats(ten, 10, feat_dim)
            ws30_t = _window_stats(ten, 30, feat_dim)
            ws10_c = _window_stats(cen, 10, N_RESPONSE)
            ws30_c = _window_stats(cen, 30, N_RESPONSE)
            fft = _fft_features(ten, 30)
            xcorr = _cross_corr_features(feat)
            covd = _cov_deviation_features(feat, normal_cov)
            v = np.ones(n, dtype=bool); v[:10] = False
            self.feature_seqs.append({"name": seq["name"], "features": np.hstack([ws5_t, ws10_t, ws30_t, ws10_c, ws30_c, fft, xcorr, covd]),
                                       "labels": seq["labels"], "n_frames": n, "has_fault": seq["has_fault"], "valid": v})
    def evaluate(self):
        X_n = np.vstack([s["features"][(s["labels"]==0)&s["valid"]] for s in self.feature_seqs]).astype(np.float32)
        X_a = np.vstack([s["features"] for s in self.feature_seqs]).astype(np.float32)
        y_a = np.concatenate([s["labels"] for s in self.feature_seqs])
        v_a = np.concatenate([s["valid"] for s in self.feature_seqs])
        zm, zs = X_n.mean(0, keepdims=True), X_n.std(0, keepdims=True)+1e-8
        X_nz, X_az = (X_n-zm)/zs, (X_a-zm)/zs
        cfg = GalaxyConfig(k=5, hidden_dim=128, latent_dim=64, seed=self.seed,
                           pretrain_epochs=100, em_iters=3, em_finetune_steps=100,
                           gravity_version="vector", score_type="vector",
                           device=self.device, verbose=False)
        g = Galaxy(cfg); g.fit(X_nz)
        scores = g.predict_score(X_az)
        return compute_metrics(scores, y_a, v_a, self.feature_seqs)

class V7Pipeline:
    """ConditionalGRU + 78D + 直接 SMM or Galaxy AE 扩展"""
    def __init__(self, device="cpu", seed=42, use_ae=True):
        self.device = device; self.seed = seed; self.use_ae = use_ae
    def train(self, sequences):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.cond_pred = ConditionalGRU().to(self.device)
        X_h, Y_r, Y_c = [], [], []
        for seq in sequences:
            feat, labels = seq["features"], seq["labels"]
            ni = np.where(labels==0)[0]
            for i in range(10, len(ni)):
                idx = ni[i]
                if np.all(labels[idx-10:idx]==0):
                    X_h.append(feat[idx-10:idx]); Y_r.append(feat[idx, RESPONSE_IDX])
                    Y_c.append(feat[idx, COMMAND_IDX])
        X_h = torch.tensor(np.array(X_h), dtype=torch.float32, device=self.device)
        Y_r_t = torch.tensor(np.array(Y_r), dtype=torch.float32, device=self.device)
        Y_c_t = torch.tensor(np.array(Y_c), dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(self.cond_pred.parameters(), lr=1e-3)
        for _ in range(100):
            loss = F.mse_loss(self.cond_pred(X_h, Y_c_t), Y_r_t)
            opt.zero_grad(); loss.backward(); opt.step()
        # 78D 特征
        normal_errors = []
        for seq in sequences:
            feat, labels = seq["features"], seq["labels"]
            mask = labels==0; n=len(feat)
            for i in range(10, n):
                if mask[i] and np.all(mask[i-10:i]):
                    with torch.no_grad():
                        xh = torch.tensor(feat[i-10:i], dtype=torch.float32, device=self.device).unsqueeze(0)
                        xc = torch.tensor(feat[i, COMMAND_IDX], dtype=torch.float32, device=self.device).unsqueeze(0)
                        normal_errors.append((feat[i, RESPONSE_IDX]-self.cond_pred(xh, xc).squeeze(0).cpu().numpy())**2)
        mu = np.mean(normal_errors, 0, keepdims=True); sd = np.std(normal_errors, 0, keepdims=True)+1e-8
        self.feature_seqs = []
        for seq in sequences:
            feat = seq["features"]; n=len(feat)
            errors = np.zeros((n, N_RESPONSE), dtype=np.float32); valid = np.zeros(n, dtype=bool)
            with torch.no_grad():
                for i in range(10, n):
                    xh = torch.tensor(feat[i-10:i], dtype=torch.float32, device=self.device).unsqueeze(0)
                    xc = torch.tensor(feat[i, COMMAND_IDX], dtype=torch.float32, device=self.device).unsqueeze(0)
                    errors[i] = (feat[i, RESPONSE_IDX]-self.cond_pred(xh, xc).squeeze(0).cpu().numpy())**2
                    valid[i]=True
            en = (errors-mu)/sd; en[:10]=0
            ws10 = _window_stats(en, 10, N_RESPONSE); ws30 = _window_stats(en, 30, N_RESPONSE)
            self.feature_seqs.append({"name": seq["name"], "features": np.hstack([ws10, ws30]),
                                       "labels": seq["labels"], "n_frames": n,
                                       "has_fault": seq["has_fault"], "valid": valid})
    def evaluate(self):
        X_n = np.vstack([s["features"][(s["labels"]==0)&s["valid"]] for s in self.feature_seqs]).astype(np.float32)
        X_a = np.vstack([s["features"] for s in self.feature_seqs]).astype(np.float32)
        y_a = np.concatenate([s["labels"] for s in self.feature_seqs])
        v_a = np.concatenate([s["valid"] for s in self.feature_seqs])
        zm, zs = X_n.mean(0, keepdims=True), X_n.std(0, keepdims=True)+1e-8
        X_nz, X_az = (X_n-zm)/zs, (X_a-zm)/zs
        if self.use_ae:
            cfg = GalaxyConfig(k=5, hidden_dim=128, latent_dim=64, seed=self.seed,
                               pretrain_epochs=100, em_iters=3, em_finetune_steps=100,
                               gravity_version="vector", score_type="vector",
                               device=self.device, verbose=False)
            g = Galaxy(cfg); g.fit(X_nz)
            scores = g.predict_score(X_az)
        else:
            Z_n = torch.tensor(X_nz, dtype=torch.float32, device=self.device)
            Z_a = torch.tensor(X_az, dtype=torch.float32, device=self.device)
            smm = fit_smm_with_em(Z_n, k=5, em_iters=3)
            scores = gof_score(Z_a, smm.means_, smm.covars_, smm.weights_.reshape(1,-1), "vector").cpu().numpy()
        return compute_metrics(scores, y_a, v_a, self.feature_seqs)

# ═══════════════════════ 主函数 ═══════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seeds", default="42,123,456", help="comma-separated seeds")
    args = parser.parse_args()
    device = args.device
    if not torch.cuda.is_available(): device = "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]

    print("Loading ALFA data...")
    sequences = load_alfa_data()

    architectures = [
        ("V3  309D + Galaxy 128→64", V3Pipeline),
        ("V7  78D + SMM direct",     partial(V7Pipeline, use_ae=False)),
        ("V7  78D + Galaxy 128→64",  partial(V7Pipeline, use_ae=True)),
    ]

    print(f"\n{'='*85}")
    print(f"  ALFA Anomaly Detection Benchmark ({len(seeds)} seeds × {len(architectures)} archs)")
    print(f"{'='*85}")

    all_results = {}
    for arch_name, ArchClass in architectures:
        metrics_per_seed = []
        for seed in seeds:
            t0 = time.time()
            pipe = ArchClass(device=device, seed=seed)
            pipe.train(sequences)
            auc, ap, f1, mean_seq = pipe.evaluate()
            elapsed = time.time()-t0
            metrics_per_seed.append((auc, ap, f1, mean_seq))
            print(f"  {arch_name:<30} seed={seed:<4}  AUC={auc:.3f}  MeanSeq={mean_seq:.3f}  ({elapsed:.0f}s)")

        aucs, aps, f1s, seqs = zip(*metrics_per_seed)
        all_results[arch_name] = {
            "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
            "ap_mean": np.mean(aps), "ap_std": np.std(aps),
            "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
            "seq_mean": np.mean(seqs), "seq_std": np.std(seqs),
        }

    # 汇总表
    print(f"\n{'='*85}")
    print(f"  Benchmark Summary (mean ± std over {len(seeds)} seeds)")
    print(f"{'='*85}")
    print(f"  {'Architecture':<35} {'Global AUC':>15} {'Mean Seq AUC':>15}")
    print(f"  {'-'*65}")
    baseline_seq = all_results[architectures[0][0]]["seq_mean"]
    for arch_name, _ in architectures:
        r = all_results[arch_name]
        d_seq = r["seq_mean"] - baseline_seq
        print(f"  {arch_name:<35} {r['auc_mean']:.3f}±{r['auc_std']:.3f}      {r['seq_mean']:.3f}±{r['seq_std']:.3f}  ({d_seq:+.3f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
