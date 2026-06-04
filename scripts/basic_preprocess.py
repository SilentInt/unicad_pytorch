"""Basic 数据集预处理：从独立 CSV 加载 → 对齐 10Hz → 去冗余 → 保存紧凑 .npz

输入: data/Basic/{seq_name}/*.csv (28 个独立传感器文件 + 主 CSV 取标签)
输出: data/Basic/preprocessed/{seq_name}.npz (47D features + labels)

每序列 ~6 MB 输入 → ~50 KB 输出，70 序列总计 ~3.5 MB。
"""

import os, sys, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "Basic"
OUT_DIR = DATA_DIR / "preprocessed"

# ─── 需要的传感器文件和列 ────────────────────────────────────

FILE_COLS = {
    "IMU":  ["IMU_GyrX", "IMU_GyrY", "IMU_GyrZ",
             "IMU_AccX", "IMU_AccY", "IMU_AccZ"],
    "ATT":  ["ATT_Roll", "ATT_Pitch", "ATT_Yaw",
             "ATT_DesRoll", "ATT_DesPitch", "ATT_DesYaw"],
    "RATE": ["RATE_R", "RATE_P", "RATE_Y", "RATE_A",
             "RATE_RDes", "RATE_PDes", "RATE_YDes", "RATE_ADes"],
    "XKF1": ["XKF1_VN", "XKF1_VE", "XKF1_VD"],
    "BARO": ["BARO_Alt"],
    "GPS":  ["GPS_Alt", "GPS_Spd", "GPS_VZ"],
    "MAG":  ["MAG_MagX", "MAG_MagY", "MAG_MagZ"],
    "VIBE": ["VIBE_VibeZ"],
    "XKF3": ["XKF3_IVN", "XKF3_IVE", "XKF3_IVD",
             "XKF3_IPN", "XKF3_IPE", "XKF3_IPD",
             "XKF3_IMX"],
    "XKF4": ["XKF4_SV", "XKF4_SM"],
}

# cmd-meas 对: 从已加载的列计算
CMD_MEAS_PAIRS = [
    ("ATT_DesRoll", "ATT_Roll"),
    ("ATT_DesPitch", "ATT_Pitch"),
    ("ATT_DesYaw", "ATT_Yaw"),
    ("RATE_RDes", "RATE_R"),
    ("RATE_PDes", "RATE_P"),
    ("RATE_YDes", "RATE_Y"),
    ("RATE_ADes", "RATE_A"),
]

# 最终 column layout: RESP_RAW (33) + CMD_MEAS (7) + CMD_RAW (7) = 47D
RESP_RAW_COLS = [
    # IMU
    "IMU_GyrX", "IMU_GyrY", "IMU_GyrZ",
    "IMU_AccX", "IMU_AccY", "IMU_AccZ",
    # ATT actual
    "ATT_Roll", "ATT_Pitch", "ATT_Yaw",
    # RATE actual
    "RATE_R", "RATE_P", "RATE_Y", "RATE_A",
    # XKF1 velocity
    "XKF1_VN", "XKF1_VE", "XKF1_VD",
    # BARO
    "BARO_Alt",
    # VIBE
    "VIBE_VibeZ",
    # XKF4
    "XKF4_SV",
    # GPS
    "GPS_Alt", "GPS_Spd", "GPS_VZ",
    # MAG
    "MAG_MagX", "MAG_MagY", "MAG_MagZ",
    # EKF innovation
    "XKF3_IVN", "XKF3_IVE", "XKF3_IVD",
    "XKF3_IPN", "XKF3_IPE", "XKF3_IPD",
    "XKF3_IMX", "XKF4_SM",
]

CMD_RAW_COLS = [
    "ATT_DesRoll", "ATT_DesPitch", "ATT_DesYaw",
    "RATE_RDes", "RATE_PDes", "RATE_YDes", "RATE_ADes",
]


def _yaw_diff(des, actual):
    d = np.asarray(des, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return (d + 180.0) % 360.0 - 180.0


def _load_sensor_csv(path, cols, fillna=0.0):
    """读取单个传感器 CSV, groupby(Time).last() 去重."""
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path, usecols=["Time"] + cols)
    df = df.groupby("Time", as_index=False, sort=True).last()
    df = df.fillna(fillna)
    return df["Time"].values.astype(np.float64), df[cols].values.astype(np.float32)


def preprocess_sequence(seq_dir, target_hz=10.0):
    """预处理单个序列: 从主 CSV 加载 → 去重 → 构造特征 → 插值 → 保存."""
    sn = os.path.basename(seq_dir)
    main_csv = os.path.join(seq_dir, f"{sn}.csv")
    if not os.path.exists(main_csv):
        return None, None

    # 1. 从主 CSV 一次读取所有需要的列
    usecols = ["Time"] + RESP_RAW_COLS + CMD_RAW_COLS + ["Status"]
    df = pd.read_csv(main_csv, usecols=usecols)
    df = df.groupby("Time", as_index=False, sort=True).last()
    if len(df) < 100:
        return None, None

    ts = df["Time"].values.astype(np.float64)
    n = len(df)

    # 2. 构造特征矩阵: RESP_RAW (33) + CMD_MEAS (7) + CMD_RAW (7) = 47D
    n_feat = len(RESP_RAW_COLS) + len(CMD_MEAS_PAIRS) + len(CMD_RAW_COLS)
    data = np.zeros((n, n_feat), dtype=np.float32)

    # 2a. RESP_RAW 列
    for j, col in enumerate(RESP_RAW_COLS):
        if col in df.columns:
            data[:, j] = df[col].fillna(0).values.astype(np.float32)

    # 2b. CMD_RAW 列
    cmd_offset = len(RESP_RAW_COLS) + len(CMD_MEAS_PAIRS)
    for j, col in enumerate(CMD_RAW_COLS):
        if col in df.columns:
            data[:, cmd_offset + j] = df[col].fillna(0).values.astype(np.float32)

    # 2c. cmd-meas errors (从原始值计算，匹配原 basic_detect 行为)
    for j, (cmd_col, resp_col) in enumerate(CMD_MEAS_PAIRS):
        cv = df[cmd_col].fillna(0).values.astype(np.float32) if cmd_col in df.columns else np.zeros(n)
        rv = df[resp_col].fillna(0).values.astype(np.float32) if resp_col in df.columns else np.zeros(n)
        if "Yaw" in cmd_col:
            data[:, len(RESP_RAW_COLS) + j] = _yaw_diff(cv, rv)
        else:
            data[:, len(RESP_RAW_COLS) + j] = cv - rv

    # 3. Yaw unwrap (必须在插值之前)
    yaw_actual_idx = RESP_RAW_COLS.index("ATT_Yaw")
    yaw_cmd_idx = cmd_offset + CMD_RAW_COLS.index("ATT_DesYaw")
    data[:, yaw_actual_idx] = np.unwrap(data[:, yaw_actual_idx], period=360.0)
    data[:, yaw_cmd_idx] = np.unwrap(data[:, yaw_cmd_idx], period=360.0)

    # 4. 线性插值到均匀时间网格
    t0, t_end = ts[0], ts[-1]
    dt = 1.0 / target_hz * 1e6
    n_new = max(int((t_end - t0) / dt) + 1, 100)
    t_grid = np.linspace(t0, t_end, n_new)

    features = np.zeros((n_new, n_feat), dtype=np.float32)
    for d in range(n_feat):
        features[:, d] = np.interp(t_grid, ts, data[:, d])

    # 5. 标签: nearest-neighbor
    label_vals = (df["Status"].fillna(0).values > 0).astype(np.int32)
    labels = np.zeros(n_new, dtype=np.int32)
    for i, tn in enumerate(t_grid):
        idx = np.searchsorted(ts, tn)
        if idx >= len(label_vals):
            idx = len(label_vals) - 1
        elif idx > 0 and tn - ts[idx - 1] < ts[idx] - tn:
            idx = idx - 1
        labels[i] = label_vals[idx]

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32), labels.astype(np.int32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-hz", type=float, default=10.0)
    p.add_argument("--max-seqs", type=int, default=0)
    a = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    names = sorted(os.listdir(DATA_DIR))
    names = [n for n in names if os.path.isdir(os.path.join(DATA_DIR, n))]
    if a.max_seqs > 0:
        names = names[:a.max_seqs]

    NR = len(RESP_RAW_COLS) + len(CMD_MEAS_PAIRS)
    NC = len(CMD_RAW_COLS)
    NF = NR + NC
    print(f"Preprocessing {len(names)} sequences → {OUT_DIR}")
    print(f"Output: {NR}D response + {NC}D command = {NF}D features @ {a.target_hz}Hz")
    print()

    total_frames = 0
    total_fault = 0
    t0 = time.time()

    for i, sn in enumerate(names):
        seq_dir = os.path.join(DATA_DIR, sn)
        out_path = os.path.join(OUT_DIR, f"{sn}.npz")

        print(f"[{i+1}/{len(names)}] {sn[:55]}...", end=" ", flush=True)
        t_seq = time.time()
        features, labels = preprocess_sequence(seq_dir, a.target_hz)

        if features is None:
            print("SKIP")
            continue

        np.savez_compressed(out_path, features=features, labels=labels)
        n_frames = len(features)
        n_fault = int(labels.sum())
        total_frames += n_frames
        total_fault += n_fault
        fsize = os.path.getsize(out_path)
        print(f"{n_frames} frames, {n_fault} fault "
              f"({n_fault/n_frames*100:.1f}%), "
              f"{fsize/1024:.0f} KB [{time.time()-t_seq:.1f}s]")

    print(f"\n{'='*60}")
    print(f"Done: {len(names)} seq, {total_frames:,} frames, "
          f"{total_fault:,} fault ({total_fault/total_frames:.1%})")
    print(f"Total time: {time.time()-t0:.1f}s")
    print(f"Output: {OUT_DIR}/")


if __name__ == "__main__":
    main()
