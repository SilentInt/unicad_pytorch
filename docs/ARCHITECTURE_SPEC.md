# 模型架构详细说明 — 供绘制架构图参考

---

## 一、整体数据流（三阶段）

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Stage 1: 因果预测                            │
│                                                                     │
│  输入: 时序窗口 x_{t-75:t-1} (75帧) + 当前指令 cmd_t                  │
│  模型: ConditionalGRU                                                │
│  输出: 预测响应 resp̂_t (13D/40D)                                     │
│  残差: e_t = (resp_t − resp̂_t)² (13D/40D)                            │
│                                                                     │
│  训练: 仅正常帧, MSE(resp̂_t, resp_t)                                 │
│  参数: ~47K (ALFA) / ~97K (Basic)                                   │
│  设备: GPU                                                          │
├─────────────────────────────────────────────────────────────────────┤
│                         Stage 2: 特征工程                            │
│                                                                     │
│  输入: 残差序列 e_t (13D/40D)                                        │
│  步骤:                                                               │
│    1. Z-score 归一化 (μ, σ 从正常帧估计)                              │
│    2. 双尺度滑动窗口: W=10 (1s), W=30 (3s)                            │
│    3. 每窗口取 [mean, std, max] → 各 39D/120D                         │
│  输出: 78D (ALFA) / 240D (Basic) 特征向量                             │
│                                                                     │
│  参数: 0 (纯计算)                                                     │
│  设备: CPU 或 GPU                                                    │
├─────────────────────────────────────────────────────────────────────┤
│                         Stage 3: 异常检测                            │
│                                                                     │
│  输入: 特征向量 x (78D/240D)                                         │
│  方法: 马氏距离                                                       │
│    μ = normal_features.mean(0)                                      │
│    Σ⁻¹ = np.linalg.inv(cov + εI)                                    │
│    score = (x − μ)ᵀ Σ⁻¹ (x − μ)                                      │
│  输出: 异常分数 (标量, 越高越异常)                                     │
│                                                                     │
│  训练: 0s (解析解)                                                    │
│  参数: d×d 协方差矩阵 (78² / 240² float32)                            │
│  设备: CPU                                                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、Stage 1 详解：ConditionalGRU

### 2.1 设计动机

无人机是因果系统：舵面指令（因）→ 传感器响应（果）。故障破坏这个因果映射。

- 正常飞行：给定指令，响应可预测
- 故障飞行：同样指令，响应偏离预期 → 预测残差增大

### 2.2 输入通道角色分离

以 **ALFA (21D)** 为例：

```
通道索引   维度   来源                         角色
────────────────────────────────────────────────────
 0 –  5    6D    IMU (角速度×3 + 加速度×3)       响应 RESP
 6 –  9    4D    nav_info cmd−meas              响应 RESP
                 (roll/pitch/yaw/airspeed
                  commanded − measured)
10 – 12    3D    nav_errors                      响应 RESP
                 (alt_error, aspd_error,
                  xtrack_error)
────────────────────────────────────────────────────
13 – 17    5D    rc-out channels1–5              指令 CMD
                 (俯仰/油门/偏航/左副翼/右副翼)
────────────────────────────────────────────────────
18 – 20    3D    vfr_hud                         上下文 CTX
                 (throttle, altitude, climb)
```

以 **Basic (47D)** 为例：

```
通道索引   维度   来源                         角色
────────────────────────────────────────────────────
 0 –  5    6D    IMU (GyrX/Y/Z, AccX/Y/Z)       响应 RESP
 6 –  8    3D    ATT actual (Roll,Pitch,Yaw)     响应 RESP
 9 – 12    4D    RATE actual (R,P,Y,A)           响应 RESP
13 – 15    3D    XKF1 velocity (VN,VE,VD)        响应 RESP
16         1D    BARO Alt                         响应 RESP
17         1D    VIBE VibeZ                       响应 RESP
18         1D    XKF4 SV                          响应 RESP
19 – 21    3D    GPS (Alt,Spd,VZ)                 响应 RESP
22 – 24    3D    MAG (MagX,MagY,MagZ)             响应 RESP
25 – 27    3D    XKF3 vel innovation (IVN/E/D)    响应 RESP
28 – 30    3D    XKF3 pos innovation (IPN/E/D)    响应 RESP
31 – 32    2D    XKF3 IMX + XKF4 SM               响应 RESP
────────────────────────────────────────────────────
33 – 39    7D    cmd−meas errors                  响应 RESP
                 ATT Des−Actual (3)
                 RATE Des−Actual (4)
────────────────────────────────────────────────────
40 – 46    7D    ATT_Des(3) + RATE_Des(4)         指令 CMD
```

### 2.3 模型结构

```
                    x_{t-75:t-1}                   cmd_t
                    (75, feat_dim)               (cmd_dim,)
                         │                           │
                         ▼                           │
              ┌───────────────────┐                  │
              │   GRU 层 1         │                  │
              │   input:  feat_dim │                  │
              │   hidden: 64/128   │                  │
              │   batch_first=True │                  │
              └────────┬──────────┘                  │
                       │                            │
                       ▼                            │
              ┌───────────────────┐                  │
              │   GRU 层 2         │                  │
              │   input:  64/128   │                  │
              │   hidden: 64/128   │                  │
              │   batch_first=True │                  │
              └────────┬──────────┘                  │
                       │                            │
                       ▼                            │
                  h_t (hidden_dim,)                  │
                  (取最后一帧的隐状态)                  │
                       │                            │
                       └──────────┬─────────────────┘
                                  │
                                  ▼
                        concat(h_t, cmd_t)
                        (hidden_dim + cmd_dim,)
                                  │
                                  ▼
                       ┌───────────────────┐
                       │   Linear + ReLU    │
                       │   in:  hidden+cmd  │
                       │   out: hidden      │
                       └────────┬──────────┘
                                │
                                ▼
                       ┌───────────────────┐
                       │   Linear           │
                       │   in:  hidden      │
                       │   out: resp_dim    │
                       └────────┬──────────┘
                                │
                                ▼
                             resp̂_t
                          (resp_dim,)
```

### 2.4 维度参数表

| 参数 | ALFA | Basic |
|------|------|-------|
| feat_dim (输入总维度) | 21 | 47 |
| resp_dim (预测目标) | 13 | 40 |
| cmd_dim (指令条件) | 5 | 7 |
| ctx_dim (上下文) | 3 | — |
| hidden_dim (GRU 隐层) | 64 | 128 |
| num_layers (GRU 层数) | 2 | 2 |
| hist_len (历史帧数) | 75 (7.5s) | 75 (7.5s) |
| 总参数量 | ~47K | ~97K |

### 2.5 训练配置

| 参数 | ALFA | Basic |
|------|------|-------|
| 训练数据 | 仅正常帧 | 仅正常帧 |
| 训练样本数 | ~36K | ~147K (采样至 200K) |
| Epochs | 100 | 100 |
| Optimizer | Adam (lr=1e-3) | Adam (lr=1e-3) |
| Scheduler | — | StepLR (step=30, γ=0.5) |
| Batch size | 全量 | 2048 |
| Loss | MSE(resp̂, resp) | MSE(resp̂, resp) |
| 设备 | GPU | GPU |

---

## 三、Stage 2 详解：特征工程

### 3.1 数据流

```
e_t (N帧 × resp_dim)
  │  每帧一个 resp_dim 维的残差向量
  │  e_t = (resp_t − resp̂_t)²
  │
  ▼
Z-score 归一化
  │  μ = mean(e_normal), σ = std(e_normal) + 1e-8
  │  e_norm = (e − μ) / σ
  │
  ├────────────────────────────────────────────┐
  │                                            │
  ▼                                            ▼
滑动窗口 W=10 (1.0s @ 10Hz)                   滑动窗口 W=30 (3.0s @ 10Hz)
  对每帧 i (i ≥ W):                             对每帧 i (i ≥ W):
    window = e_norm[i-W : i]                      window = e_norm[i-W : i]
                                                  
    feat[i, 0:resp]      = window.mean(0)         feat[i, 0:resp]      = window.mean(0)
    feat[i, resp:2*resp] = window.std(0)          feat[i, resp:2*resp] = window.std(0)
    feat[i, 2*resp:3*resp] = window.max(0)        feat[i, 2*resp:3*resp] = window.max(0)
  
  → (N, resp_dim × 3)                           → (N, resp_dim × 3)
  │                                            │
  └────────────────┬───────────────────────────┘
                   │
                   ▼
            concat → (N, resp_dim × 3 × 2)
```

### 3.2 维度表

| 步骤 | ALFA | Basic |
|------|------|-------|
| 残差维度 (resp_dim) | 13D | 40D |
| W=10 统计量 | 13×3 = 39D | 40×3 = 120D |
| W=30 统计量 | 13×3 = 39D | 40×3 = 120D |
| **最终特征维度** | **78D** | **240D** |

### 3.3 三个统计量的物理含义

| 统计量 | 捕获的信号 | 典型故障 |
|--------|-----------|---------|
| mean | 持续偏差 | 气压计慢漂、GPS 偏移 |
| std | 波动增大 | 传感器噪声、陀螺不稳定 |
| max | 瞬态冲击 | 执行器突然卡死、IMU 尖峰 |

### 3.4 双尺度的设计理由

| 窗口 | 时长 | 捕获的模式 |
|------|------|-----------|
| W=10 | 1.0s | 快速瞬变：执行器卡死瞬间、传感器尖峰 |
| W=30 | 3.0s | 持续异常：慢漂趋势、渐变故障 |

---

## 四、Stage 3 详解：马氏距离检测器

### 4.1 为什么替代 Galaxy

Galaxy SMM(k=5) 在两个数据集上均坍缩为单簇：

```
ALFA:  簇0=0, 簇1=35980, 簇2=0, 簇3=0, 簇4=0  → k=1
Basic: 簇0=0, 簇1=0,    簇2=0, 簇3=18718, 簇4=0 → k=1
```

Gravity loss = −log(force)，最小化它 ≡ 最大化引力 ≡ 所有帧压入同一簇。

k=1 时，Galaxy 的 GOF 评分等价于"到唯一簇中心的马氏距离"。直接使用解析解更简单。

### 4.2 拟合过程（离线，一次性）

```python
# 输入: X_normal (N_normal × d), d = 78 or 240
mu = X_normal.mean(axis=0)                        # (d,)         O(Nd)
cov = np.cov(X_normal, rowvar=False)              # (d, d)       O(Nd²)
cov += 0.01 * np.eye(d) * cov.trace() / d         # shrinkage 收缩估计
prec = np.linalg.inv(cov + 1e-6 * np.eye(d))      # (d, d)       O(d³) < 0.1s
```

### 4.3 推理过程（在线，逐帧）

```python
# 输入: x (d,) 单帧特征向量
diff = x - mu               # (d,)
score = diff @ prec * diff  # 标量 = Σ_ij diff_i · prec_ij · diff_j
# score 越大 → 越异常
```

### 4.4 阈值确定

```python
normal_scores = mahalanobis(X_normal, mu, prec)
threshold = np.percentile(normal_scores, 95)
# 或通过验证集扫参取最优 F1
```

---

## 五、完整前向推理流程（一帧异常检测）

```
输入: 原始传感器数据 (最近 75 帧 + 当前帧)
      shape: (76, feat_dim)

步骤 1 — ConditionalGRU 预测:
  x_hist = data[-76:-1]           # (75, feat_dim), 历史 75 帧
  cmd_t  = data[-1, CMD_IDX]      # (cmd_dim,),      当前指令
  resp̂_t = cgru(x_hist, cmd_t)    # (resp_dim,),      预测响应
  resp_t  = data[-1, RESP_IDX]    # (resp_dim,),      真实响应
  e_t = (resp_t - resp̂_t)²        # (resp_dim,),      残差

步骤 2 — 特征提取 (需要缓存最近 30 帧残差):
  residuals_buffer.append(e_t)     # 维护长度 ≥ 30 的环形缓冲
  e_norm  = (e_t - mu_resp) / sigma_resp    # z-score 归一化
  w10 = residuals_buffer[-10:]    # (10, resp_dim)
  w30 = residuals_buffer[-30:]    # (30, resp_dim)
  feat = concat([w10.mean(0), w10.std(0), w10.max(0),
                 w30.mean(0), w30.std(0), w30.max(0)])  # (d,)

步骤 3 — 异常评分:
  score = mahalanobis(feat, mu, prec)    # 标量
  alarm = score > threshold
```

---

## 六、训练流程（离线）

```
输入: 全部正常飞行序列 (仅 label=0 的帧)

Phase 0: 数据预处理
  1. 多传感器 CSV → 线性插值 → 统一 10Hz
  2. 划分响应/指令/上下文通道
  3. Yaw unwrap, cmd−meas 误差计算
  4. 仅保留正常帧 (label=0)

Phase 1: 训练 ConditionalGRU (~10s–5min GPU)
  1. 从正常帧构造训练样本:
     for each 正常帧 i (i ≥ 75):
       if [i-75:i) 全部正常:
         样本: (x[i-75:i], cmd[i]) → resp[i]
  2. 训练 GRU + 预测头, MSE loss, Adam
  3. 保存模型权重

Phase 2: 特征提取 (~5s–2min)
  1. 用训练好的 ConditionalGRU 逐帧计算残差 e_t
  2. 估算 z-score 参数: mu_resp, sigma_resp (仅正常帧)
  3. 对所有帧计算窗口统计 → 特征矩阵

Phase 3: 拟合检测器 (<1s)
  1. 取正常帧特征 X_normal
  2. 计算 mu, cov, prec
  3. 计算阈值 (正常帧分数的 95% 分位数)

输出: cgru_weights + mu + prec + threshold + zscore_params
```

---

## 七、关键消融证据速览

| 消融 | 结论 | Δ MSeq AUC |
|------|------|-----------|
| 删除条件残差 → 仅用无条件预测 | 条件残差不可替代 | **−0.014** |
| 跳过残差 → 直接聚类 GRU 隐状态 | 残差路径不可跳过 | −0.058 |
| 删除 FFT/互相关/协方差 (231D→78D) | 手工特征全冗余 | **+0.016** |
| GRU → LSTM | GRU 小样本最优 | −0.023 |
| GRU → TCN | 过深, 严重过拟合 | −0.117 |
| hist=10 → hist=75 | 长窗口覆盖完整机动 | +0.010 |
| Galaxy AE+SMM → 马氏距离 | 解析解优于神经网络聚类 | +0.006 |
| 10Hz → 25Hz 采样 | 高采样率稀释故障信号 | −0.078 |

---

## 八、部署物清单

| 文件 | 内容 | 大小 |
|------|------|------|
| `cgru.pt` | ConditionalGRU 权重 | ~200KB |
| `mu.npy` | 正常特征均值 (78D/240D) | ~1KB |
| `prec.npy` | 协方差精度矩阵 (78²/240²) | ~50KB / ~460KB |
| `zscore_mu.npy` | 残差 z-score μ (13D/40D) | <1KB |
| `zscore_sigma.npy` | 残差 z-score σ (13D/40D) | <1KB |
| `threshold.json` | 异常阈值 | <1KB |

总部署大小 < 1MB，CPU 推理延迟 < 1ms/帧。
