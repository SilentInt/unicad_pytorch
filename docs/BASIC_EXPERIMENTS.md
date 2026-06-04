# Basic 数据集异常检测实验

## 数据

ArduPilot SITL 四旋翼仿真数据，70 序列（10 正常 / 60 故障），275,604 帧（10 Hz 统一采样），6 种传感器故障类型。

预处理：主 CSV → groupby(Time).last() 去重 → 47D 特征提取 → 10 Hz 线性插值 → compact .npz

## 最终架构

```
原始 47D (10 Hz)
  │
  ▼
ConditionalGRU: predict(response_t | history_{75帧}, command_t)
  GRU(47→128, 2层, 47K params), Head(128+7→40)
  → 残差 = (真实响应 − 预测响应)² (40D)
  → z-score 归一化 (mu/sd 用正常帧估计)
  │
  ▼
双尺度窗口统计: W=10 (1s) mean/std/max → 120D
               W=30 (3s) mean/std/max → 120D
  → 240D 特征
  │
  ▼
马氏距离: (x − μ)ᵀ Σ⁻¹ (x − μ) → 异常分数
  μ = 正常帧均值, Σ = 正常帧协方差矩阵 ← 解析解, 无需训练
```

## 特征设计

| 组 | 通道 | 维度 |
|----|------|------|
| Response raw | IMU(6) + ATT_actual(3) + RATE_actual(4) + XKF1_V(3) + BARO_Alt + VIBE_Z + XKF4_SV + GPS(3) + MAG(3) + EKF创新(8) | 33D |
| cmd-meas 误差 | ATT_Des−Actual(3) + RATE_Des−Actual(4) | 7D |
| Command | ATT_Des(3) + RATE_Des(4) | 7D |
| **GRU 输入** | | **47D** |
| **预测目标 (Response)** | | **40D** |

## 检测方法消融

### 最终排名 (全量 70 序列, 147K 正常帧)

| 方法 | MeanSeq AUC | Global AUC | 训练时间 | 参数量 |
|------|-----------|-----------|---------|--------|
| **马氏距离** | **0.952** | **0.942** | **0s** | **240² 矩阵** |
| Deep SVDD v2 (两阶段+BN+EMA) | 0.939 | 0.934 | 207s | 176K |
| Deep SVDD v1 (纯 center loss) | 0.938 | 0.932 | 68s | 150K |
| Galaxy AE+SMM (k=5) | 0.938 | 0.932 | 81s | 300K |

### Deep SVDD 版本演进 (14 序列 → 70 序列)

| 版本 | 改进 | 14 序列 MeanSeq | 70 序列 MeanSeq |
|------|------|----------------|----------------|
| v1 (纯 center loss) | — | 0.979 → 坍缩为常数 | 0.830 |
| v1 + 重建约束 | 加 decoder, recon+center 联合 loss | 0.979 | 0.938 |
| v2 (两阶段+BN+EMA) | AE 预训练 → 中心 EMA 更新 → BatchNorm | — | 0.939 |

Deep SVDD 在小样本上表现极好 (14 序列 0.979, 超过马氏距离 0.956)，但数据充足后马氏距离的协方差估计更准确，非线性优势消失。

### 特征设计消融 (14 序列)

| 特征来源 | 维度 | Galaxy MeanSeq |
|---------|------|---------------|
| 窗口统计 (W=10,30) | 240D | 0.936 |
| ResidualGRU 隐状态 | 64D | 0.931 |
| ConditionalGRU 隐状态 | 135D | 0.878 |

> 残差路径 (预测→误差→统计) 不可替代。直接使用 GRU 隐状态聚类效果最差——隐状态学的是"如何预测"，不是"什么是异常"。

## 按故障类型性能 (马氏距离)

| 故障类型 | 序列数 | Mean AUC | 范围 | 检测难度 |
|---------|--------|---------|------|---------|
| Gyro Failure | 10 | **0.998** | 0.995 – 1.000 | 极低 |
| Accelerometer Failure | 10 | **0.998** | 0.991 – 0.999 | 极低 |
| RC Failure | 10 | **0.955** | 0.844 – 0.999 | 低 |
| Compass Failure | 10 | **0.954** | 0.836 – 0.990 | 低 |
| GPS Failure | 10 | **0.892** | 0.628 – 0.994 | 中 |
| Barometer Failure | 10 | **0.875** | 0.753 – 0.953 | 高 |

## SMM 聚类退化分析

Galaxy k=5 的实际簇分布 (147K 正常帧):

```
簇0:      0 (0.0%)  ← 空
簇1:      0 (0.0%)  ← 空
簇2:      0 (0.0%)  ← 空
簇3: 18,718 (100%)  ← 所有帧
簇4:      0 (0.0%)  ← 空
```

Gravity loss 将所有正常帧强行吸入同一个簇，SMM 退化为 k=1。Galaxy 异常检测实际等价于 "到唯一簇中心的马氏距离"——这正是马氏距离直接胜出的根本原因。

## 核心发现

1. **前向因果预测是核心，不可替代**：ConditionalGRU 的 command→response 因果结构是异常信号的唯一来源。跳过残差直接用隐状态聚类 (MeanSeq 0.878) 或用手工特征替代 GRU 预测都会大幅下降。

2. **窗口统计比原始残差好**：单帧残差噪声大（正常转弯时也大），W=10/30 的 mean/std/max 聚合后才能区分"正常的偶尔大残差"和"故障的持续大残差"。

3. **马氏距离是最优检测器**：240D 特征经 z-score 归一化后，正常帧近似多元高斯分布 (std≈3)，协方差矩阵精确刻画正常边界。非线性方法 (AE/Deep SVDD) 无法在此基础上增益。

4. **数据量决定方法差异**：小样本时 Deep SVDD > 马氏距离 (14 序列 0.979 vs 0.956)，大样本时马氏距离 > Deep SVDD (70 序列 0.952 vs 0.939)。越多的正常数据让协方差估计越准确，非线性方法的归纳偏置优势越小。

5. **12 个消融版本，最终收敛到 3 个组件**：
   - ConditionalGRU (~5 min 训练)
   - 双尺度窗口统计 (0 params)
   - 马氏距离 (解析解, <1s)

## 完整测试指标 (Basic 70 序列, P95 阈值)

| 指标 | 值 |
|------|-----|
| AUC | 0.9425 |
| AP | 0.9508 |
| **Precision** | **0.9331** |
| **Recall** | **0.8312** |
| **F1** | **0.8792** |

| 混淆矩阵 | 预测正常 | 预测故障 |
|---------|---------|---------|
| 真实正常 | TN = 139,692 | FP = 7,353 |
| 真实故障 | FN = 20,812 | TP = 102,497 |

- 误报率 FPR = 5.00%（P95 阈值的设计预期：允许 5% 正常帧被误报）
- 检出率 TPR = 83.12%
- 93.31% 的告警是真正的故障

> 阈值可调：提高至 P99 可降低误报率至 1%，但 Recall 会相应下降。选择取决于部署场景对虚警和漏报的容忍度。

## 最终推荐

```python
# 最简部署方案: 无需 Galaxy, 无需 SMM, 无需 GPU
mu = normal_240d_features.mean(0)
cov = np.cov(normal_240d_features.T)
prec = np.linalg.inv(cov + 1e-6 * np.eye(240))

def detect(features_240d):
    diff = features_240d - mu
    score = (diff @ prec * diff).sum(1)
    return score > threshold  # threshold = np.percentile(normal_scores, 95)
```

## 文件

| 文件 | 用途 |
|------|------|
| `scripts/basic_preprocess.py` | 一次性预处理：主 CSV → .npz (47D, 10Hz) |
| `scripts/basic_detect.py` | 主力脚本：ConditionalGRU + Galaxy |
| `scripts/basic_detect_v2.py` | 消融实验：马氏距离 / Deep SVDD / Galaxy / ResidualGRU / GRU隐状态 |
| `docs/BASIC_DATA.md` | 数据集描述 |
| `docs/BASIC_EXPERIMENTS.md` | 本文档 |
