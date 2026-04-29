# UniCAD-Torch

Galaxy 异常检测模型的 PyTorch 原生重实现——基于引力聚类框架的无监督异常检测。

> **论文**: "Towards a Unified Framework of Clustering-based Anomaly Detection"
>
> 核心思想：聚类中心对数据点施加"引力"，异常样本受到的引力较弱，因此异常分数更高。

## 特性

- **纯 PyTorch 实现** — 无 PyTorch Lightning / ADBench / TensorFlow 依赖，GPU 原生计算
- **Callback 训练框架** — 早停、检查点、进度条、LR 调度器，均可通过 CLI 标志启用
- **四阶段流水线** — 预处理 → 自编码器预训练 → 迭代 EM → GOF 评分
- **对数空间计算** — 全程 log-space 避免高维行列式溢出/下溢
- **完整 CLI** — `galaxy-train` / `galaxy-predict` / `galaxy-benchmark`，支持 YAML/JSON 配置文件
- **sklearn 风格 API** — `fit()` / `predict()` / `predict_score()` / `save()` / `load()`

## 安装

```bash
# 基础安装
uv sync

# 开发环境（含测试、lint、tqdm 等）
uv sync --extra dev
```

**依赖**: numpy, scipy, scikit-learn, torch>=2.6.0

**可选依赖**: tqdm（进度条）, pyyaml（YAML 配置）, pandas（CSV 数据）

## 快速开始

### Python API

```python
from unicad_torch import Galaxy, GalaxyConfig
from unicad_torch.callbacks import History, EarlyStopping, ModelCheckpoint, TqdmProgress

import numpy as np

# 准备数据 (N, D) float32
X_train = np.random.randn(1000, 10).astype(np.float32)

# 配置模型与回调
config = GalaxyConfig(
    k=10,                       # 聚类数
    em_iters=3,                 # EM 迭代次数
    device="cpu",               # "cuda" 可用
    verbose=True,               # 输出训练日志
    callbacks=[
        History(),              # 记录训练指标历史
        EarlyStopping(patience=10, monitor="pretrain_loss"),
        ModelCheckpoint(dirpath="ckpt/", save_best=True, monitor="score_mean"),
        TqdmProgress(),         # 进度条
    ],
)

# 训练
model = Galaxy(config)
model.fit(X_train)

# 推理
scores = model.predict_score(X_train)    # 连续异常分数，越高越异常
labels  = model.predict(X_train)         # 0/1 二值标签（基于训练阈值）

# 持久化
model.save("galaxy.pt")
loaded = Galaxy.load("galaxy.pt", device="cpu")

# 查看训练信息
print(model)                  # Galaxy(input_dim=10, k=10, threshold_=1.23)
print(model.threshold_)       # 训练阈值
print(model.fit_info_)        # {"train_score_mean": ..., "n_excluded_per_iter": [...]}
```

### 有标签训练

```python
# 提供 y_train 时，真实异常率自动替代 config.outlier_ratio
model.fit(X_train, y_train=y)  # y: 0/1 标签数组
print(model.effective_outlier_ratio_)  # 实际使用的异常率
print(model.fit_info_["outlier_ratio_source"])  # "labels" 或 "config"
```

### 命令行

```bash
# 训练（自动创建运行目录，保存模型/配置/日志）
galaxy-train --data data/Classical/38_thyroid.npz \
    --tqdm --history --early-stopping --checkpoint

# 训练并指定输出路径（传统方式，仍可用）
galaxy-train --data data/Classical/38_thyroid.npz --output galaxy.pt

# 推理
galaxy-predict --model galaxy.pt --data data/Classical/38_thyroid.npz
galaxy-predict --model galaxy.pt --data test.csv --labels
galaxy-predict --model galaxy.pt --data test.csv --output-scores scores.csv

# 基准测试
galaxy-benchmark --datasets 38_thyroid --verbose --tqdm
galaxy-benchmark --data-dir data/Classical --output results.csv

# 查看历史运行
galaxy-runs
galaxy-runs --detail 2026-04-29_14-30-22
galaxy-runs --compare run1 run2
```

也可以通过 `uv run` 运行脚本（无需安装）：

```bash
uv run python scripts/train.py --data data/Classical/38_thyroid.npz --output galaxy.pt
uv run python scripts/predict.py --model galaxy.pt --data data/Classical/38_thyroid.npz
uv run python scripts/run_benchmark.py --datasets 38_thyroid
```

---

## 模型架构

Galaxy 采用四阶段流水线，由 `Galaxy` 类编排，所有计算在配置的 torch 设备上完成：

```
原始数据 X (N, D)
    │
    ▼  Stage 1: 预处理
标准化后的 X (N, D)
    │
    ▼  Stage 2: 自编码器预训练
编码器 → 潜在表示 Z (N, D_latent) + 重建 X_hat
    │
    ▼  Stage 3: 迭代 EM（排除异常 → 微调网络 → 更新原型）
拟合后的编码器 + 原型 (means_, covars_, weights_)
    │
    ▼  Stage 4: GOF 评分
异常分数 score (N,)
```

### Stage 1: 预处理

| 方式 | 说明 | 配置 |
|------|------|------|
| z-score | 零均值单位方差（`correction=0`） | `preprocess="z-score"` |
| row-norm | 逐行 L2 归一化 | `preprocess="row-norm"` |
| 无 | 跳过预处理 | `preprocess="none"` |

### Stage 2: 自编码器预训练

两层编码器 (`input_dim → hidden_dim → latent_dim`) + 两层解码器，使用 MSE(sum) 损失，Adam + StepLR 优化。

- `pretrain=False` 可跳过此阶段
- `latent_dim=None` 时自动等于 `hidden_dim`

### Stage 3: 迭代 EM

每次 EM 迭代执行三步：

1. **排除异常** — 用当前原型计算 GOF 分数，移除 top `outlier_ratio`% 的样本
2. **微调网络** — 在过滤后的数据上用 `重建损失 + 引力损失` 微调自编码器
3. **更新原型** — 对过滤后数据的编码表示拟合 Student-t 混合模型 (ν=1)

引力损失在 log-space 计算，避免高维协方差行列式的溢出/下溢。

### Stage 4: GOF 评分

异常分数 = `-log(force)`，即引力越弱分数越高。

| 评分方式 | 公式 | 配置 |
|----------|------|------|
| 标量 | `-logsumexp(log F_ik)` | `score_type="scalar"` |
| 向量 | `-[log(‖force_vec‖) + max_log_force]` | `score_type="vector"`（默认） |

---

## Callback 系统

Callback 是训练框架的核心扩展机制，允许在训练各阶段注入自定义逻辑。

### 事件钩子

| 事件 | 触发时机 | 可用指标 |
|------|----------|----------|
| `on_fit_begin` | `Galaxy.fit()` 开始 | — |
| `on_pretrain_epoch_begin` | 每个 pretrain epoch 开始 | `epoch`, `total_epochs` |
| `on_pretrain_epoch_end` | 每个 pretrain epoch 结束 | `pretrain_loss` |
| `on_em_iter_begin` | 每个 EM 迭代开始 | `em_iter`, `total_em_iters` |
| `on_em_iter_end` | 每个 EM 迭代结束 | `n_excluded`, `score_min`, `score_max`, `score_mean` |
| `on_finetune_step_end` | 每个微调步骤结束 | `finetune_recon_loss`, `finetune_gravity_loss`, `finetune_total_loss` |
| `on_fit_end` | `Galaxy.fit()` 结束 | 所有指标 |

### 内置 Callback

#### History — 记录训练指标

```python
from unicad_torch.callbacks import History

history = History()
config = GalaxyConfig(callbacks=[history])
model = Galaxy(config)
model.fit(X)

# 训练后查看记录
print(history.history)
# {"pretrain_loss": [5.2, 4.1, 3.8, ...], "n_excluded": [5, 3], "score_mean": [2.1, 1.8], ...}
```

#### EarlyStopping — 早停

```python
from unicad_torch.callbacks import EarlyStopping

# pretrain 阶段：loss 连续 10 个 epoch 无改善则停止
es = EarlyStopping(patience=10, monitor="pretrain_loss", mode="min")

# EM 阶段：score_mean 连续 3 次迭代无改善则停止
es = EarlyStopping(patience=3, monitor="score_mean", mode="min", threshold=1e-3)
```

- `monitor` 支持: `pretrain_loss`, `score_mean`, `score_min`, `score_max`, `n_excluded`, `finetune_total_loss` 等 FitContext 中的指标
- `mode="min"` 越小越好, `mode="max"` 越大越好
- `threshold` 最小改善量

#### ModelCheckpoint — 检查点保存

```python
from unicad_torch.callbacks import ModelCheckpoint

# 每个 EM 迭代保存 latest.pt
ckpt = ModelCheckpoint(dirpath="checkpoints/", save_last=True)

# 仅保存最优检查点
ckpt = ModelCheckpoint(dirpath="checkpoints/", save_best=True, monitor="score_mean", mode="min")

# 保留 top-k 最优
ckpt = ModelCheckpoint(dirpath="checkpoints/", save_best=True, monitor="score_mean", save_top_k=3)
```

#### TqdmProgress — 进度条

```python
from unicad_torch.callbacks import TqdmProgress

# 自动为 pretrain 和 EM 阶段显示 tqdm 进度条
progress = TqdmProgress()
```

#### LRScheduler — 自定义学习率调度

```python
from unicad_torch.callbacks import LRScheduler
import torch

# 注入余弦退火调度器（替代默认的 StepLR）
scheduler_cb = LRScheduler(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=200)
```

### 自定义 Callback

继承 `Callback` 基类，重写需要的 `on_*` 方法：

```python
from unicad_torch.callbacks import Callback, FitContext

class MyCallback(Callback):
    def on_pretrain_epoch_end(self, ctx: FitContext) -> None:
        print(f"Epoch {ctx.epoch}: loss={ctx.pretrain_loss:.4f}")

    def on_em_iter_end(self, ctx: FitContext) -> None:
        print(f"EM iter {ctx.em_iter}: excluded={ctx.n_excluded}, score={ctx.score_mean:.4f}")

    def on_finetune_step_end(self, ctx: FitContext) -> None:
        if ctx.finetune_total_loss is not None and ctx.finetune_total_loss > 100:
            ctx.stop_training = True  # 信号：停止训练
```

`FitContext` 是训练循环与回调之间共享的可变上下文对象，训练循环写入指标字段，回调读取指标并可通过 `stop_training=True` 信号停止训练。

---

## 配置参数

### GalaxyConfig 完整字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **通用** | | | |
| `seed` | int | 42 | 随机种子（`Galaxy.fit()` 开头设置） |
| `k` | int | 10 | 聚类数 |
| `outlier_ratio` | float | 0.01 | 默认异常比例 [0, 1)；有标签时可能被覆盖 |
| `preprocess` | str | `"z-score"` | `"z-score"`, `"row-norm"`, `"none"` |
| `device` | str | `"cpu"` | Torch 设备（创建时验证） |
| `verbose` | bool | False | 输出训练日志 |
| **自编码器** | | | |
| `hidden_dim` | int | 128 | 隐藏层维度 |
| `latent_dim` | int \| None | None | 潜在层维度；None = 等于 hidden_dim |
| `pretrain` | bool | True | 是否进行预训练 |
| `pretrain_epochs` | int | 200 | 预训练 epoch 数 |
| `pretrain_lr` | float | 3e-3 | 预训练学习率 |
| `pretrain_batch_size` | int | 1024 | 预训练批次大小 |
| **EM** | | | |
| `em_iters` | int | 3 | EM 迭代次数 |
| `em_finetune_steps` | int | 100 | 每次 EM 迭代的微调步数 |
| `em_finetune_lr` | float | 3e-4 | 微调学习率 |
| **SMM** | | | |
| `smm_n_iter` | int | 100 | SMM EM 迭代次数 |
| `smm_tol` | float | 1e-3 | SMM 收敛容差（相对） |
| **评分** | | | |
| `gravity_version` | str | `"vector"` | `"scalar"` 或 `"vector"` |
| `score_type` | str | `"vector"` | `"scalar"` 或 `"vector"` |
| **回调** | | | |
| `callbacks` | list \| None | None | Callback 实例列表 |

### 创建配置

```python
# 默认配置
config = GalaxyConfig()

# 自定义配置
config = GalaxyConfig(k=5, em_iters=5, device="cuda", verbose=True)

# 从已有配置派生（不修改原配置）
config_v2 = config.replace(k=20, em_iters=10)

# 使用 YAML/JSON 配置文件（CLI）
# galaxy-train --config train_config.yaml --output galaxy.pt
```

YAML 配置文件示例：

```yaml
k: 10
hidden_dim: 128
latent_dim: 32
pretrain_epochs: 100
em_iters: 5
em_finetune_steps: 50
device: cuda
verbose: true
preprocess: z-score
gravity_version: vector
score_type: vector
outlier_ratio: 0.02
```

---

## CLI 命令详解

### galaxy-train — 训练模型

```bash
galaxy-train --data <数据文件> --output <模型路径> [选项]
```

| 参数 | 说明 |
|------|------|
| `--data` | 训练数据文件（.npz 或 .csv） |
| `--output` | 输出模型路径（.pt）；不指定则存到运行目录 |
| `--run-dir` | 运行根目录（默认 runs/） |
| `--run-name` | 运行标签，附加到时间戳后 |
| `--config` | 配置文件（.yaml/.yml/.json） |
| `--resume` | 从检查点恢复训练（支持运行目录路径） |
| `--k`, `--hidden-dim`, `--em-iters` ... | 覆盖 GalaxyConfig 字段 |
| `--history` | 记录训练历史 |
| `--early-stopping` | 启用早停 |
| `--early-stopping-patience` | 早停耐心值（默认 10） |
| `--early-stopping-monitor` | 早停监控指标（默认 pretrain_loss） |
| `--checkpoint` | 保存模型检查点 |
| `--checkpoint-dir` | 检查点目录（默认 checkpoints/） |
| `--checkpoint-best` | 保存最优检查点 |
| `--checkpoint-monitor` | 检查点监控指标（默认 score_mean） |
| `--tqdm` | 显示进度条 |
| `--verbose` | 输出详细日志 |

### galaxy-predict — 推理评估

```bash
galaxy-predict --model <模型路径> --data <数据文件> [选项]
galaxy-predict --model <模型路径> --data-dir <数据目录> [选项]
```

| 参数 | 说明 |
|------|------|
| `--model` | 已训练模型路径（.pt） |
| `--data` | 单个数据文件 |
| `--data-dir` | 数据目录（批量模式） |
| `--datasets` | 筛选指定数据集（批量模式） |
| `--device` | 推理设备（默认 cpu） |
| `--threshold` | 覆盖异常阈值 |
| `--labels` | 逐样本输出分数/标签 |
| `--output-scores` | 逐样本分数保存为 CSV |
| `--output` | 汇总结果保存为 CSV |

### galaxy-benchmark — 基准测试

```bash
galaxy-benchmark --datasets <数据集名> [选项]
galaxy-benchmark --data-dir <数据目录> [选项]
```

| 参数 | 说明 |
|------|------|
| `--data-dir` | 数据目录（默认 data/Classical） |
| `--datasets` | 指定数据集名称 |
| `--output` | 结果保存为 CSV |
| 其他 | 同 galaxy-train 的配置覆盖和回调参数 |

### galaxy-runs — 查看历史运行

```bash
galaxy-runs [选项]
```

| 参数 | 说明 |
|------|------|
| `--run-dir` | 运行根目录（默认 runs/） |
| `--detail` | 查看指定运行的详细信息 |
| `--compare` | 对比多次运行（空格分隔） |

---

## 数据格式

### ADBench .npz 格式

```python
data = np.load("38_thyroid.npz")
X = data["X"]  # (N, D) 特征矩阵
y = data["y"]  # (N,) 0/1 标签
```

### CSV 格式

带 `y` 列（标签可选）：

```csv
feature_0,feature_1,feature_2,y
0.5,-1.2,3.1,0
2.3,0.8,-0.5,1
...
```

### 下载数据

```bash
# 下载 Classical 类别（默认）
uv run python scripts/download_data.py --output data

# 下载特定数据集
uv run python scripts/download_data.py --output data --datasets 38_thyroid 2_annthyroid

# 下载所有类别
uv run python scripts/download_data.py --output data --all
```

---

## 模型持久化

### 保存与加载

```python
# 保存
model.save("galaxy.pt")

# 加载（可指定设备）
model = Galaxy.load("galaxy.pt", device="cuda")
```

保存内容包括：配置（不含回调）、自编码器权重、EM 原型 (means_, covars_, weights_)、阈值、预处理参数、训练元数据、History 回调数据（如有）。

### 断点恢复

```bash
# 从检查点文件恢复（传统方式）
galaxy-train --data data/Classical/38_thyroid.npz --output galaxy.pt \
    --resume checkpoints/latest.pt

# 从运行目录恢复（自动查找 checkpoints/latest.pt 或 model.pt）
galaxy-train --data data/Classical/38_thyroid.npz \
    --resume runs/2026-04-29_14-30-22
```

### 保存格式

v2 格式（扁平字典），`load()` 同时兼容 v2 和旧版嵌套 `fit_info_` 格式（会产生 DeprecationWarning）。

---

## 训练运行目录

每次 `galaxy-train` 训练自动创建独立的时间戳目录，统一保存模型、配置、日志和元数据：

```
runs/                                    # --run-dir 根目录（默认 runs/）
├── 2026-04-29_14-30-22/               # 自动生成的时间戳目录
│   ├── config.yaml                    # 训练配置（可读、可复现）
│   ├── model.pt                       # 最终模型
│   ├── history.json                   # 训练指标历史
│   ├── summary.json                   # 运行元数据
│   └── checkpoints/                   # ModelCheckpoint 输出
│       └── latest.pt
├── 2026-04-29_15-00-00_baseline/
│   └── ...
└── 2026-04-29_16-00-00_exp_k5/
    └── ...
```

### 使用方式

```bash
# 默认：自动创建 runs/YYYY-MM-DD_HH-MM-SS/ 目录
galaxy-train --data data.npz --history --checkpoint --tqdm

# 指定根目录
galaxy-train --data data.npz --run-dir experiments/

# 添加运行标签
galaxy-train --data data.npz --run-name baseline

# 传统方式仍可用：--output 直接指定输出路径
galaxy-train --data data.npz --output galaxy.pt
```

### summary.json 内容

```json
{
  "run_name": "2026-04-29_14-30-22",
  "timestamp": "2026-04-29T14:35:10+00:00",
  "fit_duration_seconds": 12.34,
  "threshold": -3.4567,
  "train_score_mean": -5.1234,
  "effective_outlier_ratio": 0.01,
  "outlier_ratio_source": "config",
  "n_samples": 197,
  "n_features": 6,
  "git_commit": "a1b2c3d4",
  "git_dirty": false,
  "unicad_torch_version": "0.2.0"
}
```

### galaxy-runs 命令

查看历史训练运行：

```bash
# 列出所有运行
galaxy-runs

# 指定根目录
galaxy-runs --run-dir experiments/

# 查看某次运行详情
galaxy-runs --detail 2026-04-29_14-30-22

# 对比多次运行
galaxy-runs --compare run1 run2
```

### 编程接口

```python
from unicad_torch.run import (
    create_run_dir, save_config, save_history, save_summary,
    find_runs, load_summary, resolve_resume_path,
)

# 创建运行目录
run_dir = create_run_dir("runs/", tag="exp1")
# → runs/2026-04-29_14-30-22_exp1/

# 保存工件
save_config(config, run_dir)
save_history(history_dict, run_dir)
save_summary(run_dir, fit_time=12.34, threshold=-3.45, n_samples=100)

# 浏览历史运行
runs = find_runs("runs/")          # 按时间倒序
summary = load_summary(runs[0])    # 读取摘要

# 恢复路径解析
ckpt_path = resolve_resume_path("runs/2026-04-29_14-30-22")
# → runs/2026-04-29_14-30-22/checkpoints/latest.pt
```

---

## 项目结构

```
unicad_torch/
├── src/unicad_torch/        # 核心库
│   ├── __init__.py          # 公共导出
│   ├── config.py            # GalaxyConfig 数据类
│   ├── galaxy.py            # Galaxy 主模型类
│   ├── em.py                # GalaxyEM 迭代 EM 核心
│   ├── model.py             # Autoencoder + 预训练
│   ├── callbacks.py         # Callback 系统
│   ├── gravity.py           # 引力计算（单数据源）
│   ├── gof.py               # GOF 评分函数
│   ├── smm_torch.py         # Student-t 混合模型
│   ├── preprocessing.py     # StandardScaler / RowScaler
│   ├── run.py               # 训练运行目录管理
│   ├── datasets.py          # 数据集发现工具
│   └── cli.py               # CLI 命令实现 + 共享工具
├── scripts/                 # 脚本入口（薄包装）
│   ├── train.py             # → cli.train_main()
│   ├── predict.py           # → cli.predict_main()
│   ├── run_benchmark.py     # → cli.benchmark_main()
│   ├── test_modules.py      # 逐模块诊断工具
│   └── download_data.py     # ADBench 数据下载
├── tests/                   # 90 个测试用例
│   ├── test_callbacks.py    # Callback 系统测试
│   ├── test_config.py       # 配置验证测试
│   ├── test_galaxy.py       # Galaxy 集成测试
│   ├── test_model.py        # 自编码器测试
│   ├── test_em.py           # (via test_galaxy)
│   ├── test_gof.py          # GOF 评分测试
│   ├── test_gravity.py      # 引力计算测试
│   ├── test_smm.py          # SMM 测试
│   ├── test_preprocessing.py# 预处理测试
│   └── test_datasets.py     # 数据集发现测试
├── pyproject.toml           # 项目配置 + console_scripts
└── CLAUDE.md                # 开发者内部参考
```

### 模块依赖关系

```
Galaxy (galaxy.py)
├── GalaxyConfig (config.py)
├── GalaxyEM (em.py)
│   ├── Autoencoder (model.py)
│   ├── SMMTorch (smm_torch.py)
│   │   └── gravity.py [mahalanobis_diag]
│   ├── gof.py [gof_score]
│   └── gravity.py [compute_log_forces, aggregate_force_*]
├── CallbackManager + FitContext (callbacks.py)
├── StandardScaler / RowScaler (preprocessing.py)
└── cli.py [load_data, build_callbacks, ...] → run.py [create_run_dir, save_*, find_runs]
```

`gravity.py` 是引力计算的单数据源，导出 `mahalanobis_diag`、`compute_log_forces`、`aggregate_force_scalar`、`aggregate_force_vector` 以及数值常量 `VAR_FLOOR` (1e-6)、`WEIGHT_FLOOR` (1e-30)、`NORM_FLOOR` (1e-30)。

---

## 开发

```bash
# 安装开发依赖
uv sync --extra dev

# 运行测试
uv run pytest tests/ -v

# 代码检查
uv run ruff check src/ scripts/ tests/
uv run ruff format --check src/ scripts/ tests/
uv run pyright src/ tests/

# 逐模块诊断（需要下载数据）
uv run python scripts/test_modules.py --datasets 38_thyroid
```

### 关键设计决策

| 决策 | 说明 |
|------|------|
| **GalaxyEM 拥有评分** | `GalaxyEM.score()` 封装编码+评分模式，Galaxy 不直接操作 EM 内部 |
| **SMM 是临时的** | 每次 `_update_prototypes()` 创建新的 `SMMTorch`，避免 load 后的陈旧状态 |
| **gof_score 是函数** | 无状态模块级函数，非类 |
| **引力版本构造时绑定** | `GalaxyEM.__init__` 将 `_compute_gravity_loss` 绑定为闭包，训练循环无 if/elif |
| **fit_info_ 是 property** | 只读属性，从权威源组装字典，无重复可变状态 |
| **对数空间评分** | GOF 分数 = `-log(force)` 而非 `1/force`，避免除零 |
| **绝对阈值** | `predict()` 使用训练阈值，不相对测试批次 |
| **标签覆盖异常率** | `y_train` 提供时自动替代 `outlier_ratio`，原始 config 不被修改 |
| **数值安全** | `compute_log_forces` 中 `log(covars)` 前 clamp 防止 `-inf`；常量统一定义于 `gravity.py` |
| **日志替代 print** | 所有模块使用 `logging.getLogger("unicad_torch.{module}")`；`verbose=True` 设置包级 logger 为 INFO |

---

## 引用

```bibtex
@article{galaxy2023,
  title={Towards a Unified Framework of Clustering-based Anomaly Detection},
  author={...},
  journal={...},
  year={2023}
}
```

## 许可证

本项目为独立 PyTorch 重实现，仅供研究使用。
