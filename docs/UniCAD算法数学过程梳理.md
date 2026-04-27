# UniCAD 算法数学过程详细梳理

---

## 一、问题定义

给定数据集 $\mathbf{X} \in \mathbb{R}^{N \times D}$，包含 $N$ 个样本、$D$ 维特征。无监督异常检测的目标是为每个样本 $\mathbf{x}_i$ 学一个异常分数 $o_i$，使得异常样本的分数高于正常样本。

---

## 二、核心目标：最大化异常感知似然 (Eq.1)

UniCAD 的理论基础是最大化**异常感知数据似然**：

$$\max \log p(\mathbf{X}|\Theta, \Phi) = \max \sum_{i=1}^{N} \delta(\mathbf{x}_i) \log \sum_{k=1}^{K} p(\mathbf{x}_i, c_i=k|\Theta, \Phi)$$

其中：
- $\Theta$：表示学习（自编码器）的参数
- $\Phi = \{\omega_k, \mu_k, \Sigma_k\}$：聚类参数（混合权重、原型、协方差）
- $\delta(\mathbf{x}_i)$：异常指示函数，异常为 0，正常为 1
- $c_i$：样本 $\mathbf{x}_i$ 的潜在聚类变量

**关键思想**：通过 $\delta(\mathbf{x}_i)$ 将异常样本从似然计算中剔除，使表示学习和聚类不受异常样本干扰。

---

## 三、联合表示学习与聚类 (Eq.2–4)

### 3.1 混合模型分解 (Eq.2)

似然用混合模型建模：

$$p(\mathbf{x}_i|\Theta, \Phi) = \sum_{k=1}^{K} \omega_k \cdot p(\mathbf{x}_i|c_i=k, \Theta, \mu_k, \Sigma_k)$$

其中 $\sum_{k=1}^{K} \omega_k = 1$。

### 3.2 Student-t 分布建模 (Eq.3)

样本先通过自编码器映射到低维表示空间：$\mathbf{z}_i = f_\Theta(\mathbf{x}_i)$，在表示空间中用**多元 Student-t 分布**建模给定聚类的条件概率：

$$p(\mathbf{x}_i|c_i=k, \Theta, \mu_k, \Sigma_k) = \frac{\Gamma(\frac{\nu+1}{2})|\Sigma_k|^{-1/2}}{\Gamma(\frac{\nu}{2})\sqrt{\nu\pi}} \left(1 + \frac{1}{\nu} D_M(\mathbf{z}_i, \mu_k)^2\right)^{-\frac{\nu+1}{2}}$$

其中：
- $D_M(\mathbf{z}_i, \mu_k) = \sqrt{(\mathbf{z}_i - \mu_k)^T \Sigma_k^{-1}(\mathbf{z}_i - \mu_k)}$ 是马氏距离
- $\nu$ 是自由度，**固定为 1**（无需交叉验证）
- Student-t 分布的重尾特性使其对异常值鲁棒

### 3.3 化简 (Eq.4)

令 $\nu=1$，利用 $\Gamma(1)=1, \Gamma(1/2)=\sqrt{\pi}$，化简为：

$$p(\mathbf{x}_i|\Theta, \Phi) = \sum_{k=1}^{K} \omega_k \cdot \frac{\pi^{-1} \cdot |\Sigma_k|^{-1/2}}{1 + D_M(\mathbf{z}_i, \mu_k)^2}$$

---

## 四、异常指示函数与异常分数 (Eq.5–6)

### 4.1 指示函数 (Eq.5)

设异常比例为 $l$，则 $\delta(\mathbf{x}_i)$ 的最优解为：对生成概率 $p(\mathbf{x}_i|\Theta,\Phi)$ 最低的 $l\%$ 样本设为 0，其余设为 1：

$$\delta(\mathbf{x}_i) = \begin{cases} 0, & \text{if } p(\mathbf{x}_i|\Theta,\Phi) \text{ is among the } l \text{ lowest} \\ 1, & \text{otherwise} \end{cases}$$

### 4.2 标量异常分数 (Eq.6)

生成概率越低，样本越可能是异常。异常分数定义为概率的倒数：

$$o_i = \frac{1}{p(\mathbf{x}_i|\Theta, \Phi)} = \frac{1}{\sum_{k=1}^{K} \omega_k \cdot \frac{\pi^{-1} \cdot |\Sigma_k|^{-1/2}}{1 + D_M(\mathbf{z}_i, \mu_k)^2}}$$

---

## 五、万有引力启发的异常分数 (Eq.7–9)

### 5.1 引力类比 (Eq.8)

定义第 $k$ 个聚类对样本 $i$ 的"引力"为：

$$\tilde{F}_{ik} = \omega_k \cdot \frac{\pi^{-1} \cdot |\Sigma_k|^{-1/2}}{1 + D_M(\mathbf{z}_i, \mu_k)^2}$$

与牛顿万有引力公式 $\vec{F}_{ik} = \frac{G \cdot m_i m_k}{r_{ik}^2}$ 类比：

| 物理量 | 类比 |
|---|---|
| 引力常数 $G$ | $\tilde{G} = \pi^{-1}$ |
| 聚类质量 $m_k$ | $\tilde{m}_k = \omega_k |\Sigma_k|^{-1/2}$ |
| 样本质量 $m_i$ | $\tilde{m}_i = 1$（统一标准化） |
| 距离 $r_{ik}$ | $\tilde{r}_{ik} = \sqrt{1 + D_M(\mathbf{z}_i, \mu_k)^2}$ |

标量分数 $o_i = 1 / \sum_k \tilde{F}_{ik}$ 相当于各力**大小之和**的倒数。

### 5.2 向量异常分数 (Eq.9) — 论文核心创新

在物理中，合力是**向量求和**，考虑方向后不同方向的力可以相互抵消。将这一思想引入异常分数：

$$o_i^V = \frac{1}{\left\| \sum_{k=1}^{K} \tilde{F}_{ik} \cdot \hat{F}_{ik} \right\|}$$

其中 $\hat{F}_{ik} = \frac{\mu_k - \mathbf{z}_i}{\|\mu_k - \mathbf{z}_i\|}$ 是从样本 $\mathbf{z}_i$ 指向聚类原型 $\mu_k$ 的单位方向向量。

**直观解释**：如果一个样本被多个方向不同的聚类吸引，力向量相互抵消，合力小，异常分数高。例如：一个同时喜欢"省钱技巧"和"奢侈品"的用户，两个方向的引力抵消，比同时喜欢两个相似奢侈品的用户更"异常"。

---

## 六、迭代优化 (Algorithm 1)

优化分两大部分交替进行：更新混合模型参数 $\Phi$（EM 算法）和更新网络参数 $\Theta$（梯度下降）。

### 6.1 E 步 — 计算后验概率 (Eq.10)

$$\tau_{ik}^{(t+1)} = \frac{\omega_k \cdot p(\mathbf{z}_i|\mu_k, \Sigma_k)}{\sum_{j=1}^{K} \omega_j \cdot p(\mathbf{z}_i|\mu_j, \Sigma_j)} = \frac{\tilde{F}_{ik}^{(t)}}{\sum_{j=1}^{K} \tilde{F}_{ij}^{(t)}}$$

同时计算精度缩放因子 (Eq.11)：

$$u_{ik}^{(t+1)} = \frac{2}{1 + D_M(\mathbf{z}_i^{(t)}, \mu_k^{(t)})}$$

这是 Student-t 分布层级表示中的隐变量期望，$\nu=1$ 时分子为 $\nu+1=2$。

### 6.2 M 步 — 更新混合模型参数 (Eq.12–14)

设正常样本数为 $n = \lfloor (1-l) \cdot N \rfloor$，仅对正常样本更新：

**混合权重** (Eq.12)：

$$\omega_k^{(t+1)} = \frac{1}{n} \sum_{i=1}^{n} \tau_{ik}^{(t+1)}$$

**聚类原型** (Eq.13)：

$$\mu_k^{(t+1)} = \frac{\sum_{i=1}^{n} \tau_{ik}^{(t+1)} u_{ik}^{(t+1)} \mathbf{z}_i}{\sum_{i=1}^{n} \tau_{ik}^{(t+1)} u_{ik}^{(t+1)}}$$

这是后验概率加权（再乘以 $u_{ik}$ 缩放因子）的样本均值。$u_{ik}$ 的作用是：距离原型越远的点，$u_{ik}$ 越小，对原型更新的贡献越低——这就是 Student-t 分布对异常值鲁棒的数学来源。

**协方差矩阵** (Eq.14)：

$$\Sigma_k^{(t+1)} = \frac{\sum_{i=1}^{n} \tau_{ik}^{(t+1)} u_{ik}^{(t+1)} (\mathbf{z}_i - \mu_k^{(t+1)})(\mathbf{z}_i - \mu_k^{(t+1)})^T}{\sum_{j=1}^{K} \tau_{ij}^{(t+1)}}$$

（代码中用对角协方差，实际计算只取对角线元素。）

### 6.3 更新网络参数 $\Theta$ (Eq.15)

联合损失函数：

$$\mathcal{L} = -J(\Theta, \Phi) + g(\Theta)$$

- $J(\Theta, \Phi) = \log p(\mathbf{X}|\Theta, \Phi)$：重力损失（负对数似然）
- $g(\Theta) = \|x - \hat{x}\|^2$：重构损失，防止"捷径解"

**重力损失的具体形式**（对应代码 `em.py:96-108`）：

**标量版本**：

$$\mathcal{L}_{\text{gravity}} = -\sum_i \log\left(\sum_{k=1}^{K} \frac{\omega_k}{\pi \sqrt{|\Sigma_k|} \cdot (1 + D_M(\mathbf{z}_i, \mu_k)^2)}\right)$$

**向量版本**：

$$\mathcal{L}_{\text{gravity}} = -\sum_i \log\left\|\sum_{k=1}^{K} \frac{\omega_k}{\pi \sqrt{|\Sigma_k|} \cdot (1 + D_M(\mathbf{z}_i, \mu_k)^2)} \cdot \hat{F}_{ik}\right\|$$

总损失 = 重构损失 + 重力损失，用 Adam 优化器更新 $\Theta$。

---

## 七、完整算法流程

对应代码 `galaxy.py` 中的四阶段流水线：

```
Stage 1: 预处理
    z-score 标准化 或 L2 行归一化

Stage 2: 自编码器预训练
    最小化 MSE(sum) 重构损失，Adam + StepLR，200 epochs

Stage 3: 迭代 EM（em_iters 轮，默认 3 轮）
    每轮：
      (a) 排除异常：对完整 X 重新计算 GOF 分数，
          移除 top outlier_ratio% 的样本 → X_filtered
      (b) 更新网络 Θ：在 X_filtered 上用
          重构损失 + 重力损失微调自编码器
      (c) 更新原型 Φ：在 X_filtered 的编码 Z 上
          拟合 Student-t 混合模型（EM 内循环至收敛）

Stage 4: GOF 评分
    编码输入 → 计算每个样本的引力分数
    标量模式：1 / Σ_k F_ik
    向量模式：1 / ‖Σ_k F_ik · û_ik‖
```

---

## 八、数值稳定性细节

`em.py:88-91` 和 `gof.py:43-46` 中，`det_covars`（$|\Sigma_k|$）必须在 **float64** 下计算。当潜在维度 $d=128$ 时，128 个小方差值相乘在 float32 下会下溢为零，导致分母为零、分数爆炸。这是数值稳定性的硬性要求，不是优化。

---

## 九、EM 推导细节（附录 B 补充）

### E 步

后验概率（Eq.16）：

$$\tau_{ik} = \frac{\omega_k \cdot p(\mathbf{z}_i|\mu_k, \Sigma_k)}{\sum_{j=1}^{K} \omega_j \cdot p(\mathbf{z}_i|\mu_j, \Sigma_j)}$$

精度缩放因子（Eq.17），$\nu=1$：

$$u_{ik} = \frac{\nu + 1}{\nu + D_M(\mathbf{z}_i, \mu_k)^2} = \frac{2}{1 + D_M(\mathbf{z}_i, \mu_k)^2}$$

### M 步

对似然函数（Eq.18）：

$$L(\omega, \mu, \Sigma) = \sum_{i=1}^{N} \sum_{k=1}^{K} \omega_k \cdot \frac{\pi^{-1} \cdot |\Sigma_k|^{-1/2}}{1 + (\mathbf{z}_i - \mu_k)^T \Sigma_k^{-1} (\mathbf{z}_i - \mu_k)}$$

**求 $\omega_k$**：引入拉格朗日乘子 $\lambda$（约束 $\sum_k \omega_k = 1$），构造（Eq.19）：

$$L'(\omega, \mu, \Sigma, \lambda) = L(\omega, \mu, \Sigma) + \lambda\left(1 - \sum_{k=1}^{K} \omega_k\right)$$

对 $\omega_k$ 求导（Eq.20-21）：

$$\frac{\partial L'}{\partial \omega_k} = \sum_i \frac{\tau_{ik}}{\omega_k} - \lambda = 0$$

利用约束条件解得 $\lambda = N$（Eq.22-23）：

$$\omega_k = \frac{\sum_i \tau_{ik}}{N}$$

**求 $\mu_k$**：对条件期望 $Q(\mu_k, \Sigma_k)$（Eq.24）求导：

$$Q(\mu_k, \Sigma_k) = \sum_{i=1}^{N} \tau_{ik} \left(-\log\pi - \frac{1}{2}\log|\Sigma_k| + \frac{1}{2}\log u_{ik} - \frac{1}{2} u_{ik} (\mathbf{z}_i - \mu_k)^T \Sigma_k^{-1} (\mathbf{z}_i - \mu_k)\right)$$

对 $\mu_k$ 求导令为 0（Eq.25-26）：

$$\mu_k^{(t+1)} = \frac{\sum_{i=1}^{n} \tau_{ik}^{(t+1)} u_{ik}^{(t+1)} \mathbf{z}_i}{\sum_{i=1}^{n} \tau_{ik}^{(t+1)} u_{ik}^{(t+1)}}$$

**求 $\Sigma_k$**：对 $\Sigma_k^{-1}$ 求导令为 0（Eq.27-28）：

$$\Sigma_k^{(t+1)} = \frac{\sum_{i=1}^{n} \tau_{ik}^{(t+1)} u_{ik}^{(t+1)} (\mathbf{z}_i - \mu_k^{(t+1)})(\mathbf{z}_i - \mu_k^{(t+1)})^T}{\sum_{j=1}^{K} \tau_{ij}^{(t+1)}}$$

---

## 十、数学总结

UniCAD 的数学核心是一条从**概率建模**到**引力类比**的逻辑链：

1. **概率框架**：最大化异常感知似然 → 统一表示学习、聚类、异常检测
2. **Student-t 混合模型**：用重尾分布建模表示空间 → 天然抗异常
3. **指示函数**：$\delta(\mathbf{x}_i)$ 显式排除异常 → 保护学习和聚类
4. **异常分数**：$1/p(\mathbf{x}_i)$ → 生成概率的倒数
5. **引力类比**：将分数重新解读为万有引力 → 引入向量求和
6. **向量分数**：考虑力的方向 → 捕获跨聚类的复杂关系，性能持续优于标量版
