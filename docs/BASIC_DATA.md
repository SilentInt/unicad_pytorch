# Basic 数据集描述

## 概述

**Basic**: ArduPilot SITL (Software In The Loop) 仿真数据集，70 个飞行序列，7 种故障类型，~10.5M 帧总量。

| 属性 | 值 |
|---|---|
| 序列总数 | 70 |
| 纯正常序列 | 10 条 |
| 含故障序列 | 60 条 |
| 故障类型 | 6 种 (RC / GPS / Accelerometer / Gyro / Compass / Barometer) |
| 仿真日期 | 2022-07-26 ~ 2022-08-01 |
| 飞行器 | ArduPilot SITL 四旋翼 (Quadcopter) |
| 飞控 | ArduPilot (DataFlash 日志) |

---

## 目录结构

```
data/Basic/
├── {datetime} ({Failure Type})/
│   ├── {datetime} ({Failure Type}).csv     (主合并文件, 含标签)
│   ├── AHR2.csv       (姿态与航向参考系统)
│   ├── ATT.csv        (目标/实际姿态)
│   ├── BARO.csv       (气压计)
│   ├── BAT.csv        (电池)
│   ├── CTUN.csv       (控制器调参)
│   ├── GPA.csv        (GPS 精度)
│   ├── GPS.csv        (GPS 定位)
│   ├── IMU.csv        (惯性测量单元)
│   ├── MAG.csv        (磁力计/罗盘)
│   ├── MAV.csv        (MAVLink 遥测)
│   ├── MOTB.csv       (电机/推力)
│   ├── POS.csv        (位置估计)
│   ├── PSCD.csv       (位置控制 Down)
│   ├── PSCE.csv       (位置控制 East)
│   ├── PSCN.csv       (位置控制 North)
│   ├── RATE.csv       (角速率/角速率控制)
│   ├── SIM.csv        (仿真真值)
│   ├── TERR.csv       (地形跟随)
│   ├── VIBE.csv       (振动)
│   ├── XKF1.csv       (EKF 姿态/位置)
│   ├── XKF2.csv       (EKF 加速度/磁场)
│   ├── XKF3.csv       (EKF 新息)
│   ├── XKF4.csv       (EKF 传感器健康)
│   ├── XKF5.csv       (EKF 测距仪)
│   ├── XKQ.csv        (EKF 四元数)
│   ├── XKTV.csv       (EKF 偏航测试)
│   ├── XKY0.csv       (EKF 偏航估计 0)
│   └── XKY1.csv       (EKF 偏航估计 1)
```

每个序列目录包含 **29 个 CSV 文件**：1 个主合并文件 + 28 个独立传感器日志。

---

## CSV 格式

### 主合并文件 (`{datetime} ({Failure Type}).csv`)

**所有传感器数据已合并为单表，带故障标签。**

- **行数**: 83K ~ 340K 行/序列（见表末帧统计）
- **列数**: 242 列 (Time + 240 变量 + Status)
- **采样机制**: ArduPilot DataFlash 日志合并输出，不同传感器以不同频率写入同一文件。每个微秒时间戳下可能有多个日志行（不同传感器在同一仿真步内先后更新），导致大量连续行共享同一时间戳
- **唯一时间戳占比**: ~8.8%（平均 11.4 行/时间戳）
- **数据质量**: 无 NaN，46 列常值

**Status 标签列 (最后一列):**
- `0` → 正常飞行
- `1` → 故障激活中

### 独立传感器 CSV

每个传感器也有单独 CSV，仅包含该传感器的字段，按传感器原生频率记录：

| 文件 | 行数范围 | 有效采样率 | 列数 | 关键字段 |
|------|---------|-----------|------|---------|
| IMU.csv | 10K ~ 43K | ~50 Hz | 14 | GyrX/Y/Z, AccX/Y/Z |
| BARO.csv | 4K ~ 17K | ~20 Hz | 9 | Alt, Press, Temp |
| VIBE.csv | 4K ~ 17K | ~20 Hz | 5 | VibeX/Y/Z |
| XKF1.csv | 4K ~ 17K | ~20 Hz | 15 | Roll, Pitch, Yaw, VN/VE/VD |
| XKF2.csv | 4K ~ 17K | ~20 Hz | 13 | AX/AY/AZ, MN/ME/MD |
| XKF3.csv | 4K ~ 17K | ~20 Hz | 13 | IVN/IVE/IVD (新息) |
| XKF4.csv | 4K ~ 17K | ~20 Hz | 12 | SV/SP/SH (传感器健康) |
| MAG.csv | 6K ~ 26K | ~30 Hz | 12 | MagX/Y/Z |
| AHR2.csv | 2K ~ 11K | ~10 Hz | 10 | Roll, Pitch, Yaw, Alt |
| ATT.csv | 2K ~ 11K | ~10 Hz | 9 | DesRoll/Roll, DesPitch/Pitch |
| CTUN.csv | 2K ~ 11K | ~10 Hz | 12 | ThI, ThO, Alt |
| RATE.csv | 2K ~ 11K | ~10 Hz | 13 | RDes/R, PDes/P, YDes/Y |
| GPS.csv | 2K ~ 11K | ~10 Hz | 12 | Lat, Lng, Alt, Spd |
| BAT.csv | 1K ~ 5K | ~5 Hz | 12 | Volt, Curr |
| POS.csv | 1K ~ 5K | ~5 Hz | 5 | Lat, Lng, Alt |
| MAV.csv | 1K ~ 5K | ~5 Hz | 2 | txp, rxp |
| MOTB.csv | 1K ~ 5K | ~5 Hz | 5 | LiftMax, BatVolt, FailFlags |
| SIM.csv | 2K ~ 11K | ~10 Hz | 8 | Roll, Pitch, Yaw, Alt (真值) |
| TERR.csv | 1K ~ 5K | ~5 Hz | 8 | Status, TerrH, CHeight |
| GPA.csv | 1K ~ 5K | ~5 Hz | 5 | VDop, HAcc, VAcc |
| PSCD/PSCE/PSCN | 1K ~ 5K | ~5 Hz | 各 8 | 位置控制 D/E/N 分量 |
| XKQ.csv | 2K ~ 11K | ~10 Hz | 5 | Q1-Q4 (四元数) |
| XKTV.csv | 1K ~ 5K | ~5 Hz | 2 | TVS, TVD |
| XKY0.csv | 2K ~ 11K | ~10 Hz | 11 | 偏航估计 0 |
| XKY1.csv | 2K ~ 11K | ~10 Hz | 6 | 偏航估计 1 |
| XKF5.csv | 1K ~ 5K | ~5 Hz | 10 | 测距仪/地形高度 |

---

## 主合并文件变量分组

主 CSV 的 240 个变量按前缀分组，共 28 组：

### 惯性测量 (IMU)
| 变量 | 物理含义 | 单位 |
|------|---------|------|
| IMU_GyrX, IMU_GyrY, IMU_GyrZ | 三轴陀螺仪角速度 | rad/s |
| IMU_AccX, IMU_AccY, IMU_AccZ | 三轴加速度 | m/s² |
| IMU_EG, IMU_EA | 陀螺/加速度计错误标志 | — |
| IMU_T | IMU 温度 | °C |
| IMU_GHz, IMU_AHz | 陀螺/加速度计采样率 | Hz |

### 姿态 (ATT, AHR2)
| 变量 | 物理含义 | 单位 |
|------|---------|------|
| ATT_DesRoll, ATT_Roll | 目标/实际滚转角 | ° |
| ATT_DesPitch, ATT_Pitch | 目标/实际俯仰角 | ° |
| ATT_DesYaw, ATT_Yaw | 目标/实际偏航角 | ° |
| ATT_ErrRP, ATT_ErrYaw | 滚转-俯仰/偏航跟踪误差 | — |
| AHR2_Roll, AHR2_Pitch, AHR2_Yaw | AHR2 姿态估计 | ° |
| AHR2_Alt | AHR2 高度估计 | m |
| AHR2_Q1~Q4 | AHR2 姿态四元数 | — |
| AHR2_Lat, AHR2_Lng | AHR2 位置估计 | ° |

### 角速率控制 (RATE)
| 变量 | 物理含义 | 单位 |
|------|---------|------|
| RATE_RDes, RATE_R, RATE_ROut | 滚转目标/实际/输出 | rad/s |
| RATE_PDes, RATE_P, RATE_POut | 俯仰目标/实际/输出 | rad/s |
| RATE_YDes, RATE_Y, RATE_YOut | 偏航目标/实际/输出 | rad/s |
| RATE_ADes, RATE_A, RATE_AOut | 油门目标/实际/输出 | — |

### 导航与位置
| 变量组 | 物理含义 | 关键变量 |
|--------|---------|---------|
| GPS | GPS 定位 | Lat, Lng, Alt, Spd, GCrs, VZ |
| POS | 位置估计 | Lat, Lng, Alt, RelHomeAlt |
| CTUN | 控制器调参 | ThI, ThO, Alt, DAlt, BAlt, CRt |
| GPA | GPS 精度 | VDop, HAcc, VAcc, SAcc |

### EKF (扩展卡尔曼滤波, XKF1~XKF5, XKQ, XKTV, XKY0, XKY1)
| 变量组 | 物理含义 | 关键变量 |
|--------|---------|---------|
| XKF1 | EKF 姿态/位置/速度 | Roll, Pitch, Yaw, VN/VE/VD, PN/PE/PD |
| XKF2 | EKF 加速度/磁场 | AX/AY/AZ, MN/ME/MD, MX/MY/MZ |
| XKF3 | EKF 新息 (innovation) | IVN/IVE/IVD, IPN/IPE/IPD, IMX/IMY/IMZ |
| XKF4 | EKF 传感器健康 | SV/SP/SH/SM, errRP, OFN/OFE |
| XKF5 | EKF 测距仪 | HAGL, offset, rng, Herr |

### 环境传感器
| 变量组 | 物理含义 | 关键变量 |
|--------|---------|---------|
| BARO | 气压计 | Alt, Press, Temp, CRt |
| MAG | 磁力计 | MagX/Y/Z, OfsX/Y/Z |
| VIBE | 振动 | VibeX/Y/Z |

### 仿真真值 (SIM)
| 变量 | 物理含义 |
|------|---------|
| SIM_Roll, SIM_Pitch, SIM_Yaw | 真实姿态 |
| SIM_Alt, SIM_Lat, SIM_Lng | 真实位置 |
| SIM_Q1~Q4 | 真实四元数 |

### 执行器与动力
| 变量组 | 物理含义 | 关键变量 |
|--------|---------|---------|
| MOTB | 电机/推力 | LiftMax, BatVolt, ThLimit, ThrAvMx, FailFlags |
| BAT | 电池 | Volt, VoltR, Curr, CurrTot, EnrgTot, Temp, Res, RemPct |

### 位置控制 (PSCD/PSCE/PSCN)
| 变量前缀 | 含义 | 关键变量 |
|----------|------|---------|
| PSCD | Down 方向控制 | TPD, PD, DVD, TVD, VD, DAD, TAD, AD |
| PSCE | East 方向控制 | 同上 (E 方向) |
| PSCN | North 方向控制 | 同上 (N 方向) |

### 通信与地形
| 变量组 | 物理含义 | 关键变量 |
|--------|---------|---------|
| MAV | MAVLink 遥测 | txp, rxp (收发包数) |
| TERR | 地形跟随 | Status, TerrH, CHeight, Pending, Loaded |

---

## 常值列 (46 列, 无判别力)

以下字段在正常飞行序列中保持常值：

```
ATT_AEKF, BARO_Offset, BARO_GndTemp, BAT_Volt, BAT_Temp,
GPA_VDop, GPA_HAcc, GPA_VAcc, GPA_SAcc, GPA_YAcc,
GPS_NSats, GPS_HDop, GPS_Yaw, GPS_U,
IMU_EG, IMU_EA,
MAG_OfsX/Y/Z, MAG_MOX/Y/Z,
MOTB_ThLimit, MOTB_FailFlags,
TERR_Status, TERR_Spacing, TERR_ROfs,
XKF1_OH, XKF2_VWN/VWE, XKF2_IDX/IDY,
XKF3_IVT, XKF4_SVT, XKF4_FS/TS/GPS,
XKF5_NI/FIX/FIY/AFI/offset/RI/rng/Herr
```

注: SITL 仿真中部分传感器参数被固定，实际飞行中这些值会变化。

---

## 故障类型分布

| 故障类型 | 序列数 | 含义 | 注入机制 |
|---------|--------|------|---------|
| No Failure | 10 | 正常飞行 | — |
| RC Failure | 10 | 遥控信号丢失 | RC 输入失效, 切换至失控保护模式 |
| GPS Failure | 10 | GPS 信号丢失 | GPS 数据停止更新 |
| Accelerometer Failure | 10 | 加速度计故障 | 加速度计读数偏移/冻结 |
| Gyro Failure | 10 | 陀螺仪故障 | 陀螺仪读数偏移/冻结 |
| Compass Failure | 10 | 罗盘/磁力计故障 | 磁力计读数偏移/冻结 |
| Barometer Failure | 10 | 气压计故障 | 气压计读数偏移/冻结 |

所有故障均为**传感器/执行器硬件故障**，模拟真实无人机传感器失效场景。

---

## 帧级别统计

| 故障类型 | 总帧数 (10 序列合计) | 故障帧数 | 故障占比 | 帧数范围/序列 | 故障注入时间范围 |
|---------|---------------------|---------|---------|-------------|----------------|
| No Failure | ~1,565,615 | 0 | 0% | 91K ~ 299K | — |
| RC Failure | ~1,164,942 | 506,638 | 43.5% | 91K ~ 184K | 120 ~ 328s |
| GPS Failure | ~1,169,940 | 338,722 | 29.0% | 83K ~ 205K | 117 ~ 434s |
| Accelerometer Failure | ~1,971,721 | 1,185,983 | 60.1% | 139K ~ 303K | 86 ~ 311s |
| Gyro Failure | ~1,891,299 | 1,138,120 | 60.2% | 133K ~ 340K | 80 ~ 317s |
| Compass Failure | ~1,570,924 | 852,689 | 54.3% | 104K ~ 247K | 99 ~ 311s |
| Barometer Failure | ~1,436,501 | 770,888 | 53.7% | 92K ~ 223K | 100 ~ 240s |
| **总计** | **~10,770,942** | **~4,793,040** | **44.5%** | — | — |

故障注入模式：所有故障序列以正常飞行开始 → 飞行一段时间后注入故障 → 持续至序列结束。注入时机因序列而异（80s ~ 434s），注入后不再恢复。

### 按序列帧数明细

| # | 序列 | 帧数 | 故障帧 | 占比 | 故障@ |
|---|------|------|--------|------|-------|
| 1 | 2022-07-26 05-49-12 (No Failure) | 126,990 | 0 | 0% | — |
| 2 | 2022-07-26 06-25-08 (RC Failure) | 108,850 | 51,600 | 47.4% | 144.6s |
| 3 | 2022-07-26 06-46-30 (GPS Failure) | 90,527 | 44,262 | 48.9% | 116.8s |
| 4 | 2022-07-26 17-02-53 (Accelerometer Failure) | 152,596 | 100,198 | 65.7% | 133.6s |
| 5 | 2022-07-26 18-11-42 (Gyro Failure) | 159,455 | 112,508 | 70.6% | 118.6s |
| 6 | 2022-07-26 18-26-44 (Compass Failure) | 126,872 | 76,891 | 60.6% | 126.2s |
| 7 | 2022-07-26 18-37-31 (Barometer Failure) | 123,593 | 72,131 | 58.4% | 129.9s |
| 8 | 2022-07-26 20-33-38 (No Failure) | 91,484 | 0 | 0% | — |
| 9 | 2022-07-26 20-43-46 (RC Failure) | 93,300 | 45,598 | 48.9% | 120.4s |
| 10 | 2022-07-26 20-51-21 (GPS Failure) | 82,705 | 27,396 | 33.1% | 139.6s |
| — | ... (其余 60 序列见上方表格汇总) | | | | |

---

## 按采集日期分组 (推荐划分)

| 日期 | 序列数 | 建议用途 |
|------|--------|---------|
| 2022-07-26 | 14 | Train (含全部 7 种类型) |
| 2022-07-27 | 14 | Train |
| 2022-07-28 | 10 | Val |
| 2022-07-29 | 13 | Val/Test |
| 2022-07-30 | 10 | Test |
| 2022-07-31 | 7 | Test |
| 2022-08-01 | 4 | Test |

> 注: 2022-08-01 仅 4 序列。若按日期顺序划分，前 4 天 (07-26 ~ 07-29) 作为 train，后 3 天作为 test。

---

## 数据读取要点

1. **时间戳为微秒**: `Time` 列单位为 μs，需除以 1e6 转换为秒
2. **日志合并格式**: 主 CSV 中多个传感器日志行共享同一时间戳，导致约 91% 的行具有重复时间戳。可对同一时间戳取平均、取首行、或先插值到统一网格后使用
3. **独立 CSV 为原生采样**: 各传感器独立 CSV 不含填充/合并，建议直接基于独立 CSV 进行时序建模
4. **Status 为帧级标签**: 主 CSV 最后一列 0/1 标注故障状态，独立 CSV 无标签列
5. **SITL 特征**: 46 列保持常值，在实际硬件数据中这些列会有变化，注意迁移时的特征分布漂移
6. **故障注入**: 所有故障序列以正常飞行开始，中途注入故障，正常/故障比例因类型而异（29%~60%）
7. **仿真真值**: SIM 组变量提供真实位姿真值，可用于评估 EKF 估计精度

---

## 与 ALFA 数据集的对比

| 属性 | Basic (本数据集) | ALFA |
|------|-----------------|------|
| 来源 | ArduPilot SITL 仿真 | Pixhawk 实飞 |
| 飞行器 | 四旋翼 | Carbon Z 固定翼 |
| 序列数 | 70 | 46 |
| 故障类型 | 6 种传感器故障 | 4 种执行器故障 |
| 采样率 | ~100 Hz (主CSV) | 10 Hz (统一采样) |
| 数据格式 | 多日志合并 + 独立CSV | 多Topic独立CSV |
| 特征维度 | 240 (196 有效) | 21 |
| 总帧数 | ~10.8M | ~47K |
| 总故障帧 | ~4.8M (44.5%) | ~7.7K (16.3%) |
| 标签粒度 | 帧级 0/1 | 帧级 0/1 |
| 常值列 | 46 (SITL 固有限制) | 0 |
