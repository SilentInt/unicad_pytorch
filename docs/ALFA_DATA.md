# ALFA 数据集描述

## 概述

**ALFA (AIR Lab Fault Detection Dataset)**: Carbon Z 固定翼无人机飞行数据，Pixhawk 飞控，2018 年采集于匹兹堡 CMU AIR Lab。本文档仅描述 `processed/` 目录中已处理的 CSV 数据。

| 属性 | 值 |
|---|---|
| 序列总数 | 46 |
| 纯正常序列 | 10 条 |
| 含故障序列 | 36 条 |
| 采集日期 | 2018-07-18, 07-30, 09-11, 10-05, 10-18 |
| 飞行器 | Carbon Z 固定翼 |
| 飞控 | Pixhawk (MAVLink via MAVROS) |

---

## 目录结构

```
data/ALFA/
├── carbonZ_{date}-{time}_{subseq}_{fault_label}/
│   ├── *-mavros-imu-data_raw.csv          (必有)
│   ├── *-mavros-nav_info-roll.csv          (必有)
│   ├── *-mavros-nav_info-pitch.csv         (必有)
│   ├── *-mavros-nav_info-yaw.csv           (必有)
│   ├── *-mavros-nav_info-airspeed.csv      (必有)
│   ├── *-mavros-nav_info-errors.csv        (必有)
│   ├── *-mavros-rc-out.csv                 (必有)
│   ├── *-mavros-vfr_hud.csv               (必有)
│   └── *-failure_status-{component}.csv   (仅故障序列)
```

每个序列是一个目录，内含 8（正常）或 9+（故障）个 CSV 文件。文件名前缀与目录名相同。

---

## CSV 通用格式

所有 CSV 遵循 ROS `rostopic echo -b` 导出格式：

- **首列**: `%time` — 纳秒级 Unix 时间戳 (int64)
- **后续列**: `field.*` 命名的传感器字段
- 无行号，无引号，纯数值
- 时间戳**不均匀**（受 ROS 发布频率和系统调度影响）

---

## 传感器 Topic 详解

### 1. IMU (`*-mavros-imu-data_raw.csv`)

- **采样率**: ~10 Hz
- **行数范围**: 264 ~ 2334 行/序列
- **关键列** (6 维，用于异常检测):

| 列名 | 物理含义 | 单位 |
|---|---|---|
| `field.angular_velocity.x` | 滚转角速度 | rad/s |
| `field.angular_velocity.y` | 俯仰角速度 | rad/s |
| `field.angular_velocity.z` | 偏航角速度 | rad/s |
| `field.linear_acceleration.x` | 前向加速度 | m/s² |
| `field.linear_acceleration.y` | 侧向加速度 | m/s² |
| `field.linear_acceleration.z` | 垂直加速度 | m/s² |

- **其他列**: `field.orientation.*` (4 维四元数，全零/无效), `field.*_covariance` (9 维协方差，常值) — **不推荐使用**

### 2. 导航跟踪误差 (`*-mavros-nav_info-{roll,pitch,yaw,airspeed}.csv`)

- **采样率**: ~19 Hz
- 每个文件 2 列：`field.commanded` (指令值), `field.measured` (实测值)
- **推荐特征**: `commanded - measured` (跟踪误差)

| 文件 | 推荐特征 | 物理含义 | 单位 |
|---|---|---|---|
| `nav_info-roll` | roll_err = cmd - meas | 滚转跟踪误差 | ° |
| `nav_info-pitch` | pitch_err = cmd - meas | 俯仰跟踪误差 | ° |
| `nav_info-yaw` | yaw_err = cmd - meas | 偏航跟踪误差 | ° |
| `nav_info-airspeed` | aspd_err = cmd - meas | 空速跟踪误差 | m/s |

### 3. 导航误差 (`*-mavros-nav_info-errors.csv`)

- **采样率**: ~19 Hz
- 3 列 + 1 列距离:

| 列名 | 物理含义 | 单位 |
|---|---|---|
| `field.alt_error` | 高度误差 | m |
| `field.aspd_error` | 空速误差 | m/s |
| `field.xtrack_error` | 航迹交叉误差 | m |
| `field.wp_dist` | 到航路点距离 | m (不推荐用作异常特征) |

### 4. 执行器输出 (`*-mavros-rc-out.csv`)

- **采样率**: ~3 Hz
- 8 通道 PWM 值:

| 列名 | 对应 | 说明 |
|---|---|---|
| `field.channels0` | — | 通常为常值 1500 (未使用) |
| `field.channels1` | 俯仰 | 升降舵 PWM |
| `field.channels2` | 油门 | 发动机 PWM |
| `field.channels3` | 偏航 | 方向舵 PWM |
| `field.channels4` | 左副翼 | PWM |
| `field.channels5` | 右副翼 | PWM |
| `field.channels6` | — | 通常为常值 1500 |
| `field.channels7` | — | 通常为常值 2036 |

推荐使用 channels1~5 (5 维)，channels0/6/7 为常值可丢弃。

### 5. 飞行参数 (`*-mavros-vfr_hud.csv`)

- **采样率**: ~2.5 Hz

| 列名 | 物理含义 | 单位 |
|---|---|---|
| `field.throttle` | 油门百分比 | 0~1 |
| `field.altitude` | 气压高度 | m |
| `field.climb` | 爬升率 | m/s |
| `field.airspeed` | 空速 | m/s |
| `field.groundspeed` | 地速 | m/s |
| `field.heading` | 航向 | ° |

推荐使用 throttle, altitude, climb (3 维)；airspeed/groundspeed 与 nav_info 重复。

### 6. 故障标签 (`*-failure_status-{component}.csv`)

- **仅存在于故障序列**，正常序列无此文件
- 2 列: `%time`, `field.data` (值始终为 `1`)
- **语义**: 文件中每一行表示该时刻故障**正在发生**
- **关键特性**: 文件仅包含故障持续期间的条目，故障未激活时无任何行
- 同一序列可有多个故障文件（组合故障），各自独立标注时间区间

| 故障类型 | failure_status 后缀 | 含义 |
|---|---|---|
| 发动机 | `engines` | 发动机失效 / 推力丧失 |
| 升降舵 | `elevator` | 升降舵卡死/失效 |
| 副翼 | `aileron` | 副翼卡死/失效 (左/右/双) |
| 方向舵 | `rudder` | 方向舵卡死/失效 (左/右/零位) |

---

## 推荐 22 维特征向量 (对齐至 10 Hz)

对齐策略：以 IMU 时间戳为主轴（~10 Hz），其余 topic 通过线性插值映射到相同时间网格。

| # | 特征 | 来源 | 采样率 |
|---|---|---|---|
| 0 | angular_velocity.x | IMU | 10 Hz |
| 1 | angular_velocity.y | IMU | 10 Hz |
| 2 | angular_velocity.z | IMU | 10 Hz |
| 3 | linear_acceleration.x | IMU | 10 Hz |
| 4 | linear_acceleration.y | IMU | 10 Hz |
| 5 | linear_acceleration.z | IMU | 10 Hz |
| 6 | roll_err = cmd - meas | nav_info-roll | 19→10 Hz |
| 7 | pitch_err = cmd - meas | nav_info-pitch | 19→10 Hz |
| 8 | yaw_err = cmd - meas | nav_info-yaw | 19→10 Hz |
| 9 | aspd_err = cmd - meas | nav_info-airspeed | 19→10 Hz |
| 10 | alt_error | nav_info-errors | 19→10 Hz |
| 11 | aspd_error | nav_info-errors | 19→10 Hz |
| 12 | xtrack_error | nav_info-errors | 19→10 Hz |
| 13 | channels1 (俯仰) | rc-out | 3→10 Hz |
| 14 | channels2 (油门) | rc-out | 3→10 Hz |
| 15 | channels3 (偏航) | rc-out | 3→10 Hz |
| 16 | channels4 (左副翼) | rc-out | 3→10 Hz |
| 17 | channels5 (右副翼) | rc-out | 3→10 Hz |
| 18 | throttle | vfr_hud | 2.5→10 Hz |
| 19 | altitude | vfr_hud | 2.5→10 Hz |
| 20 | climb | vfr_hud | 2.5→10 Hz |
| 21 | — | (channels0 保留位) | — |

> 注: channels0 通常为常值 1500，可替换为其他有意义特征或直接丢弃为 21 维。

---

## 故障类型分布

### 按故障类型统计

| 故障类型 | 序列数 | 说明 |
|---|---|---|
| no_failure | 10 | 纯正常飞行 |
| engine_failure | 2 | 发动机失效 |
| engine_failure_with_emr_traj | 14 | 发动机失效 + 紧急航迹 |
| elevator_failure | 2 | 升降舵失效 |
| aileron_failure (含组合) | 7 | 副翼失效 (左/右/双) |
| rudder_failure | 3 | 方向舵失效 (左/右) |
| rudder + aileron 组合 | 1 | 方向舵零位 + 左副翼 |
| both_ailerons | 1 | 双副翼失效 |

### 按采集日期分组 (推荐划分 train/val/test)

| 日期 | 序列数 | 包含故障类型 | 建议用途 |
|---|---|---|---|
| 2018-07-18 | 4 | engine, no_failure | Train |
| 2018-07-30 | 8 | engine, no_failure | Train |
| 2018-09-11 | 16 | 全部类型 | Val |
| 2018-10-05 | 8 | engine, aileron, no_failure | Test |
| 2018-10-18 | 7 | engine, no_failure | Test |

### 帧级别标注生成

故障标签需要从 `failure_status` 时间区间映射到对齐后的帧时间戳：

```
对每帧 t:
  label[t] = 1 if 任一 failure_status.csv 中存在行 t_fault 使得 |t - t_fault| < 0.5/fps
           = 0 otherwise
```

正常序列全部帧 label=0。

---

## 数据读取要点

1. **时间戳以纳秒存储**: `%time` 列为 int64，需除以 1e9 得到秒
2. **各 topic 采样率不同**: 必须插值对齐到统一时间网格
3. **rc_out 和 vfr_hud 低频**: 插值时注意不要引入超前信息（实际应用需因果插值）
4. **failure_status 仅含故障区间**: 缺少该文件 = 全程正常
5. **组合故障**: `*_rudder_zero__left_aileron_*` 有两个 failure_status 文件，任一激活即为故障帧
6. **orientation 字段全零**: IMU 中的四元数为默认值，不可用
7. **channels0/6/7 常值**: rc_out 中这三个通道几乎不变，无判别力
