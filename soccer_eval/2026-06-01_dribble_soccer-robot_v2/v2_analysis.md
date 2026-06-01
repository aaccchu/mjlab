# v2 Soccer Robot 验证分析

## 1. 概述

### v2 核心改动
1. **球 condim=3→6**：启用 rolling friction（friction[2]），球不再无限滑动
2. **足↔球 ContactSensor**：检测左右脚踝与球的接触力
3. **kick_contact 奖励（weight=0.3）**：鼓励策略用脚触球
4. **sim 参数**：nconmax=100, contact_sensor_maxmatch=128

### 训练配置
- 环境数：8192
- 迭代数：5000
- 物理步长：0.005s，decimation=4，控制频率 50Hz
- 策略网络：MLP 512→256→128，ELU 激活
- PPO：adaptive LR（desired_kl=0.01），entropy_coef=0.005
- 训练总耗时：约 3 小时（RTX 3090）
- Run ID：2026-06-01_11-09-25，wandb run gvuf7h8r

---

## 2. 定量评估

### 2.1 v2 最终模型（model_4999，512 episodes，seed=42）

| 指标 | 值 | 95% CI |
|------|------|--------|
| success_rate | 99.8% | [99.4%, 100%] |
| fall_rate | 0.2% | [-0.2%, 0.6%] |
| possession_rate | 95.9% | [95.5%, 96.2%] |
| ball_to_target_error | 0.20m | [0.19m, 0.21m] |
| time_to_goal | 122 步 (2.4s) | [120, 123] |

### 2.2 v2 中途快照（model_1400，28% 训练进度）

| 指标 | 值 | 95% CI |
|------|------|--------|
| success_rate | 99.6% | [99.1%, 100%] |
| fall_rate | 0.4% | [-0.2%, 0.9%] |
| possession_rate | 89.6% | [88.0%, 91.3%] |
| ball_to_target_error | 0.23m | [0.20m, 0.25m] |
| time_to_goal | 207 步 (4.1s) | [202, 211] |

### 2.3 D-vs-B 对比（v1 策略 on v2 物理 vs v2 策略 on v2 物理）

| 指标 | B: v1 on v2 physics | D: v2 on v2 physics | 差异 |
|------|--------------------|--------------------|------|
| success_rate | 18.4% | 99.8% | +81.4pp |
| ball_to_target_error | 2.57m | 0.20m | -92% |
| possession_rate | 67.8% | 95.9% | +28.1pp |
| time_to_goal | 245 步 (4.9s) | 122 步 (2.4s) | -50% |
| fall_rate | 0.2% | 0.2% | 相同 |

**关键发现**：v1 策略在 v2 物理下仍能行走（fall_rate 相同），但带球能力严重退化。
v1 学到的是"推球后球无限滑动"的策略，在有摩擦的 v2 物理下完全失效。

### 2.4 v1 vs v2 总对比

| 指标 | v1 最终 (model_2999) | v2 最终 (model_4999) | 改善 |
|------|---------------------|---------------------|------|
| success_rate | 6.5% | 99.8% | +93.3pp |
| ball_to_target_error | 3.79m | 0.20m | -95% |
| fell_over | 12.5% | 0.2% | -98% |
| possession_rate | — | 95.9% | — |
| time_to_goal | — | 122 步 | — |

---

## 3. 训练曲线分析

### 关键指标随迭代的变化

| 阶段 | iter | Mean reward | success | ball_error | fell_over | 特征 |
|------|------|-------------|---------|------------|-----------|------|
| 随机 | 0 | -5.6 | 0% | — | 834 | 立即摔倒 |
| 学站 | ~200 | ~10 | 0% | — | ~50 | 学会平衡 |
| 学走 | ~500 | ~30 | 3% | 3.5m | ~1 | 开始接近球 |
| 学带球 | ~1000 | ~97 | 61% | 2.0m | ~1 | 快速学会带球方向 |
| 精细化 | ~1400 | ~113 | 91% | 0.48m | ~1 | 高成功率 |
| 收敛 | ~2500 | ~131 | 95% | 0.27m | ~0.5 | 精度提升 |
| 最终 | ~5000 | ~140 | 95% | 0.34m | ~0.5 | 步态质量优化 |

**观察**：
- 策略学习顺序：站立 → 行走 → 接近球 → 带球方向 → 精确控球
- 核心能力在 ~1400 iter 已基本收敛（99.6% eval success）
- 后续 3600 iter 主要改善：time_to_goal（207→122 步）、possession（90%→96%）
- kick_contact 信号始终很弱（~0.002），策略更多用身体推球而非精确踢球

---

## 4. 定性观察（验证视频）

### 03_robustness_farball_model4999
- **测试条件**：spawn_dist=(2.5, 4.0)m，target_dist=(6.0, 9.0)m（均超出训练分布）
- **观察**：
  - G1 能稳定走向远处的球（3-4m 外），不摔不穿模
  - 到达球后能正确判断目标方向并带球推进
  - 长距离带球（6-9m）过程中保持控球，球不会跑偏太远
  - 策略对初始条件有良好泛化，不依赖训练时的固定距离
- **结论**：策略具备超分布泛化能力

### 04_diverse_longrollout_model4999
- **测试条件**：默认参数，341 帧连续 rollout（~6.8s，包含多次 episode）
- **观察**：
  - 多次 episode 重置后 G1 持续稳定行走+带球
  - 每次新目标方向不同，策略能快速调整行进方向
  - 步态自然流畅，无抖动或异常动作
  - 行为不退化，展示策略鲁棒性
- **结论**：策略在持续运行中保持稳定

### 05_training_progression
- **model_0（随机策略）**：G1 立即摔倒，四肢乱动，无任何有意义行为
- **model_300（300 iter）**：学会站立平衡，双腿微微弯曲保持重心，但不会移动
- **model_1400（1400 iter）**：稳定行走，能走向球并带球推进，步态已成型
- **model_4999（最终）**：步态更流畅自然，带球路径更直接，转向更果断
- **结论**：清晰展示 RL 四阶段演化 — 摔倒→站立→走向球→精确带球

### 06_v1_vs_v2_physics
- **测试条件**：v1 checkpoint (model_2999) 在 v2 物理环境下运行
- **观察**：
  - G1 能正常行走（v1 的步态在 v2 物理下仍有效）
  - 走向球后尝试推球，但球因摩擦不再无限滑动
  - 球只移动很短距离就停下，策略反复推球但无法有效带球到目标
  - 表现为"原地反复蹭球"而非"带球推进"
- **结论**：v1 策略依赖无摩擦物理的"推一下滑很远"特性，在真实摩擦下失效

---

## 5. 物理验证：condim=6 rolling friction

### 实验验证（tests/test_soccer_ball_roll.py）
- condim=3 + friction[2]=0.01：球以 6m/s 初速滚动 **无限远**（摩擦完全失效）
- condim=6 + friction[2]=0.01：球以 6m/s 初速滚动 **~9m** 停下
- condim=6 + friction[2]=0.1：球以 6m/s 初速滚动 **~3m** 停下
- 混合 condim（地面=3，球=6）：取 max=6，行为与 6/6 相同

### 对训练的影响
- v1（condim=3）：球无摩擦 → 策略学到"轻推一下球就滑到目标" → 不需要精确控球
- v2（condim=6）：球有摩擦 → 策略必须持续带球推进 → 学会真正的控球技能
- 这解释了 v2 的 possession_rate 95.9%（持续贴近球）vs v1 的低控球

---

## 6. 局限性 + v3 方向

### 当前局限
1. **kick_contact 信号弱**：策略更多用身体/腿部推球，而非精确的脚底踢球
2. **无对抗**：单机器人环境，无防守者干扰
3. **固定球场**：无地形变化、无风力干扰
4. **球惯性简化**：使用实心球惯性（vs 真实空心足球壳体惯性差 40%）
5. **无射门**：只有带球到目标点，无球门+守门员场景
6. **观测无延迟**：真实传感器有 1-3 帧延迟

### v3 可能方向
1. **Reward weight curriculum**：按训练阶段动态调整权重（站立→走路→带球→精确控球）
2. **球惯性修正**：使用空心壳体惯性（inertiaFromGeom="false"）
3. **Domain randomization 增强**：球摩擦/质量/弹性随机化
4. **射门任务**：带球到射门区 + 大力射门到球门
5. **多智能体**：加入防守者，学习对抗带球
6. **观测延迟**：模拟真实传感器延迟

---

## 7. 文件清单

```
soccer_eval/2026-06-01_dribble_soccer-robot_v2/
├── README.md
├── v2_analysis.md                          ← 本文档
├── 01_midtrain_condim6_model1400/
│   ├── dribble_v2_model1400.mp4
│   ├── eval_v2_iter1400.csv
│   └── frame_01~04.png
├── 02_final_default_model5000/
│   ├── dribble_v2_model4999.mp4
│   ├── eval_v2_model4999.csv
│   ├── eval_v1_on_v2physics.csv
│   └── frame_01~04.png
├── 03_robustness_farball_model4999/
│   ├── dribble_v2_farball.mp4
│   └── frame_01~04.png
├── 04_diverse_longrollout_model4999/
│   ├── dribble_v2_longrollout.mp4
│   └── frame_01~04.png
├── 05_training_progression/
│   ├── dribble_model_0.mp4
│   ├── dribble_model_300.mp4
│   ├── dribble_model_1400.mp4
│   ├── dribble_model_4999.mp4
│   └── *_frame_01~04.png
└── 06_v1_vs_v2_physics/
    ├── dribble_v1_on_v2physics.mp4
    └── frame_01~04.png
```
