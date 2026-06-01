# 可视化验证集 — soccer-robot v2（G1 端到端带球 + condim=6 物理修复 + 接触奖励）

本目录汇总 v2 版策略（`Mjlab-Soccer-Unitree-G1`，5000 迭代训练，
run `2026-06-01_11-09-25`）的所有可视化验证结果。

v2 相对 v1 的核心改动：
1. **球 condim=3→6**：启用 rolling friction，球不再无限滑动
2. **足↔球 ContactSensor**：检测脚与球的接触力
3. **kick_contact 奖励（weight=0.3）**：鼓励用脚触球
4. **nconmax=100 + contact_sensor_maxmatch=128**：适配 condim=6 约束增加

渲染命令模板：
```sh
MUJOCO_GL=egl uv run play Mjlab-Soccer-Unitree-G1 --agent trained \
  --checkpoint-file <model_XXXX.pt> --num-envs 1 --video True \
  --video-length 400 --video-height 480 --video-width 640
```

评估命令模板：
```sh
MUJOCO_GL=egl uv run evaluate-soccer \
  --checkpoint-file <model_XXXX.pt> --seed 42 --output-csv <path.csv>
```

## 子文件夹说明

### 01_midtrain_condim6_model1400
- **着重点:** 训练中途（28%，1400 迭代）快照，验证 condim=6 物理下策略已快速收敛
- checkpoint: model_1400.pt
- 评估结果（512 episodes，seed=42）：
  - success_rate: 99.6%（v1 最终仅 6.5%）
  - fall_rate: 0.4%
  - possession_rate: 89.6%
  - ball_to_target_error: 0.23m
  - time_to_goal: 207 步（~4.1s）
- 效果: 球有真实滚动摩擦，策略学会精确带球到目标点

### 02_final_default_model5000
- **着重点:** 最终训练完成（5000 迭代）标准评估 + D-vs-B 对比
- checkpoint: model_4999.pt
- 评估结果（512 episodes，seed=42）：
  - success_rate: 99.8%
  - fall_rate: 0.2%
  - possession_rate: 95.9%
  - ball_to_target_error: 0.20m
  - time_to_goal: 122 步（~2.4s）
- D-vs-B 对比（v1 checkpoint on v2 physics）：
  - success_rate: 18.4%（vs v2 的 99.8%）
  - ball_to_target_error: 2.57m（vs v2 的 0.20m）
  - possession_rate: 67.8%（vs v2 的 95.9%）
- 结论: v2 策略在 v2 物理下远优于 v1 策略，证明重训的必要性

## D-vs-B 对比（核心验证）

| 指标 | B: v1 on v2 physics | D: v2 on v2 physics | 差异 |
|------|--------------------|--------------------|------|
| success_rate | 18.4% | 99.8% | +81.4pp |
| ball_to_target_error | 2.57m | 0.20m | -92% |
| possession_rate | 67.8% | 95.9% | +28.1pp |
| time_to_goal | 245 步 | 122 步 | -50% |
| fall_rate | 0.2% | 0.2% | 相同 |

v1 策略在 v2 物理下仍能行走（fall_rate 相同），但带球能力严重退化 —
因为 v1 学到的是"推球后球无限滑动"的策略，在有摩擦的 v2 物理下失效。

## v1 vs v2 最终对比

| 指标 | v1 (model_2999) | v2 (model_4999) | 改善 |
|------|----------------|-----------------|------|
| success_rate | 6.5% | 99.8% | +93pp |
| ball_to_target_error | 3.79m | 0.20m | -95% |
| possession_rate | — | 95.9% | — |
| time_to_goal | — | 122 步(2.4s) | — |
| fell_over | 12.5% | 0.2% | -98% |

核心原因：condim=6 使球有真实摩擦，策略学会精确控球而非依赖无摩擦滑动。
