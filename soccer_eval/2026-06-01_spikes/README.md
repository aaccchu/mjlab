# 可视化验证集 — v2.5 Feasibility Spikes

本目录汇总 v2.5 前置验证(Spike)中产生策略的可视化结果。每个子目录对应一个
spike 实验，包含一段策略 rollout 视频(`spike_<id>-step-0.mp4`)和 4 张抽帧
(`frame_01~04.png`)。

完整实验记录见 `../../soccer_robot_v2_experiment.md`，方案见 `../../soccer_robot_v2-5.md`。

> **渲染说明:** 这些视频用 `scripts/render_spike_videos.py` 生成。该脚本**重建各
> spike 训练时的 env**(spawn/target 分布、reward 改动、waist std)，而非默认
> registry env —— 因为 F 系列改了距离分布、A 系列改了 gaze/pose reward，用默认
> env 播放无法展示训练时的真实行为。

渲染命令:
```sh
MUJOCO_GL=egl uv run python scripts/render_spike_videos.py --spike all
# 或单个: --spike a1 / a2 / f1 / f2
```

## 子文件夹说明

### a1 — Spike A gaze (weight=1.0, 根因待验证)
- **着重点:** 高 gaze weight + 松 waist 约束下的转头行为
- checkpoint: `2026-06-01_15-54-10_spike_a/model_5498.pt`
- 训练态 fall_fraction 15.2%、success 91.3%、gaze reward 0.56(转头明显)
- evaluate(默认 env): success 99.8% / fall 0.2% —— 标准条件下不摔
- 看点: 转头幅度大，但训练分布下平衡受扰

### a2 — Spike A gaze (weight=0.3, 根因确认 ✓)
- **着重点:** 降 weight + 收紧 waist std 后的转头行为
- checkpoint: `2026-06-01_18-09-51_spike_a_a2/model_5998.pt`
- 训练态 fall_fraction 7.0%(<10% 目标)、success 94.5%、gaze reward 0.45
- 结论: learned gaze 可行，weight=0.3 是「能转头又不破坏平衡」的平衡点
- 看点: 转头幅度比 a1 小但更稳定

### f1 — Spike F 全场尺度 (4m, 接近通过)
- **着重点:** 两阶段 reward(approach_delta + heading_to_ball)在温和扩大距离下
- checkpoint: `2026-06-01_16-07-40_spike_f_f1/model_6998.pt`
- spawn 0.6-4m / target 2-8m，success 67.4%(目标 70%)
- 看点: 机器人能走向较远的球并带到目标

### f2 — Spike F 全场尺度 (9m, 崩溃)
- **着重点:** 全场覆盖距离下两阶段 reward 的极限
- checkpoint: `2026-06-01_16-57-26_spike_f_f2/model_6998.pt`
- spawn 0.6-9m / target 2-12m，success 仅 18.4%(目标 40%)
- 结论: 直接全场训练不行，证明 v3 需要 distance curriculum
- 看点: 远距离 approach 耗时长，大量 episode 超时，球常未被推动

## Spike 结论速查

| Spike | 问题 | 结果 | v3 决策 |
|-------|------|------|---------|
| A (gaze) | G1 能学会转头不摔? | 可行(weight≤0.3) | learned gaze 方案 B + 保守 weight |
| F (全场) | 两阶段 reward 泛化全场? | 4m 行 / 9m 崩 | reward 结构正确，需 distance curriculum |

> Spike B(相机视野)是渲染测试无策略视频，结果见
> `../spike_b_camera_visibility/` 和实验记录。

### g1_mos92_walk — Spike G Step 1 (MOS92 基础行走 ✓)
- **着重点:** MOS92 18-DOF 人形在 mjlab 中能否行走(位置控制)
- checkpoint: `mos92_velocity/2026-06-01_20-34-38/model_999.pt`
- 1000 iter 结果: mean_reward 83.2, fell_over 0.0, track_lin_vel 1.57
- PD 参数: armature=0.01, stiffness≈39.5, damping≈2.51(xml 默认值)
- 结论: MOS92 能在 flat terrain 上稳定行走，速度跟踪良好
- 看点: 简化碰撞几何体(无 mesh)，步态简单但有效

### g2_mos92_dribble — Spike G Step 2 (MOS92 带球 ✓)
- **着重点:** MOS92 能否在 soccer env 中带球到目标点
- checkpoint: `mos92_velocity/2026-06-01_22-58-05/model_2999.pt`
- 3000 iter 结果: kick_contact=0.82, dribble_success=4.14, fell_over=0.08
- 关键修复: spawn_dist 从 0.6-1.5m 缩放到 0.3-0.8m 适配 MOS92 体型
- 结论: MOS92 能稳定带球,参数需按体型缩放

### a_mos92_gaze — Spike A-MOS92 (Neck Gaze, Partial Pass)
- **着重点:** neck_yaw 追踪球方向是否破坏带球/平衡
- checkpoint: `mos92_velocity/2026-06-02_00-17-56/model_4998.pt`
- 2000 iter fine-tune: gaze=0.69, kick_contact=0.76, fell_over=0.33
- 结论: 能转头追踪球(-7% 带球退化),但稳定性下降(fell_over 0.08→0.33)
- v3 决策: learned gaze 可行,需降低 weight 或加动作平滑惩罚

### a_mos92_gaze_v2 — Spike A-MOS92 三组对照 (最优配置)
- **着重点:** 找 MOS92 gaze 的 weight/neck_std 平衡点
- checkpoint: `mos92_velocity/2026-06-02_13-33-44/model_3998.pt` (G3 最优)
- G3(weight=1.0 + tight neck_std=1.0): gaze=0.70, dribble=3.95, fell_over=0.29
- 结论: 与 G1 相反,MOS92 上紧 neck 约束比降 weight 更有效
- neck_yaw 实测转头: 平均 5.7°/最大 54°(`scripts/measure_neck_yaw.py`)

### e_mos92_goal — Spike E-MOS92 (Goal-Scoring, PASS)
- **着重点:** 能否把球踢进球门(v3 启动硬门槛)
- checkpoint: `mos92_velocity/2026-06-02_14-41-20/model_4000.pt`
- goal_rate 全程 >10%(峰值 24%) → **v3 §0.1b 门槛 1 通过,v3 可启动**
- ⚠️ goal reward 过强,upright 0.94→0.56、fell_over 0.08→0.21,姿态退化
- v3 决策: Stage 1 需降 goal weight 或加 upright/alive 权重防退化
