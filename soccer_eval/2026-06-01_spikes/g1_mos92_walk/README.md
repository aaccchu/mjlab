# Spike G-1: MOS92 基础行走验证

## 结论: PASS ✓

MOS92 (20-DOF) 在 mjlab 中稳定行走,确认为 v3 训练平台。

## 实验版本

### G-1 (xml 默认参数, 18-DOF)
- mean_reward: 83.2, fell_over: 0.0
- checkpoint: `logs/rsl_rl/mos92_velocity/2026-06-01_20-34-38/model_999.pt`

### G-1b (调优参数 + neck + 手臂贴身, 20-DOF) ← 最终版
- mean_reward: 78.3, fell_over: 0.04
- checkpoint: `logs/rsl_rl/mos92_velocity/2026-06-01_21-18-21/model_999.pt`
- video: `mos92_walk_v2-step-0.mp4`

## 关键改进 (G-1 → G-1b)
- PD 参数: MOS9-AMP-main 调优值 (armature 0.028/0.048, kp 47/105)
- Mesh 可视化: 21 STL 加回 xml
- Neck joints: neck_yaw(±90°) + neck_pitch(±28.6°) 用于 v3 gaze
- 手臂姿态: shoulder_roll=±1.4 (贴身)

## 验证清单
- [x] MOS92 能稳定行走 (fell_over≈0)
- [x] 速度跟踪良好 (track_lin_vel=1.55)
- [x] 调优 PD 参数收敛正常
- [x] Neck joints 不破坏行走稳定性
- [x] Mesh 渲染正常,机器人外形可见
