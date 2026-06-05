# Spike A-MOS92: Neck Gaze Verification

## 结果: PARTIAL PASS

MOS92 能用 neck_yaw 追踪球方向,带球能力基本保持,但稳定性下降。

## 关键指标

| 指标 | G-2b (baseline) | A-MOS92 (gaze) |
|------|----------------|----------------|
| gaze_at_ball | — | 0.69 |
| kick_contact | 0.82 | 0.76 |
| dribble_success | 4.14 | 3.84 |
| fell_over | 0.08 | 0.33 |

## v3 决策

走 learned gaze (方案 B),需要更长训练或降低 gaze weight 改善稳定性。

## 文件

- `mos92_gaze-step-0.mp4` — 带球+转头视频

## Checkpoint

`logs/rsl_rl/mos92_velocity/2026-06-02_00-17-56/model_4998.pt`
