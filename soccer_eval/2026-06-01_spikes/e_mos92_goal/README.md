# Spike E-MOS92: Goal-Scoring 闭环

## 结果: PASS(critical path 门槛通过,但需调权重)

MOS92 能把球踢进球门。goal_rate 全程 >10% 门槛(峰值 24%),
**v3 §0.1b 门槛 1 通过 → v3 可启动**。

## 关键指标

| 指标 | G-2b 基线 | E 峰值 | E step4000 |
|------|----------|--------|-----------|
| goal_rate | — | 0.24 | 0.125 |
| kick_contact | 0.82 | — | 0.47 |
| dribble_success | 4.14 | — | 2.34 |
| fell_over | 0.08 | — | 0.21 |
| upright | 0.94 | — | 0.56 |

## 关键问题(v3 必须解决)

goal reward(weight 10+2)过强,压倒平衡 reward —— upright 从 0.94 掉到 0.56。
goal_rate 峰值在早期(step 3972),之后过度优化导致姿态退化。
最佳 checkpoint 是 step ~4000,非最后一步。

## v3 决策

- v3 可启动(门槛通过)
- Stage 1 需降低 goal weight(如 5/1)或加 upright/alive 权重防姿态退化

## 文件

- `mos92_goal-step-0.mp4` — step4000 进球视频

## Checkpoint(最佳)

`logs/rsl_rl/mos92_velocity/2026-06-02_14-41-20/model_4000.pt`
