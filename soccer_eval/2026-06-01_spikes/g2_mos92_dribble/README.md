# Spike G-2: MOS92 Soccer Dribble

## 结果: PASS

MOS92 能在 soccer env 中稳定带球到目标点。

## 关键指标 (3000 iter)

| 指标 | 值 |
|------|-----|
| kick_contact | 0.82 |
| dribble_success | 4.14 |
| fell_over | 0.08 |
| mean_reward | 224.4 |

## 文件

- `mos92_dribble-step-0.mp4` — G-2a (失败, kick_contact=0)
- `mos92_dribble_v2-step-0.mp4` — G-2b (成功, 参数适配后)

## 关键修复

G1 默认 spawn_dist=0.6-1.5m 对 MOS92(0.45m) 太远。
缩放到 0.3-0.8m + approach_radius=0.25m 后成功。

## Checkpoint

`logs/rsl_rl/mos92_velocity/2026-06-01_22-58-05/model_2999.pt`
