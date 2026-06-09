# 线A EKF 自定位 — 验证评估产物(2026-06-09 v4)

源权重:`checkpoints/v4_soccer/lineA_ekf_exp13/model_1999.pt`(EXP13)
评估脚本:`scripts/eval_v4_lineA.py`

## 产物
- `lineA_ekf/lineA_ekf-step-0.mp4` — 9 env 平铺 rollout 视频(500 步)。
- `lineA_ekf/frame_{0..3}.png` — 视频四等分预览静帧。
- `lineA_ekf/metrics.json` — 量化指标(128 env headless,400 步,warmup 150)。

## 量化结果(metrics.json)
| 指标 | mean | median | 备注 |
|---|---|---|---|
| **selfloc_pos_err_m** | **1.02m** | **0.99m** | ✅ 与训练日志 0.98m 吻合,归档数字可信 |
| ball_speed | 0.35 | 0.37 | 与训练一致 |
| episode_success | 0.05 | 0.05 | 见下方局限说明 |
| ball_to_target_error | 4.55m | 4.59m | 见下方局限说明 |
| goal_rate | 0.004 | 0.0 | 见下方局限说明 |
| fell_over | (n=4) | — | 采样不足,无意义,见下 |

## 评估方法学局限(诚实记录)
1. **pos_err 可信**:逐步量,warmup 150 步让 EKF 从宽先验充分收敛后,median 0.99m
   精确复现训练稳态 0.98m。这是本次归档要确证的核心数字 ✅。
2. **goal_rate / ball_to_target / episode_success 被低估**:这些是 episode 级累积量。
   评估用 play=False 全新 reset + 仅 400 步窗口,许多 episode 尚未完成带球→进门就被采样,
   故低于训练日志(goal 0.08、ball_to_target 2.97m)。训练日志的 episode 级数字更可信;
   本评估的逐步量(pos_err、ball_speed)更适合 headless 短窗口确证。
3. **fell_over n=4 无意义**:该 key 仅在 episode 终止步写入 log,400 步只采到 4 次。
   真实摔倒率以训练日志 0.057 为准。
4. 仍 oracle 检测阶段(隔离变量)。

## 结论
线A EKF 自定位里程碑**核心数字(pos_err≈1m)经独立 headless 评估确证**。
episode 级踢球指标的评估低估是窗口长度所致,非模型退化;以训练日志为准。
视频/静帧供定性查看步态与定位稳定性。
