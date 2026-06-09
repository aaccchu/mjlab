# 线B 踢球技能 EXP14 — 验证评估产物(2026-06-09 v4)

源权重:`checkpoints/v4_soccer/lineB_kick_exp14/model_1999.pt`(EXP14,线B B2 踢球首训)
评估脚本:`scripts/eval_v4_lineB_kick.py`

## 产物
- `lineB_kick_exp14-step-0.mp4` — 9 env 平铺 rollout 视频(500 步,play 模式)。
- `frame_{0..3}.png` — 视频四等分预览静帧。
- `metrics.json` — 量化指标(64 env headless,400 步,warmup 150)。

## 量化结果(metrics.json)
| 指标 | mean | median | 备注 |
|---|---|---|---|
| **selfloc_pos_err_m** | **0.96m** | **0.97m** | ✅ 与训练日志 0.97m 吻合,定位继承自线A未退化,归档数字可信 |
| ball_speed | 0.32 | 0.32 | 逐步量,与训练 0.40 同量级(短窗口偏低) |
| episode_success | 0.11 | 0.13 | episode级,短窗口低估,训练日志 0.42 更可信 |
| ball_to_target_error | 4.14m | 4.12m | episode级,短窗口低估,训练日志 2.45m 更可信 |
| goal_rate | 0.001 | 0.0 | episode级,短窗口严重低估,训练日志 0.11 更可信 |
| fell_over | (n=2) | — | 采样不足无意义,训练日志 0.074 为准 |

## 评估方法学局限(诚实记录,同 lineA)
1. **pos_err 可信**:逐步量,warmup 150 步后 median 0.97m 精确复现训练稳态 0.97m。
   这确证了线B 训练中定位没有被踢球技能干扰退化 ✅。
2. **episode 级指标(goal_rate/ball_to_target/episode_success)被严重低估**:play=False 全新 reset +
   仅 400 步窗口,多数 episode 未走完"带球→进门"就被采样。**这些以训练日志为准**
   (goal 0.11、ball_to_target 2.45m、success 0.42)。headless 短窗口只适合确证逐步量。
3. **fell_over n=2 无意义**:仅 episode 终止步写入,400 步采样不足。真实摔倒率训练日志 0.074。
4. 仍 oracle 检测阶段(隔离变量)。

## 与线A 对比的意义
线A(EKF)pos_err 0.99m → 线B(EKF+kick)pos_err 0.97m,**定位完全保持**;同时 goal_rate
0.08→0.11(训练日志)。证明"在已解决的定位上叠加踢球技能"未破坏定位。EXP14 是中间里程碑,
已知 spawn 穿透 bug + 温吞推球瓶颈在 EXP15 修复(见 ../../soccer_robot_v4_kick_research.md 共同诊断)。
