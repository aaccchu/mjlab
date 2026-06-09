# v4 soccer — 关键节点验证评估产物索引(2026-06-09)

本目录留痕 v4 两条线的关键节点评估。每个子目录有独立 README + 视频 + 静帧 + metrics.json。

| 子目录 | 节点 | 源权重 | 核心确证 |
|---|---|---|---|
| `lineA_ekf/` | 线A EKF 自定位(EXP13) | `checkpoints/v4_soccer/lineA_ekf_exp13/model_1999.pt` | pos_err median **0.99m** ✅ |
| `lineB_kick_exp14/` | 线B 踢球技能首训(EXP14) | `checkpoints/v4_soccer/lineB_kick_exp14/model_1999.pt` | pos_err **0.97m**(定位未退化)+ goal_rate 0.08→0.11(训练日志) |

## 配套方法论文档(repo 根目录)
- `soccer_robot_v4_localization_method.md` — 线A 自定位方法论(EKF/UKF、对称消歧、主动视觉调研)。
- `soccer_robot_v4_kick_research.md` — 线B 踢球链调研 + codex 经验 + **共同诊断**(两条独立来源交叉印证)。
- `soccer_robot_v4_experiment.md` — 实验流水账(EXP1-15)。

## 评估方法学局限(两个节点通用,诚实记录)
1. **逐步量可信**:selfloc_pos_err_m、ball_speed 是逐步量,warmup 150 步后精确复现训练稳态,
   归档数字可信(线A 0.99m / 线B 0.97m)。
2. **episode 级指标被低估**:goal_rate / ball_to_target / episode_success 是 episode 级累积量,
   play=False 全新 reset + 仅 400 步窗口下许多 episode 未走完带球→进门就被采样,**以训练日志为准**
   (线B:goal 0.11、ball_to_target 2.45m、success 0.42)。headless 短窗口只适合确证逐步量。
3. **fell_over 采样不足**(n=2-4):仅 episode 终止步写入,真实摔倒率以训练日志为准(线A 0.057 / 线B 0.074)。
4. 仍 oracle 检测阶段(隔离变量)。

## 结论
- 线A EKF 自定位 pos_err≈1m **经独立 headless 评估确证**。
- 线B 在叠加踢球技能后**定位完全保持**(0.99→0.97m),证明多技能未互相破坏。
- episode 级踢球指标的评估低估是窗口长度所致,非模型退化。视频/静帧供定性查看步态与稳定性。

