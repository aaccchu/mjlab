# EXP19 评估留痕 — e2e EKF kick + OOB 修复 + 强软边界 shaping

**代表性部署模型。** 源权重:`checkpoints/v4_soccer/kick_oob_exp19/model_1999.pt`
(= `logs/rsl_rl/mos92_velocity/2026-06-10_04-51-15_spike_v4_e2e_ekf_kick_oob_soft2/model_1999.pt`)。
全指标对照 + 综合评分见 `metrics.json`。

## 核心确证(末100稳态,四轮同法对照)
- 真实射门率 **0.325**(EXP16-19 最高);out_of_bounds **0.294**(< EXP16 0.345)。
- 在"进球 + 不出界"两核心维度 Pareto 优于 EXP16;综合分 0.681 ≈ EXP16 0.686(噪声内)。

## EXP16-19 排名
EXP16 (0.686) ≈ **EXP19 (0.681)** > EXP18 (0.409) > EXP17 (0.380)。
EXP17 OOB 0.256 最低,是 time_out 修复机制验证最佳(非部署最佳)。

## 方法学局限
沿用本目录 README 的评估方法学:episode 级指标绝对值以训练日志为准,逐步量(pos_err/ball_speed)可信;
fell_over 启动 transient 已排除;out_of_bounds 0.294 是 time_out 修复后的结构地板。
详见 `soccer_robot_v4_experiment.md`「EXP16-19 多指标综合评价」+ `v4评估指标详解.md §6`。
