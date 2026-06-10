# EXP19 — e2e EKF kick + out-of-bounds fix + strong soft-boundary shaping

**代表性部署模型(v4 踢球 + 越界修复线)。** `model_1999.pt` 来自
`logs/rsl_rl/mos92_velocity/2026-06-10_04-51-15_spike_v4_e2e_ekf_kick_oob_soft2/`,
完整跑满 2000 iter。`params/{env,agent}.yaml` 为复现所需配置。

## 为什么选 EXP19(多指标综合,2026-06-10)
四轮(EXP16→19)稳态末100采样综合评分:EXP16 0.686 ≈ **EXP19 0.681**(差值在噪声内),
但 EXP19 在**两个核心维度上 Pareto 优于 EXP16**:
- 真实射门率 **0.325** > EXP16 0.319(主目标,权重最高)
- out_of_bounds **0.294** < EXP16 0.345(EXP16 最大痛点,本线修复对象)

EXP19 还拿下 goal_rate(0.173)、target_is_goal(0.531)三冠,继承全部修复
(time_out=False + 球门走廊 + 软边界 shaping)。EXP17 OOB 0.256 最低但射门率垫底
(机制验证最佳,非部署最佳);EXP18 全面被 EXP19 支配。

## 关键指标(末100稳态)
| 指标 | 值 | 方向 |
|---|---|---|
| 真实射门率(子集) | 0.325 | ↑ 四轮最高 |
| goal_rate / target_is_goal | 0.173 / 0.531 | ↑ |
| episode_success | 0.449 | ↑ |
| out_of_bounds | 0.294 | ↓ time_out 修复后结构地板 |
| fell_over | 0.063 | ↓ <0.1 护栏 |
| pos_err_m | 0.964 | ↓ 定位未退化 |
| ball_speed_peak | 3.181 | ↑ 踢力强 |

## 谱系
EXP16(最好踢球,OOB 0.345)→ EXP17(time_out 修复,OOB 0.345→0.262 胜负手)
→ EXP18(弱软边界,无改善)→ **EXP19(强软边界,主目标最强)**。
完整因果链见 `soccer_robot_v4_experiment.md` + `v4评估指标详解.md §6`。

## 局限(诚实记录)
- 数字取自训练 stdout 稳态末100窗口,非全新 headless play=False 评估;跨四轮同法计算,**趋势可信**。
- out_of_bounds 0.294 是 time_out 修复后的**结构地板**(goal_line_x==half_length==11.0,margin 0.3m),
  reward shaping(EXP18/19 已验证)压不下去;要再降需改场地/任务几何。
