# EXP16 — e2e EKF kick(踢球链最佳基线,越界修复前)

`model_1999.pt` 来自 `logs/rsl_rl/mos92_velocity/2026-06-09_22-19-34_spike_v4_e2e_ekf_kick/`,
完整跑满 2000 iter。`params/{env,agent}.yaml` 为复现所需配置;`git/mjlab.diff` 为归档时
git 版本信息。

## 定位
EXP16 是 v4 踢球链(线B)在**越界修复前**的最佳一轮:std0.85 + kick threshold0.3 的组合,
真实射门率 0.319、fell_over 0.044(四轮最稳)、robot_to_ball 0.914(追球最高效)。
它是 EXP17-19 越界修复线的**起点 / 对照基线**——OOB 0.345 正是后续要修的痛点。

## 关键指标(末100稳态)
| 指标 | 值 | 备注 |
|---|---|---|
| 真实射门率(子集) | 0.319 | goal_rate 0.154 / target_is_goal 0.483 |
| episode_success | 0.479 | 四轮最高 |
| possession | 478.7 | 四轮最高 |
| robot_to_ball | 0.914 | ↓ 四轮最优(追球最准) |
| ball_to_tgt_err | 2.347 | ↓ 四轮最优 |
| ball_stuck_s | 0.825 | ↓ 四轮最优 |
| fell_over | 0.044 | ↓ 四轮最稳 |
| **out_of_bounds** | **0.345** | ↑ 四轮最差 → EXP17-19 修复对象 |
| pos_err_m | 0.913 | ↓ 定位最好 |
| ball_speed_peak | 2.957 | 踢力(EXP17-19 略强) |

## 与代表性部署模型(EXP19)的关系
多指标综合评分 EXP16(0.686)≈ EXP19(0.681),但 EXP19 在**进球 + 不出界两核心维度
Pareto 优于 EXP16**(射门率 0.325>0.319 且 OOB 0.294<0.345),故部署模型选 EXP19
(见 `checkpoints/v4_soccer/kick_oob_exp19/`)。EXP16 作为**修复前最佳基线**归档,
次要指标(success/possession/robot_to_ball/ball_to_tgt)仍是四轮最好,价值在于对照。

完整谱系与因果链见 `soccer_robot_v4_experiment.md`「EXP16-19 多指标综合评价」+ `v4评估指标详解.md §6`。
