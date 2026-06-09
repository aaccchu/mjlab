# lineB_kick_exp14 — 线B B2 踢球技能首训(关键节点留痕)

> 归档于 2026-06-09。源:`logs/rsl_rl/mos92_velocity/2026-06-09_19-54-04_spike_v4_e2e_ekf_kick`
> 训练脚本:`scripts/spike_v4_e2e_ekf_kick.py` | env:`mos92_soccer_e2e_dualcam_ekf_kick_env_cfg`

## 这是什么
线A(EKF 自定位,见 `../lineA_ekf_exp13`)收官后,首次专门攻"近脚踢球技能"的端到端 RL 训练。
从 EXP13(线A)bootstrap,加两件事:
1. **near_foot_spawn_fraction=0.5**:一半 episode 把球直接 spawn 在脚前(密集练习"球在脚边→踢门"),
   另一半保留完整接近+带球任务(不遗忘行走/接近/定位)。
2. **dribble_kick_impulse 奖励**:在触球步奖励"朝球门方向的球速"(踢球质量),二元 kick_contact 教不了。

## 指标(末100采样,model_1999)
| 维度 | 指标 | EXP14 | vs EXP13 |
|---|---|---|---|
| 踢球 | goal_rate | **0.110** | +37%(0.08) |
| | episode_success | 0.42 | +0.02 |
| | ball_to_tgt_err | 2.45m | -0.52m(更近门) |
| | ball_speed | 0.40 | — |
| 护栏 | pos_err_m | 0.97 | 持平(定位没退化) |
| | fell_over | 0.074 | +0.017 |
| | out_of_bounds | 0.28 | +0.07 ⚠️ |

## 结论与已知问题
- **方向有效**:goal_rate +37%、球更接近门,证明"近脚 spawn + 冲量奖励"能教踢球。
- **卡 0.11 量级**:ball_speed 卡 0.39-0.40,是"温吞推球"局部最优症状(冲量奖励无门槛 + bootstrap
  继承坍缩动作 std 抑制爆发探索)——见 `../../soccer_robot_v4_kick_research.md` 共同诊断。
- **真 bug(后续 EXP15 修)**:near_foot_dist 下界 0.08m 从 root 量,球(半径0.11)spawn 进脚几何→
  穿透弹飞→out_of_bounds 升到 0.28。EXP15 修为 0.25-0.40m + rear_spawn 置 0。

## 状态
**中间里程碑,非最终**。EXP15 在此基础上做调研驱动改进(速度门槛奖励 + 重开探索 + spawn 修复)。
