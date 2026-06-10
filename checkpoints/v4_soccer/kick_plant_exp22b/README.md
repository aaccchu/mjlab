# EXP22b — support-foot plant(预接触窗口)+ finish 豁免 + aim 三件套

**12h 自主推进窗口(2026-06-10/11)的最终模型,v4 当前最佳。**
`model_1999.pt` 来自 `logs/rsl_rl/mos92_velocity/2026-06-11_01-36-51_spike_v4_e2e_ekf_kick_plant_b/`,
完整 2000 iter。`params/` + `git/mjlab.diff` 为复现信息。

## 谱系
EXP20b(aim 三件套,射门率 0.368)→ EXP21(走廊 x 豁免,SHORT 44.6%→23.3%)
→ EXP22(接触帧 plant,双重稀疏止损)→ **EXP22b(预接触窗口 plant)**。

## 关键指标(末300/末100稳态)
| 指标 | EXP20b(窗口起点) | EXP22b | 判定 |
|---|---|---|---|
| 真实射门率 | 0.368 | **0.395-0.416**(中途稳定0.42-0.43带) | ✅ v4 首上 0.40 带 |
| SCORED(forensics) | 31.6% | **37.0%** | ✅ 三轮连升 |
| NEVER_ARRIVED | 17.6% | **16.7%**(中场球速 0.69→0.73) | ✅ 到达问题解决 |
| pos_err | 0.882 | **0.886-0.899** | ✅ 守住 |
| robot_to_ball | 0.599 | **0.59-0.65** | ✅ |
| ball_to_tgt_err | 1.88 | **1.78-1.83** | ✅ 新高 |
| fell_over | 0.063 | 0.059 | ✅ |
| out_of_bounds | 0.202 | 0.20-0.22 | ✅ 守住 ≤0.22 |
| ball_speed_peak | 2.70 | 2.62-2.69 | ⚠️ 缓降(>2.5 护栏内,连续三轮缓降需盯) |

## 已知顽固瓶颈(下一窗口的主攻方向)
**近门发力**:近门(x>7)kick 速度三轮卡 0.47-0.52(中场 0.73),kicks/episode 卡 9,
SHORT 回升到 40.6%(到达变快后门口轻推重新成为主导败因)。这不是惩罚结构问题
(EXP21 已豁免走廊)而是技能局部最优:轻推有稳定 goal_progress 正反馈,大力射门
有摔倒/出界风险。候选方案(未验证):近门区 kick_impulse 权重梯度 / shot_clearance
奖励(球离脚速度>1.5 才计 progress)。

完整判读链:`soccer_robot_v4_experiment.md` EXP20b 解剖 → EXP21 → EXP22/22b 各节;
指标方法学:`v4评估指标详解.md §7`(finish 漏斗 + forensics 四分类)。
