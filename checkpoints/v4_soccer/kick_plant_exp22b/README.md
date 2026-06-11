# EXP22b — support-foot plant(预接触窗口)+ finish 豁免 + aim 三件套

**12h 自主推进窗口(2026-06-10/11)的最终训练产物。**
`model_1999.pt` 来自 `logs/rsl_rl/mos92_velocity/2026-06-11_01-36-51_spike_v4_e2e_ekf_kick_plant_b/`,
完整 2000 iter。`params/` + `git/mjlab.diff` 为复现信息。

> **⚠️ 2026-06-11 审计更正(codex 发现)**:本 README 初版引用的 forensics 数字
> (SCORED 37.0%、NEVER_ARRIVED 16.7%、中场球速 0.73)实测来自 **EXP20b 权重**
> (sed 改造临时脚本时静默不匹配,跑错模型)。下表为更正后数字。
> **"v4 当前最佳"的地位存疑**:forensics 口径下 EXP22b(SCORED 28.6%)未优于
> EXP21(34.2%);训练日志口径(0.395-0.416)仍是历史最高,两口径缺口的原因
> 未定(评估样本噪声/分布shift)。结论以 EXP23 及后续更大样本对决为准。

## 谱系
EXP20b(aim 三件套,射门率 0.368)→ EXP21(走廊 x 豁免,SHORT 44.6%→23.3%)
→ EXP22(接触帧 plant,双重稀疏止损)→ **EXP22b(预接触窗口 plant)**。

## 关键指标
**训练日志口径(末300/末100稳态,可信)**:
| 指标 | EXP20b(窗口起点) | EXP22b |
|---|---|---|
| 真实射门率 | 0.368 | **0.395-0.416**(中途 0.42-0.43 带) |
| pos_err | 0.882 | 0.886-0.899 |
| robot_to_ball | 0.599 | 0.59-0.65 |
| ball_to_tgt_err | 1.88 | **1.78-1.83**(新高) |
| fell_over | 0.063 | 0.059 |
| out_of_bounds | 0.202 | 0.20-0.22 |
| ball_speed_peak | 2.70 | 2.62-2.69 ⚠️ 三轮缓降需盯 |

**forensics 口径(更正后,kick_plant_exp22b 权重实测,192 episodes)**:
| | EXP20b | EXP21 | EXP22b |
|---|---|---|---|
| SCORED | 31.6% | **34.2%** | 28.6% |
| SHORT | 44.6% | 23.3% | 30.7% |
| NEVER_ARRIVED | 17.6% | 35.2% | 36.5% |
| 近门球速中位 | 0.47 | 0.52 | 0.42 |
| 中场球速中位 | 0.69 | 0.60 | 0.57 |

## 已知顽固瓶颈(下一窗口的主攻方向)
**近门发力**:近门 kick 速度按更正数据 0.42-0.52 区间徘徊(EXP23 设计前提在真数据下
更成立)。这不是惩罚结构问题(EXP21 已豁免走廊)而是技能局部最优:轻推有稳定
goal_progress 正反馈,大力射门有摔倒/出界风险。EXP23 = 近门 kick_impulse 增益×3。

完整判读链:`soccer_robot_v4_experiment.md` EXP20b 解剖 → EXP21 → EXP22/22b 各节 +
「EXP22b forensics 测量错误更正」;指标方法学:`v4评估指标详解.md §7`。
