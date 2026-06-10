# EXP20b — aim-the-kick(v4 踢球链当前最佳模型)

`model_1999.pt` 来自 `logs/rsl_rl/mos92_velocity/2026-06-10_20-22-43_spike_v4_e2e_ekf_kick_aim2/`
(EXP20b = EXP20 同配置续训 2000 iter,bootstrap 自 EXP20 model_1999;EXP20 bootstrap 自 EXP19)。
`params/{env,agent}.yaml` 为复现配置;`git/mjlab.diff` 为归档时 git 版本信息。

## 为什么是当前最佳
**真实射门率 0.38-0.41**(EXP16-19 平台 0.32-0.33,+20%,v4 单次最大增益),且 5 项历史最好:
| 指标 | EXP16-19 | EXP20b |
|---|---|---|
| 真实射门率 | 0.32-0.33 | **0.38-0.41** |
| ball_to_tgt_err | 2.7-2.8 | **1.88** |
| robot_to_ball | 1.0-1.2 | **0.60** |
| out_of_bounds | 0.26"地板" | **0.20** |
| pos_err_m | 0.91-0.96 | **0.85-0.88** |
| episode_success | 0.45 | **0.58** |
| fell_over | 0.04-0.06 | 0.063 |
| ball_speed_peak | 3.2 | 2.70 ⚠️ 缓降需盯 |

## 谱系与改动
EXP19(OOB 修复线最佳)→ **EXP20**(三件套:① 全局修 `_gaze_uv_visible` pitch 符号 bug——
正 neck_pitch=低头,旧公式教抬头;② EXP20 env 放开 neck_pitch pose std→1.0;③ 新增
kick_lateral_alignment 摆正再踢)→ **EXP20b**(同配置续训)。
env=`mos92_soccer_e2e_dualcam_ekf_kick_aim_env_cfg`,脚本=`spike_v4_e2e_ekf_kick_aim{,2}.py`。
全程无 std 重开(codex"低速贴球吸引子"教训:噪声微调 50-200 iter 可洗掉踢球链)。

**机制验证**:瞄准修好后,出界/追飞球/落点散这些此前"各自为战"的症状一起好转——证明
"OOB 0.26 地板"和"pos_err 0.9 平台"其实都是踢歪的副作用。低头修复还突破了 ~1.25m 脚下
盲区(评估视频出现 ball IN VIEW 0.7m 帧)。

## 评估产物
`soccer_eval/2026-06-10_v4/kick_aim_exp20b/`:三段 dribble-arc 演示 + 8 联画 POV/信念视频
(scored:true 真实进球)+ metrics.json + pov_belief.json + README。

## 局限
训练 log 稳态口径(同法对照可信);仍 oracle 球检测;ball_speed_peak 缓降破 2.5 应停;
0.5-0.6 终极目标物理可达性未跑上界探针。
