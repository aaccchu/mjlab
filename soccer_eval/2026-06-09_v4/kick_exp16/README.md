# EXP16 评估留痕 — e2e EKF kick(踢球链最佳基线,越界修复前)

**踢球链修复前最佳基线 / EXP17-19 越界修复线的对照。** 源权重:
`checkpoints/v4_soccer/kick_exp16/model_1999.pt`
(= `logs/rsl_rl/mos92_velocity/2026-06-09_22-19-34_spike_v4_e2e_ekf_kick/model_1999.pt`)。
全指标 + 综合评分见 `metrics.json`。

## 产物(三段代表性 dribble-arc 演示)
由 `scripts/eval_v4_kick_exp16.py`(共用 `scripts/_v4_scenario_video.py`)生成。
单 env0、play=False(逐 episode reset 出变化)、GT-pose 课程拉到最终态(纯 EKF)、
关 corruption/push、goal_target_fraction=1.0(每 episode 朝球门):
- `kick_exp16_approach.mp4` — 机器人走向球(接近)。
- `kick_exp16_strike.mp4` — 起脚踢球(峰值球速时刻)。
- `kick_exp16_goalward.mp4` — 球被驱向球门(本段**捕到真实进球**,`scenarios.json` 中 `scored: true`)。
- `*_<scenario>.png` — 每段中点静帧;`scenarios.json` — 选取窗口与进球标记。

## 产物(机器人视觉 POV + 自定位信念演示)
由 `scripts/eval_v4_pov_belief_exp16.py`(共用 `scripts/_v4_pov_belief_video.py`)生成,
完整一个 episode 的三联画(31/14s,30fps):
- `kick_exp16_pov_belief.mp4` — 左:机器人头部相机 POV(`head_cam_rgb`,策略 CNN 实际输入);
  中:俯视球场 GT 位姿(绿)vs 策略融合信念(红 x,即策略消费的 `_last_xy_n`)+ 球(白)+ 目标(黄星),带拖尾;
  右:定位误差(米)与融合视野覆盖率(uniq_frac)随时间曲线。
- `pov_belief.json` — episode 统计(是否进球、首/末/均定位误差、起始/最大覆盖率)。
- **本段叙事**:**start_coverage 0.04(刚 reset 几乎看不到场地信息,定位误差 ~5.3m)→
  机器人转身扫视,覆盖率升、误差在 ~step40 骤降到 ~0.3m → 带球射门并进球(`scored: true`,
  step414 结束)**。正是"一开始看不到→调整后看到→踢球进门"的完整链路。

## 核心确证(末100稳态,四轮同法对照)
- 真实射门率 0.319、episode_success 0.479(四轮最高)、robot_to_ball 0.914(四轮最优)。
- **out_of_bounds 0.345(四轮最差)= EXP17-19 修复对象**。

## 与代表性部署模型(EXP19)对照
综合分 EXP16(0.686)≈ EXP19(0.681);EXP19 在进球+不出界两核心维度 Pareto 优于 EXP16,
故部署选 EXP19(`../kick_oob_exp19/`)。EXP16 归档为修复前最佳基线。

## 方法学局限
数字取自训练 stdout 稳态末100窗口(四轮同法,趋势可信),非全新 headless 评估;
视频为单 env0 定性演示(非打分);out_of_bounds 是稳态率(0.345),单段 demo 不必然体现。
详见 `soccer_robot_v4_experiment.md`「EXP16-19 多指标综合评价」+ `v4评估指标详解.md §6`。
