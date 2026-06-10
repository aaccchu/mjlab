# EXP19 评估留痕 — e2e EKF kick + OOB 修复 + 强软边界 shaping

**代表性部署模型。** 源权重:`checkpoints/v4_soccer/kick_oob_exp19/model_1999.pt`
(= `logs/rsl_rl/mos92_velocity/2026-06-10_04-51-15_spike_v4_e2e_ekf_kick_oob_soft2/model_1999.pt`)。
全指标对照 + 综合评分见 `metrics.json`。

## 产物(三段代表性 dribble-arc 演示)
由 `scripts/eval_v4_kick_oob_exp19.py`(共用 `scripts/_v4_scenario_video.py`)生成。
单 env0、play=False(逐 episode reset 出变化)、GT-pose 课程拉到最终态(纯 EKF)、
关 corruption/push、goal_target_fraction=1.0(每 episode 朝球门):
- `kick_oob_exp19_approach.mp4` — 机器人走向球(接近)。
- `kick_oob_exp19_strike.mp4` — 起脚踢球(峰值球速时刻)。
- `kick_oob_exp19_goalward.mp4` — 球被驱向球门(`scenarios.json` 中 `scored` 标记是否捕到真实进球)。
- `*_<scenario>.png` — 每段中点静帧;`scenarios.json` — 选取窗口与进球标记。
- `metrics.json` — EXP16-19 四轮全指标对照 + 加权综合评分。
- 训练时 git 版本信息:`checkpoints/v4_soccer/kick_oob_exp19/git/mjlab.diff`。

## 产物(机器人视觉 POV + 自定位信念演示)
由 `scripts/eval_v4_pov_belief_exp19.py`(共用 `scripts/_v4_pov_belief_video.py`)生成,
完整一个 episode 的三联画(31s,30fps):
- `kick_oob_exp19_pov_belief.mp4` — 左:机器人头部相机 POV(`head_cam_rgb`,策略 CNN 实际输入);
  中:俯视球场 GT 位姿(绿)vs 策略融合信念(红 x,即策略消费的 `_last_xy_n`)+ 球(白)+ 目标(黄星),带拖尾;
  右:定位误差(米)与融合视野覆盖率(uniq_frac)随时间曲线。
- `pov_belief.json` — episode 统计(是否进球、首/末/均定位误差、起始/最大覆盖率)。
- **本段叙事**:start_coverage 0.13 → max_coverage 0.35,均定位误差 0.30m(信念全程贴 GT,定位稳)。
  本段为单 env0 最深带球(12 次尝试未进球,如实标 `scored: false`);进球片段见三段 dribble-arc 的 `*_goalward.mp4`。

## 核心确证(末100稳态,四轮同法对照)
- 真实射门率 **0.325**(EXP16-19 最高);out_of_bounds **0.294**(< EXP16 0.345)。
- 在"进球 + 不出界"两核心维度 Pareto 优于 EXP16;综合分 0.681 ≈ EXP16 0.686(噪声内)。

## EXP16-19 排名
EXP16 (0.686) ≈ **EXP19 (0.681)** > EXP18 (0.409) > EXP17 (0.380)。
EXP17 OOB 0.256 最低,是 time_out 修复机制验证最佳(非部署最佳)。

## 方法学局限
沿用本目录 README 的评估方法学:episode 级指标绝对值以训练日志为准,逐步量(pos_err/ball_speed)可信;
fell_over 启动 transient 已排除;out_of_bounds 0.294 是 time_out 修复后的结构地板。
视频为单 env0 定性演示(非打分用)。详见 `soccer_robot_v4_experiment.md`「EXP16-19 多指标综合评价」+ `v4评估指标详解.md §6`。
