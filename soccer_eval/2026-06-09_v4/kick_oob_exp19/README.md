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
完整一个 episode 的**四联画**(33s,30fps):
- `kick_oob_exp19_pov_belief.mp4` —
  1. **RGB POV**(`head_cam_rgb`,自定位 CNN 实际输入);
  2. **深度 POV**(`head_cam` 深度,球感知 CNN 输入,turbo 伪彩、近处亮);
  3. **俯视球场**:机器人 GT 位姿(绿)vs EKF 信念(红 x,即策略消费的 `_mu`),**真实球(白)vs 策略相信的球位置(橙菱形,连线示偏差)**+ 目标(黄星),带拖尾;
  4. **曲线**:定位误差(米)+ 融合视野覆盖率(青)+ 当前帧可见率(浅蓝)随时间。
- `pov_belief.json` — episode 统计(是否进球、首/末/均定位误差、**相信的球位置平均误差**、起始/最大覆盖率)。
- **关键说明**:开局必然"盲视"——reset 时 EKF 处于宽先验、相机看不到任何关键点(`start_visible_now=0`、`start_coverage 0.04`),
  视频**第一帧没有球门、没有关键点**,定位误差高(本段起始 **~6.2m**);随后转身扫视、EKF 收敛、误差下降。
- **本段叙事**(`scored: false`):起始定位误差 ~6.2m → 收敛,球带到球门附近;
  本段单 env0 未进球(EXP19 控球率低于 EXP16,单环境进球偶发,如实标注);进球片段见三段 dribble-arc 的 `*_goalward.mp4`。
- **诚实标注**:橙点偏差 = **自定位误差传播到球上**(球的自我朝向向量用的是 GT;球感知隐含在 CNN 里,
  没有可直接读取的"球坐标估计"输出)。即此偏差反映"我以为我在哪"的误差,不是独立的球感知误差。

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
