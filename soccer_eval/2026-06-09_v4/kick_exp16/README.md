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
完整一个 episode 的**四联画**(33s,30fps):
- `kick_exp16_pov_belief.mp4` —
  1. **RGB POV**(`head_cam_rgb`,自定位 CNN 实际输入);标注"ball IN VIEW / BLIND SPOT (距离)";
  2. **深度 POV**(`head_cam` 深度,球感知 CNN 输入,**裁剪到 3m 再归一化**——与策略 `camera_depth` 输入一致,近处亮);
  3. **俯视球场**:机器人 GT 位姿(绿)vs EKF 信念(红 x,即策略消费的 `_mu`),**真实球(白)vs 策略相信的球位置(橙菱形,连线示偏差)**+ 目标(黄星),带拖尾;
  4. **曲线**:定位误差(米)+ 融合视野覆盖率(青)+ 当前帧可见率(浅蓝)随时间。
- `pov_belief.json` — episode 统计(是否进球、首/末/均定位误差、**相信的球位置平均误差**、起始/最大覆盖率、**ball_in_view_frac**)。
- **⚠️ 球的视野盲区(2026-06-10 代码审查确认,关键认知)**:头部相机**高 0.79m、俯角≈0°(平视)、fovy 60°**,
  地面球(z=0.07)**仅在距离 >~1.25m 时进入视野**;带球/踢球(球<1.25m)球落入脚下盲区,RGB+深度都看不到球。
  本段 `ball_in_view_frac=0.20`——**大部分带球时间球在盲区**。所以触球时策略**不靠当前帧看球**,而是:
  ①接近时(1–3m)看到球建立位置 ②之后靠 EKF 自定位 + RGB 6 帧时序记忆推算球相对位置。这是设计约束非 bug
  (`mos92_foot_zone_blind_spot`)。视频里 ball IN VIEW 的帧(接近阶段)能在 RGB 看到橙球、深度看到近处亮 blob。
- **关键说明**:开局必然"盲视"——episode reset 时 EKF 处于宽先验、相机看不到任何关键点(`start_visible_now=0`),
  所以视频**第一帧画面里没有球门、没有关键点**,定位误差高;随后机器人转身扫视球场线/球门,EKF 收敛、误差下降。
- **本段叙事**(`scored: true`):起始定位误差高 → 收敛 → 带球射门**进球**;
  橙菱形(相信的球)逐渐贴合白点(真实球),正是定位变准的体现。
- **诚实标注**:橙点偏差 = **自定位误差传播到球上**(球的自我朝向向量用的是 GT;球感知隐含在 CNN 里,
  没有可直接读取的"球坐标估计"输出)。即此偏差反映"我以为我在哪"的误差,不是独立的球感知误差。

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
