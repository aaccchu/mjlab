# EXP20b 评估留痕 — aim-the-kick(gaze pitch 符号修复 + 放开低头 + 横向对齐)

**当前最佳部署模型(v4 踢球链)。** 源权重:`checkpoints/v4_soccer/kick_aim_exp20b/model_1999.pt`
(= `logs/rsl_rl/mos92_velocity/2026-06-10_20-22-43_spike_v4_e2e_ekf_kick_aim2/model_1999.pt`)。
全指标 + 与 EXP16-19 对照见 `metrics.json`。

## 核心数字(训练 log 稳态,多窗口)
- **真实射门率 0.38-0.41**(末400 0.387 / 末50 0.410 / 中段峰 0.416)——较 EXP16-19 平台 0.32-0.33
  **+20%,v4 单次最大增益**。
- 同时 5 项历史最好:ball_to_tgt_err **1.88**、robot_to_ball **0.60**、pos_err **0.85-0.88**、
  episode_success **0.58**、out_of_bounds **0.20**(此前的 0.26"结构地板"被瞄准改善打破)。
- 护栏:fell_over 0.063 ✅;**ball_speed_peak 3.2→2.70 缓降**(守住红线但需盯,见局限)。

## 改了什么(EXP20 三件套,详见实验 log"EXP20 调研与瓶颈诊断")
1. **全局 bug 修复**:`_gaze_uv_visible` 垂直符号 `elev−neck_pitch → elev+neck_pitch`。
   实测渲染验证正 neck_pitch=低头;旧公式教策略对近球**抬头**(codex EXP7C3 同款 bug)。
2. **放开 neck_pitch**(pose std 0.1/0.15→1.0):让修对符号的 gaze 奖励真能把头压下去。
3. **kick_lateral_alignment**(w=1.0):球在身前 x_b∈[0.15,1.0]m 时奖励 |y_b|→0,摆正再踢
   (codex C21l2 实测 13× 杠杆)。

## 产物
- 三段 dribble-arc 演示:`*_approach/strike/goalward.mp4` + 中点静帧 + `scenarios.json`
  (本次 rollout 含 1 真实进球;goalward 段窗口未覆盖进球帧,`scored:false` 如实标注)。
- 8 联画 POV+信念视频:`kick_aim_exp20b_pov_belief.mp4`(33s,**scored:true 真实进球**)+
  `pov_belief.json`。**本模型亮点**:视频中出现 **"ball IN VIEW (0.7 m)"** 帧——低头修复让球在
  0.7m 仍可见,**突破了约束 EXP16/19 的 ~1.25m 脚下盲区线**;RGB POV 里球占满画面下方,
  深度 POV 球为清晰近处亮团。first_ball_sighting step111、18 次踢球接触、累计里程 7.5m。
- 复现:`MUJOCO_GL=egl uv run python scripts/eval_v4_kick_aim_exp20b.py`(三段)/
  `scripts/eval_v4_pov_belief_exp20b.py`(8 联画)。
- 训练时 git 版本信息:`checkpoints/v4_soccer/kick_aim_exp20b/git/mjlab.diff`。
- 面板逐指标详解见 `../../2026-06-09_v4/kick_exp16/kick_exp16_pov_belief.md`(定义相同)。

## 方法学局限(诚实记录)
- 数字取自训练 stdout 稳态窗口(与 EXP16-19 同法,趋势可信),非全新 headless 评估。
- **ball_speed_peak 缓降(3.2→2.7)**:对齐奖励可能轻微 trade 踢力;后续训练若破 2.5 应停,
  转踢腿参考轨迹或提 kick_impulse 权重。
- 仍 oracle 球检测(B1 真检测器未动);0.5-0.6 终极目标的物理可达性(20s episode)未跑上界探针校准。
