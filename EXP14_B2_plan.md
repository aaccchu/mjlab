# EXP14 — 线B B2:近脚窗口踢球动作技能训练

## 目标
把"球在脚边→踢进门"的动作技能单独练出来。codex 上界 0.68 vs 实际 0.018/0.08——
缺的是动作技能,不是感知(oracle 阶段球位完美)、不是奖励主轴(goal_progress 已是球速朝门投影)。

## 根因(已验证)
1. `spawn_dist_range=(0.6,1.5)`:球永远 ≥0.6m 生成,**策略几乎从不在"球在脚边"状态开始学习**,
   踢击窗口(x≈0.08-0.15m)样本极稀疏。
2. `dribble_kick_contact` 弱:只奖励"有无脚-球接触"(布尔 found>0),不奖励踢击质量(传给球的冲量/Δv)。
   轻碰和爆发踢一样分,学不出"用力踢向门"。

## 改动(3 处,均最小化、可回退)

### 1. 近脚 spawn 模式(dribble_command.py)
- 新 cfg 参数 `near_foot_spawn_fraction: float = 0.0`(默认 0,不影响现有 env)、
  `near_foot_dist_range=(0.08,0.20)`、`near_foot_half_angle`(机器人正前方扇区)。
- 在 `_resample_command` 的 spawn 逻辑里(line 198-222),类比现有 `rear_spawn_fraction`,
  对一部分 episode 把球放在**机器人正前方踢击窗口**(heading 方向、距离 0.08-0.20m)。
- 球初速保持 0(已有 `ball_init_speed_range`,默认 0),让策略练"从静止把球踢出"。

### 2. 冲量踢击奖励(rewards.py)
- 新函数 `dribble_kick_impulse`:检测脚-球接触事件那一步,奖励**球速增量 Δv_ball 朝球门方向的投影**
  (clamp ≥0)。复用 `foot_ball_contact` 传感器 + `command.ball_lin_vel_w`。
- 比现有 binary kick_contact 更教"在正确时刻用力踢向门"。保留旧 kick_contact(低权重)。

### 3. 新 env builder + spike(env_cfgs.py + scripts/)
- `mos92_soccer_e2e_dualcam_ekf_kick_env_cfg`:派生 EKF env(继承线A 成果),
  设 `near_foot_spawn_fraction≈0.5`(一半 episode 练踢击,一半保持完整任务防遗忘),
  加 `dribble_kick_impulse` 奖励(weight 待调,初始 ~1.5)。
- `scripts/spike_v4_e2e_ekf_kick.py`:从 EXP13 model_1999 bootstrap(继承定位+步态+接近)。

## 评估(多指标,scripts/readout_v4_metrics.py)
- 主判据:goal_rate 能否从 0.08 显著上升(目标逼近 0.2),episode_success 上升。
- 球动学:ball_speed↑、ball_to_target_err↓、ball_stuck_s↓。
- 护栏:fell_over 保持低(<0.1)、pos_err 不退化(线A 不能被踢球训练破坏)。
- 对照:EXP13(goal 0.08,success 0.40,pos_err 0.98)。

## 风险与缓解
- **技能干扰**(codex C8 射门退化、EXP10 破稳):用"一半 episode 完整任务"防遗忘;
  若仍退化,改用更小 near_foot_fraction 或后续 B3 蒸馏。
- **奖励黑客**(轻碰刷接触):用冲量奖励(需真 Δv)+ 保留 ball_trapped/sticking 重罚。
- 全程 oracle 检测(隔离变量)。bootstrap 同 88 维,权重全载。

## 不做(范围控制)
- 不做 B1 ball belief(oracle 阶段球位完美,推迟到真检测器阶段)。
- 不做踢腿参考轨迹(先试 spawn+impulse 的纯 RL,不行再加 imitation)。
- 不动线A EKF 代码。
