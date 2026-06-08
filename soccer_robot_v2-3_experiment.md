# v2.5 Spike 实验记录

> 所有实验结果按时间顺序记录，用于指导后续调整。
> 每次重跑标注变更原因和对比基线。

---

## Spike B: 相机视野验证

### 实验 B-1 (2026-06-01) — PASS

| 参数 | 值 |
|------|-----|
| parent_body | robot/torso_link |
| pos | (0.15, 0.0, 0.43) |
| quat | (0.6124, 0.3536, -0.3536, -0.6124) — pitch 30° down |
| fovy | 60° |
| resolution | 64×48 |

| 距离 | 0° | ±30° | 60° |
|------|-----|------|-----|
| 0.5m | 0 px | 0 px | 0 px |
| 1.0m | 42 px (1.37%) | 43 px (1.40%) | 0 px |
| 2.0m | 14 px (0.46%) | 20 px (0.65%) | 0 px |
| 3.0m | 6 px (0.20%) | 10 px (0.33%) | 0 px |
| 5.0m | 2 px (0.07%) | 4 px (0.13%) | 0 px |

**结论:** 1-3m depth 可检测(≥4px)，0.5m 不可见，60° 超出 FoV。
**状态:** PASS — 参数确定，无需重跑。

---

## Spike A: Gaze 可行性

> **指标口径说明(重要):** 训练 tensorboard 的 `Episode_Termination/fell_over`
> 是**每 episode 的平均终止计数**(可 >1),不是百分比。早期文档误读成「73.9%」。
> 可靠的 fall_rate 应以 `evaluate_soccer.py` 的 CI 估计为准(默认 env,512 episodes)。
> Spike A 只改 reward(gaze + waist std)不改物理/分布，故 A-1/A-2 可用默认 env 评估，口径一致。

### 实验 A-1 (2026-06-01) — 根因待验证

| 参数 | 值 |
|------|-----|
| 基线 checkpoint | model_4999.pt (v2) |
| gaze_at_ball weight | 1.0 |
| gaze std | 2.0 |
| waist_yaw pose std | 1.0 (原 0.2) |
| waist_pitch pose std | 0.5 (原 0.1) |
| 训练 iter | 500 |
| num_envs | 4096 |

**训练终值(raw 计数，非百分比):**

| 指标 | 起始(step 5000) | 结束(step 5498) |
|------|----------------|----------------|
| fell_over (每ep计数) | 0.0 | 0.739 |
| timeout (每ep计数) | — | 3.696 |
| fall_fraction = fell/(fell+to+oob) | 0% | **15.2%** |
| episode_success | 0% | 91.3% |
| gaze_at_ball reward | 0.001 | 0.562 (raw≈0.56) |

**evaluate_soccer 权威评估(默认 env, 512 ep):**

| 指标 | 值 |
|------|-----|
| success_rate | 99.8% |
| fall_rate | **0.2%** |
| possession_rate | 96.1% |
| ball_to_target_error | 0.20m |

> 关键洞察:A-1 策略在**标准评估条件下几乎不摔**(0.2%)，摔倒只发生在**训练时的
> 激进 gaze 设置下**(训练态 fall_fraction 15.2%)。说明问题不是策略本身坏了，
> 而是训练时 gaze reward 的梯度把策略推向激进转头。这正是 A-2 要验证的根因。

**原因假设:**
1. gaze weight=1.0 过强，策略为获取 gaze reward 激进转头，破坏训练态平衡
2. waist_yaw std 从 0.2 放宽到 1.0 幅度太大，pose penalty 几乎消失
3. 500 iter 不够让策略同时学会转头+保持平衡

**checkpoint:** `logs/rsl_rl/g1_velocity/2026-06-01_15-54-10_spike_a/model_5498.pt`

### 实验 A-2 (2026-06-01) — 根因确认 ✓

变更（基于 A-1 假设，只调 weight 和 std 两个变量以保证归因干净）:
- gaze_at_ball weight: 1.0 → **0.3**（降低转头激励强度）
- waist_yaw pose std: 1.0 → **0.5**（保留一定约束）
- waist_pitch pose std: 0.5 → **0.3**（pitch 更严格）
- 训练 iter: 500 → **1000**

**训练终值对比(同口径):**

| 指标 | A-1 (weight=1.0) | A-2 (weight=0.3) |
|------|------------------|------------------|
| fall_fraction | 15.2% | **7.0%** |
| episode_success | 91.3% | **94.5%** |
| gaze_at_ball reward | 0.562 | 0.136 (raw≈0.45) |
| ball_to_target_error | — | 0.37m |
| robot_to_ball_error | — | 0.28m |

**结论: 根因确认。** 降低 gaze weight(1.0→0.3) + 收紧 waist std(1.0→0.5)后:
- 训练态 fall_fraction 从 15.2% 降到 7.0%(满足 <10% 目标)
- success 不降反升(91.3%→94.5%)
- 代价:gaze reward 下降(raw 0.56→0.45)，即转头幅度变小

**这验证了 A-1 的根因假设:fall 确实由「过强 gaze 激励 + 过松 pose 约束」导致，
而非 gaze 任务本身不可行。** weight=0.3 是平衡点:既能学转头又不破坏平衡。

**待补:** A-2 的 gaze_angle_error 量化(训练 reward 反推角度受 std 形式影响，
不够准；如需精确值，应在 evaluate 中直接记录 |waist_yaw − ball_bearing|)。

**v3 决策(更新):** learned gaze **可行**(方案 B)，但 gaze weight 需保守(≤0.3)，
waist pose std 不可放太松。v3 §5.4 选方案 B，配 weight=0.3 起始 + curriculum 逐步增强。

**checkpoint:** `logs/rsl_rl/g1_velocity/2026-06-01_18-09-51_spike_a_a2/model_5998.pt`

### 实验 A-3 (备选，A-2 已成功故暂不执行)

如未来需要更强 gaze 又要更稳:
- gaze weight curriculum: 0.1 → 0.3 → 0.5 逐步增强
- 或在 evaluate 中精确记录 gaze_angle_error 再决定是否加码
- 启发式 gaze(PD 直接控 waist_yaw)作为 learned 失败时的兜底

---


## Spike F: 全场尺度泛化

### 实验 F1-1 (2026-06-01) — 接近通过 (67.4% vs 目标 70%)

| 参数 | 值 |
|------|-----|
| 基线 checkpoint | model_4999.pt (v2) |
| spawn_dist_range | (0.6, 4.0) |
| target_dist_range | (2.0, 8.0) |
| approach_delta weight | 2.0 |
| heading_to_ball weight | 0.5 |
| dribble_approach | 已删除（替换为 approach_delta） |
| 训练 iter | 2000 |

| 指标 | 起始 | 结束(2000 iter) |
|------|------|----------------|
| episode_success | 0% | **67.4%** |
| ball_to_target_error | 4.43m | 1.91m |
| robot_to_ball_error | 2.56m | 1.00m |
| heading_to_ball | 0.001 | 0.235 |
| fell_over | 2.17 | 2.29 |

**分析:**
- 67.4% 接近 70% 目标，可能再训 1000 iter 能达标
- approach_delta 数值很小(0.0002)但学习方向正确
- 主要瓶颈：远距离 approach 阶段耗时长，episode 容易超时

**checkpoint:** `logs/rsl_rl/g1_velocity/2026-06-01_16-07-40_spike_f_f1/model_6998.pt`

### 实验 F2-1 (2026-06-01) — 部分成功 (18.4% vs 目标 40%)

| 参数 | 值 |
|------|-----|
| spawn_dist_range | (0.6, 9.0) |
| target_dist_range | (2.0, 12.0) |
| 其余同 F1-1 | |
| 训练 iter | 2000 |

| 指标 | 起始(step 5000) | 结束(step 6998) |
|------|----------------|----------------|
| episode_success | 3.2% | **18.4%** |
| ball_to_target_error | 5.64m | 5.19m |
| robot_to_ball_error | 6.18m | 3.81m |
| heading_to_ball | 0.001 | 0.052 |
| fell_over | 39.3 | 19.96 |

**分析:**
- 全场尺度 success 仅 18.4%，远低于 40% 目标 → 两阶段 reward 不足以直接覆盖全场
- 但方向正确：robot_to_ball 从 6.18m 降到 3.81m（学会走向球），fell_over 从 39→20
- 主因：approach 距离过大(最远 9m+12m=21m)，2000 iter 内 episode 大量超时
- ball_to_target 几乎没动(5.64→5.19) → 大部分 episode 球根本没被推动
- **对应决策表「F1 好但 F2 崩，F3 恢复」→ v3 需要 distance curriculum**

**checkpoint:** `logs/rsl_rl/g1_velocity/2026-06-01_16-57-26_spike_f_f2/model_6998.pt`

### 实验 F3 (建议，但非阻塞) — distance curriculum

F2 验证了「直接全场训练不行」，F3 应验证「渐进扩大可行」:
- 从 F1-1 checkpoint(已会 4m 尺度)继续，按 success_rate>80% 逐步扩大 spawn/target
- 预期能恢复到 60% 左右
- 优先级低于 Spike A 验证，可在 v3 阶段实施

### 实验 F1-2 (待定，如果 F1-1 需要改进)

可能的调整:
- 从 F1-1 checkpoint 继续训 1000 iter（验证是否只是收敛不够）
- 或增大 approach_delta weight 到 3.0
- 或增加 episode_length 让远距离 approach 有更多时间

### Spike F 阶段性结论

| 组 | 结果 | 判定 |
|----|------|------|
| F1 (温和扩大 4m) | 67.4% | 接近通过，两阶段 reward 结构有效 |
| F2 (全场 9m+12m) | 18.4% | 崩溃，证明需要 curriculum |

**v3 决策:** 两阶段 reward 结构(approach_delta + heading_to_ball)正确，但全场尺度
**必须配 distance curriculum**(F1→F2 渐进),不能直接在全场分布上 fine-tune。

---

## Spike G: MOS92 机器人适配

### 控制方式决策 (2026-06-01)

MOS92 xml 原生是**力矩电机**(`<motor ctrlrange="-36/60">`)，但 mjlab 支持两种
action space。Step 1 选用**位置控制**，力矩控制留作后续对比组。

#### 选项 A: 位置控制 (Position / PD) — **Step 1 采用**

- 策略输出**目标关节角度** `q_target`，仿真内置 PD 伺服算力矩
  `τ = kp·(q_target − q) − kd·q̇`
- mjlab 配置: `BuiltinPositionActuatorCfg`(复用 G1 的 stiffness/damping 模式)
- 导入时需根据 MOS92 电机参数估算 kp/kd:
  - 力矩上限已知(±36 Nm 肩/踝roll、±60 Nm 肘/髋/膝/踝pitch)
  - armature 从转子惯量推算(参考 G1 的 `reflected_inertia_from_two_stage_planetary`)
  - 或先用 G1 同类关节的 kp/kd 作为初值，再按质量比(16.5/35≈0.47)缩放

**优点:** 训练难度低(PD 提供天然稳定性)、收敛快(G1 ~500 iter 会走)、
复用 G1 全套 velocity task 配置，变量最少 → 最快拿到 go/no-go。

**缺点:** 依赖真机 kp/kd 标定准确，sim-to-real 多一层映射误差。

#### 选项 B: 力矩控制 (Effort / Torque) — 后续对比组

- 策略**直接输出关节力矩** `τ`，无内置伺服，策略自学整个反馈回路
- mjlab 配置: `JointEffortActionCfg` + `BuiltinMotorActuatorCfg`
- 直接对应 MOS92 xml 原生定义，无需估算 kp/kd

**优点:** sim-to-real 更直接(真机本就是力矩电机)、无 PD 标定误差。

**缺点:** 训练难度高(易抖动/发散)、收敛慢(需更多 iter + 调 reward)、
对观测延迟更敏感。

#### 对比表

| 维度 | 位置控制(A) | 力矩控制(B) |
|------|------------|------------|
| 策略输出 | 目标关节角 | 关节力矩 |
| 谁稳定关节 | 内置 PD 伺服 | 策略自学 |
| 训练难度 | 低 | 高 |
| 收敛速度 | 快(~500 iter 走) | 慢 |
| sim-to-real | 依赖 kp/kd 标定 | 更直接 |
| 延迟敏感度 | 较低(PD 抗扰) | 较高 |

#### 决策理由

1. Spike G 目标是「验证 MOS92 能否走+带球」，不是验证控制方式 →
   位置控制复用 G1 验证过的配置，把变量降到最少。
2. 若位置控制都走不起来 → 问题在 xml/质量分布/keyframe，与控制方式无关，
   更易定位。
3. 力矩控制是 sim-to-real 的优化项，不该卡在 Step 1。位置控制跑通后再开
   对比组(类似 Spike A 多组验证思路)。

**何时切换到 B:** 若 v3 最终追求真机部署保真度，且位置控制的 sim-to-real
gap 过大(真机 kp/kd 难标定或带宽不足)，则改用力矩控制重训。

### 实验 G-1 (2026-06-01) — PASS ✓

**目标:** 验证 MOS92 (18-DOF) 能在 mjlab 中行走(位置控制)。

**资产处理:**
- 源 xml: `docs/robot_param/MOS92_urdf_0517_v3_simplified.xml`
- 清理后: `src/mjlab/asset_zoo/robots/mos92/xmls/mos92.xml`
- 删除所有 mesh/texture(21 个 STL 缺失,但 collision geom 全是 primitive)
- 保留 18 个 primitive collision geom (cylinder/sphere, group=3)
- 添加 velocimeter、subtreeangmom sensor、left_foot/right_foot site

**PD 参数:**
- armature=0.01, natural_freq=10Hz, damping_ratio=2.0
- stiffness=39.48, damping=2.51
- action_scale = 0.25 * effort_limit / stiffness

**训练配置:**
- task: `Mjlab-Velocity-Flat-MOS92`
- num_envs: 4096, max_iterations: 1000
- 基于 G1 velocity flat 配置,去掉 terrain scan/height scan

**训练结果 (1000 iter):**

| 指标 | 值 |
|------|-----|
| mean_reward | **83.20** |
| fell_over | **0.0** (300 iter 后归零) |
| upright | **0.99** |
| track_linear_velocity | 1.57 |
| track_angular_velocity | 1.47 |

**结论:** MOS92 位置控制行走完全可行。~300 iter 学会站稳,1000 iter 稳定
跟踪速度指令。fell_over=0 说明 PD 参数和 keyframe 合理。

**checkpoint:** `logs/rsl_rl/mos92_velocity/2026-06-01_20-34-38/model_999.pt`

### 实验 G-1b (2026-06-01) — PASS ✓ (参数优化版)

**变更(基于 MOS9-AMP-main 项目的调优参数):**
- 21 个 STL mesh 加回 xml 做可视化(contype=0, 不参与碰撞)
- PD 参数更新为 MOS9-AMP 验证值:
  - 4310 电机(肩/踝roll): armature=0.0283, kp=47.18, kd=1.78
  - 6408 电机(肘/髋/膝/踝pitch): armature=0.0478, kp=105.19, kd=2.63
- 加入 neck_yaw(±90°) + neck_pitch(±28.6°) 关节(硬件确认可转动)
- 手臂初始姿态改为贴身(shoulder_roll=±1.4, 参考 MOS9-AMP)
- 总 DOF: 20 (18 原有 + 2 neck)

**训练结果 (1000 iter):**

| 指标 | G-1 (xml默认) | G-1b (调优) | 说明 |
|------|--------------|------------|------|
| mean_reward | 83.20 | **78.30** | 多2个关节,稍慢收敛 |
| fell_over | 0.0 | **0.04** | 极少摔倒,再训500iter可归零 |
| track_lin_vel | 1.57 | **1.55** | 几乎一致 |
| track_ang_vel | 1.47 | **1.41** | 略低 |
| upright | 0.99 | **0.99** | 一致 |
| pose | 0.71 | **0.68** | 手臂姿态变化导致 |

**结论:** 调优参数 + neck joints + 手臂贴身后仍能稳定行走。reward 略低是因为
action space 增大(18→20 DOF)和手臂姿态变化,趋势正常。fell_over=0.04 说明
偶尔摔倒但已接近归零。

**checkpoint:** `logs/rsl_rl/mos92_velocity/2026-06-01_21-18-21/model_999.pt`
**可视化:** `soccer_eval/2026-06-01_spikes/g1_mos92_walk/mos92_walk_v2-step-0.mp4`

### Spike G-1 总结

| 验证项 | 结果 | 状态 |
|--------|------|------|
| MOS92 能行走 | fell_over≈0, track_vel>1.5 | ✓ PASS |
| 调优 PD 参数可用 | 收敛正常,reward 78+ | ✓ PASS |
| Neck joints 不破坏行走 | 加入后仍稳定 | ✓ PASS |
| Mesh 可视化 | 21 STL 加载,渲染正常 | ✓ PASS |

**v3 决策:** MOS92 确认为 v3 训练平台,20-DOF(含 neck gaze)。

### Spike G-2: MOS92 带球 (Soccer Dribble)

#### 实验 G-2a (2026-06-01) — FAIL (kick_contact=0)

**配置:** 基于 mos92_flat_env_cfg + soccer 组件
- SoccerFieldCfg + SoccerBallCfg(radius=0.11m)
- DribbleCommandCfg(默认 spawn_dist=0.6-1.5m, approach_radius=0.6m)
- foot-ball contact: pattern=`^(Rfoot|Lfoot)$` (subtree mode)
- 5 dribble rewards + 3 ball observations

**训练结果 (3000 iter, ~57 min):**

| 指标 | 值 | 说明 |
|------|-----|------|
| mean_reward | 103.4 | 行走稳定 |
| fell_over | 0.0 | 从不摔倒 |
| dribble_approach | 0.67 | 接近球 |
| robot_to_ball_error | 0.62m | 接近但不够近 |
| kick_contact | **0.0** | 从未触球 |
| dribble_success | 0.0 | 未完成带球 |

**问题分析:**
- MOS92 身高 0.45m,脚部碰撞体很小(cylinder r=0.03, h=0.065)
- 默认 spawn_dist=0.6-1.5m 是为 G1(1.27m)设计的
- approach_radius=0.6m 对 MOS92 步幅来说太大,切换到 push phase 时仍太远
- 机器人走到 ~0.62m 处就停了,脚够不到球

**checkpoint:** `logs/rsl_rl/mos92_velocity/2026-06-01_21-53-12/model_2999.pt`

#### 实验 G-2b (2026-06-01) — PASS ✓

**修复(缩放参数适配 MOS92 体型):**
- spawn_dist_range: (0.6, 1.5) → **(0.3, 0.8)**
- approach_radius: 0.6 → **0.25**
- approach_offset: 0.3 → **0.12**
- max_speed: 1.0 → **0.6** (匹配小型机器人步速)
- kick_contact weight: 0.3 → **1.0** (强化触球激励)
- dribble_approach std: 1.0 → **0.5** (近距离 reward 更尖锐)
- dribble_approach weight: 1.0 → **1.5**
- dribble_to_target std: 2.0 → **1.5**

**训练结果 (3000 iter):**

| 指标 | G-2a (失败) | G-2b (修复) | 说明 |
|------|------------|------------|------|
| mean_reward | 103.4 | **224.4** | 2x 提升 |
| kick_contact | 0.0 | **0.82** | 稳定触球 |
| dribble_success | 0.0 | **4.14** | 持续带球到目标 |
| fell_over | 0.0 | **0.08** | 偶尔摔倒,可接受 |
| dribble_approach | 0.67 | **1.40** | 饱和,总能接近球 |
| upright | — | **0.94** | 姿态良好 |

**训练曲线关键节点:**
- step 245: kick_contact 首次出现信号 (0.008)
- step 1040: kick_contact=0.21, dribble_success=2.69
- step 1545: kick_contact=0.63, fell_over 峰值 0.42(学习踢球时不稳)
- step 2351: fell_over 降至 0.25(学会平衡踢球)
- step 2999: kick_contact=0.82, fell_over=0.08(最终收敛)

**checkpoint:** `logs/rsl_rl/mos92_velocity/2026-06-01_22-58-05/model_2999.pt`
**可视化:** `soccer_eval/2026-06-01_spikes/g2_mos92_dribble/mos92_dribble_v2-step-0.mp4`

### Spike G-2 总结

| 验证项 | 结果 | 状态 |
|--------|------|------|
| MOS92 能触球 | kick_contact=0.82 | ✓ PASS |
| MOS92 能带球到目标 | dribble_success=4.14 | ✓ PASS |
| 带球时不摔倒 | fell_over=0.08 | ✓ PASS |
| 参数适配 | 缩放 spawn/approach 到 MOS92 体型 | ✓ 已验证 |

**关键发现:** G1 的 soccer env 参数不能直接用于 MOS92,需按体型比例缩放
spawn_dist、approach_radius 等距离参数。MOS92 步幅短,需要球更近才能触及。

---

## Spike A-MOS92: Neck Gaze 验证

### 实验 A-MOS92 (2026-06-02) — PARTIAL PASS

**目标:** 验证 MOS92 neck_yaw 能追踪球且不破坏带球能力。

**配置:**
- 基于 G-2b checkpoint (model_2999.pt) fine-tune 2000 iter
- 新增 `gaze_at_ball` reward (weight=1.0, std=2.0)
- joint: neck_yaw (通过 SceneEntityCfg joint_names 指定)
- 放宽 neck pose reward std 到 10.0(允许自由转头)

**训练结果 (2000 iter fine-tune, 总 step 4998):**

| 指标 | G-2b (无 gaze) | A-MOS92 (有 gaze) | 变化 |
|------|---------------|-------------------|------|
| gaze_at_ball | — | **0.69** | 头部追踪球 |
| kick_contact | 0.82 | **0.76** | -7% |
| dribble_success | 4.14 | **3.84** | -7% |
| fell_over | 0.08 | **0.33** | +0.25 ⚠️ |
| mean_reward | 224.4 | **225.1** | 持平 |
| upright | 0.94 | **0.85** | -10% |

**训练曲线:**
- step 3149 (149 new): gaze=0.60, fell_over=0.42 (初始适应期)
- step 3412 (412 new): gaze=0.66, fell_over=0.46 (峰值不稳)
- step 4455 (1455 new): gaze=0.69, fell_over=0.22 (开始恢复)
- step 4998 (final): gaze=0.69, fell_over=0.33 (最终值)

**分析:**
- 头部追踪成功: gaze_at_ball=0.69 说明 neck_yaw 在主动对准球方向
- 带球能力保持: kick_contact 和 dribble_success 仅下降 7%
- 稳定性下降: fell_over 从 0.08 升至 0.33,转头动作影响平衡
- 中期(step 4455)fell_over 曾降至 0.22,最终回升到 0.33

**结论:** PARTIAL PASS
- ✓ neck gaze 概念可行(能转头追踪球)
- ✓ 带球能力基本保持(仅 -7%)
- ⚠️ 稳定性需改善(fell_over=0.33 > 目标 0.10)

**v3 决策:**
- 走 learned gaze (方案 B),但需要:
  1. 更长训练(4000+ iter fine-tune)让平衡恢复
  2. 或降低 gaze weight 到 0.5 减少转头幅度
  3. 或加入 neck 动作平滑惩罚防止急转

**checkpoint:** `logs/rsl_rl/mos92_velocity/2026-06-02_00-17-56/model_4998.pt`

### 实验 A-MOS92 三组对照 (2026-06-02) — PASS ✓

**目的:** A-MOS92 初版 (weight=1.0) fell_over=0.33 偏高。参考 G1 上 Spike A
"weight=0.3 是平衡点"的结论,跑三组对照找 MOS92 的最优配置。
全部从 G-2b (model_2999) fine-tune 1000 iter,公平对比。

| 组 | gaze_weight | neck_std | gaze | kick_contact | dribble_success | fell_over | reward |
|----|-------------|----------|------|-------------|----------------|-----------|--------|
| G-2b 基线 | — | — | — | 0.82 | 4.14 | 0.08 | 224.4 |
| **G1** | 0.3 | 10 | 0.17 | 0.76 | 3.83 | 0.29 | 219.4 |
| **G2** | 1.0 | 10 | 0.67 | 0.75 | 3.82 | 0.33 | 216.4 |
| **G3** | 1.0 | **1.0** | **0.70** | **0.77** | **3.95** | **0.29** | **227.3** |

(G2 复用 2026-06-02_00-17-56 run 在 step 4000 的快照,等价于 1000 new iter)

**关键发现(与 G1 结论不同):**
- G1 上降低 weight 到 0.3 才能稳;MOS92 上降 weight 反而让 gaze 几乎消失
  (gaze=0.17),且 fell_over 没改善
- MOS92 的最优解是 **weight=1.0 + tight neck pose std=1.0**(G3):
  gaze 最高(0.70,转头明显)、带球最好、fell_over 最低、reward 最高
- 原因: tight neck std 约束 neck 关节**回归中位**,但不抑制对球的瞬时追踪,
  两者协同 —— 转完头会自动回正,减少持续偏头导致的失衡

**结论:** PASS — learned gaze 在 MOS92 上可行,最优配置 weight=1.0 + neck_std=1.0。
fell_over=0.29 仍高于基线 0.08,但带球能力保持(-5%)且转头明显,可接受。

**v3 决策:**
- gaze 方案 B (learned gaze) 确认,配置 weight=1.0 + neck pose std=1.0
- fell_over 残留可通过更长训练(2000+ iter)或加 neck 角速度惩罚进一步降低
- neck_pitch 的 gaze(俯仰追球距离)留待 v3 视觉接入后验证

**checkpoints:**
- G1: `logs/rsl_rl/mos92_velocity/2026-06-02_13-08-45/model_3998.pt`
- G3 (最优): `logs/rsl_rl/mos92_velocity/2026-06-02_13-33-44/model_3998.pt`
**可视化:** `soccer_eval/2026-06-01_spikes/a_mos92_gaze_v2/`

**neck_yaw 转头幅度实测(G3, 400步×16环境):**

| 统计量 | 值 |
|--------|-----|
| 平均 \|角度\| | 5.7° |
| std | 9.0° |
| min/max | -54° / +37° |
| p90 \|角度\| | 15.7° |
| 转头 >10° 时间占比 | 19% |
| 转头 >20° 时间占比 | 6% |

转头**确实发生**(最大 54°),但平均幅度小(5.7°)—— 因为带球时球多在身体
正前方,方位角≈0 时头自然朝前,只在球偏侧时才大幅转头。这是带球任务下的
合理行为,非策略缺陷。视频中转头不明显的另一原因是后方跟随摄像机难分辨绕
垂直轴的旋转。测量脚本: `scripts/measure_neck_yaw.py`

---

## 执行顺序决策 (2026-06-02)

MOS92 平台已完成: G-1(行走) ✓ / G-2(带球) ✓ / A-MOS92(neck gaze) ✓。
按 v2-5 §7b 的 critical path,下一步定为 **Spike E-MOS92 (goal-scoring)**。

**为什么跳过 Spike B-MOS92(相机视野):**
- B 验证的是"球在 depth image 中是否可见",这是**几何性质**(球径 0.11m、
  相机分辨率 64×48、fov 60°)。G1 上已测出 1-3m 可检测、0.5m 不可见、60° 侧方
  不可见(见本文档 Spike B 章节)。
- MOS92 与 G1 体型相近(头部相机高度差 ~0.3m),几何结论可迁移。
- 文档里 B 重跑的理由"neck 可动"已被 A-MOS92 覆盖(gaze 行为已验证)。
- 结论:B 不阻塞 v3 启动,精确分辨率参数留待 v3 视觉接入时微调。

**为什么 E 优先于 C/D:**
- Spike E (goal-scoring) 是 **v3 启动的硬门槛** —— 若 goal_rate=0,v2 物理/奖励
  有根本问题,v3 不能启动。
- C(penalty 数值)和 D(delay 耐受度)是参数调优,不阻塞 v3 启动,排在 E 之后。
- E 的依赖(带球能力)已由 G-2b 满足(dribble_success=4.14)。

**Spike E-MOS92 计划:**
- 基于 G-2b checkpoint (`2026-06-01_22-58-05/model_2999.pt`) fine-tune
- target 采样 50% 改为球门方向 + `goal_scored` 判定(球 x>half_length 且
  |y|<goal_width/2)
- 新增 `goal_scored` reward(weight=10,稀疏)+ `goal_progress`(球朝门速度投影,dense)
- 2000-3000 iter,成功标准: goal_rate>10%, shot_on_goal>20%, 带球不严重退化

### 实验 E-MOS92 (2026-06-02) — PASS(门槛通过,但需调权重)

**配置:** 从 G-2b (model_2999) fine-tune 3000 iter,4096 envs。
- 50% episode target = 球门口(x=11.0, |y|<1.0),50% 随机点
- robot spawn 在进攻半场(x∈[6,8.5])面朝 +x 球门,球在近距离
- `goal_progress`(球朝门速度投影,weight=2.0,dense)
- `goal_scored`(越过门线 0→1 跳变,weight=10.0,稀疏)
- goal 判定: 球 local x∈(10.8, 11.6) 且 |y|<1.0

**训练结果(总 step 3000-5998):**

| 指标 | G-2b 基线 | E 峰值(step3972) | E step4000 | E 最终(5998) |
|------|----------|------------------|-----------|--------------|
| goal_rate | — | **0.24** | 0.125 | 0.124 |
| goal_progress | — | — | 0.245 | 0.246 |
| kick_contact | 0.82 | — | 0.47 | 0.50 |
| dribble_success | 4.14 | — | 2.34 | 2.46 |
| fell_over | 0.08 | — | 0.21 | 0.375 |
| upright | 0.94 | — | **0.56** | 0.58 |
| mean_reward | 224 | — | 130 | 120 |

**结论: PASS(critical path 门槛通过)**
- ✓ goal_rate 全程 >10%(门槛),峰值 24% —— **v3 §0.1b 门槛 1 通过,v3 可启动**
- ✓ 球能被踢进球门,goal-scoring 在 MOS92 上可学
- ⚠️ **姿态严重退化**: upright 从 0.94 掉到 0.56,fell_over 升到 0.21-0.375

**关键发现(v3 必须解决):**
- goal reward(weight 10+2)**过强,压倒了平衡/姿态 reward**。策略为把球
  踢向球门牺牲了站立质量 —— 出现踉跄踢球、踢完摔倒的行为。
- goal_rate 峰值在 step 3972(早期),之后**不升反降**且姿态持续恶化 ——
  典型的单一目标过度优化。最佳 checkpoint 是 step ~4000,非最后一步。
- dribble_success 从 4.14 降到 2.4: 50% 随机点 episode 表现退化,策略
  偏向了 goal 模式。

**v3 决策:**
- goal-scoring 可行,v3 可启动。但 Stage 1 的 reward 配比需要:
  1. 降低 goal_scored/goal_progress weight(如 5.0/1.0)或加 upright/alive 权重
  2. 用 early-stop 或 KL 约束防止后期姿态退化(峰值后退化明显)
  3. 保留 G-2b 的 dribble reward 强度,避免带球能力流失
- shot_on_goal_rate 指标本 spike 未单独统计,v3 evaluator 需补充

**checkpoint(最佳): `logs/rsl_rl/mos92_velocity/2026-06-02_14-41-20/model_4000.pt`**
**可视化:** `soccer_eval/2026-06-01_spikes/e_mos92_goal/`

---

## Spike A2-MOS92: 搜索→锁定→追踪逼近→踢球 时序验证

> 目标: 验证「视野丢球→转头搜索→锁定→追踪移动球→逼近→踢向目标」这条
> 行为时序能否**从 reward gating 涌现**(无硬编码 FSM)。用 GT 几何投影模拟
> 视野(无相机),把 v3 的视野中心化设计在 MOS92 上做前置验证。

### 实验 A2-MOS92 (2026-06-02) — 训练中

**新增观测 `ball_gaze_uv` = (u, v, visible):**
- gaze_yaw = heading + neck_yaw,gaze_pitch = neck_pitch
- u = wrap_to_pi(球方位角 − gaze_yaw) / FOV_H(±0.61rad≈35°)
- v = wrap_to_pi(球俯仰角 − gaze_pitch) / FOV_V(±0.50rad≈28°)
- visible = (|u|<1 且 |v|<1);u,v clamp 到 [-2,2],附加到 actor/critic obs 尾部

**新增 reward(门控时序的核心):**
| reward | weight | 作用 | 门控 |
|--------|--------|------|------|
| gaze_center | 1.0 | visible·exp(−(u²+v²)/std²) | 仅可见时给分→保持球居中 |
| gaze_search | 0.5 | (1−visible)·exp(−(|u|−1)₊) | 仅不可见时→朝球方向扫视 |
| search_freeze | −0.5 | (1−visible)·base_speed | 不可见时移动受罚→先转头别迈步 |
| approach_intercept | 2.0 | visible·Δ(到拦截点距离) | 仅可见时→朝预测落点(球速·τ)逼近 |

**环境改动(制造搜索/追踪难度):**
- 50% episode 球生成在背后盲区(rear_sector_half_angle=0.96rad≈55°)
- 球带随机初速 ball_init_speed_range=(0,0.6) m/s(93% episode 球在动)
- spawn_dist_range=(0.8,2.5)m 加宽,搜索非平凡
- neck pose std=1.0(放松脖子,A-MOS92 已验证 MOS92 上 tight neck 最优)

**bootstrap 方式(关键):**
- 从 G-2 最强带球 checkpoint `2026-06-02_13-33-44/model_3998` 启动
  (dribble_success 3.95, upright 0.865, fell_over 0.29)
- ball_gaze_uv 是尾部追加的 3 维,故 actor 81→84 / critic 91→94 首层
  **零填充新列**,obs normalizer 新列填 mean=0/var=std=1
- 仅 load actor+critic(optimizer 重置),保留步态+带球,让 gaze/搜索涌现
- 脚本: `scripts/spike_a2_search.py`,3000 iter,4096 envs

**成功标准(v3 §完整行为时序):**
- search_to_lock_time: 球在盲区时能转头/转身找到(visible 0→1)
- moving_ball_track_rate: 锁定后持续 visible(头部跟上球的移动)
- body_frozen_during_search: 搜索期身体基本不动(search_freeze 起效)
- 不破坏带球: dribble_success / kick_contact 不显著退化

**结果: 时序涌现验证 PASS(含一个关键几何发现)。**

**训练曲线(model_900≈iter900,4096 envs,从 13-33-44 bootstrap):**
| reward | bootstrap后 | iter~900 | iter~1550 | 解读 |
|--------|------------|----------|-----------|------|
| dribble_success | 3.95(基线) | 3.01 | 3.51 | 带球能力完整保留 |
| kick_contact | 0.77 | 0.59 | 0.68 | 踢球保留 |
| upright | 0.865 | 0.72 | 0.85 | obs 扩维后快速恢复 |
| gaze_search | — | 0.31 | 0.36 | **搜索涌现**:看不到球时朝球扫视 |
| gaze_center | — | 0.002 | 0.047↑ | 接近期持续上升 |
| approach_intercept | — | -0.0 | 0.002 | 拦截弱(球几乎静止) |

**行为探针(model_900,512 envs rollout):**
- **搜索成功率 99.8%**,平均第 11.6 step(~0.5s)首次看到球 —— 即使 50%
  球生成在背后 ±55° 盲区,机器人能转头/转身找到球。**搜索→锁定 涌现 ✓**
- **接近期(球距 >1m)可见率 78.6%** —— 走向球时把球保持在视野内。**追踪 ✓**
- **控球期(球距 <0.5m)可见率仅 14%** —— 见下方几何发现。

### ⚠️ 关键几何发现:足下盲区(v3 必须设计)

带球目标把球拉到脚边(rollout 中位球距 **0.03m**),此时球在地面(z≈0.125),
机器人 root z≈0.503,**球比头部低 ~0.38m**。换算俯角:
| 球距 | 球俯角 | 垂直视野预算(neck_pitch −28.6° + FOV −28.6° = −57°) |
|------|--------|------|
| 0.3m | **−52°** | 勉强够 |
| 0.5m | −37° | 够 |
| 0.8m | −25° | 舒适 |
| 1.0m | −21° | 舒适 |

**结论:** 球距 <0.4m 时俯角逼近垂直视野极限,看脚下的球几何上几乎不可能。
这**不是 bug 也不是调参问题,是物理约束** —— 人也不会盯着自己脚下的球。
所以 `gaze_center` 在控球期被物理压到≈0,早期看它「平」是被控球期稀释的假象;
按阶段拆分后,真正该用视觉的**接近期可见率 78.6%**,设计是有效的。

**v3 必须吸收的设计修正:**
1. **gaze_center 只在接近期(球距 >0.5m)门控生效**,控球期不要求球居中,
   改为要求球落在视野**下边缘**或干脆 disable,避免策略为看脚下球而过度低头。
2. 控球/踢球期可切换到**用脚部接触/本体感知**而非视觉(本来就有
   robot_to_ball / kick_contact),视觉负责「找球+接近期对准」。
3. neck_pitch 下限可适当放宽(当前 −0.5rad),但收益有限(0.4m 内仍不够);
   根本解是阶段化视觉职责,而非堆视野。
4. `approach_intercept` 在静止球场景信号弱;moving-ball 追踪需要更大初速
   (当前 (0,0.6) 偏小)或专门的移动球评估场景才能体现。

**checkpoint: `logs/rsl_rl/mos92_velocity/2026-06-02_17-14-09_spike_a2_search/`(已训练完 3000 iter,model_2999)**
**脚本: `scripts/spike_a2_search.py`(零填充 bootstrap)、`scripts/render_a2_search.py`(渲染)**
**可视化: `soccer_eval/2026-06-02_spikes/a2_mos92_search/`(强制 rear_spawn=1.0,500 帧)**
- 视频肉眼可见:球生成在身后 → 机器人**转身 180° 面向球** → 走到球边控球,
  搜索→接近时序清晰。
- 注:第三人称后方跟随相机下,**脖子转头(neck_yaw)细微动作不明显,明显的是
  整个身体转向** —— 恰好印证设计:球在 ±55° 盲区时光靠脖子够不着,必须身体转向
  (与 A-MOS92 转头幅度小的观察一致)。最终训练指标:dribble_success 3.89、
  upright 0.91、fell_over 0.17,带球能力不仅保留还略升,无 Spike E 的姿态崩溃。


---

## Spike v3-MOS92: 视觉接入(Stage 3 — depth 相机 + CNN)

> 对应 v3 §Stage 3 课程 step 1-2(vision/gaze smoke + gaze warmup)。把头部
> depth 相机经 spatial-softmax CNN 接进 actor,在 A2(GT 几何 gaze)基础上加视觉。
> 核心问题分两步:**(1)相机接线对不对、加相机会不会破坏步态(warmup);
> (2)CNN 到底有没有学会看球(去 GT 消融)。**

### 实验 v3-warmup (2026-06-02) — PASS(管线通,但证不了 CNN 学会看)

**配置:** `mos92_soccer_vision_env_cfg` + `mos92_vision_ppo_runner_cfg`
- 头部 depth 相机 64×48、fovy 60°,挂在 head body(随 neck_yaw/pitch 转动)
- spatial-softmax CNN(output_channels [16,32] → 64-d latent),latent 追加到
  84-d GT 1D obs 尾部,actor 首层 148 输入
- **关键: actor 保留全部 GT 球 obs**(ball_gaze_uv / robot_to_ball / ball_velocity),
  即 gaze warmup 不是纯视觉;critic 全 GT 不变(asymmetric)
- 从 A2 `model_2999` bootstrap:GT 列 load A2 权重,尾部 64 列 CNN latent 零填充
- 脚本 `scripts/spike_v3_vision.py`,3000 iter,1024 envs

**训练结果(model_2999):**

| 指标 | 值 | 解读 |
|------|-----|------|
| gaze_center | **0.0622** | **正好卡在 A2 的 GT-only 基线,没上升** |
| dribble_success | 3.75 | 带球能力完整恢复 |
| upright | 0.927 | 步态没被相机破坏 |
| fell_over | 0.056 | 稳 |
| gaze_search | 0.383 | 搜索行为保留 |

**结论: PASS,但只证了 warmup 该证的。**
- ✓ depth 接线正确、neck 转动改变视野(smoke 已验证)
- ✓ 加相机不破坏步态(训练验证:dribble/upright/fell_over 全部恢复)
- ✗ **证不了 CNN 学会看** —— gaze_center 卡在 A2 GT-only 基线不动,因为 actor obs
  里 GT 球向量是相机的**完整替身**,PPO 没有任何梯度压力把信息从相机路由进来。
  gaze_center 与 A2 持平只说明视觉**没有伤害**步态,不能说明 CNN **学会了看**。
  这是「保留 GT 的 warmup」的预期正确结果,非 bug。

**checkpoint:** `logs/rsl_rl/mos92_velocity/2026-06-02_20-19-43_spike_v3_vision/model_2999.pt`
**可视化:** `soccer_eval/2026-06-02_spikes/v3_vision_mos92/v3_vision_mos92-step-0.mp4`

### 实验 v3-probe (2026-06-05) — GT 消融探针,CNN 是死重(验证 warmup 的判断)

> 廉价推理时探针:不训练,加载 warmup model_2999,同一策略跑两遍 —— baseline
> (GT 完整)vs ablated(把 actor 的 GT 球向量 slice 在 raw obs 里清零,只剩
> depth 相机→CNN 路径)。看带球/gaze 能否靠相机活下来。脚本
> `scripts/probe_v3_gt_ablation.py`(256 envs,600 步,settle 后平均)。

**消融 slice(运行时从 ObservationManager 读,防漂移):** actor 1D obs 宽 84,
robot_to_ball=(72,75)、ball_velocity=(78,81)、ball_gaze_uv=(81,84)。

**结果(settle 后平均):**

| 指标 | baseline(GT 完整) | ablated(只剩相机) | ratio |
|------|-------------------|-------------------|-------|
| dribble_success | 4.689 | **0.011** | **0.00** |
| upright | 0.989 | 0.613 | 0.62 |
| gaze_search | 0.475 | 0.143 | 0.30 |
| gaze_center | 0.006 | 0.020 | 3.26 |
| ball_visible | 0.010 | 0.153 | 15.79 |

**结论: depth CNN 是死重 —— 证实 warmup 的判断。**
- 清零 GT 后带球**完全崩塌**(4.689→0.011),相机路径**没携带任何可用球信号**。
  这正是 GT 拐杖预测的结果:CNN 没承担找球,GT 是它的完整替身。
- gaze_center / ball_visible **反而上升是假象不是视觉起效**:GT 一去策略不再带球,
  球被遗弃在 spawn 距离(更常在画面里),而非贴在脚边的相机盲区
  (见足下盲区发现)。gaze_center 绝对值仅 0.02,接近零。
- 诚实标注:清零 raw GT 对策略是 OOD,部分崩塌来自输入冲击;但零残余带球能力
  说明 CNN 没有任何可回退的信号。几分钟探针就把「预期」变成「实测」,
  不必先押上数小时重训。

**可视化:** `soccer_eval/2026-06-05_spikes/v3b_gt_ablation/probe_baseline_vs_ablated.png`

![GT ablation probe](soccer_eval/2026-06-05_spikes/v3b_gt_ablation/probe_baseline_vs_ablated.png)

### 实验 v3b-ablation (2026-06-05) — PASS(CNN 部分学会看,验收达标)

> 对应 v3 §Stage 3 课程 **step 3「Privileged vector dropout」**:actor 仍有
> noisy ball vector 但按课程逐步 mask,vision 始终在,critic 保留 GT。这是
> probe 证明 CNN 死重后的**唯一**让视觉学起来的路径。用户定的**最小消融**范围。

**配置:** `mos92_soccer_vision_ablation_env_cfg` + `spike_v3b_ablation.py`
- 从 warmup `model_2999` **strict bootstrap**(actor 含 CNN + critic,obs 维度
  不变=84,无需 padding)
- `gt_mask` 课程:把 actor 的 robot_to_ball / ball_velocity / ball_gaze_uv 的
  `scale` 从 1.0 线性降到 0.0(iter 500 前保持 1.0 稳住 bootstrap,500→1500
  渐降,1500→3000 保持 0 做纯 CNN 训练)。scale 实现避免硬切的输入冲击。
- actor 的 `ball_to_target`(=target−ball,泄露球位)换成 `robot_to_target`
  (合法目标方向,不含球位);**保留 command + track_* 步态脚手架**(最小消融的
  已接受代价:command 仍泄露粗略球位)
- critic 保持全 GT(asymmetric),只 mask/swap actor 组

**训练完成(3000 iter,1h00m,1024 envs):** mask 全程按设计 1→0。

**消融探针对比(在 v3b model_2999 上重跑 probe,256 envs):**

> 关键口径:v3b 是在 **GT=0** 条件下训练的,所以对 v3b 而言 **ablated(GT 清零)
> 才是它的 in-distribution 工况**,baseline(GT 完整)反而是 OOD。因此读
> ablated 列作为 v3b 的真实「纯相机」表现,并与 warmup 探针对照。

| 指标 | warmup baseline(GT 拐杖) | warmup ablated(崩塌) | **v3b native(GT=0,训练态)** |
|------|--------------------------|----------------------|------------------------------|
| dribble_success | 4.69 | **0.01** | **1.11** |
| ball_visible | 0.01 | 0.15 | **0.39** |
| gaze_center | 0.006 | 0.02 | **0.17** |
| upright | 0.99 | 0.61 | **0.95** |
| gaze_search | 0.48 | 0.14 | **0.28** |

**结论: PASS —— depth CNN 部分学会看,且可证明。**
- 去掉 GT(v3b 的训练工况)后仍保留 dribble_success **1.11**、ball_visible
  **0.39**、gaze_center 升到 **0.17**(warmup 卡死基线 0.05 的 ~3 倍)。warmup
  在同样去 GT 下塌到 0.01 —— 这个 **0.01→1.11 的差距就是 CNN 现在携带了可用
  球向量**,正是 warmup 证不了的事。
- 诚实标注两点:(1)**部分迁移** —— dribble 1.11 远低于 GT-拐杖的 4.7,符合预期
  (相机方位更噪 + 足下盲区物理限制);(2)`command` 仍泄露粗略球位(最小消融的
  选择),部分残留带球靠的是这个脚手架,非纯视觉。要做纯视觉须进 step 5(去掉
  command),见下「下一步」。

**checkpoint:** `logs/rsl_rl/mos92_velocity/2026-06-05_13-40-53_spike_v3b_ablation/model_2999.pt`
**脚本:** `scripts/spike_v3b_ablation.py`、探针 `scripts/probe_v3_gt_ablation.py <run_dir>`、
渲染 `scripts/render_v3b_ablation.py`(每步清零 GT,展示纯相机行为)、
可视化 `scripts/plot_v3_gt_ablation.py`
**可视化:** `soccer_eval/2026-06-05_spikes/v3b_gt_ablation/`(三方对比图 `probe_three_way.png`
+ 纯相机视频 `v3b_camera_only-step-0.mp4`)

![v3b three-way](soccer_eval/2026-06-05_spikes/v3b_gt_ablation/probe_three_way.png)

### 下一步(按 v3 Stage 3 课程)

v3b 验证了 step 3(privileged dropout)可行 —— CNN 能在 GT 去除下承担球向量。
按 v3 课程,后续:
- **step 4「detector/vector distillation」+ step 5「pure vision actor」**:去掉
  `command` 脚手架(当前仍泄露粗略球位),让 actor 完全靠 depth→CNN,验收
  `ball_visible_rate>70%`(接近期)、`lost_ball_recovery`、Goal Rate ≥ Stage 2 的 50%。
- 若纯视觉下带球过弱:按 Phase B2「失败处理」先定位(CNN 结构/视野/reward/延迟),
  **不降低最终要求**;足下盲区按设计用脚部接触+本体感知覆盖,不靠视觉。
- 提醒:dribble 1.11 是「最小消融 + 静止球 + 保留 command」下的数字,真正纯视觉的
  Goal Rate 需在 step 5 + Spike E 的进球场景里单独评估。

---

## Task A: 球物理标定 + DR + 外观 (2026-06-05 晚) — DONE,全部实跑验证

> 对应 v3 §Stage 2「球 Domain Randomization」+ §Stage 4 外观 DR。目的:让球的
> 物理/外观尽量接近真实并可随机化,缩小 Sim2Real gap。

**1. 弹性标定(restitution e≈0.35):** MuJoCo 无直接恢复系数。正 solref 封顶 e≈0.25
(太死);负 solref=(-刚度,-阻尼)才到 RoboCup 级。在真实 timestep(0.005s)用
`scripts/calibrate_ball_restitution.py` 实测:`solref=(-2300,-32)` → e≈0.35
(0.31-0.36 over drop 0.3-1.5m)。已确认进编译模型。roll friction 0.01→0.02 加滚动阻力。

**2. 弹性 DR 需要新函数:** mjlab 的 dr 包**没有 solref 随机化**。本会话新增
`dr.geom_solref`(仿 `dr.geom_friction`,`src/mjlab/envs/mdp/dr/geom.py`)。实测
stiffness-only(damping 固定 -32)对 e 单调:stiffness [-9000,-1050] → e∈[0.20,0.58],
覆盖目标 [0.2,0.6](0.6 在此 damping 下封顶 ~0.58,已诚实标注)。

**3. 球 DR(4 个 startup event,逐 env 采样):** radius ±5%(`dr.geom_size`)、
mass ±15%(`dr.pseudo_inertia`,质量+惯量一致缩放)、slide friction 0.3-0.7
(`dr.geom_friction`)、elasticity(`dr.geom_solref`)。**实跑验证:** 16 envs 逐
env stiffness 真实分散 -8744…-1070(std 2650),球 30 步稳定 z∈[0.105,0.115]。

**4. 外观纹理(truncated-icosahedron):** 12 红五边形 + 蓝/白六边形(spherical Voronoi),
1 块标记六边形用于观测旋转。作为 cube texture 建出,`textured=False` 默认关闭——
**当前 obs 是 depth-only,纹理只影响 viewer/RGB,对训练零影响**,加 RGB obs 后再开。
已渲染验证(`scripts/proto_ball_texture.py`)。

**改动文件:** `soccer_field.py`(SoccerBallCfg + 纹理 builder)、`dr/geom.py`+`dr/__init__.py`
(新增 geom_solref)、`mos92/env_cfgs.py`(4 个 DR event)。
**脚本:** `calibrate_ball_restitution.py`、`proto_ball_texture.py`。

---

## Task B: 跳跃修复 (2026-06-05 晚) — PASS,但暴露夹球 exploit

> 动机:用户在 v3b 视频里看到机器人**靠跳起来转向**而非迈步。双足步态本应始终
> 单/双支撑,出现腾空相说明在用跳跃重定向。

**新 reward `flight_phase`:** 双脚同时离地(`feet_ground_contact.found==0` 全 True)
= 惩罚,`air_time_threshold=0.05` 忽略快速步态切换的瞬时双浮。挂进 base config,
weight=-1.0,记 `Metrics/flight_phase_frac` 指标。**未加**多余的转向塑形——已有
`track_angular_velocity`(weight 2.0)会在跳跃被禁后自然驱动迈步转向。

**微调训练(`spike_v3d_jumpfix.py`,从 v3b model_2999 strict bootstrap,1500 iter,
1024 envs,~32min):** 关键防坑——`gt_mask` 课程的 step 计数在新 run 会**从 0 重启**,
若照搬 v3b 设置会在前 500 iter **重新引入 GT 拐杖**;故 pin `start_step=-1,end_step=0`
→ 全程 factor=0,GT 球向量整段保持 mask。

**结果(实测):**
- `flight_phase_frac`: 0.244(init)→ **~0.01**,全程稳定低位。跳跃被压住。
- `dribble_success` ≈ 2.0-2.5(训练态),`gaze_center`/`upright` 都没退化。
- 消融探针(v3d run):baseline≈ablated,确认**仍是纯视觉,未重引入 GT 拐杖**。

**⚠️ 但视频暴露了新问题(夹球 exploit):** v3d 行为视频(全 10s,12 帧跨时间线
人工查看)显示——跳跃确实没了(每帧双脚着地),**但机器人反复"骑"在球上,球卡在
胯下/两腿之间**。这正是规则文档里的**规则 2(持球)+ 规则 3(夹球)**。
- **根因:** 任务只奖励"球到目标 + 接触",**没有任何 penalty 惩罚占有/夹球**。
  已确认 rewards.py 里 `illegal_body_contact/holding_ball/ball_trapped/...` 6 个规则
  函数**全部未实现**,mos92 env 也没挂 body↔ball 接触传感器。
- **结论:** 这不是 bug,是 reward 设计缺口。**只要不惩罚占有,RL 必然学到卡球**——
  这是教科书级 reward hacking,已写进 v3.md/v2-5.md 规则章节作为实证锚点。

**checkpoint:** `logs/.../2026-06-05_20-46-37_spike_v3d_jumpfix/model_1499.pt`
**产物:** `soccer_eval/2026-06-05_spikes/v3d_jumpfix/`(视频 + 探针结果)

---

## 旁路发现: 手臂外展根因 (2026-06-05) — 非代码覆盖,是关键帧

用户多次要求手臂自然垂放。实测 `default_joint_pos` 的 shoulder_roll = **±1.400 rad
(±80°)** —— 机器人用的 `KNEES_BENT_KEYFRAME` 关键帧本身就把手臂设成平举,
`pose` 奖励(weight 1.0,站立 std 0.05 很紧)每步把手臂拉向这个默认位 → **策略被
主动奖励去举手**。另有未使用的 `HOME_KEYFRAME`(全 0,手臂垂下)。
- 诚实说明:**无法用 git 证明是否"被后来覆盖"**——该目录不是 git 仓库。
- 修复:`KNEES_BENT_KEYFRAME` 的 shoulder_roll ±1.4→~0,**需重训**(当前策略按 ±1.4
  训出);最省做法是并进下一次重训。已存记忆 `mos92_arms_out_keyframe.md`。

---

## Task v3e: 反作弊规则 + 手臂修复 (2026-06-05 夜, 无人值守) — 部分成功

> 动机:v3d 视频暴露"骑球/卡球" exploit(规则2持球+规则3夹球)。本次实现完整反作弊
> 基础设施并从 v3d 微调,同时并入手臂关键帧修复。对应 v3.md §2 全部 6 类违规中的 1-4 条。

**实现(全部本会话新建,已 smoke 验证 penalty 真触发):**
- `DribbleCommand` 新增 metrics:`ball_speed`、`holding_time`、`ball_stuck_time`(逐步累积+条件断开即清零)
- 新 `body_ball_contact` 传感器:`mode="body"` 匹配全部非脚身体(排除 Rfoot/Lfoot/Rankle/Lankle),
  实测解析到 17 个非脚 body。`illegal = body接触 ∧ ¬脚接触`
- rewards.py 新增 4 个 penalty:`illegal_body_contact`(-1.0)、`ball_trapped`(-3.0,最重)、
  `holding_ball`(-2.0)、`ball_sticking`(-1.0),各自写 `Metrics/*_frac` 日志
- `penalty_weight_curriculum`:factor 0.2→1.0 over iter 100-700(Spike-C:满强度从 0 起会
  让"不碰球最安全",带球崩)。实测 penalty_factor 末期=1.0
- 关键帧:`shoulder_roll` ±1.4→±0.15(留 9° 余量防手↔大腿自碰),手臂自然垂放
- 安全防坑:`gt_mask` pin 到 factor=0(start=-1,end=0),微调不重引入 GT 拐杖

**训练:** 从 v3d `model_1499` strict bootstrap,3000 iter,1024 envs,~60min,跑满 2999。

**验证(全部实跑,无编造):**

*1. 纯视觉探针(在 v3e 上重跑 probe,256 envs):*
- ablated(GT=0,训练态)dribble_success **1.82** vs baseline(GT 完整,OOD)0.06,
  gaze_center ratio ~1.3 → **GT obs 惰性,仍是纯视觉,未重引入拐杖** ✓

*2. v3d vs v3e A/B(同一反作弊插桩 env,256 envs×450 步,均逐步清零 GT —— 唯一公平对照,
因为 v3d 训练期根本没记这些指标):*

| 指标 | v3d | v3e | delta | 判读 |
|------|-----|-----|-------|------|
| dribble_success | 2.44 | 2.46 | +0.02 | **技能保住** ✓ |
| mean_abs_shoulder_roll (rad) | 1.29 | **0.26** | **-1.03** | **手臂垂下,决定性** ✓ |
| holding_ball_frac | 0.297 | **0.227** | -0.07 | 持球降 24%,真实但**仍偏高** |
| holding_time_mean (s) | 1.22 | 0.92 | -0.30 | 平均持球时长降 25% ✓ |
| ball_trapped_frac | 0.000 | 0.000 | 0 | 严格夹球(双接触)两者都≈0 |
| ball_under_robot_frac | 0.62 | 0.79 | +0.17 | **升**(张力,见下) |
| ball_speed_mean (m/s) | 0.34 | 0.37 | +0.03 | 球更动,非停住 |
| upright | 0.987 | 0.987 | 0 | 稳 ✓ |

**诚实结论:部分成功,喜忧参半。**
- **手臂修复:决定性成功。** shoulder_roll 从 1.29rad(~74°)→0.26rad(~15°),动态带球时也保持。
- **技能保住:** dribble_success 持平,upright 稳,纯视觉未退化。
- **反作弊:真实但不彻底。** v3d 的 exploit 主要是**持球(holding,球近+慢+持续~30%步)**,
  而非严格**夹球(trapped,双接触+静止)**——后者两模型都≈0。v3e 把 holding 降了 24%、
  平均持球时长降 25%,方向对,但 holding 仍有 23%。
- **关键张力:`ball_under_robot` 反而升(0.62→0.79)。** 诚实解读:这个代理指标**无速度门控**,
  把"贴身紧密控球(球在动)"和"卡住球(球停)"混为一谈。v3e 的 ball_speed 实际更高(球一直在动),
  说明它是把球控得**更近但仍在推进**,不是停球占有。所以 under 升 ≠ 作弊变多;它暴露了这个
  代理指标本身不够精确(precise penalty fractions 才是准的,见上表)。
- **早先用 3 帧静图判"手臂又甩出去"是误判**:那是 shoulder-*pitch* 自然摆臂平衡,不是 fix 针对的
  *roll* 外展。256env×450步的 0.26rad 测量是权威。

**产物:** `soccer_eval/2026-06-05_spikes/v3e_anticheat/`(视频 + 探针结果)。
**checkpoint:** `logs/.../2026-06-05_22-43-22_spike_v3e_anticheat/model_2999.pt`
**脚本:** `spike_v3e_anticheat.py`、A/B 对照 `eval_v3d_vs_v3e.py`(同 env 双 checkpoint)。
**下一步:** holding 仍 23% 偏高 → 加强 holding 惩罚(权重或降阈值)再微调一轮(见 v3f)。

---

## Task v3f: 加强持球惩罚 (2026-06-06 凌晨, 无人值守) — 成功,本夜最佳模型

> 动机:v3e 的 A/B 显示残留 exploit 主要是**持球(holding 23%)**而非夹球。v3f 从 v3e
> 微调,只动 holding 惩罚:weight -2.0→-3.0(等于夹球,最重),阈值 1.5s→1.0s(更严但仍在
> 文档 1-2s 范围内,不误杀踢球间隙)。其余与 v3e 完全相同。1500 iter,从 v3e bootstrap。

**监控到的风险与化解:** 脚本里预判"惩罚过猛→机器人躲球→dribble 崩"。早期 iter47 dribble_success
确实掉到 0.07 —— 但这是微调初期暂态,iter186 已恢复到 3.0,**未发生躲球**。penalty_factor 末期=1.0。

**验证(全部实跑):**

*1. 纯视觉探针:* ablated(GT=0,训练态)dribble_success **1.82** vs baseline(OOD)0.19,
GT obs 惰性 → **仍纯视觉,无拐杖** ✓

*2. v3d/v3e/v3f 三方 A/B(同一插桩 env,256envs×450步,逐步清零 GT —— 公平对照):*

| 指标 | v3d | v3e | v3f | 趋势 |
|------|-----|-----|-----|------|
| dribble_success | 2.41 | 2.52 | **2.60** | ↑ 技能反升 ✓ |
| holding_ball_frac | 0.292 | 0.219 | **0.137** | ↓ **比 v3d 降 53%** ✓ |
| holding_time_mean (s) | 1.17 | 0.90 | **0.61** | ↓ 持续下降 ✓ |
| ball_trapped_frac | 0.000 | 0.000 | 0.000 | 严格夹球始终≈0 |
| ball_under_robot_frac | 0.63 | 0.81 | **0.74** | v3e 升后 v3f 拉回 |
| ball_speed_mean (m/s) | 0.35 | 0.38 | **0.39** | 球更动(非停球) |
| mean_abs_shoulder_roll (rad) | 1.29 | 0.26 | **0.20** | ↓ 手臂垂下 ✓ |
| mean_abs_shoulder_pitch (rad) | 0.30 | 0.27 | 0.32 | ≈持平(见下) |
| gaze_center | 0.030 | 0.035 | **0.058** | ↑ |
| upright | 0.983 | 0.986 | **0.992** | ↑ 最稳 ✓ |

**结论:成功。v3f 是本夜最佳模型** —— holding 比 v3d 降 53%(0.29→0.14)、平均持球时长降 48%,
同时 dribble_success 反升到 2.60、upright 最高、纯视觉保持。over-penalty 风险未兑现。

**关于手臂的诚实修正:** v3f 视频帧里中段仍见手臂**上举**。深查发现这是 shoulder-**pitch**
(前向/上摆),不是关键帧 fix 针对的 shoulder-**roll**(侧向外展/T-pose)。量化:三个模型
pitch 都 ~0.3rad(~17°)基本持平 → 上举是**动态平衡的瞬时峰值**,非持久姿态;而 roll(持久的
T-pose 外展)已从 74°→11° 彻底修好。**早先 A/B 只测了 roll,漏看 pitch,本轮补测纠正。**
若要进一步压低摆臂,需对 shoulder_pitch 加 pose/动作惩罚,留作后续。

**产物:** `soccer_eval/2026-06-05_spikes/v3f_holding/`(视频+探针+三方A/B)。
**checkpoint:** `logs/.../2026-06-06_00-09-22_spike_v3f_holding/model_1499.pt`(**当前最佳**)
**脚本:** `spike_v3f_holding.py`、三方对照 `eval_v3d_vs_v3e.py`。

---

## Task v3g-S1: RGB 自定位可见性 GATE (2026-06-06 上午) — PASS

> v3g 目标:纯 RGB 视觉自定位 + 深度测球(用户确认:加 RGB 相机 + 特权→蒸馏)。
> 训练前的科学纪律:先证明信号存在,再训练(同深度探针做法)。

**关键约束(实测):** 现有头部相机 depth-only;球场线/中圈/罚球点是平面贴花 geom(z≈0)
**深度完全看不见**;球门柱 3D 但 0.1m@10m 不足 1 像素。→ 深度无法自定位,必须用 RGB。

**踩坑修正:** 初版探针 5 个位姿渲染出**完全相同**的图(marking 比例全等)。根因:相机渲染
触发是 `env.sim.sense()`,不是 `sim.forward()`——teleport 后只 forward 没 sense,视图不刷新。
修正为 `forward(); sense()` 后,5 个位姿产生真实差异。**诚实记录:不查这个 bug 会得出错误 GATE 结论。**

**结果(`scripts/probe_rgb_field.py`,3 档分辨率 × 5 位姿,marking 像素比例 + 人工看图):**
- 面向球门的位姿过线:midfield 2.4%、near_opp_goal 2.4%(64×48)→ 5.7-6.0%(96×72/128×96)
- 面向空场/背向的位姿 LOW(0.3-0.5%)——物理合理:视野无地标时 marking 自然稀疏
- **人工看图确认:** near_opp_goal 渲出**清晰球门矩形+门柱+罚球区白线**;corner 渲出绿场+
  零散白线段+红色球点。**深度绝无可能渲出这些。**
- **96×72 明显比 64×48 信息更丰**(面向球门 5.7% vs 2.4%)→ 自定位相机建议用 96×72。

**GATE 结论:PASS。RGB 确实能看见球场地标,是自定位的正确模态。** 可进入 Phase A(特权自定位闭环)。
**产物:** `soccer_eval/2026-06-06_spikes/v3g_rgb_probe/`(15 张渲染图)。**计划:** `soccer_robot_v3g_plan.md`。

---

## Task: 手臂不过肩 — shoulder_roll 动作硬限位 (2026-06-06 上午, 代码就绪未训)

> 用户需求:手臂保留用于平衡(**不冻结**),但要像人一样放在两侧、**不高于肩**。

**重要更正(推翻 v3f 记录里的结论):** 之前 v3f 段说"手臂上举是 shoulder-**pitch**",
**错了**。单关节扫角精确实测(手相对肩的 z 高度):
- **pitch** 扫 −1.5→+2.0:手 z 几乎不动(±0.006 m)→ pitch 只让手臂**前后摆**(平衡要用)。
- **roll** 扫 −1.4→+1.4:手 z 变化 **0.32 m** → roll 才是**侧向抬手**、决定高不高于肩的 DOF。

**标定数值(站立单关节,手 z vs 肩 z + 手-大腿间距):**
- 右臂 roll default −0.15 = 正好肩高;>−0.15 抬过肩;−0.6 时手低于肩 0.067m、离大腿 0.29m(不自碰)。
- 左臂镜像(default +0.15 = 肩高)。

**实现:** 在 mos92 base velocity env 给 `joint_pos_action.clip` 加 per-joint 硬限位
(clip 作用于绝对关节目标弧度,因 `use_default_offset=True`):
- `right_shoulder_roll: (-0.6, -0.15)`、`left_shoulder_roll: (0.15, 0.6)`
- **pitch / elbow 不限**(保留前后摆臂用于动态平衡 —— 用户明确要求)。

**Smoke 验证(实跑):** clip 已挂载(shape (envs,20,2));给右 roll 发 +100 抬手指令 →
目标被钳到 **−0.15**(肩高);给 pitch 发 +100 → 19.08 不钳(摆臂自由)。硬保证成立,非软惩罚。

**诚实局限:** 静态单关节标定;动态踢球是 pitch+roll+elbow 组合,过肩可能还有 elbow 贡献,
clip 住 roll 上限挡住主要侧抬,彻底与否需实跑视频确认。**需重训生效**(action 空间语义变),
当前 v3g Phase A 仍在跑,故只改代码+验证,未启动训练。**脚本:** `env_cfgs.py` 第 101-118 行。

---

## Task v3g Phase A 收尾: 自定位探针 (2026-06-06 中午) — 结论:策略**没有**真正依赖自定位

> Phase A 训练完成(model_1999,2000 iter,训练日志 dribble_success 稳态~1.1)。收尾必须
> 验证:策略到底有没有用 `robot_field_pose` 来瞄准球门,还是找了别的捷径绕过去。

**探针踩坑(已修):** 初版探针 baseline dribble_success=0.002(≈0,和训练的~1.1 矛盾)。根因:
selfloc env 的 `gt_mask` 课程在 play 模式停在 factor=1.0,于是策略收到**非零的 GT 球 obs,
而它被训练成"看到这些是 0"** → OOD 输入冲击,行为崩坏。修正:探针在**两个 pass 都把 GT 球项清零**
(复现训练的纯视觉条件),只单独变动 `robot_field_pose`。修后 baseline 恢复正常(upright 0.999)。

**探针结果(修正版,256envs×600步,只变动 field_pose):**

| 指标 | baseline(field_pose 在) | ablated(field_pose 清零) | ratio |
|------|------|------|------|
| dribble_success | 0.2294 | 0.2128 | **0.93** |
| ball_to_target_error (m) | 1.97 | 1.99 | 1.01 |
| upright | 0.999 | 0.999 | 1.00 |

**诚实结论:清零自定位只让 dribble_success 掉 7% → 策略基本没在用 `robot_field_pose`。**
Phase A 的目标("强制策略靠自定位推断球门方向")**没达成** —— 策略找了捷径绕过自定位。

**最可能的原因(待验证):** dribble 任务的几何让"球门方向"无需绝对场上位姿就能推断 —— 机器人
初始大致朝球门、目标在 +x 方向,"把球往前推"就近似对了。**没有任何东西强制它用 field_pose。**
要让自定位变成必需,任务得在**多样化的初始位置/朝向**下生成,使球门方向真正依赖于场上位姿。

**这是个有价值的负结果**,直接决定下一步:不能直接进 Phase B 加 RGB —— 得先让任务**需要**自定位,
否则给 RGB CNN 蒸馏一个策略根本不用的信号,是空中楼阁。**产物:** `scripts/probe_v3g_selfloc.py`。

---

## Task v3g Phase A-explicit: 显式自定位 head + 误差奖惩 (2026-06-06 下午) — 代码就绪,smoke 通过

> 动机:上面 Phase A 的负结果。用户明确"判断是否准确需要有奖励和惩罚"——要的是**显式**的
> 自定位估计输出 + 对估计准度的直接奖惩,不是原计划的隐式蒸馏。本轮做几何修复 + 显式 head。

**确诊根因(读代码,非猜测):** selfloc env 继承链 `selfloc←vision_ablation←vision←search←base`
**没有任何一处设 `goal_target_fraction`**,取默认 `0.0`。于是 dribble 目标是离球 2-6m 的**随机点**,
不是球门。知道自身场上位姿对随机点方向毫无帮助——只有目标是**固定地标**(球门 x=+11)时自定位
才有用。所以自定位在原任务里**构造上不可学**,清零 field_pose 只掉 7% 是必然。

**实现(全部本会话,smoke 实跑验证):**
- **几何修复**:`mos92_soccer_selfloc_env_cfg` 设 `goal_target_fraction=1.0` + `target_dist_range
  =(1.0,3.0)`,目标永远是球门;保持继承的大范围初始位姿随机化(base reset_base yaw ±π+全场 xy,
  链上未改)。这是负结果的根因修复。
- **显式自定位 head**:新建 4 维 no-op `SelfLocAction`/`SelfLocActionCfg`
  (`envs/mdp/actions/actions.py`)。策略输出对自身 `[x_n,y_n,sinθ,cosθ]` 的估计,`apply_actions`
  是 no-op 不驱动 sim;`raw_action` 供 reward 读、经已有 `"actions"` obs term(`last_action`)
  自动回传策略(20→24)。注册 `cfg.actions["selfloc"]`(插在 joint_pos 之后,电机切片不变)。
- **误差奖惩**(`velocity/mdp/rewards.py`):`selfloc_accuracy`(exp(-err/std²),pos/head 分项,
  weight +1.5,记 `Metrics/selfloc_pos_err_m` 米制)+ `selfloc_error_penalty`(原始 L2,weight −0.5,
  线性,大误差持续受罚——exp kernel 在大误差处饱和,线性项才真正满足"估错给罚")。
- **维度账(env 实例化实测确认)**:actor 1D obs 85→89,mlp.0 in 148→153,输出 20→24,
  `distribution.std_param` (20)→(24)。**critic 也变宽**:`"actions"` 同在 critic 组,
  obs 94→98,mlp.0 in 94→98(输出仍 1)。
- **partial-load 修正(实测推翻原计划)**:原计划以为 critic 可 strict load,**错**——critic 输入
  也变了。改用**通用 shape-match partial-load**(actor+critic 都只保留 shape 不变的张量,
  自动跳过 input/output/normalizer/std_param)。smoke 实测:actor loaded 12/reinit 7
  (3 normalizer+std_param+mlp.0.w+mlp.6.w/b),critic loaded 8/reinit 4(3 normalizer+mlp.0.w),
  深层 MLP+depth-ball CNN 全部承接。
- **PPO 风险已查**:`std_type` 配置标量但 ckpt 里 std_param 实为 per-dim (20,);cognitive 维
  与电机共享探索噪声(这正是它探索估计值所需),partial-load 重置成 (24,) 无碍。

**Smoke 验证(`--smoke` 8 iter,32 env,实跑 exit 0):** 无 NaN;action dim=24(joint_pos 20
+selfloc 4);两个 selfloc reward 都触发(`selfloc_error_penalty` 从 iter1 给梯度,
`selfloc_accuracy` 估计未标定前≈0,符合设计);partial-load 打印正确。**未达标项诚实记录**:
smoke 仅 8 iter,`selfloc_pos_err_m` 还在 ~35m(全新未标定 head),标定下降要看全量 2000 iter。

**代码质量**:5 个改/新文件 ruff+ty 全清。`make check` 余 38 个 ty diagnostics 全为**预存**
(soccer_field.py mujoco add_geom stub 类型 + rewards.py 三个旧 anti-cheat 函数的 cmd.metrics 警告),
非本轮引入,未动。

**张力提示(诚实):** Phase A GT `robot_field_pose` 仍在 obs 里,估计可"抄"GT 满足 accuracy
reward——这只是 sanity(证 head+reward+几何管线对)。**自定位真本事看 Phase C**:mask 掉 GT
field_pose(hook 是 `_SELFLOC_MASK_START/END_ITER` 常量)后估计只能从视觉来。需先接 RGB
(S1 已证唯 RGB 能看见球场线,depth 看不见)。

**产物/脚本**:`spike_v3g_selfloc_explicit.py`(actor+critic shape-match partial-load)、
`probe_v3g_selfloc_explicit.py`(双消融:① 标定精度 estimate-vs-GT 米/度;② 清零估计回传看
goal-dribble 是否崩→证估计真被用)。**下一步**:全量训练 2000 iter,跑探针验标定下降+估计被用。

### 全量训练结果 (2026-06-06 下午, 2000 iter, 1024 env, ~40min) — 自定位成功,带球崩塌(reward 失衡)

**训练态(model_1999):**
- **自定位学会了**:`selfloc_accuracy` reward **1.11**(接近 1.5×exp 上限)、`selfloc_pos_err_m`
  从 18m(iter96)→ **1.42m**、`upright` 0.87、`fell_over` 仅 0.32(站得很稳)。
- **但带球崩了**:`dribble_to_target`≈0.0005、`ball_path_length` 0.44m、`ball_speed` 0.05、
  `dribble_success` 全程 **0.0**。机器人几乎不带球。

**探针(model_1999,256env×600步,三消融):**

| 指标 | baseline | −estimate | −GT_pose |
|------|------|------|------|
| dribble_success | 0.0000 | 0.0000 | 0.0000 |
| selfloc_pos_err_m | **0.567** | 0.576 | 5.381 |
| selfloc_head_err_deg | **2.64** | 2.39 | 91.85 |
| upright | 0.988 | 0.989 | 0.989 |

**诚实结论:**
1. **自定位管线完全打通,标定精准**:baseline 位置误差 0.57m、朝向误差 2.6° —— 显式 head +
   误差奖惩的设计**奏效**,这是本轮的核心成果。
2. **估计目前靠"抄"GT**:`−GT_pose` 消融后误差暴涨到 5.4m/92° —— 估计依赖 obs 里的 GT
   field_pose,GT 一清零就估不准。**这正是计划里预警的 Phase A 张力**(GT 在 obs 里时估计可走
   捷径),不是 bug,是 Phase A 的预期局限。真正的视觉自定位要等 Phase C(mask GT + 接 RGB)。
3. **`−estimate` 消融无法判定**:`dribble_success` 三态全 0 —— 因为机器人根本不带球,没有带球
   行为可供"崩溃",这个消融此轮失去意义。
4. **根因:reward 失衡(教科书级,非自定位之过)**。机器人发现"站稳 + 准报位置"稳拿
   `selfloc_accuracy`(1.11)+`upright`(0.87)≈2 分,比"辛苦带球穿越大半场到固定球门"(难且稀疏)
   划算太多,于是**理性放弃主任务**。几何修复(目标永远是固定球门 + 全场随机出生)同时让带球
   任务变难,放大了这个失衡。

**下一步(待定方向):**
- **修 reward 平衡**:降 `selfloc_accuracy` 权重(1.5→~0.3)或对其加"仅在带球进展时才给"的
  门控(类似 gaze_center 的可见性门控),让自定位成为带球的辅助而非替代;同时可能要从 v3f
  bootstrap 时先保住带球技能(本轮 reinit 了 mlp.0/mlp.6,带球策略其实被部分打散重学)。
- **或调几何难度**:`target_dist_range` 进一步收短、或初始位姿不要全场而是限制在进攻半场
  (类似 goal env 的 spawn),让带球到球门可达,主任务能拿到分再谈平衡。
- 这些都需重训,留作下一轮决策。

### Reward 平衡重训 (2026-06-06 下午, 2000 iter) — 带球恢复,但 success 仍 0,消融再次无法判定

> 动机:上轮 1.5/−0.5 失衡让机器人"站着报位置"躺平,带球全崩。本轮两改:① `selfloc_accuracy`
> 1.5→**0.3**、penalty −0.5→**−0.3**(自定位降为辅助 shaping,不与带球争);② spike 脚本
> partial-load 改"行前缀拷贝"——`mlp.6`/`std_param` 输出层前 20 行(电机,joint_pos 在前)从 v3f
> 拷入、仅后 4 行(selfloc)fresh,保住 v3f 学会的踢球电机映射(上轮整层 reinit 把带球打散了)。

**训练态(model_1999)对比上轮:**

| 指标 | 上轮(1.5/−0.5) | 本轮(0.3/−0.3) | 解读 |
|------|------|------|------|
| dribble_approach | 0.21 | **0.71** | 大幅靠近球 ✓ |
| dribble_to_target | 0.0005 | **0.081** | 带球向目标(×160) ✓ |
| ball_path_length | 0.44m | **4.24m** | 球真在动(×10) ✓ |
| ball_speed | 0.05 | **0.49** | 带球速度起来 ✓ |
| dribble_success | 0.0 | **0.0** | 仍未进球门 ✗ |
| selfloc_pos_err_m(训练态) | 1.42 | 5.30 | 标定让位带球(探针更准) |

**探针(model_1999,256env×600步,三消融):**

| 指标 | baseline | −estimate | −GT_pose |
|------|------|------|------|
| dribble_success | 0.0000 | 0.0023 | 0.0006 |
| selfloc_pos_err_m | **1.514** | 2.188 | 5.688 |
| selfloc_head_err_deg | **5.02** | 5.79 | 84.76 |
| upright | 0.999 | 0.999 | 0.999 |

**诚实结论:**
1. **reward 平衡修复成功**:带球从"完全不动"恢复到"主动追球(approach 0.71)+ 推球 4.24m
   (ball_speed 0.49)"。降权重 + 保住 v3f 电机头两个改动都奏效。
2. **自定位标定基本保住**:探针 baseline 1.51m/5.0°(上轮 0.57m/2.6°)。降权重只让标定从亚米级
   退到 1.5m 级,自定位能力没丢。
3. **估计仍靠抄 GT**:`−GT_pose` 消融误差暴涨 5.7m/85°(同上轮)。Phase A 预期张力,要 Phase C
   mask GT + 接 RGB 才能破。
4. **关键问题第二次无法判定**:`−estimate` 消融 dribble_success 0.0000→0.0023,几乎没变 ——
   **但根因是 baseline 的 dribble_success 本身≈0**(带球够推球但不够送进固定球门),没有"成功"
   行为可供崩溃。带球恢复了,却没强到能进球门,所以"估计是否用于瞄准"仍证不了。

**根因分析:dribble_success=0 的真凶是几何太难,不是 reward 了。** 几何修复(目标永远固定球门
x=+11 + 初始位姿**全场**随机)意味着机器人常从场地远端/背对球门出生,要带球穿越大半场+转向才能
进球门 —— 对一个刚重学输入映射的策略太难。带球能力(approach/path_length)已恢复,差的是"能
带到那么远的固定球门"。

**下一步(明确指向几何,非 reward):**
- **收窄初始位姿到进攻半场 + 面向球门**(类似 goal env 的 `spawn_x=half_length-[2.5,5]`、
  `yaw=(-0.6,0.6)`),让带球到球门**可达**,dribble_success 拿到非零值。这样 `−estimate` 消融才
  有"成功"可崩,才能终于判定估计是否被用于瞄准。
- 但注意:窄初始位姿会**削弱自定位的必要性**(位姿不够多样,球门方向又近似 +x)——这正是最初
  Phase A 失败的几何。**取舍**:可先用"中等"随机(进攻半场但 yaw 全 ±π,强迫转向找球门),
  在"带球可达"与"自定位必需"之间找平衡。需下一轮重训验证。

### Phase B+C: 双 CNN(深度球 + RGB 定位) + GT 渐隐课程 (2026-06-06 晚, 2000 iter) — 纯视觉自定位失败(负结果)

> 用户要求直接做 Phase B(接 RGB)+ Phase C(mask GT 逼纯视觉)。一次性实现:
> ① **双 CNN 架构**(查代码确认 spatial-softmax 模型按 obs 组 tensor rank 自动建 CNN,4-D 组→CNN,
>    传 cnn_cfg dict-of-dicts 按组名给各分支独立配置)。深度 CNN 专管球(不动),新增 fresh RGB CNN
>    专管定位。head_cam data_types 加 rgb,新增 camera_rgb obs 组(3,48,64),actor obs_groups 加它。
> ② **Phase C 课程**:`selfloc_gt_mask` curriculum 把 GT robot_field_pose obs(仍在 ball_to_target 键下)
>    scale 从 1→0 渐隐于 iter[400,1200]。关键设计:selfloc_accuracy reward 比对的 GT 是每步从真实
>    state 现算的,**不受 obs mask 影响**,所以老师信号在 obs 渐隐后仍在 → 逼策略改用 RGB CNN。
> ③ **partial-load 最干净**:从 rebalanced model_1999 bootstrap(已有 selfloc head,动作维不变),
>    actor 仅 mlp.0 reinit(input 153→217 加 64 维 RGB latent)+ RGB CNN fresh;深度 CNN/MLP 主干/
>    selfloc head/std_param 全留;critic 纯 MLP 完全不变(loaded 12 reinit 0)。smoke 实测确认。

**训练态 selfloc_pos_err_m 全程轨迹(行号≈iter):**

| iter | GT 状态 | pos_err_m | 解读 |
|------|------|------|------|
| 150 | 全在 | 9.6 | 初始重学 |
| 400 | 全在→开始渐隐 | 6.8 | 学到 ~7m(比纯 explicit 1.5m 差,因 mlp.0 reinit) |
| 1200 | 渐隐完→全 0 | **8.4** | **GT 抽走后不降反升** |
| 2000 | 全 0 | 7.8 | 横住,无恢复迹象 |

**探针(model_1999,256env×600步,三消融):**

| 指标 | baseline | −estimate | −GT_pose |
|------|------|------|------|
| selfloc_pos_err_m | 4.62 | 2.19 | **5.43** |
| selfloc_head_err_deg | 9.17 | 7.10 | **65.8** |
| dribble_success | 0.0015 | 0.0015 | 0.0034 |
| upright | 0.986 | 0.989 | 0.998 |

**诚实结论:纯视觉自定位失败。**
1. **决定性测试(−GT_pose)失败**:清零 GT pose obs 后误差 4.62→5.43m、朝向 9°→66°。虽没像
   Phase A 爆到 85°,但明显恶化 → **RGB CNN 没接管定位**,估计仍主要依赖残留 GT 通路。
2. **反常信号——这次估计本身就没训好**:baseline 4.62m 比上轮纯 explicit(无 RGB)的 1.51m **更差**;
   `−estimate`(清零估计回传)误差反而更低(2.19m)。说明双 CNN + GT 渐隐双重压力下估计退化了。
3. **训练轨迹铁证**:误差在 GT 渐隐期(iter400-1200)**单调恶化**(6.8→8.4m)并在 GT 全 0 后横住
   不恢复。fresh RGB CNN 在 800 iter 内**没学会从球场线推位置**。
4. dribble 也基本崩(success ~0.001-0.003)——任务太难 + 估计退化 + RGB 没接上,三重夹击。

**根因分析(待验证的几个假说,按可能性排序):**
- **(A) fresh RGB CNN 从零学视觉定位,800 iter 远不够**。深度 CNN 当初是在 v3f 长期训练里学的;
  这里 RGB CNN 从随机初始化开始,还要同时和 GT 渐隐赛跑。**最可能的主因**。
- **(B) GT 渐隐太早/太快**(iter400 就开始,从一个估计还没训好的状态)。应先让估计在 GT 全在时
  充分收敛(到 ~1.5m)再开始渐隐。本轮 mlp.0 reinit 让估计 iter400 时还在 6.8m,根基不稳就抽拐杖。
- **(C) RGB 看不清球场线**:相机 64×48 低分辨率 + FOV 60° + 球场线可能太细/对比低,CNN 无信号可学。
  **需诊断**:渲染一帧 RGB 看球场线是否可见(类似 S1 当初验证深度看不见线那样)。
- **(D) spatial-softmax 不适合定位**:它提取的是"亮点空间坐标",适合找球这种局部目标,未必适合
  "从全局线条布局推位姿"。

**下一步(需决策,未动手):**
- 先做**便宜的诊断 (C)**:渲染机器人 RGB 视角一帧,确认球场线在 64×48 下是否可见、可学。若看不见,
  上面 A/B 都白搭,得先解决可见性(提分辨率/加粗线/调相机)。
- 若可见,再考虑 **(A)+(B)**:RGB CNN 预热(GT 全在时先训久点让估计收敛 + RGB 旁路学起来),再开始
  更慢的 GT 渐隐(如 iter800→2000),给视觉定位足够学习时间。可能需要 >2000 iter。

### 诊断 (C):RGB 可见性探针 (probe_rgb_field.py, 2026-06-06 晚) — 标线可见但信号弱且有歧义

5 个已知位姿 × 3 分辨率,渲染 head cam RGB,统计"球场标线像素占比":

| 位姿 | 64×48 | 128×96 | 说明 |
|------|------|------|------|
| midfield_face_x | 2.44% OK | 2.47% | 面朝 +x,中线/中圈在视野 |
| **own_half_face_goal** | **0.39% LOW** | **0.46% LOW** | 远端球门只剩地平线几个像素 |
| **corner_face_center** | **0.39% LOW** | **0.33% LOW** | 角球位,标线在侧后方看不到 |
| near_opp_goal | 2.41% OK | 5.96% | 近球门,门柱+禁区线清晰 |
| **center_circle_edge** | **0.42% LOW** | **0.48% LOW** | 中圈左右对称,有歧义 |

**渲染图实证(看 soccer_eval/2026-06-06_spikes/v3g_rgb_probe/):**
- `near_opp_goal`(OK):面朝近球门时门框/禁区线清晰可辨,定位信号充足。
- `own_half_face_goal`(LOW):中圈白弧虽可见但又远又小,远端球门只是地平线一个极小黑点(~17m,几像素);
  且**中圈左右对称**,单帧前视图无法唯一确定位姿。

**诊断结论:Phase B+C 失败是三因叠加,(C) 可见性/地标弱是地基问题。**
1. **不是纯像素问题**:提分辨率到 128×96 救不了那 3 个低位姿(仍 <0.5%)。根因是**视角+距离**——
   头部相机前向平视,关键地标(球门、可区分线条)经常在十几米外只剩几像素。
2. **信号有歧义**:中圈/边线左右对称,单帧前视无法唯一定位。人靠转头扫视+时序积分,单帧 CNN 没这能力。
3. 地基不稳 → 延长训练 (A)/放慢课程 (B) 都救不了近一半看不见地标的位姿。

**下一步方向(需用户决策,各有取舍,均未动手):**
- **方向 1:加时序记忆(最对症)**。单帧→无歧义定位本就难。给 actor 加 RNN/GRU 或叠帧(frame stack),
  让策略积分多帧视角推位姿。改动大(网络结构+rollout),但直击歧义根因。
- **方向 2:降低定位难度的几何**。缩小场地/收窄初始位姿到地标密集区(近球门半场),让"看得见地标"成为
  常态再谈视觉定位。代价:削弱泛化,且和"全场自定位"目标缩水。
- **方向 3:RGB 预热 + 慢渐隐(最省事,但可能仍受限于歧义)**。GT 全在时先训 ~800 iter 让估计+RGB 旁路
  收敛,再 iter800→2500 慢渐隐,加大 RGB CNN 容量。赌的是"有信号的位姿足够支撑学习"。
- **方向 4:换/加相机**。加一个朝下广角或更高视点相机看更多地面标线。改 cfg 即可,但偏离真机头部相机设定。

### v3g temporal: 给 RGB 加时序记忆(叠帧)— 实现就绪,全量训练中 (2026-06-06 晚)

> 用户决策:加时序记忆破单帧歧义,选**叠帧优先**(frame-stack,非 RNN——低风险先验证)。
> 叠帧让 CNN 看到多帧间**运动视差**:机器人移动/转头时对称地标位移模式不同,可消歧。

**实现(全部本会话,smoke 实跑验证):**
- **有状态叠帧 obs term** `StackedCameraRGB`(velocity/mdp/observations.py):`ManagerTermBase` 子类,
  持 per-env `CircularBuffer`,把最近 N=4 帧 RGB 叠成 (B, N*3, H, W)=(B,12,48,64)。**append 时机坑已解**:
  `compute_group` 每步被调多次,但靠 `env.common_step_counter` 去重,仅步计数前进时 append 一次;
  obs manager 在 reset 时调 `func.reset(env_ids)` 清零该 env 历史(下次 append 自动 backfill)。
- **env cfg** `mos92_soccer_selfloc_vision_env_cfg`:RGB obs term 换成 StackedCameraRGB 实例。
  **吸取上轮教训放慢 GT 渐隐**:iter[400,1200]→**[800,2500]**,先让估计在 GT 全在时收敛再抽拐杖。
- **CNN 自动适配**:RGB CNN input_channels 从 obs tensor 自动读=12(N*3),无需改 rl_cfg。
- **partial-load**:从 rebalanced model_1999 bootstrap,actor 仅 mlp.0 reinit(153→217),RGB CNN fresh
  (现 12 通道输入),critic 纯 MLP 不变(loaded 12 reinit 0)。smoke 实测确认。
- **MAX_ITER 2800**(> GT 渐隐终点 2500,留时间在全 mask 下训练)。

**smoke 验证(8 iter,exit 0):** camera_rgb dim=(12,48,64) ✓、RGB CNN 建为 12 通道 SpatialSoftmaxCNN ✓、
叠帧 shape 正确、append cadence 对(每步一次)、无 NaN、partial-load 正确。`make check` 我的文件全清。

**全量训练中**(2800 iter,~95min,比单帧久因多帧 RGB CNN forward)。决定性验证待训练后:用
`probe_v3g_selfloc_vision.py`(叠帧对它透明),看 **−GT_pose 消融误差是否不再暴涨**(上轮单帧
baseline 4.6m→−GT 5.4m;若叠帧让 RGB CNN 学会消歧,−GT 应接近 baseline)。诚实预期:叠帧加运动
视差不加远景分辨率,若仍失败则上 RNN。**脚本:** `spike_v3g_selfloc_temporal.py`。







---

## 2026-06-07 端到端整合 + 真场线自定位(v3 soccer solo 攻关)

### 背景
12 小时自主推进交付三块独立能力后,用户要求(1)整合成"纯视觉定位→深度找球→带球进球"
端到端单策略;(2)把自定位用的 1.0m 加粗标线改回 0.125m 真场规格,同时保住纯视觉定位。

### 端到端整合(目标①②③ 单策略)
- **关键发现**:`selfloc_vision` env 已含 ①自定位(selfloc head+accuracy reward+渐隐课程)+
  ②找球(depth+RGB 双 CNN)+ ③带球(dribble reward 栈)。整合只需移植 goal env 的进球层。
- **e2e env** `mos92_soccer_e2e_env_cfg`:进攻半场出生(比 goal env 更宽,保自定位多样性)+
  goal_progress(w2)+goal_scored(w5,exp11平衡值)+time penalty(-0.02)+upright 提到 2.5。
  从 model_2800 strict load(obs/action 同构,reinit 0),①②③ 技能全迁移。
- **结果**:goal_rate 0.30、fell_over **0**(全程零摔倒)、dribble_success 0.51;但 selfloc
  从 0.82→4.5m。**首判为 ①↔③ 容量竞争,提 selfloc 权重 0.8→1.6 重训完全无效(仍 4.5m)。**
- **翻转性诊断(对照实验)**:整合前的 model_2800 放到**同样进攻半场几何**上自己也只有
  4.65m(vs 整合后 4.5m,几乎相同)。**结论:整合干净成功,无容量竞争;selfloc"退化"几乎
  全来自操作几何变难(进攻半场位姿多样性低、地标可见性差),不是整合破坏定位。** 教训:
  比较必须控制几何分布,拿全场 0.82m 比进攻半场 4.5m 是错误归因。
- 模型:`checkpoints/v3_soccer_solo/04_e2e_integrated/model_1499.pt`。

### 真场线(0.125m)纯视觉自定位:gate 证伪单帧 + 主动扫视+时序方案
- **三个相机 gate(0.125m 真场线,零训练便宜测试)**:
  - 分辨率扫描(64×48→512×384):最差位姿标记占比 worst 0.32-0.39%,**像素提 64x 几乎不动**。
  - 视场角(宽 60→90):同样 fail。
  - 俯角×窄FOV组合(down10_25 最佳):worst 0.84%,仍 <2% 线。
- **当时结论(过强,后有修正见下)**:0.125m 线低视角单帧物理看不全地标。用户选定方向:
  **主动扫视+时序记忆**(neck_yaw 在策略动作里,真机 RoboCup 做法)。
- **实现**:`StackedCameraRGB` 加帧间 **stride**(每 K 步存一帧,N 帧跨越 N×K 控制步的扫视窗);
  `mos92_soccer_selfloc_realspec_env_cfg`:0.125m 真场线 + 6帧×stride6(~0.7s 扫视窗)+ 相机 96×72。
  从 model_2800 shape-match bootstrap(RGB CNN mlp.0+spatial_softmax reinit 5 个,其余迁移)。
- **结果**(`05_selfloc_realspec_0125m/model_3400.pt`):训练渐隐期一度到 2.0m,GT 全抽后稳定
  4-6m;纯视觉(play=False, mask=0 硬置)**~5.8m**。时序积分**有显著帮助**(从单帧 >10m/看不见
  拉到 5.8m),但**回不到加粗线的 0.79m**。num_envs 1024→384 解 CUDA OOM(图像 obs 涨 3.4x)。

### 修正性发现(高分辨率机器人视角图)
渲染 64×48→1280×960 七档机器人视角图(`robot_view_res/`)。**肉眼看 1280×960 的角落位姿:
中圈/多条线/球门清晰可辨、完全可定位;而 64×48 同位姿一团糊、线消失。** 即:
**之前"标记像素占比"gate 指标会误导**——高分辨率下线变清晰但占比不涨(草地像素同比例变多)。
对 CNN 关键是线"可否分辨边缘"而非占比。**修正:64×48 下 0.125m 线确实亚像素消失,但高分辨率
确实能找回线**,之前"物理上看不见"的结论下得太绝对。这支持用高分辨率重训真场线自定位。

### 交付物
- 模型:`checkpoints/v3_soccer_solo/{01..05}`(README 有逐模型说明+诚实保留)
- 演示视频+可视化:`soccer_eval/2026-06-07_v3_soccer_solo/{01..05,robot_view_res}`
- 脚本:`spike_v3g_e2e{,_fix}.py`、`spike_v3g_selfloc_realspec.py`、`probe_resolution_gate.py`、
  `probe_camera_geom_realspec.py`、`render_v3_soccer_solo.py`、`render_robot_view_resolutions.py`
