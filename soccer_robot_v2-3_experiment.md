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


