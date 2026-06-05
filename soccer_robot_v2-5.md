# Soccer Robot v2.5 — v3 前置验证(Feasibility Spikes)

> 状态:**规划文档**。在 v2-core 训练完成后、v3 正式启动前执行。
> 目标:用最小成本验证 v3 方案中**不确定性最高的几个假设**,
> 避免在错误前提上投入 2-3 周。每个 spike 独立,失败不阻塞其他 spike。

---

## 0. 为什么需要 v2.5

v3 方案有多个**未验证的关键假设**,任何一个为假都会导致 v3 大改或失败:

| 假设 | 风险 | 如果为假的后果 |
|------|------|---------------|
| G1 能学会转头找球且不摔倒 | waist 耦合躯干,转头破坏平衡 | learned gaze 不可行,v3 核心路径需重设计 |
| 深度相机能在训练中看到球 | 低分辨率+球太小+遮挡 | vision policy 无信号,Phase B 白费 |
| 规则 penalty 不会杀死进球能力 | 密集 penalty vs 稀疏 goal reward | 策略学会"不碰球最安全" |
| 策略能泛化到全场尺度 | v2 训练分布 <6m,实际需 9m+10.5m | 全场任务完全失败,需要任务重构 |

v2.5 **不训练完整策略**,只做 3-6 个短实验(每个 <1 天),产出是 go/no-go 决策。

---

## 1. Spike A: Gaze 可行性(决定 G1 上的 v3 核心路径)

### 目的

验证:策略能否学会通过 waist_yaw/pitch 转动相机朝向球,且不破坏行走平衡?

**重要前提:** 本 spike 仅对 G1 有效。MOS92 无 waist joints,因此如果 Spike G
确认 v3 最终在 MOS92 上执行,Spike A 的结果仅作为参考(验证"主动转头"概念是否
在某种硬件上可行),不直接指导 v3 实施。

**决策逻辑:**
- 如果 Spike G 确认 MOS92 neck 可改 revolute → Spike A 的方法论可迁移到 neck joints
- 如果 Spike G 确认 MOS92 neck 不可改 → Spike A 结果无实际用途,v3 走固定前视方案
- 如果 Spike G 失败(MOS92 无法行走)→ v3 暂在 G1 上执行,Spike A 结果直接有效

### 为 v3 提供的参考

- v3 §5.4 主动视线控制:方案 A(启发式) vs 方案 B(learned gaze) 的选择
- v3 Phase B 的时间估算和可行性
- waist joints 的 pose reward 该保留多少

### 实验设计

**环境:** 沿用 v2 soccer env,不加视觉/规则,GT 观测。
**改动(~30 行):**

1. 把 waist_yaw/pitch 的 pose reward std 从 0.3/0.2 放宽到 1.0/0.5
   (允许转头,但仍轻微约束防止极端姿态)
2. 加一个 `gaze_at_ball` reward:

```python
def gaze_at_ball(env, asset_cfg, command_name) -> Tensor:
    """奖励头部(torso_link 前方)朝向球。"""
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    # 球在机器人 body frame 中的向量(用 robot_to_ball 的逻辑)
    vec_w = cmd.ball_pos_w - robot.data.root_link_pos_w
    ball_pos_b = quat_apply(quat_inv(robot.data.root_link_quat_w), vec_w)
    # 球方位角(body frame 中)
    ball_bearing = torch.atan2(ball_pos_b[:, 1], ball_pos_b[:, 0])
    # 当前 waist_yaw 关节角度
    waist_yaw = robot.data.joint_pos[:, waist_yaw_idx]
    # 奖励 waist_yaw 接近 ball_bearing
    angle_error = torch.abs(waist_yaw - ball_bearing)
    return torch.exp(-2.0 * angle_error)
```

3. 挂 reward:`gaze_at_ball` weight=1.0

**训练:** 从 v2 checkpoint fine-tune,500-1000 iter,8192 envs。

### 观察指标

| 指标 | 成功标准 | 失败标准 |
|------|---------|---------|
| fall_rate | < 10% | > 30% |
| gaze_angle_error (mean) | < 30° | > 60° |
| target_success | > 30%(不严重退化) | < 15%(转头导致不会走) |
| waist_yaw 动作幅度 | 有明显转动(std > 0.2 rad) | 几乎不动(std < 0.05) |

### 决策

| 结果 | v3 路径 |
|------|---------|
| 成功:能转头且不摔 | v3 走 learned gaze(方案 B) |
| 部分成功:能转但退化严重 | v3 用 teacher warmup 更长,或限制 yaw 范围 |
| 失败:转头就摔 | v3 走启发式 gaze(方案 A 作为最终方案),或探索固定 waist_roll + 只学 yaw |

**实验结果 (2026-06-01):**

| 指标 | 起始(step 5000) | 结束(step 5498) | 判定 |
|------|----------------|----------------|------|
| fell_over | 0% | **73.9%** | 严重退化 |
| episode_success | 0% | 91.3% | 未摔时仍能完成 |
| gaze_at_ball reward | 0.001 | 0.562 | 学会了转头 |

**结论: 部分成功** — 策略确实学会了通过 waist_yaw 转向球,但 fall_rate 从 0% 飙升到 73.9%。
v3 路径: 需要更长的 teacher warmup(>500 iter)、限制 waist_yaw 范围(±45°→±30°)、
或者分阶段先稳定行走再加 gaze reward。也可考虑降低 gaze_at_ball weight 到 0.3。

### 文件改动

| 文件 | 改动 |
|------|------|
| `rewards.py` | 新增 `gaze_at_ball` 函数 |
| `scripts/spike_a_gaze.py` | 训练脚本 |
| checkpoint | `logs/rsl_rl/g1_velocity/2026-06-01_15-54-10_spike_a/model_5498.pt` |

---

## 2. Spike B: 相机视野验证(渲染,不训练)

### 目的

验证:CameraSensorCfg 挂在机器人头部/上身,球在不同距离/角度时,
depth image 里球的像素覆盖、深度值、是否被遮挡。

**平台差异:**
- G1: 挂在 `robot/torso_link`（head_link 不是独立 body）
- MOS92: 挂在 `robot/head`（head 是独立 body,虽然 neck 是 fixed joint）
- 如果 Spike G 已完成,应同时测试两个平台的相机视野

### 为 v3 提供的参考

- v3 §6 (5.1-5.4):相机参数(分辨率/fov/俯仰角)的最终选择
- v3 Phase 0 的门槛 4(camera smoke test)
- 是否需要 RGB-depth 而非 depth-only
- 是否需要提高分辨率(64×48 → 96×72)

### 实验设计

**环境:** v2 soccer env + CameraSensorCfg,1-4 envs,不训练。
**改动(~20 行 env_cfgs + ~50 行测试脚本):**

1. 加 CameraSensorCfg:
```python
head_cam = CameraSensorCfg(
    name="head_depth",
    parent_body="robot/torso_link",
    pos=(0.05, 0.0, 0.35),   # torso_link 上方 35cm(头部位置)
    quat=(...),                # 朝前下方 20°
    width=64, height=48,
    data_types=("depth", "segmentation"),
    use_textures=False,
    use_shadows=False,
)
```

2. 写测试脚本:固定机器人站立,把球放在不同位置,截取 depth + seg image:
   - 距离:{0.5, 1.0, 2.0, 3.0, 5.0} m
   - 方位:{正前方, 左 30°, 左 60°, 右 30°}
   - waist_yaw:{0°, 30°, 60°}(验证转头是否改变视野)

3. 对每个位置记录:
   - 球在 depth image 中的像素数
   - 球的 depth 值 vs 背景
   - segmentation 中球的 mask 面积
   - 是否被机器人自身遮挡(腿/手臂)

### 产出

**实验已完成 (2026-06-01)**

相机配置: `parent_body=robot/torso_link`, `pos=(0.15, 0, 0.43)`, pitch 30° down, fovy=60°, 64×48

| 距离 | 角度 0° | 角度 ±30° | 角度 60° | 备注 |
|------|---------|-----------|----------|------|
| 0.5m | 0 px | 0 px | 0 px | 球在脚下,完全不可见 |
| 1.0m | 42 px (1.37%) | 43 px (1.40%) | 0 px | 可检测,信号充足 |
| 2.0m | 14 px (0.46%) | 20 px (0.65%) | 0 px | 可检测,但较小 |
| 3.0m | 6 px (0.20%) | 10 px (0.33%) | 0 px | 边缘可检测(≥4px) |
| 5.0m | 2 px (0.07%) | 4 px (0.13%) | 0 px | 极小,不可靠 |

**关键发现:**
- 水平 FoV 约 ±35°,60° 侧方完全不可见 → gaze 对侧方球至关重要
- 0.5m 不可见 → 近距离带球时需要 proprioception 而非 vision
- 5m 处仅 2-4px → 远距离找球需要更高分辨率或 gaze 扫描
- 自身遮挡问题已通过前移相机(x=0.15)解决

### 决策

| 结果 | v3 影响 |
|------|---------|
| 1-3m depth 可见(6-42px) | **GO**: depth-only 可行,64×48 在 1-3m 够用 |
| 0.5m 不可见 | 近距离带球阶段不依赖 vision,用 proprioception |
| 60° 不可见 | **强化 Spike A 必要性**: gaze 是扩展视野的唯一手段 |
| 5m 仅 2px | 远距离找球需 96×72 或 gaze 扫描,v3 §6 需调整 |

### 文件改动

| 文件 | 改动 |
|------|------|
| `env_cfgs.py` | 加 CameraSensorCfg (~8 行,可用临时配置) |
| `tests/test_camera_visibility.py` | 新建 (~60 行) |

---

## 3. Spike C: 规则 Penalty 数值平衡(短训练)

### 目的

验证:加入规则 penalty 后,策略是否仍然会主动接近球和踢球?
找到 penalty_weight 的安全范围。

### 为 v3 提供的参考

- v3 §2 规则权重的初始值和 curriculum 范围
- v3 坑 6/7 的具体边界在哪
- penalty 是否会 false-positive 杀死合法带球

### v2 关键发现对本 spike 的影响

**v2 的 kick_contact 信号极弱(~0.002)**,策略实际上是用身体/腿部推球而非脚踢球。
这意味着 illegal_body_contact penalty 会直接惩罚策略当前的**主要控球方式**。

因此本 spike 需要额外测试一个变体:**先提高 kick_contact reward 再加 penalty**。

### 必须覆盖的 6 类违规(权威分类 → 为什么违规 → RL exploit)

Spike C 只先验证最易误杀的 2 条(illegal_contact / holding),但 v3 最终要覆盖全部 6 类。
每类都标注**为什么是违规**和 **RL 最爱学的作弊解**,因为 penalty 的真正对手就是这些 exploit:

| # | 违规 | 工程判定 | 为什么违规 | RL 坏策略 | 权重 |
|---|------|---------|-----------|----------|------|
| 1 | 非法身体接触 | `body_contact ∧ ¬foot_contact` | 破坏"踢球"本质;非脚更稳→exploit | 大腿推球、躯干挡球贴走 | -1.0 |
| 2 | 持球 Holding | `dist<0.4 ∧ speed<0.2 ∧ >1.5s` | 足球是动态控制非占有 | 把球卡住不动 | -2.0 |
| 3 | 夹球 Trapping | `foot ∧ body ∧ speed≈0` | **最严重**,彻底破坏任务 | 两脚夹/身体包住球 | **-3.0** |
| 4 | 粘球 Sticking | `contact ∧ speed<0.05 ∧ >1.0s` | 接触但不推进 | 贴球不踢、球抖不前进 | -1.0 |
| 5 | 危险高踢 | `foot_contact ∧ foot_z>阈值` | 多人赛撞对手 | 高抬腿、上踢、失控摆腿 | -1.5 |
| 6 | 冲撞 Charging | `contact ∧ robot_speed>阈值` | 高速撞球非控制 | 全速撞球不减速 | -0.5 |

**Holding(长时间占有,看 dist+时长) vs Sticking(接触但无推进,看 speed≈0)** —— 两者机制不同,判据不同。

> **⚠️ 已观测到的真实 exploit(2026-06-05, v3d):** v3d 跳跃修复策略的行为视频里,机器人反复
> **"骑"在球上(球卡在胯下)** —— 正是规则 2(持球)+ 规则 3(夹球)。证明只要任务只奖励
> "球到目标"而不惩罚占有,RL **必然**学到卡球。这是 Spike C 高 penalty 权重的实证依据,
> 不是预防性猜测。证据:`soccer_eval/2026-06-05_spikes/v3d_jumpfix/`。完整分类见 v3.md §2。

### 实验设计

**环境:** v2 soccer env + body_ball contact sensor + 2-3 个规则 penalty。
**只加最可能出问题的两个规则:**
- `illegal_body_contact`(weight=-1.0)
- `holding_ball`(weight=-2.0, threshold=1.5s)

**不加:** dangerous_kick, ball_stuck(这些不太可能误杀)

**两组实验:**

**C-1: 直接加 penalty(验证最坏情况)**
- 扫描 penalty_weight: {0.1, 0.3, 0.5, 1.0, 2.0}
- 从 v2 checkpoint fine-tune,每个 weight 跑 500 iter

**C-2: 先强化 kick_contact 再加 penalty(验证正确路径)**
- 先把 kick_contact weight 从 0.3 提到 2.0,fine-tune 500 iter(让策略学会用脚）
- 然后在此基础上加 penalty_weight=0.5,再跑 500 iter
- 对比 C-1 中 penalty_weight=0.5 的结果

**C-2 的意义:** 如果 C-1 全崩但 C-2 存活,说明 v3 Stage 1 的正确顺序是
"先强化 kick_contact → 再加 illegal_contact penalty",而非同时加。

### 观察指标

**C-1 表:**

| penalty_weight | target_success | kick_contact_rate | illegal_rate | episode_length | body_push_rate | 判断 |
|---------------|---------------|-------------------|-------------|---------------|---------------|------|
| 0.1 | ? | ? | ? | ? | ? | 太弱? |
| 0.3 | ? | ? | ? | ? | ? | |
| 0.5 | ? | ? | ? | ? | ? | |
| 1.0 | ? | ? | ? | ? | ? | |
| 2.0 | ? | ? | ? | ? | ? | 太强? |

**C-2 表:**

| 阶段 | target_success | kick_contact_rate | illegal_rate | body_push_rate |
|------|---------------|-------------------|-------------|---------------|
| v2 baseline | 99.8% | ~0.2% | N/A | ~高 |
| +kick_contact=2.0 (500 iter) | ? | ? | ? | ? |
| +penalty_weight=0.5 (500 iter) | ? | ? | ? | ? |

**关键监控:**
- `body_push_rate`(新增）= 非脚部位接触球的比例,v2 baseline 应该很高
- `episode_length` 变长 = 策略在回避球(penalty 过强)
- `kick_contact_rate` 下降 = 策略不敢碰球
- `kick_contact_rate` 上升(C-2) = 策略学会用脚替代身体推球
- `illegal_rate` 不下降 = penalty 太弱
- `target_success` 崩塌 = penalty 和 task reward 失衡

### 决策

| 结果 | v3 影响 |
|------|---------|
| C-1 在 0.3-0.5 两者平衡 | v3 Stage 1 从 0.3 开始 |
| C-1 全崩,C-2 存活 | v3 Stage 1 必须先强化 kick_contact(Phase 0.5),再加 penalty |
| C-1 只有 0.1 才不崩,C-2 也崩 | illegal_contact 的 body pattern 需要重新设计(可能腿部推球也应算合法) |
| C-1 在 2.0 也不崩 | v3 可以更激进,Stage 1 直接用 1.0 |
| false-positive 很多 | 需要收紧 body_ball sensor 的 pattern(排除更多 body) |

### 文件改动

| 文件 | 改动 |
|------|------|
| `env_cfgs.py` | 加 body_ball sensor + 2 个 penalty rewards (~25 行) |
| `rewards.py` | 加 `illegal_body_contact` + `holding_ball_penalty` (~25 行) |
| `dribble_command.py` | 加 holding_time 追踪 (~10 行) |

---

## 4. Spike D: 观测延迟耐受度(短训练)

### 目的

验证:策略在多大的观测延迟下仍能踢球?找到 delay 的"悬崖边"。

### 为 v3 提供的参考

- v3 §4.5.1 的 delay_max_lag 目标值是否现实
- v3 Stage 2 的 delay curriculum 需要多长的 ramp
- 是否需要 history_length > 3 或 RNN

### 实验设计

**环境:** v2 soccer env,加 observation delay。
**扫描 delay_max_lag:** {0, 2, 4, 8, 12, 18}(物理步,即 0/10/20/40/60/90 ms)

**重要实现细节:** `DelayBuffer` 必须在 env 初始化时创建（`delay_max_lag>0`）。
因此所有实验都需要用 `delay_max_lag=18`（最大值）初始化 buffer,
然后通过 `set_lags()` 控制实际 active lag。不能在已有 env 上动态加 delay。

**两种方式:**
1. **Zero-shot 测试:** 用 `delay_max_lag=18` 初始化 env,加载 v2 checkpoint,
   设置不同 active lag 跑评估(不重训)。注意：v2 策略从未见过 delayed obs,
   这测试的是策略的即时鲁棒性,不是学习能力。
2. **Fine-tune 测试:** 同样用 `delay_max_lag=18` 初始化,设置目标 active lag,
   从 v2 checkpoint 各训 500 iter,看策略能恢复到什么程度。

**同时测 history_length 的效果:** 对 delay=8 和 delay=18,
分别用 history_length={1, 3, 5} 比较。

### 观察指标

| delay_max_lag | ms | zero-shot success | fine-tune success | fall_rate |
|--------------|-----|-------------------|-------------------|-----------|
| 0 | 0 | baseline | baseline | baseline |
| 2 | 10 | ? | ? | ? |
| 4 | 20 | ? | ? | ? |
| 8 | 40 | ? | ? | ? |
| 12 | 60 | ? | ? | ? |
| 18 | 90 | ? | ? | ? |

### 决策

| 结果 | v3 影响 |
|------|---------|
| delay=8 fine-tune 可恢复到 >80% | v3 用 delay_max_lag=8 作为主目标 |
| delay=18 fine-tune 仍 >50% | v3 §4.5.1 的 18 步目标可行 |
| delay>8 崩塌且 history=5 也救不回 | 需要 RNN/transformer,或降低延迟目标 |
| zero-shot 在 delay=4 就崩 | delay curriculum 必须非常缓慢 |

### 文件改动

| 文件 | 改动 |
|------|------|
| `env_cfgs.py` | 加 obs delay 配置(~5 行,或用 CLI override) |
| `evaluate_soccer.py` | 跑评估即可,无需改 |

---

## 5. Spike E: Goal-Scoring 闭环(v2.8 最小实现)

### 目的

验证:v2 策略能否学会把球踢进球门(不只是到达 target 点)?
同时产出 v3 所需的 v2-goal checkpoint。

### 为 v3 提供的参考

- v3 §0.1b 门槛 1:v2-goal checkpoint 是否存在
- v3 §0.1b 门槛 2:goal evaluator 是否可用
- v3 所有 Stage 的 Goal Rate baseline

### 实验设计

**改动(~60 行):**

1. `DribbleCommand` 的 target 采样改为:50% 随机点 + 50% 球门口方向
2. 新增 `goal_scored` 判定:球 x > half_length 且 |y| < goal_width/2
3. 新增 `goal_scored` reward(weight=10.0,稀疏)
4. 新增 `goal_progress` reward:球朝球门方向的速度投影(dense shaping)
5. evaluate_soccer.py 加 `goal_rate` / `shot_on_goal_rate` 指标

**训练:** 从 Spike F 最佳 checkpoint fine-tune（如果 F 已完成且成功），
否则从 v2 model_4999 fine-tune。2000-3000 iter。

**与 Spike F 的关系:** Spike E 的 goal-scoring 需要全场 approach 能力。
如果 Spike F 成功（F1 或 F3 > 60% success），E 应基于 F 的 checkpoint + reward 结构。
如果 Spike F 未完成或失败，E 退回到 v2 的近距离分布（spawn_dist 0.6-1.5m），
只验证"近距离能否进球"这个更弱的问题。

### 观察指标

| 指标 | 目标 |
|------|------|
| goal_rate | > 10%(非零即成功) |
| shot_on_goal_rate | > 20% |
| target_success(随机点) | 不严重退化(> 30%) |

### 决策

| 结果 | v3 影响 |
|------|---------|
| goal_rate > 10% | v3 §0.1b 门槛 1 通过,可以启动 |
| goal_rate ≈ 0% 但 shot_on_goal > 0 | 需要更多训练或更强的 goal shaping |
| 完全不进球 | v2 物理/奖励有根本问题,v3 不能启动 |

### 文件改动

| 文件 | 改动 |
|------|------|
| `dribble_command.py` | target 采样 + goal_scored 判定 (~30 行) |
| `rewards.py` | goal_progress + goal_scored reward (~20 行) |
| `env_cfgs.py` | 挂 goal rewards (~10 行) |
| `evaluate_soccer.py` | 加 goal metrics (~20 行) |

---

## 6. Spike F: 全场尺度泛化(Approach + Dribble 两阶段)

### 目的

验证：把任务拆成 approach（走向球）+ dribble（带球到目标）两个 phase，
能否让策略泛化到全场尺度（spawn_dist 9m + target_dist 10.5m）？

### 为 v3 提供的参考

- v3 全场任务的 reward 结构设计
- 是否需要 curriculum 逐步扩大距离，还是两阶段 reward 本身就够
- approach 阶段的 reward shaping 是否足够驱动远距离寻球

### 背景

v2 训练分布：spawn_dist=(0.6, 1.5)m，target_dist=(2.0, 6.0)m。
v2 robustness 测试：spawn_dist=(2.5, 4.0)m，target_dist=(6.0, 9.0)m 仍有效。
实际需求：机器人从球场边缘（9m）走到球，再带球 10.5m 到球门。

核心问题：observation 中 robot_to_ball / ball_to_target 向量幅度远超训练分布，
策略从未见过这种 scale 的输入。

**v2 reward 结构的根本限制：**
当前 `dribble_approach` 用 `exp(-dist²/std²)` 且 std=1.0。
在 9m 距离时 reward = exp(-81) ≈ 0 — 完全没有梯度信号。
Spike F 的 approach reward 必须改用**线性递减**或**大 std**才能在远距离提供信号。

### 实验设计

**方案：两阶段 reward + 扩大分布**

Phase 切换条件：`distance_to_ball < 0.8m`

**Approach reward 设计（替代 v2 的 exp 形式）：**
```python
# 方案 A: 距离递减（每步 reward = 上一步距离 - 当前距离）
approach_reward = prev_dist_to_ball - curr_dist_to_ball  # 正=接近,负=远离

# 方案 B: 大 std 的 exp（std=5.0,9m 时 reward=exp(-3.24)≈0.04,仍有信号）
approach_reward = exp(-dist² / 25.0)
```
建议用方案 A（距离递减），因为它在任意距离都有等强度信号，不依赖 std 调参。

| Phase | 活跃 reward | 权重 | 说明 |
|-------|------------|------|------|
| Approach | approach_delta (距离递减) | 2.0 | 每步 reward = -Δdist_to_ball |
| Approach | heading_to_ball (朝向球) | 0.5 | 鼓励面朝球走 |
| Dribble | ball_to_target (原 v2) | 1.0 | 带球到目标 |
| Dribble | kick_contact (原 v2) | 0.3 | 脚触球 |
| 全程 | alive + pose + smooth | 原值 | 基础行走质量 |

**训练分布（3 组对比）：**

| 组 | spawn_dist | target_dist | 说明 |
|----|-----------|-------------|------|
| F1 | (0.6, 4.0) | (2.0, 8.0) | 温和扩大 |
| F2 | (0.6, 9.0) | (2.0, 12.0) | 全场覆盖 |
| F3 | curriculum: 从 F1 → F2 | 按 success_rate > 80% 扩大 | 渐进式 |

**训练：** 从 v2 checkpoint (model_4999) fine-tune，每组 2000 iter。

### 观察指标

| 指标 | F1 目标 | F2 目标 | F3 目标 |
|------|---------|---------|---------|
| success_rate (全分布 eval) | > 70% | > 40% | > 60% |
| success_rate (近距离 eval) | > 90% | > 80% | > 90% |
| approach_time (9m 距离) | < 300 步 | < 300 步 | < 300 步 |
| fall_rate | < 5% | < 10% | < 5% |

**评估时额外测试全场场景：** spawn=(−9,0), ball=(0,0), target=(10.5,0)

**F1 实验结果 (2026-06-01):**

| 指标 | 起始 | 结束(2000 iter) | 判定 |
|------|------|----------------|------|
| episode_success | 0% | **67.4%** | 接近目标(70%) |
| ball_to_target_error | 4.43m | 1.91m | 显著改善 |
| robot_to_ball_error | 2.56m | 1.00m | 学会走向球 |
| heading_to_ball | 0.001 | 0.235 | 学会朝向球 |
| fell_over | 2.17 | 2.29 | 稳定 |

F1 结论：两阶段 reward 结构有效，67.4% 接近 70% 目标。
approach_delta 数值很小（~0.0002/step）但方向正确。
checkpoint: `logs/rsl_rl/g1_velocity/2026-06-01_16-07-40_spike_f_f1/model_6998.pt`

### 决策

| 结果 | v3 影响 |
|------|---------|
| F2 直接 >40% success | 两阶段 reward 足够，v3 不需要 distance curriculum |
| F1 好但 F2 崩，F3 恢复 | v3 需要 distance curriculum，但两阶段 reward 结构正确 |
| 三组都 <20% | 需要更根本的改动：obs normalization / 子策略切换 / HRL |
| approach 阶段 fall_rate 高 | 远距离行走本身有问题，需先强化 locomotion |

### 文件改动

| 文件 | 改动 |
|------|------|
| `rewards.py` | 新增 approach_ball + heading_to_ball reward (~30 行) |
| `env_cfgs.py` | 两阶段 reward 配置 + phase 切换逻辑 (~40 行) |
| `dribble_command.py` | 扩大 spawn/target 分布 (~10 行) |
| `evaluate_soccer.py` | 加 approach_time 指标 + 全场场景 eval (~20 行) |

---

## 6b. Spike G: MOS92 机器人适配与验证

### 目的

将训练平台从 Unitree G1 切换到实际目标机器人 MOS92,验证：
1. MOS92 能否在 mjlab 中稳定行走
2. MOS92 能否完成 v2 级别的带球任务
3. 确定 MOS92 特有的约束对 v3 方案的影响

### 为 v3 提供的参考

- v3 所有 stage 的实际执行平台确认
- gaze 方案的根本性重设计（MOS92 无 waist joints,neck 是 fixed）
- 执行器模型（力矩控制 vs 位置控制）对 DR/delay 的影响
- 更小体型对踢球动力学的影响

### MOS92 vs G1 关键差异

| 特征 | G1 (当前训练平台) | MOS92 (目标机器人) |
|------|-------------------|-------------------|
| 总质量 | ~35kg | ~16.5kg |
| 站高 | 0.78m | 0.45m |
| 自由度 | 29 (含 waist×3, wrist×4) | 20 (含 neck×2, 无 waist/wrist) |
| 头部控制 | 无独立关节(挂在 torso_link) | **neck_yaw + neck_pitch** (revolute, 已确认) |
| 腰部 | waist_yaw/pitch/roll (可转头) | **无**(转头由 neck 负责) |
| 手臂 | shoulder_pitch/roll/yaw + elbow + wrist×2 | shoulder_pitch/roll + elbow |
| 腿部 | hip_pitch/roll/yaw + knee + ankle_pitch/roll | 同(6 DOF/leg) |
| 执行器类型 | PD 位置控制 (kp/kd) | 力矩电机 (ctrlrange ±36/60 Nm) |
| 脚碰撞体 | 7 geoms/foot | 3 geoms/foot (cylinder×3) |
| 参考文件 | `asset_zoo/robots/unitree_g1/` | `docs/robot_param/MOS92_urdf_0517_v3_simplified.xml` |

### 对 v3 方案的根本性影响

**1. Gaze 方案已确定(2026-06-01 更新):**
- MOS92 neck 已确认可转动,xml 中加入 neck_yaw(±90°) + neck_pitch(±28.6°)
- v3 gaze 方案:**策略输出 neck_yaw/pitch 主动转头找球**
- 优势:转头不影响躯干重心/行走稳定性(vs G1 waist 方案会破坏平衡)
- Spike A-MOS92 将验证这一方案

**2. 执行器模型不同:**
- G1：PD 位置控制,action = target joint position (`JointPositionActionCfg`)
- MOS92：力矩电机,action = torque (`JointEffortActionCfg` + `BuiltinMotorActuatorCfg`)
- mjlab 已有两种 action type,不需要新开发框架代码
- 但力矩控制的 action scale、reward tuning、训练稳定性与位置控制差异大
- 可选方案：在 MOS92 上也用 PD 位置控制（mjlab 的 `BuiltinPositionActuatorCfg`
  会自动加 PD servo），只是 kp/kd 需要根据 MOS92 电机参数重新计算

**3. 体型差异影响踢球动力学:**
- MOS92 站高 0.45m vs G1 0.78m → 脚到球的相对高度不同
- MOS92 总质量 16.5kg vs G1 ~35kg → 踢球力量/球速不同
- **球尺寸是 RoboCup 标准（直径 22cm, 0.43kg），不能改小**
  - 球顶高度 0.22m ≈ MOS92 站高的 49%（vs G1 的 28%）
  - 几何验证：MOS92 脚顶面约 0.022m 高,球心 0.11m → 脚接触球下半球,可以踢
  - 但球对前方视线的遮挡比 G1 严重得多（球占视野比例更大）
  - 踢球时腿需要抬得更高才能越过球心 → dangerous_kick 阈值可能需要调整
- Spike G-2 需要验证：MOS92 能否有效控球,还是只能"推球"

**4. Spike A (Gaze) 在 MOS92 上需要重新验证:**
- G1 Spike A 测试的是 waist_yaw/pitch 转头 → MOS92 用 neck_yaw/pitch
- MOS92 方案理论上更优(转头不影响躯干),但需要实验确认
- 预期:即使 gaze weight=1.0 也不应摔倒(neck 质量小,不影响重心)

### 实验设计

**Step G-1: MOS92 资产导入 + 基础行走（~1 天）**

1. 将 MOS92 xml 转为 mjlab asset_zoo 格式:
   - 创建 `src/mjlab/asset_zoo/robots/mos92/` 目录
   - 定义 actuator groups + constants（参考 G1 结构）
   - 处理力矩控制 vs 位置控制的差异
2. 在 flat terrain 上训练基础行走:
   - 复用 velocity task 配置,调整 action scale/pose reward
   - 目标：稳定行走 + 转向,fall_rate < 5%
3. 验证 keyframe/init pose 正确性

**Step G-2: MOS92 带球验证（~1 天）**

1. 将 soccer env 适配到 MOS92:
   - 调整 foot_ball_contact sensor 的 body pattern
   - 调整 spawn_dist/target_dist（可能需要缩小,因为机器人更小）
   - 调整 ball_radius 或保持不变（验证比例是否合理）
2. 从 scratch 训练 dribble（不能用 G1 checkpoint）:
   - 2000-3000 iter,观察是否收敛
   - 对比 G1 的学习曲线

**Step G-3: Gaze 验证（已合并到 Spike A-MOS92）**

1. ~~确认 MOS92 neck 是否可以改为 revolute~~ → **已确认,neck_yaw/pitch 已加入 xml**
2. Gaze 验证合并到 Spike A-MOS92 重跑中(见 §7b)
3. 验证 neck_yaw/pitch 作为 gaze action 是否影响行走稳定性

### 观察指标

| 指标 | G-1 目标 | G-2 目标 |
|------|---------|---------|
| 行走 fall_rate | < 5% | < 5% |
| 行走 tracking_error | < 0.3 m/s | N/A |
| dribble success_rate | N/A | > 50%（2000 iter） |
| kick_contact_rate | N/A | > v2 baseline（~0.2%） |
| 训练收敛速度 | ~500 iter 学会走 | ~1500 iter 学会带球 |

### 决策

| 结果 | v3 影响 |
|------|---------|
| G-1 行走成功 | MOS92 可作为 v3 训练平台 |
| G-1 行走失败 | 需要调试 actuator model 或 xml 参数,阻塞后续 |
| G-2 带球成功 | v3 直接在 MOS92 上执行 |
| G-2 带球成功但 kick_contact 更弱 | MOS92 脚型可能不适合踢球,需要调整 foot collision |
| G-3 neck 可改 revolute | v3 gaze 方案用 neck joints（比 G1 waist 更合理） | **已确认 ✓** |
| G-3 neck 不可改 | v3 gaze 只能靠整体转身 + 固定前视相机,大幅简化 | N/A |

### 文件改动

| 文件 | 改动 |
|------|------|
| `asset_zoo/robots/mos92/` | 新建目录：xml, constants, __init__ (~200 行) |
| `asset_zoo/robots/mos92/mos92_constants.py` | actuator groups, keyframes (~100 行) |
| `config/mos92/env_cfgs.py` | MOS92 velocity + soccer env cfg (~150 行) |
| `docs/robot_param/` | 参考源文件 |

---

## 7. 执行顺序和时间

> **2026-06-01 更新:** Spike G-1 已通过,确认 MOS92 neck 可改为 revolute。
> 后续所有 spike 统一在 MOS92 平台上执行。详见 §7b。

```
Day 1 AM: Spike B（相机验证,不训练,1-2 小时）— G1 ✓ 已完成
Day 1 PM: Spike A（gaze 可行性,500 iter,2-3 小时）— G1 ✓ 已完成
Day 1 PM: Spike G Step 1（MOS92 资产导入,并行）— ✓ 已完成
Day 2 AM: Spike F（全场尺度,3 组×2000 iter）— G1 ✓ 已完成
Day 2 AM: Spike G Step 2（MOS92 行走训练,并行）
Day 2 PM: Spike C（penalty 数值扫描,5×500 iter 并行）
Day 2-3:  Spike D（delay 耐受度扫描）
Day 3:    Spike E（goal-scoring,基于 F 的全场 approach reward）
Day 3:    Spike G Step 3（MOS92 带球 + gaze 确认）
Day 3-4:  汇总结果,写入 stage_eval.md,决定 v3 路径
```

**依赖关系:**
- Spike E 依赖 Spike F 的 approach reward 设计（全场尺度下才能有效射门）
- Spike G Step 2/3 依赖 Step 1（资产导入）
- Spike A 的价值取决于 Spike G 的结果（MOS92 是否有 gaze joints）
- 其余 spike 互相独立

---

## 7b. MOS92 平台迁移后的更新计划 (2026-06-01)

### 背景

Spike G-1 已通过:MOS92 在 mjlab 中稳定行走(fell_over=0, mean_reward=83.2)。
同时确认:**MOS92 头部硬件支持 neck_yaw + neck_pitch 转动**,xml 中已加入关节。

这意味着:
- v3 目标平台确认为 MOS92(20-DOF: 18 原有 + 2 neck)
- Gaze 方案:用 **neck_yaw/pitch** 主动转头(优于 G1 的 waist 方案,转头不影响躯干)
- 后续所有 spike 统一在 MOS92 上执行,G1 结果仅作历史参考

### 各 Spike 受影响分析与重跑方案

#### Spike A (Gaze) → 必须重跑 ★★★

**原因:** 核心机制完全改变。G1 用 waist_yaw/pitch(转躯干)→ MOS92 用
neck_yaw/pitch(只转头)。MOS92 方案理论上更好:转头不影响平衡和行走。

**MOS92 版实验设计:**
- 基线: MOS92 velocity checkpoint(G-1 完成后)
- 在 velocity task 上加 `gaze_at_ball` reward,控制 neck_yaw/pitch
- neck joints 作为 action space 的一部分(已在 env_cfgs 中)
- pose reward 中 neck_yaw/pitch 的 std 控制转头幅度约束

**实验组:**
| 组 | gaze weight | neck_yaw pose std | neck_pitch pose std | 预期 |
|----|------------|-------------------|--------------------|----|
| A-M1 | 0.3 | 0.3 | 0.2 | 保守,对标 G1 A-2 |
| A-M2 | 1.0 | 0.5 | 0.3 | 激进,G1 版崩溃了但 MOS92 应该稳 |
| A-M3 | 1.0 | 1.0 | 0.5 | 极限,验证 neck 转头是否真的不影响平衡 |

**关键假设:** MOS92 用 neck 转头不影响躯干重心 → 即使 weight=1.0 也不应该摔倒。
如果 A-M2/M3 也不摔,说明 neck gaze 方案本质优于 G1 waist gaze。

**验收:** fall_fraction < 5% + gaze reward > 0.3 即通过。

#### Spike B (相机视野) → 需要重跑 ★★

**原因:** MOS92 身高 0.45m vs G1 0.78m,相机挂载位置完全不同。
且 MOS92 neck 可动 → 相机视野随 neck 姿态变化,需要重新评估。

**MOS92 版实验设计:**
- 相机挂载在 `head` body 上(随 neck 转动)
- 测试: neck_yaw=0/±30°/±60° × neck_pitch=0/±15° 的视野覆盖
- 球在 0.5/1/2/3/5m 距离的像素覆盖
- pos/quat 需要根据 MOS92 头部几何重新设计

**关键差异:**
- MOS92 相机更矮(~0.45+0.24=0.69m head top vs G1 ~1.0m)
- 但 MOS92 neck 可独立转动 → 主动 gaze 覆盖范围更大
- 球在视野中的比例更大(因为离球更近)

#### Spike C (Penalty 平衡) → 直接在 MOS92 上做 ★

**原因:** C 尚未执行。直接在 MOS92 上做,不需要先在 G1 上跑。

**改动:** 从 MOS92 soccer checkpoint(G-2 完成后)出发,扫描 penalty 权重。
kick_contact 的 foot body pattern 需要改为 MOS92 的 `Rfoot/Lfoot`。

#### Spike D (延迟耐受度) → 直接在 MOS92 上做 ★

**原因:** D 尚未执行。MOS92 电机延迟特性可能与 G1 不同,直接测 MOS92。

#### Spike E (Goal-Scoring) → 直接在 MOS92 上做 ★★

**原因:** E 尚未执行。MOS92 体型更小,踢球动力学不同,需要验证能否进球。
依赖 G-2(MOS92 带球)通过后执行。

#### Spike F (全场尺度) → 不重跑 (结论平台无关)

**原因:** F 验证的是 reward 结构(approach_delta + heading_to_ball)在远距离下
是否衰减。结论"需要 distance curriculum"是数学性质,与机器人平台无关。
MOS92 的全场任务直接继承 F 的 reward 设计。

#### Spike G (MOS92 适配) → 继续执行 G-2/G-3

G-1 已通过。下一步:
- **G-2:** MOS92 带球(soccer field + dribble command)
- **G-3:** 不再需要"确认 neck 可否改 revolute"(已确认),
  改为: Spike A-MOS92 gaze 验证(合并到 A 重跑中)

### 更新后的执行顺序

```
[已完成] Spike B-G1, A-G1, F-G1, G-1 — G1 平台的历史验证
[已完成] G-1 (MOS92 行走) ✓
[已完成] Spike A-MOS92 (gaze with neck joints) ✓ — 最优 weight=1.0+tight neck_std
[已完成] G-2 (MOS92 带球) ✓ — kick_contact=0.82, dribble=4.14
[已完成] Spike E-MOS92 (goal-scoring) ✓ — goal_rate 峰值 24%, v3 门槛通过
         (Spike B-MOS92 跳过: G1 相机几何结论可迁移,见 v2_experiment.md)
[已完成] Spike A2-MOS92 (视野中心化 + 搜索→追踪→踢 时序) ✓ — 时序涌现验证通过
         搜索成功率 99.8%,接近期(>1m)可见率 78.6%;发现「足下盲区」几何约束
[之后]   Spike C-MOS92 (penalty 扫描)
[之后]   Spike D-MOS92 (delay 耐受度)
```

**Critical path 已打通:** A-MOS92 ✓ > G-2 ✓ > E-MOS92 ✓ > A2-MOS92 ✓ → **v3 可启动**。
A2-MOS92 已验证 v3 §Stage3 的"视野中心化 gaze + 搜索/追踪时序"设计可从 reward
gating 涌现(详见 soccer_robot_v3.md "完整行为时序"章节),并修正了 gaze_center
的阶段门控(足下盲区 → KICK 期置 0,改用脚部/本体感知)。为 Stage 3 vision 铺路。
C/D 是参数调优,不阻塞 v3 启动。

**总计 3-4 天,产出:**
- 7 个 go/no-go 决策
- v2-goal checkpoint（如果 Spike E 成功）
- MOS92 基础行走 + 带球 checkpoint（如果 Spike G 成功）
- v3 的具体参数选择（delay 范围、penalty 范围、相机参数、gaze 路径）
- v3 全场任务的 reward 结构方案（Spike F）
- v3 的目标平台确认 + gaze 方案最终选择（Spike G）

**Critical path:** Spike E（goal-scoring）是 v3 启动的硬门槛。如果 E 失败
（goal_rate=0），v3 不能启动,需要先解决 v2 级别的进球问题。

---

## 8. 产出汇总表(填入 stage_eval.md)

| Spike | 问题 | G1 结果 | MOS92 计划 | v3 决策 |
|-------|------|---------|-----------|---------|
| A: Gaze | 能学会转头? | ✓ weight≤0.3 可行 | **重跑**: neck_yaw/pitch,预期更好 | neck gaze 方案 |
| B: Camera | 球在 depth 里可见? | ✓ 1-3m 可检测 | **重跑**: 头部更矮+neck 可动 | 分辨率/fov 重新确认 |
| C: Penalty | 规则会杀死进球? | 未做 | 直接在 MOS92 上做 | penalty_weight 起始值 |
| D: Delay | 能忍受多少延迟? | 未做 | 直接在 MOS92 上做 | delay_max_lag 目标 |
| E: Goal | 能学会进球? | 未做 | 直接在 MOS92 上做 | v3 是否可启动 |
| F: Full-field | 两阶段能泛化全场? | ✓ 4m行/9m崩 | 不重跑(结论平台无关) | 需 distance curriculum |
| G: MOS92 | 目标机器人能走+带球? | N/A | G-1✓ / G-2 待做 | MOS92 确认为 v3 平台 |

---

## 9. 关键原则

1. **每个 spike 独立失败不阻塞其他。** Spike A 失败不影响 Spike B-E。
2. **不追求完美,追求信息。** 500 iter 不会收敛到最优,但足够看趋势。
3. **失败是有价值的。** "learned gaze 不可行"比"训了 2 周发现不行"好得多。
4. **spike 代码不需要 production quality。** 可以硬编码、可以用临时配置、
   可以不过 make check。目的是快速获取信息,不是交付代码。
5. **所有 spike 的观察数据都记录到 stage_eval.md。** 包括失败的,
   供 v3 实施时参考。
6. **v2 分析是 baseline,不是假设。** Spike C/F 的实验设计基于 v2 的实际行为
   (身体推球、exp reward 在远距离无信号),而非理想化假设。

---

## 10. v2 分析驱动的修正摘要

| v2 发现 | 影响的 spike/v3 部分 | 修正 |
|---------|---------------------|------|
| kick_contact ~0.002,策略用身体推球 | Spike C, v3 Stage 1 | Spike C 加 C-2 组(先强化 kick 再加 penalty); v3 Stage 1 分 1a/1b |
| dribble_approach exp(-d²/1.0) 在 >3m 无信号 | Spike F | approach reward 改用距离递减(Δdist)而非 exp |
| locomotion 对物理鲁棒,控球对物理敏感 | v3 Stage 2 DR | DR 重点应在球物理(摩擦/弹性),而非机器人质量 |
| 1400 iter 即收敛核心能力 | 所有 spike | spike 的 500-2000 iter 预算是充足的 |
| v2 策略泛化到 (2.5,4.0)/(6.0,9.0) 仍有效 | Spike F | F1(温和扩大)大概率成功,F2(全场)才是真正的考验 |
| 目标机器人 MOS92 与 G1 差异巨大 | Spike G, v3 全局 | 新增 Spike G; v3 gaze/actuator/体型全部需要平台适配 |
