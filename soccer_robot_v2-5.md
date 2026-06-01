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

### 文件改动

| 文件 | 改动 |
|------|------|
| `env_cfgs.py` | 放宽 waist pose std,加 gaze_at_ball reward (~10 行) |
| `rewards.py` | 新增 `gaze_at_ball` 函数 (~15 行) |
| `mdp/__init__.py` | 导出 (~1 行) |

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

一个截图表格 + 结论:

| 距离 | 球像素数(depth) | 球 seg 面积 | 可检测? | 备注 |
|------|----------------|-------------|---------|------|
| 0.5m | ? | ? | ? | |
| 1.0m | ? | ? | ? | |
| ... | | | | |

### 决策

| 结果 | v3 影响 |
|------|---------|
| 3m 内 depth 可见(>4px) | depth-only 可行,64×48 够 |
| 3m 内 depth 不可见但 seg 可见 | 需要 segmentation 辅助或 RGB |
| 需要 >3m 找球 | 提升分辨率到 96×72,或加远距离 RGB |
| 自身遮挡严重 | 调整相机位置/俯仰角 |

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
| 自由度 | 29 (含 waist×3, wrist×4) | 18 (无 waist, 无 wrist) |
| 头部控制 | 无独立关节(挂在 torso_link) | neck_yaw + neck_pitch (**fixed** joints) |
| 腰部 | waist_yaw/pitch/roll (可转头) | **无** |
| 手臂 | shoulder_pitch/roll/yaw + elbow + wrist×2 | shoulder_pitch/roll + elbow |
| 腿部 | hip_pitch/roll/yaw + knee + ankle_pitch/roll | 同(6 DOF/leg) |
| 执行器类型 | PD 位置控制 (kp/kd) | 力矩电机 (ctrlrange ±36/60 Nm) |
| 脚碰撞体 | 7 geoms/foot | 3 geoms/foot (cylinder×3) |
| 参考文件 | `asset_zoo/robots/unitree_g1/` | `docs/robot_param/MOS92_urdf_0517_v3_simplified.xml` |

### 对 v3 方案的根本性影响

**1. Gaze 方案必须重设计:**
- G1 方案：用 waist_yaw/pitch 转动 torso 来改变相机朝向
- MOS92 现实：无 waist joints,neck 是 fixed → **无法主动转头**
- 选项 A：相机固定在 head 上,只能靠整体转身改变视野
- 选项 B：将 neck_yaw/pitch 从 fixed 改为 revolute（需硬件确认是否可行）
- 选项 C：用全景相机/多相机替代单目主动 gaze

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

**4. Spike A (Gaze) 对 MOS92 不适用:**
- Spike A 测试的是 waist_yaw/pitch 转头 → MOS92 没有这些关节
- 需要替换为：测试 MOS92 能否通过整体转身找球,或验证 neck 改为 revolute 的可行性

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

**Step G-3: Gaze 方案确认（~0.5 天）**

1. 确认 MOS92 neck 是否可以改为 revolute（需要硬件团队确认）
2. 如果可以：测试 neck_yaw/pitch 作为 gaze action 的可行性
3. 如果不可以：测试"整体转身找球"策略的效果（approach 阶段自然朝向球）

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
| G-3 neck 可改 revolute | v3 gaze 方案用 neck joints（比 G1 waist 更合理） |
| G-3 neck 不可改 | v3 gaze 只能靠整体转身 + 固定前视相机,大幅简化 |

### 文件改动

| 文件 | 改动 |
|------|------|
| `asset_zoo/robots/mos92/` | 新建目录：xml, constants, __init__ (~200 行) |
| `asset_zoo/robots/mos92/mos92_constants.py` | actuator groups, keyframes (~100 行) |
| `config/mos92/env_cfgs.py` | MOS92 velocity + soccer env cfg (~150 行) |
| `docs/robot_param/` | 参考源文件 |

---

## 7. 执行顺序和时间

```
Day 1 AM: Spike B（相机验证,不训练,1-2 小时）
Day 1 PM: Spike A（gaze 可行性,500 iter,2-3 小时）— 仅 G1,视 Spike G 结果决定价值
Day 1 PM: Spike G Step 1（MOS92 资产导入,并行）
Day 2 AM: Spike F（全场尺度,3 组×2000 iter）— 先于 Spike E,因为 E 需要全场能力
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

| Spike | 问题 | 结果 | v3 决策 |
|-------|------|------|---------|
| A: Gaze | G1 能学会转头? | TBD | learned vs heuristic gaze (仅 G1 适用) |
| B: Camera | 球在 depth 里可见? | TBD | 分辨率/fov/是否需要 RGB |
| C: Penalty | 规则会杀死进球? | TBD | penalty_weight 起始值 + kick_contact 前置 |
| D: Delay | 能忍受多少延迟? | TBD | delay_max_lag 目标 + history 需求 |
| E: Goal | 能学会进球? | TBD | v3 是否可启动 |
| F: Full-field | 两阶段能泛化全场? | TBD | reward 结构 + 是否需要 distance curriculum |
| G: MOS92 | 目标机器人能走+带球? | TBD | v3 平台选择 + gaze 方案重设计 |

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
