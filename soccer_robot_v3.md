# Soccer Robot v3 — 感知真实 + 规则约束 + 课程联动

> 状态:**规划文档**。依赖 v2-core 物理验收 + **v2.8 goal-scoring checkpoint**
> 通过后启动;如果当前 v2 只有 target dribble,必须先补 v2.8。
> 目标:在 v2.8 已能完成"逐步控球、推进、射门、进球"的基础上,三轴继续推进 —
> (1) 感知从 GT 退化到视觉,(2) 行为从无约束到符合比赛规则,
> (3) 环境从理想到有噪声/延迟。三者通过统一课程联动,使策略具备
> Sim2Real + 比赛合规,且最终仍以 **Goal Rate(进球率)** 为主指标。

---

## 0. 核心认知(v3 的本质)

```
v3 = 让策略在"看不清 + 不确定 + 有延迟 + 必须守规矩"的世界中仍能进球
```

v2.8 应证明"真实物理下能逐步踢进球门"。但即使 v2.8 做到这一点,
策略仍偏"全知 + 无规则":
- 它知道球的精确坐标(真机只有相机)
- 它可以用身体任何部位碰球(比赛不允许)
- 它可以夹住球不动(比赛犯规)
- 它可以高抬腿踢球(危险动作)

v3 要同时解决三个维度的真实性:

| 维度 | v2 | v3 |
|------|-----|-----|
| 感知 | GT 精确坐标 | 深度相机 + 噪声 + 延迟 |
| 行为约束 | 无规则,怎么碰都行 | 必须用脚、不能夹球、不能高踢 |
| 环境鲁棒 | 理想物理,无扰动 | 执行器延迟 + 质量随机 + 外力推 |

**三者必须课程联动,不能同时加,并且不能丢掉 v2.8 的最终任务:进球。**

```
训练难度 = f(控制难度, 感知难度, 规则约束强度)
```

**主指标定义:**
- `Target Success`:课程早期指标,只证明能把球带到任意目标点。
- `Goal Rate`:最终主指标,球穿过对方球门线且在门柱之间。
- `Shot-on-Goal Rate`:辅助指标,射门轨迹穿过球门口投影。

v3 任一阶段可以暂时用 `Target Success` 做 warmup,但最终验收必须回到
`Goal Rate / Shot-on-Goal Rate / Time-to-Score`。

---

## 0.1 关键发现:mjlab 已有的基础设施(不需要从零构建)

经过代码调研,v3 所需的大部分底层能力 mjlab **已经提供**:

| 能力 | 现状 | 位置 |
|------|------|------|
| GPU 批量相机渲染 | ✅ 已有 `CameraSensorCfg`,RGB/depth/seg,`[B,H,W,C]` | `sensor/camera_sensor.py` |
| 视觉 RL 完整示例 | ✅ yam manipulation 已跑通 depth+RGB policy | `tasks/manipulation/config/yam/` |
| 观测噪声 | ✅ `GaussianNoiseCfg`/`UniformNoiseCfg`,per-term | `utils/noise/` |
| 观测延迟 | ✅ `ObservationTermCfg.delay_min/max_lag` + `DelayBuffer` | `managers/observation_manager.py` |
| 观测历史/堆叠 | ✅ `ObservationTermCfg.history_length` + `CircularBuffer` | 同上 |
| 执行器延迟 | ✅ `ActuatorCfg.delay_min/max_lag` + `DelayBuffer` | `actuator/actuator.py` |
| 课程管理器 | ✅ `CurriculumManager` + `CurriculumTermCfg` | `managers/curriculum_manager.py` |
| 速度课程示例 | ✅ `commands_vel` 按 step 递增命令范围 | `tasks/velocity/mdp/curriculums.py` |
| Domain Rand: 摩擦 | ✅ `geom_friction`/`pair_friction` | `envs/mdp/dr/geom.py`, `pair.py` |
| Domain Rand: 质量/惯量/COM | ✅ `body_mass`/`body_com_offset`/`pseudo_inertia` | `envs/mdp/dr/body.py` |
| Domain Rand: 关节 | ✅ `joint_damping`/`armature`/`friction`/`stiffness`/`limits` | `envs/mdp/dr/joint.py` |
| Domain Rand: PD 增益 | ✅ `pd_gains`/`effort_limits` | `envs/mdp/dr/actuator.py` |
| Domain Rand: 编码器偏差 | ✅ `encoder_bias` | `envs/mdp/dr/joint.py` |
| Domain Rand: 相机内参/位姿 | ✅ `camera.py` dr | `envs/mdp/dr/camera.py` |
| 外力推扰 | ✅ `push_robot` event | velocity task 已用 |
| Raycast 深度替代 | ✅ `RayCastSensorCfg` + `PinholeCameraPatternCfg` | `sensor/raycast_sensor.py` |

**缺失(需新建):**

| 能力 | 状态 | 工作量 |
|------|------|--------|
| 执行器/动作噪声(torque noise) | ❌ 不存在 | ~30 行,加到 ActuatorCfg 或 action term |
| Reward 权重调度(curriculum 改 weight) | ❌ 无现成 term | ~40 行 curriculum term 函数 |
| Motion blur 模拟 | ❌ 需后处理 kernel | v3-core 不做,用 delay 近似 |
| 丢帧模拟 | ⚠️ 可用 delay_hold_prob 近似 | 配置即可,无需新代码 |
| 规则惩罚 reward terms | ❌ 需新建 | ~80 行,纯 reward 函数 |
| 身体↔球接触传感器 | ❌ 需新建(区分合法/非法接触) | ~10 行 cfg |
| 持球/夹球时间追踪 | ❌ 需在 DribbleCommand 中追踪 | ~30 行 |
| Depth dropout + patch dropout | ❌ 需自定义 noise model | ~40 行 |
| 主动视线动作 + 球丢失记忆 | ❌ 需新增 gaze action/head + 搜索奖励/记忆 | ~80 行 |
| Metric-driven curriculum | ❌ 需新 curriculum term | ~40 行 |

**结论:v3 的大量工作是"配置 + 写 reward/curriculum term 函数",不是"造轮子"。
但视觉、gaze、规则代理和 goal-scoring evaluation 都有硬前置验证,不能按
"几行配置"假设必然可行。**

---

## 0.1b v3 启动前硬门槛(避免在错误基础上叠难度)

v3 只有在下面检查通过后才启动训练:

1. **v2-goal checkpoint 已存在**:不是只会 target dribble,而是能在 GT 观测下
   完成"接近球→多次触球推进→射门→进球"闭环。若当前只有 target success,
   v3 不启动,先回到 v2.8 实现 goal-scoring command/reward/evaluator。
2. **Goal evaluator 已实现**:`goal_scored`、`shot_on_goal`、`time_to_score`、
   `touches_before_goal` 能离线评估同一个 checkpoint。
3. **规则代理已映射到规则文本**:每条 penalty 都有"规则依据 → 可测代理 →
   阈值来源/不确定性"记录。没有依据的阈值只能标为 engineering heuristic,
   不能写成比赛规则。
4. **相机/gaze smoke test 已通过**:球在 `{0.5,1,2,3,5}` m 的像素覆盖、
   深度值、视野位置都记录;腰部动作对相机姿态的影响被截图验证。
5. **主动 gaze action 可编译**:确认 waist/gaze 控制能作为策略动作的一部分。
   若当前 actuator/action 按组无法干净隔离 waist joints,必须先改 action term
   或 actuator/action 配置,不能退回长期启发式 gaze。
   **MOS92 注意:** MOS92 无 waist joints。如果 neck 不可改为 revolute,
   此门槛自动降级为"固定前视相机 + 整体转身",Stage 3 的 gaze 目标相应简化。
6. **延迟 buffer 预分配策略明确**:observation/action delay 的最大 buffer
   在初始化时就创建,课程只调实际 lag,不在训练中从 0 动态变成 >0。
7. **目标平台 MOS92 基础能力已验证**:MOS92 在 mjlab 中能稳定行走 + 带球。
   如果 v2.5 Spike G 失败,v3 暂时继续在 G1 上开发,MOS92 适配作为独立工作流。

---

## 0.1c 交付层级与失败回退(避免 Phase B 卡死全项目)

v3 的最终目标是 learned gaze + vision goal scoring,但工程上必须把可交付物分层。
每一层都要能独立保存 checkpoint、评估并写入 `stage_eval.md`:

| 层级 | 内容 | 是否算最终 v3 | 价值 |
|------|------|---------------|------|
| **v3-pre** | 补齐 v2.8 goal-scoring checkpoint + evaluator | 否 | 解决 v3 启动前置 |
| **v3-A minimal** | GT 观测 + 规则代理 + 本体鲁棒 + Goal Rate 验收 | 否,但可独立交付 | 得到"合法且鲁棒"的进球策略 |
| **v3-B diagnostic** | 启发式/teacher gaze + depth/RGB-depth actor | 否 | 诊断视觉输入是否足够,不代表策略会主动找球 |
| **v3-B final** | pure vision actor + teacher-off learned gaze action | 是 | 满足"策略主动转头找球并进球" |
| **v3-C** | v3-B final + 全随机化 + 严格规则 | 比赛候选 | Sim2Real/比赛鲁棒版本 |

**重要取舍:**
- 如果 learned gaze 暂时不收敛,可以产出 `v3-B diagnostic`,但文档和报告必须明确
  它使用了 teacher/heuristic gaze,不能叫最终 v3。
- `v3-A minimal` 不能因为 Phase B 失败而废弃。它是规则合规和鲁棒控制的有效成果。
- `v3-B final` 的验收必须 teacher-off;否则只是视觉感知 smoke test,不是主动视线策略。

---

## 0.2 设计哲学(贯穿 v3)

1. **不在感知困难时加控制困难**:每个 stage 只增加一个维度的难度。
2. **GT 状态永远保留给 critic**:asymmetric actor-critic,actor 看视觉/噪声观测,
   critic 看完美状态。这是 sim2real 的标准做法(参考 yam vision task 已有实现)。
3. **可回退**:每个 stage 的 checkpoint 独立保存,任何 stage 失败可回退到上一个。
4. **不过早加视觉**:Stage 1-2 用 GT 状态训练控制能力,Stage 3 才引入视觉。
5. **课程是配置,不是代码**:用 `CurriculumTermCfg` 函数 + stage 列表,
   不写 if/else 硬编码。
6. **规则用 penalty 引导,不用硬限制**:不 block action,让 RL 自己学到
   "违规不好"。penalty 强度通过 curriculum 从弱到强递增。
7. **penalty ≈ task reward 的 20%~50%**:太大策略不敢动,太小策略学会违规最优。
   通过 curriculum 逐步加强,让策略先学会踢球再学会守规矩。
8. **代理指标不冒充官方规则**:MSL/IFAB 规则文本和仿真可测量指标之间存在
   抽象误差。文档必须保留这个误差,不要把 `0.4m/1.5s/0.4m height`
   之类工程阈值写成官方阈值。
9. **训练指标和最终指标分层**:Target Success 可用于课程,但任何 v3 结论都必须
   同时报 Goal Rate。否则会回退成"会到点,不会进球"。

---

## 1. 四阶段课程设计(Perception × Rules × Robustness 三轴联动)

### 为什么必须联动

| 情况 | 结果 |
|------|------|
| 感知很难 + 规则很严 | RL 完全不收敛(既看不清又不能乱碰) |
| 感知简单 + 无规则 | 学到依赖完美感知的违规策略 |
| 先学控制 → 再加规则 → 最后退化感知 | ✅ 每步只增一个维度难度 |

**正确顺序:**
0. 先补齐/保留 v2.8 的进球闭环(GT + 无规则/弱规则)— v2-goal 已补齐并验收
0.5. 全场尺度泛化(approach+dribble 两阶段 reward,覆盖 9m+10.5m)— v2.5 Spike F 验证
1. 加规则约束(GT + 有规则)— 学会"合法地推进和射门"
   - 1a: 先强化 kick_contact(让策略学会用脚)
   - 1b: 再加 illegal_contact/holding penalty
2. 加本体噪声(GT + 有规则 + 延迟/噪声)— 学会"鲁棒且合法地进球"
3. 换视觉(depth/RGB-depth + 有规则 + 噪声)— 学会"看着踢、守规矩、能进球"
4. 全随机化(视觉 + 严格规则 + 鲁棒扰动)— 比赛候选策略

**v2 分析对顺序的影响:**
- v2 证明 locomotion 对物理变化鲁棒(v1 步态在 v2 物理下仍有效),但控球对物理敏感
- v2 的 reward 结构让策略自然发现"站→走→接近→带球"顺序,但 penalty 可能打断
- 全场尺度(Step 0.5)应在规则之前解决,因为规则 penalty 在远距离 approach 阶段无意义

**全场尺度对 reward 结构的影响（Step 0.5 → 后续所有 Stage 继承）:**
- v2 的 `dribble_approach` 用 `exp(-dist²/1.0)`,在 >3m 时 reward≈0,必须替换
- Step 0.5 确定的 approach reward（距离递减 Δdist 或大 std exp）将成为后续所有
  Stage 的基础 reward 结构,不再使用 v2 的 exp 形式
- `dribble_ball_to_target` 同理：std=2.0 在 10.5m 时 reward=exp(-27.6)≈0,
  也需要改为距离递减或大幅增大 std
- 文件清单中 `rewards.py` 的改动量应包含 approach/to_target reward 重写（+20 行）

---

### Stage 1: 规则引入(GT 状态 + 逐步加规则)

**目标:** 在完美感知下学会"合法地推进和射门",不能只到 target 点。

**v2 关键发现(必须在 Stage 1 前解决):**
v2 策略的 kick_contact 信号极弱(~0.002),实际控球方式是用身体/腿部推球。
这意味着 illegal_body_contact penalty 会直接惩罚策略当前的主要控球方式。
**Stage 1 不能假设策略已经会用脚踢球 — 它需要先学会用脚。**

**正确顺序(Stage 1 内部分两步):**
1. **Step 1a — 强化脚触球(~1000 iter):** 把 kick_contact weight 从 0.3 提到 2.0+,
   让策略从"身体推球"转型为"脚踢球"。此时不加 illegal_contact penalty。
   验收:kick_contact_rate > 30%,target_success 不严重退化。
2. **Step 1b — 加规则 penalty:** 在策略已学会用脚的基础上,逐步加 penalty。
   此时 illegal_contact penalty 惩罚的是"偶尔的身体碰球",而非"主要控球方式"。

**观测:** GT 状态,无噪声,无延迟
**执行器:** 无延迟

**规则 penalty(Step 1b 逐步开启,权重从 0.2 → 1.0):**
```python
rule_penalties = {
    "illegal_contact": -1.0,   # 非脚部位碰球
    "holding_ball": -2.0,      # 控球过久(>1.5s 近距离低速)
    "ball_trapped": -3.0,      # 夹球(多接触点 + 球速≈0)
    "dangerous_kick": -1.5,    # 高抬腿踢球(foot_z > 0.4m)
    "ball_stuck": -1.0,        # 粘球(接触中但球不动)
}
```

**Reward 结构:**
```
total_reward = task_reward + shaping_reward + rule_penalty × penalty_weight
```

**课程:** `penalty_weight` 从 0.2 线性增到 1.0(前 1000 iter 宽松,后面严格)

**验收:**
- **Step 1a:** kick_contact_rate > 30%(从 v2 的 ~0.2% 提升）,target_success > 80%
- **Step 1b:** Goal Rate 不低于 v2-goal 的 70%(允许规则引入带来退化,但不能丢掉进球能力)
- Shot-on-Goal Rate > v2-goal 的 80%
- Target Success 可作为辅助课程指标,但不能作为主验收
- illegal_contact_rate < 10%
- holding_violation_rate < 5%
- 视频确认:用脚触球、多次推进、射门进球,不夹球,不高踢

**产出:** 合规控制 checkpoint

---

### Stage 2: 本体鲁棒(GT + 规则 + 噪声/延迟)

**目标:** 在执行器延迟 + 观测噪声 + 外力推扰下仍能合法控球并完成进球闭环。

**在 Stage 1 基础上叠加:**

**观测噪声:**
```python
joint_pos_noise = GaussianNoiseCfg(operation="add", mean=0.0, std=0.01)
joint_vel_noise = GaussianNoiseCfg(operation="add", mean=0.0, std=0.05)
ball_pos_noise = GaussianNoiseCfg(operation="add", mean=0.0, std=0.05)
ball_vel_noise = GaussianNoiseCfg(operation="add", mean=0.0, std=0.1)
```

**执行器延迟:**
```python
delay_min_lag=1, delay_max_lag=3  # 5-15ms
```

**Domain Randomization:**
```python
events["pseudo_inertia"] = ...   # 质量/惯量/COM 联合扰动
events["pd_gains"] = ...         # kp/kd ±20%
events["joint_friction"] = ...   # 关节摩擦 ±30%
events["push_robot"] = ...       # 外力推扰(已有,加大)
events["ball_friction"] = ...    # 球 rolling friction ±50%(控球敏感,见下）
events["ball_mass"] = ...        # 球质量 ±20%
```

**v2 分析启示:球物理 DR 比机器人 DR 更关键。**
v2 D-vs-B 对比证明:locomotion 对物理变化鲁棒(v1 步态在 v2 物理下 fall_rate 不变),
但控球对球物理高度敏感(v1 控球在 v2 摩擦下从 6.5% → 18.4% success)。
因此 Stage 2 的 DR 重点应放在球摩擦/弹性/质量,而非机器人质量/惯量。
机器人 DR 仍需做(为 Sim2Real),但球 DR 的范围需要更谨慎地扫描。

**RoboCup 标准球参数(当前已正确配置):**
```python
# src/mjlab/terrains/soccer_field.py — SoccerBallCfg
# ✅ 已实现并实测标定(2026-06-05, Task A),数值非占位:
radius = 0.11          # 22cm 直径,RoboCup Humanoid League 标准
mass = 0.43            # kg (规则范围 0.35-0.45)
friction = (0.5, 0.02, 0.02)  # (slide, spin, roll);roll 0.01→0.02 加滚动阻力,球不会无限滑
condim = 6             # 启用 rolling friction
# 弹性 e≈0.35:MuJoCo 无直接恢复系数,靠 solref 产生。正 solref 封顶 e≈0.25(太死),
# 负 solref=(-刚度,-阻尼)才能到 RoboCup 级弹跳。在真实 timestep(0.005s)用
# scripts/calibrate_ball_restitution.py 实测标定:
solref = (-2300.0, -32.0)   # → e≈0.35(0.31-0.36 over drop 0.3-1.5m);改 timestep 需重标定
solimp = (0.95, 0.99, 0.0001, 0.5, 2.0)
# 外观:truncated-icosahedron 纹理(12 红五边形 + 蓝/白六边形,1 块标记用于观测旋转),
# textured=False 默认关闭(当前 obs 是 depth-only,纹理只影响 viewer/RGB,对训练零影响);
# 加 RGB obs 后再开。已渲染验证,见 scripts/proto_ball_texture.py。
```

**Stage 4 球 Domain Randomization(✅ 已实现,Task A,startup 模式逐 env 采样):**
```python
# src/mjlab/tasks/velocity/config/mos92/env_cfgs.py — 4 个 startup EventTermCfg
# 复用 mjlab dr.* 函数,弹性 DR 需要的 dr.geom_solref 是 Task A 新增的:
events["ball_radius"]     = dr.geom_size      # ±5% scale (axis 0)
events["ball_mass"]       = dr.pseudo_inertia # ±15% mass+inertia 一致缩放 (alpha)
events["ball_friction"]   = dr.geom_friction  # slide 0.3-0.7
events["ball_elasticity"] = dr.geom_solref    # 新增函数!stiffness [-9000,-1050] → e∈[0.2,0.6]
# 已实跑验证:逐 env stiffness 真实分散(-8744…-1070),球 30 步稳定。
```

**规则:** penalty_weight = 1.0(完整强度)
**课程:** 噪声/延迟从 0 线性增到目标值

**验收:**
- 在最大噪声/延迟下 Goal Rate 不低于 Stage 1 的 70%
- Shot-on-Goal Rate 不低于 Stage 1 的 75%
- 规则违规率不因噪声显著上升(< 15%)
- Fall rate < 10%

**产出:** 鲁棒合规 checkpoint

---

### Stage 3: 视觉感知(depth + 规则 + 噪声)

**目标:** 用视觉替代 actor 的 GT 球状态,在规则约束下从像素找球、推进、射门。

**架构:** Asymmetric Actor-Critic
```
Actor obs  = [proprioception] + [vision] + [command/goal direction]
Critic obs = [proprioception] + [GT_ball_pos/vel] + [rule_state] + [goal state]
```

**视线控制（2026-06-02 已确认: MOS92 neck 可转动 + gaze 行为已验证）:**
- **MOS92 方案:** 策略输出 neck_yaw/pitch 动作主动找球
  - neck_yaw: ±90°（左右转头）
  - neck_pitch: ±28.6°（上下点头）
  - 优势: 转头不影响躯干重心,不破坏行走稳定性(vs G1 waist 方案)
  - Spike A-MOS92 已验证(2026-06-02): learned gaze 可行,最优配置
    **gaze_weight=1.0 + neck pose std=1.0**。带球时 gaze=0.70、神经偏头
    平均 5.7°/最大 54°、fell_over=0.29。与 G1 结论相反: MOS92 上**紧 neck
    约束**(让头转完自动回中)比降低 weight 更有效。
- **G1 方案(历史参考):** 策略输出 waist_yaw/pitch 动作主动找球（Spike A 验证）

**视野中心化 gaze 设计(v3 新增,2026-06-02):**

目标: 不只是让 neck 朝球的"身体系方位角",而是让球落在**相机图像的中心区域**,
并在球离开视野时主动转头搜索。这是 A-MOS92 gaze reward 的进化版。

A-MOS92 的 `gaze_at_ball` reward 只对齐 `neck_yaw` 和身体系 ball bearing
(2D 水平角),没有用到 neck_pitch,也不保证球真在相机视锥内。v3 改为
**基于相机帧的视野中心化**:

1. **观测:** 把球在相机图像中的归一化坐标 `(u, v) ∈ [-1,1]²` 作为 gaze 状态
   (GT 阶段从 ball_pos 投影到相机内参算出;vision 阶段从 detector 得到)。
   球不可见时 `(u,v)` 置为哨兵值 + `ball_visible=0` flag。

2. **视野中心化 reward(替代/叠加 gaze_at_ball):**
   ```python
   # 球可见时: 奖励球接近图像中心
   center_reward = visible * exp(-(u² + v²) / std_uv²)   # std_uv ~ 0.4
   # 同时用 neck_yaw + neck_pitch 两个自由度对准
   #   u 偏差 → 驱动 neck_yaw, v 偏差 → 驱动 neck_pitch
   ```
   这样 neck_pitch 也被激励(近球时低头、远球时平视),而非 A-MOS92 只用 yaw。

3. **球丢失主动搜索 reward:** 球不可见(visible=0)时,奖励 neck 朝
   "上一次已知球方位 / command goal 方向"扫描,鼓励主动找回球:
   ```python
   search_reward = (1 - visible) * exp(-|neck_yaw - last_known_bearing|)
   ```
   配合 ball memory(保留最近 N 帧 bearing/range)防止"球一出视野就乱转"。

4. **行进中保持球在视野:** 走向球和踢球过程中,`center_reward` 持续生效,
   使策略学会"边走边把球钉在视野中心",而不是只在站定时看球。

**验收补充指标:**
- `ball_in_view_center_rate`: 球落在图像中心 ±40% 区域的时间占比 > 60%
- `ball_visible_rate`: 球在视锥内的时间占比(行进+踢球全程)> 70%
- `lost_ball_recovery_time`: 球丢失后重新进入视野的平均步数 < 30

**实现位置:** `rewards.py` 新增 `gaze_center` + `gaze_search`(~40 行),
观测 `ball_uv` + `ball_visible`(~20 行)。GT 阶段先验证 reward 形状,
vision 阶段接 detector 输出。先在 Spike E 之后、Stage 3 gaze warmup 时做
一个 GT 版 spike(Spike A2-MOS92)快速验证视野中心化 reward 能否收敛。

**完整行为时序: 搜索 → 锁定 → 追踪逼近 → 踢球(v3 新增,2026-06-02):**

期望策略涌现出一个连贯的行为序列(不硬编码状态机,用 reward gating 让
策略自己学出来,符合 §"policy 自主输出搜索动作"原则):

```
[SEARCH]  球不可见 → 只转头扫描; 头到极限仍找不到 → 原地转身(只转 yaw)
   ↓ (ball_visible 0→1)
[LOCK]    球进入视野 → neck 把球钉在图像中心, 观察球的运动方向/速度
   ↓ (持续可见, 距离仍远)
[APPROACH] 朝球移动, 头持续追踪保持球居中, 边走边按球的移动方向调整路线
   ↓ (distance_to_ball < kick_range)
[KICK]    用脚把球踢向目标点/球门
```

**阶段 1 — SEARCH(球不可见, 两级搜索):**
- **第一级(只转头):** 身体和脚保持静止(velocity command ≈ 0),
  只输出 neck_yaw/pitch 扫描。reward 奖励 neck 朝 last_known_bearing
  或未探索方位转动。
- **第二级(原地转身):** neck_yaw 已达极限(±90°)且累计搜索超过 T 步
  仍 ball_visible=0 → 解锁原地 yaw 旋转(只转身、不位移),
  覆盖相机+转头都够不到的正后方 ±55° 死区。
  - 几何依据(A-MOS92/Spike B 实测): neck ±90° + 相机 fov ±35°
    = 可视方位 ±125°, 正后方约 110° 宽的扇区必须靠转身覆盖。
- **冻结实现:** 用 `search_freeze` gate —— ball_visible=0 时把
  locomotion reward 的 tracking 项权重压到 ~0、并惩罚 base 线速度,
  使"找球前不要乱跑";但允许(第二级)角速度,避免死锁。
- **关键: 不位移≠不能转身。** 严格"脚不动"只在第一级;第二级允许原地
  转身是为了打破死区,这是经确认的设计取舍。

**阶段 2 — LOCK(球刚进入视野):**
- ball_visible 0→1 的瞬间,gaze 切换到 center reward(球钉在图像中心)。
- 观测里提供球的 image-frame 速度 `(du, dv)` / 世界系 ball_vel,
  让策略"看到球在动",为后续追踪移动的球做准备。
- 此阶段仍可短暂不位移(确认球的运动方向),但不强制冻结。

**阶段 3 — APPROACH(追踪逼近, 球可能在动):**
- locomotion tracking reward 恢复正常权重,策略朝球移动。
- **头部持续追踪:** center reward 全程生效,要求边走边把球保持在视野中心;
  `ball_visible_rate`(行进段)是硬验收指标(>70%)。
- **按球的运动方向调整路线:** 因为球可能在滚动,approach 的目标点应是
  球的**预测拦截点**(ball_pos + ball_vel·τ),而非当前球位。reward 用
  robot→(预测球位)的距离递减 + heading 对齐。这让策略学会"提前量"。
- 球若在逼近中再次离开视野(被自身遮挡/急转)→ 退回 SEARCH 的第一级
  (只转头找回),不重新转身,避免抖动。

**阶段 4 — KICK(踢球到目标):**
- distance_to_ball < kick_range 时,kick_contact + ball_to_target/goal_progress
  reward 主导(复用 Spike E / G-2b 的踢球 reward)。
- 静止球是 approach 的退化情形(ball_vel≈0, 预测点=当前点),自动覆盖,
  无需特判 —— 正如你指出的"球静止就更方便"。

**reward gating 机制(不硬编码 FSM):**

各 reward 项的权重由 `ball_visible` 和 `distance_to_ball` 动态门控,
四个阶段从权重曲线里**自然涌现**:

| reward 项 | SEARCH | LOCK | APPROACH | KICK |
|-----------|--------|------|----------|------|
| gaze_search (朝未知方位扫描) | 高 | 0 | 0 | 0 |
| gaze_center (球居视野中心) | 0 | 高 | 高 | **0**(见下) |
| base_vel_penalty (惩罚乱跑) | 高 | 中 | 0 | 0 |
| approach (朝预测球位逼近) | 0 | 低 | 高 | 0 |
| kick_contact + to_target | 0 | 0 | 低 | 高 |

gate 实现: `w_eff = w_base * sigmoid(...)`,以 ball_visible 和
distance 为输入做平滑过渡,避免阶段切换处的奖励突变。

> **⚠️ Spike A2-MOS92 实证修正(2026-06-02):足下盲区**
> 控球/踢球期球被拉到脚边(实测中位球距 0.03m),球在地面比头部低 ~0.38m,
> 俯角 −37°~−52°,**逼近垂直视野极限(neck −28.6° + FOV −28.6°=−57°),几何上
> 几乎看不到脚下的球**。这不是 bug 是物理约束。因此:
> - `gaze_center` 仅在**接近期(球距 >0.5m)门控生效**,KICK 期置 0,不要求球居中,
>   否则策略会为盯脚下球过度低头、破坏平衡。
> - 控球/踢球改用**脚部接触 + 本体感知**(robot_to_ball / kick_contact)而非视觉。
> - Spike A2 实测:搜索成功率 99.8%(首见≈11.6 step),**接近期(>1m)可见率
>   78.6%**,控球期(<0.5m)14% —— 证明「搜索+接近期对准」设计有效,足下盲区
>   是预期内的、应当用非视觉感知覆盖的区间。


**新增/扩展验收指标:**
- `search_to_lock_time`: 从球不可见到锁定的平均步数(分球在前方/后方两类统计)
- `moving_ball_track_rate`: 球运动时仍保持视野中心的时间占比
- `intercept_success_rate`: 对运动球的拦截/触球成功率
- `body_frozen_during_search`: SEARCH 第一级 base 位移 < 阈值的时间占比(验证"脚不乱动")

**先验 spike(Stage 3 前):** Spike A2-MOS92 用 GT 球位 + 人为让球
出视野/滚动,验证 reward gating 能否涌现"搜索→锁定→追踪→踢"序列,
再进入 vision 训练。

**规则:** penalty_weight = 1.0(维持 Stage 2 强度)
**感知噪声:** depth/RGB-depth noise + obs delay(逐步加)

**课程:**
1. **Vision/gaze smoke**:冻结/复用 Stage 2 locomotion,验证相机能看到球和球门方向,
   并验证策略 gaze action 能改变视野。
2. **Gaze imitation warmup**:短期用 GT/detector 生成 gaze target,训练策略的
   gaze head 模仿,让它学会把球放进视野;不改变腿部策略。
3. **Privileged vector dropout**:actor 仍可看 noisy ball vector,但按概率逐步 mask;
   vision 始终存在,critic 保留 GT。
4. **Detector/vector distillation**:从 depth/RGB-depth 中提取或学习 ball bearing/range,
   逐步替代 GT ball vector。不要写 `α·GT + (1-α)·depth`,向量和图像不能直接插值。
5. **Pure vision actor + learned gaze**:actor 不再获得 GT ball pos/vel,策略同时输出
   locomotion actions 和 gaze/waist actions。

**验收:**
- Pure vision actor 下 Goal Rate 不低于 Stage 2 的 50%(初期可低,但必须非零且稳定上升)
- Shot-on-Goal Rate 不低于 Stage 2 的 60%
- learned-gaze 模式下 `ball_visible_rate`、`lost_ball_recovery_rate` 必须达标;
  不能用 detector/heuristic gaze 的结果替代
- 规则违规率 < 15%
- 视频确认:从画面找球 → 走过去 → 多次触球推进 → 射门进球

**产出:** 视觉合规策略 checkpoint

---

### Stage 4: 比赛级(全随机化 + 严格规则 + 视觉)

**目标:** 最大强度 domain randomization + 严格规则下,策略仍能合法视觉进球。

**在 Stage 3 基础上叠加:**
- 相机 DR(fovy/pos/quat)
- 球外观 DR(颜色/大小)—— 大小/物理 DR ✅ 已实现(Task A:radius/mass/friction/elasticity);
  颜色/纹理 DR 待加 RGB obs 后再开(当前 depth-only,纹理对训练无影响,见 §1 球配置)
- 更大执行器延迟(delay_max_lag=5)
- 更大外力推扰(±1.0 m/s)
- torque noise

**规则:** penalty_weight = 2.0(严格模拟裁判)
**新增终止条件:** holding_time > 3.0s → terminate episode

**验收:**
- 全随机化下 Goal Rate > 15%(初始门槛;后续按 v2-goal 基线提高)
- Shot-on-Goal Rate > 25%
- 规则违规率 < 5%(比赛级合规)
- 10 个种子方差 < 30%

**产出:** Sim2Real + 比赛合规候选 checkpoint

---

## 2. 规则代理(Rule Penalties — 不是独立模块,是 reward terms)

### 设计原则

**不建独立的 `RuleEngine` class。** 原因:

1. mjlab 已有 `RewardManager` — 规则惩罚本质就是负 reward terms
2. 已有 `ContactSensor` — 接触分类不需要新模块,只需新的 sensor 配置
3. 已有 `DribbleCommand.metrics` — 持球时间追踪加几行即可
4. 独立模块 = 额外的状态管理 + 生命周期 + 测试负担,没有收益

**正确做法:** 每条规则 = 一个 reward term 函数,挂到 `cfg.rewards` 里,
权重通过 curriculum 调度。和 `kick_contact`、`dribble_approach` 完全同构。

### 违规行为分类(权威规则 → 为什么违规 → RL exploit)

足球是"**动态控制**"不是"占有"。以下 6 类是 v3 必须覆盖的违规,每类标注**为什么违规**和
**RL 最容易学到的作弊解**(reward hacking 高发区,设计 penalty 时要正面对抗):

| # | 违规 | 判定代理(工程) | 为什么违规 | RL 典型坏策略 | 权重 |
|---|------|----------------|-----------|--------------|------|
| 1 | 非法身体接触 | `body_contact and not foot_contact` | 破坏"踢球"本质;非脚部位比脚更稳定→exploit | "贴球走"用大腿推、"压球前进"用躯干挡 | -1.0 |
| 2 | 持球过久 Holding | `dist<0.4 ∧ speed<0.2 ∧ 持续>1.5s` | 足球是动态控制非占有;防"抱球走" | 把球卡住不动 = 最优解 | -2.0 |
| 3 | 夹球 Trapping | `foot_contact ∧ body_contact ∧ speed≈0` | **最严重**:完全破坏任务 | 两脚夹球、脚+小腿夹、身体包住球 | **-3.0** |
| 4 | 粘球 Sticking | `contact ∧ speed<0.05 ∧ 持续>1.0s` | 接触但不推进 = 消极控球 | 一直贴球不踢、球抖动不前进 | -1.0 |
| 5 | 危险高踢 | `foot_contact ∧ foot_z>阈值` | 多人赛会撞对手;humanoid 尤严 | 高抬腿、向上踢、失控摆腿 | -1.5 |
| 6 | 冲撞 Charging | `contact ∧ robot_speed>阈值` | 高速撞球而非控制 | 全速撞球、不减速直冲(简单粗暴但有效,RL 爱学) | -0.5 |

**Holding vs Sticking(两者都"球不动",机制不同,不要混为一谈):**
- **Holding**:长时间**占有**——球在控制范围内但不让它自由移动 → 判据看 `dist + 持续时间`
- **Sticking**:**接触但无推进**——贴着球但不产生有效运动 → 判据看 `contact + speed≈0`

> **⚠️ 经验验证(2026-06-05, v3d 实测):** 这套分类不是纸上假设。v3d 跳跃修复策略的行为视频里,
> 机器人反复**"骑"在球上(球卡在胯下/两腿之间)** —— 正是规则 2(持球)+规则 3(夹球)描述的
> exploit。说明:**只要只奖励"球到目标"而不惩罚占有,RL 必然学到把球卡住。** 规则 2/3 的高权重
> (-2.0 / -3.0)是对抗这个**已观测到的真实 exploit**,不是预防性猜测。证据见
> `soccer_eval/2026-06-05_spikes/v3d_jumpfix/`。

### 规则严谨性:官方规则 → 可测代理 → 工程阈值

v3 的 rule penalties 是**可测代理**,不是官方规则文本本身。每条规则落地前必须
写清楚:

| 规则主题 | 官方/比赛含义 | v3 可测代理 | 阈值性质 |
|---------|---------------|-------------|----------|
| 非法身体触球 | 不允许靠非预期身体部位获利/危险接触 | 非脚 body↔ball contact | 代理,需视频校准 |
| 持球/夹球 | 不能把球固定/藏住/不可争抢 | ball near + low speed + 持续时间 | 工程阈值 |
| 危险动作 | 高踢/高速接触可能危险 | foot_z + foot-ball contact + ball speed | 工程阈值 |
| 消极粘球 | 接触球但不推进 | contact + ball_speed≈0 + duration | 工程阈值 |

要求:
- 阈值如 `0.4m`、`1.5s`、`foot_z>0.4m` 必须标记为 engineering heuristic,
  不能写成"MSL 官方阈值"。
- 每次调整阈值都要保存违规片段视频,人工确认 penalty 触发的是"真实违规"而非
  合法带球。
- 验收报告同时给 rate 和 false-positive 样例,否则高 penalty 可能误杀合法控球。

---

### 规则 1: 非法接触(Illegal Body Contact)

**判定:** 非脚部位碰到球 = 犯规

**实现:** 新增一个 body↔ball ContactSensor,排除脚部:

```python
# 身体(非脚)↔球接触传感器
body_ball_cfg = ContactSensorCfg(
    name="body_ball_contact",
    primary=ContactMatch(
        mode="subtree",
        pattern=r"^(torso_link|.*_hip_.*|.*_knee_.*|.*_shoulder_.*|.*_elbow_.*)$",
        entity="robot",
    ),
    secondary=ContactMatch(mode="geom", pattern="ball_geom", entity="ball"),
    fields=("found",),
    reduce="any",
    num_slots=1,
)
```

**Reward term:**
```python
def illegal_body_contact(
    env: ManagerBasedRlEnv, sensor_name: str
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    return (sensor.data.found.sum(dim=-1) > 0).float()
```

**配置:** `weight=-1.0`(Stage 1 时 × penalty_weight=0.2,最终 × 2.0)

---

### 规则 2: 持球过久(Holding Ball)

**判定:**
```
ball_dist < 0.4m AND ball_speed < 0.2 m/s AND duration > 1.5s
```

**实现:** 在 `DribbleCommand._update_metrics` 中追踪:

```python
# 新增 metrics
self.metrics["holding_time"] = torch.zeros(self.num_envs, device=self.device)

# _update_metrics 中:
ball_near = (robot_to_ball < 0.4)
ball_slow = (ball_speed < 0.2)
is_holding = ball_near & ball_slow
self.metrics["holding_time"] = torch.where(
    is_holding,
    self.metrics["holding_time"] + self.dt,
    torch.zeros_like(self.metrics["holding_time"]),
)
```

**Reward term:**
```python
def holding_ball_penalty(
    env: ManagerBasedRlEnv, command_name: str, threshold: float = 1.5
) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    return (cmd.metrics["holding_time"] > threshold).float()
```

**配置:** `weight=-2.0`
**可选终止:** Stage 4 时 holding_time > 3.0s → terminate episode

---

### 规则 3: 夹球(Ball Trapped)

**判定:** 多个接触点 + 球速 ≈ 0(球被身体几何包住)

**简化工程版**(不需要接触法向量分析):
```
foot_ball_contact AND body_ball_contact AND ball_speed < 0.05
```

**Reward term:**
```python
def ball_trapped_penalty(
    env: ManagerBasedRlEnv,
    foot_sensor: str,
    body_sensor: str,
    command_name: str,
    speed_threshold: float = 0.05,
) -> torch.Tensor:
    foot: ContactSensor = env.scene[foot_sensor]
    body: ContactSensor = env.scene[body_sensor]
    cmd = env.command_manager.get_term(command_name)
    ball_speed = cmd.metrics.get("ball_speed", torch.zeros(...))
    both_contact = (foot.data.found.sum(-1) > 0) & (body.data.found.sum(-1) > 0)
    ball_stuck = ball_speed < speed_threshold
    return (both_contact & ball_stuck).float()
```

**配置:** `weight=-3.0`(最重的惩罚,因为夹球是最严重的犯规)

---

### 规则 4: 危险高踢(Dangerous Kick)

**判定:** 踢球时脚高度 > 阈值(对其他机器人有危险)

**阈值说明(平台相关):**
- G1 (站高 0.78m): foot_z > 0.4m (约 51% 身高)
- MOS92 (站高 0.45m): foot_z > 0.22m (约 49% 身高,等比例缩放)
- 球心高度 0.11m,要踢到球上半部分脚需抬到 >0.11m
- MOS92 的 dangerous_kick 阈值需要在 Spike G 中通过视频确认合理性

**Reward term:**
```python
def dangerous_kick_penalty(
    env: ManagerBasedRlEnv,
    foot_sensor: str,
    asset_cfg: SceneEntityCfg,
    height_threshold: float = 0.4,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[foot_sensor]
    foot_pos = env.scene[asset_cfg.name].data.body_pos_w  # 脚的世界坐标
    # 取左右脚 z 坐标的最大值
    foot_z = foot_pos[:, foot_indices, 2].max(dim=-1).values
    has_contact = sensor.data.found.sum(-1) > 0
    is_high = foot_z > height_threshold
    return (has_contact & is_high).float()
```

**配置:** `weight=-1.5`

---

### 规则 5: 粘球(Ball Must Move)

**判定:** 机器人接触球但球不动(消极控球)

```
contact_with_ball AND ball_speed < 0.05 AND duration > 1.0s
```

**Reward term:**
```python
def ball_stuck_penalty(
    env: ManagerBasedRlEnv,
    foot_sensor: str,
    command_name: str,
    speed_threshold: float = 0.05,
    time_threshold: float = 1.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[foot_sensor]
    cmd = env.command_manager.get_term(command_name)
    has_contact = sensor.data.found.sum(-1) > 0
    ball_slow = cmd.metrics.get("ball_speed", torch.zeros(...)) < speed_threshold
    stuck_time = cmd.metrics.get("ball_stuck_time", torch.zeros(...))
    return (has_contact & ball_slow & (stuck_time > time_threshold)).float()
```

**配置:** `weight=-1.0`

---

### 规则 6: 冲撞(Charging) — v3-extended,当前无对手

当前是单机器人任务,没有对手。此规则留到多智能体阶段(v4+)。
但可以提前加一个**自身速度过快时碰球的惩罚**作为近似:

```python
def excessive_speed_contact(
    env: ManagerBasedRlEnv,
    foot_sensor: str,
    asset_cfg: SceneEntityCfg,
    speed_threshold: float = 2.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[foot_sensor]
    robot_vel = env.scene[asset_cfg.name].data.root_lin_vel_w[:, :2]
    speed = torch.norm(robot_vel, dim=-1)
    has_contact = sensor.data.found.sum(-1) > 0
    return (has_contact & (speed > speed_threshold)).float()
```

**配置:** `weight=-0.5`(轻惩罚,鼓励控制接近速度)

---

### Reward 总结构

```python
total_reward = (
    # Task rewards (正向; goal_scored 是最终主奖励,to_target 只是课程 shaping)
    task_rewards(goal_scored, shot_on_goal, to_target, ball_velocity, approach, kick_contact)
    # Rule penalties (负向,权重通过 curriculum 调度)
    + penalty_weight × (
        illegal_contact × (-1.0)
        + holding_ball × (-2.0)
        + ball_trapped × (-3.0)
        + dangerous_kick × (-1.5)
        + ball_stuck × (-1.0)
        + excessive_speed × (-0.5)
    )
)
```

**关键:kick_contact 权重必须在 penalty 之前提升。**
v2 分析显示 kick_contact=0.3 时策略选择用身体推球(更容易获得 approach/to_target reward)。
Stage 1a 需要 kick_contact ≥ 2.0 才能让"用脚"成为 reward-optimal 策略。
否则 illegal_contact penalty 会惩罚策略唯一会的控球方式,导致"不碰球最安全"。

**关键:approach reward 在全场尺度下必须重新设计。**
v2 的 `dribble_approach` 用 `exp(-dist²/std²)` 且 std=1.0,在 >3m 时 reward≈0。
v3 全场任务(9m approach)需要改用距离递减 reward(`prev_dist - curr_dist`)或
大幅增大 std(≥5.0)。v2.5 Spike F 将验证最佳方案。

**penalty_weight 课程:**
| Stage | penalty_weight | 含义 |
|-------|---------------|------|
| 1 (规则引入) | 0.2 → 1.0 | 允许稍微违规,先学踢球 |
| 2 (本体鲁棒) | 1.0 | 标准强度 |
| 3 (视觉) | 1.0 | 维持(不在感知困难时加规则难度) |
| 4 (比赛级) | 2.0 | 严格模拟裁判 |

---

## 3. Reward 权重调度器(统一 curriculum term)

mjlab 的 `RewardTermCfg.weight` 每步被 `RewardManager.compute()` 读取,
运行时修改立即生效。只需写一个 curriculum term 函数:

```python
# src/mjlab/tasks/velocity/mdp/curriculums.py (追加)

@dataclass
class RewardStage:
    step_threshold: int
    weights: dict[str, float]

def reward_weights_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: Sequence[int],
    stages: tuple[RewardStage, ...],
) -> dict[str, torch.Tensor]:
    """按训练步数切换 reward 权重。"""
    current_step = env.common_step_counter
    active_stage = stages[0]
    for stage in stages:
        if current_step >= stage.step_threshold:
            active_stage = stage
    for name, weight in active_stage.weights.items():
        term_cfg = env.reward_manager.get_term_cfg(name)
        term_cfg.weight = weight
    return {"reward_stage": torch.tensor(
        stages.index(active_stage), device=env.device
    )}
```

**配置示例(Stage 1→2 过渡):**
```python
cfg.curriculum["reward_weights"] = CurriculumTermCfg(
    func=mdp.reward_weights_curriculum,
    params={"stages": (
        RewardStage(step_threshold=0, weights={
            "approach": 3.0, "kick_contact": 0.5, "to_target": 2.0,
        }),
        RewardStage(step_threshold=500_000, weights={
            "approach": 1.0, "kick_contact": 2.0, "to_target": 4.0,
        }),
        RewardStage(step_threshold=1_500_000, weights={
            "approach": 0.5, "kick_contact": 1.5, "to_target": 5.0,
            "goal_scored": 3.0,
        }),
    )},
)
```

**Gating 变体(基于指标而非步数):**
```python
def reward_gating_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: Sequence[int],
    gates: dict[str, tuple[str, float, float]],
) -> dict[str, torch.Tensor]:
    """当某 metric 达标时启用/调整 reward。
    gates: {reward_name: (metric_name, threshold, target_weight)}
    """
    for reward_name, (metric_name, threshold, target_weight) in gates.items():
        metric_val = env.extras.get(metric_name, 0.0)
        term_cfg = env.reward_manager.get_term_cfg(reward_name)
        if metric_val >= threshold:
            term_cfg.weight = target_weight
        else:
            term_cfg.weight = 0.0
    return {}
```

---

### 3.1 运行时可改 vs 必须预分配(实现边界)

不是所有 curriculum 参数都能在训练中随意从 0 切到非 0:

| 参数 | 可运行时改? | 原因/做法 |
|------|-------------|-----------|
| reward weight | ✅ | `RewardManager.compute()` 每步读取 `term_cfg.weight` |
| 噪声 std/prob | ✅ | noise object 已存在时可改字段 |
| obs/action 实际 lag | ✅/⚠️ | buffer 必须初始化时存在;课程只调用 `set_lags` 或调采样范围 |
| `delay_max_lag: 0→N` | ❌ | manager 初始化时 `delay_max_lag>0` 才创建 `DelayBuffer` |
| history_length | ❌ | buffer shape 初始化后固定 |
| 新增/删除 obs term | ❌ | 需要重建 env |

因此 v3 的延迟课程必须这样做。最大 buffer 可以按最坏情况预分配,但训练时的
active lag 必须通过退化曲线实测,不能把最大值当默认可用值:

```python
# 初始化时就分配最大 buffer
depth_obs.delay_min_lag = 0
depth_obs.delay_max_lag = 18

# curriculum 中只改变当前采样到的 active lag 或 DelayBuffer.set_lags()
# 不能训练中把 delay_max_lag 从 0 改成 18
```

如果需要从"无延迟"开始,做法是 `set_lags(0)` 保持 buffer 存在,而不是配置
`delay_max_lag=0`。

---

## 4. 感知噪声课程(新 curriculum term)

同理,观测噪声强度也可以通过 curriculum 逐步增加:

```python
def perception_noise_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: Sequence[int],
    obs_group: str,
    term_name: str,
    noise_schedule: tuple[tuple[int, float], ...],  # (step, std)
) -> dict[str, torch.Tensor]:
    """按训练步数逐步增加观测噪声。"""
    current_step = env.common_step_counter
    target_std = noise_schedule[0][1]
    for step_threshold, std in noise_schedule:
        if current_step >= step_threshold:
            target_std = std
    obs_group_cfg = env.observation_manager.get_group_cfg(obs_group)
    term_cfg = obs_group_cfg.terms[term_name]
    if term_cfg.noise is not None:
        term_cfg.noise.std = target_std
    return {"perception_noise_std": torch.tensor(target_std, device=env.device)}
```

---

## 4.5 POMDP 建模 + 视觉失败模式(Sim2Real 致命补丁)

> 以下是对 v3 方案的关键补充。不补这些,仿真里能跑但真机一定退化。

### 4.5.1 观测-决策频率错位(最大 Sim2Real 杀手)

**问题:** 当前方案隐含假设 obs_t → action_t → physics_t 同步。
但真实系统是异步多频率的:

```
camera (30Hz, 延迟 30-50ms)
  → perception (再延迟 20-50ms)
    → policy (50Hz)
      → actuator (再延迟 10-20ms)
```

策略用的永远是"过期信息"。总延迟 60-120ms = 12-24 个物理步。

**不补的后果:** 仿真中球位置≈当前真实值,策略学到精确 timing;
真机中球位置是 100ms 前的,踢空/timing 错/控球抖动。

**解决方案:** 已有基础设施可以覆盖,但延迟数值必须按课程扫描,不能直接把
100ms 当成默认训练条件:

```python
# 深度观测:预分配 camera pipeline 最大延迟 buffer
depth_obs = ObservationTermCfg(
    func=mdp.camera_depth,
    delay_min_lag=0,
    delay_max_lag=18,  # 只表示 buffer 上限;active lag 由 curriculum 控制
    history_length=3,
)

# 本体感知:模拟 IMU/encoder 延迟 ~5-10ms = 1-2 步
joint_pos_obs = ObservationTermCfg(
    func=mdp.joint_pos_rel,
    delay_min_lag=1, delay_max_lag=2,
    history_length=3,
)
```

**本质升级:** MDP → POMDP + memory。`history_length=3` 让策略能从
历史帧推断当前状态(隐式学会预测)。这不需要新代码 — mjlab 的
`CircularBuffer` + `DelayBuffer` 已经实现了完整的 POMDP 观测管线。

**课程:** 延迟从 0 逐步增到真实值候选(和 Stage 2 的 actuator delay 同步),但
`DelayBuffer` 必须在初始化时按最大 lag 预分配。建议先扫:
```python
# 初始化: delay_max_lag=18
# 课程/评估: active_lag 0 → 4 → 8 → 12 → 18
```

**验收方式:** 每个 active lag 都跑同一批 seed,画退化曲线:
- `Goal Rate(active_lag)`;
- `Shot-on-Goal Rate(active_lag)`;
- `kick_miss_rate(active_lag)`:脚到球但没形成有效球速;
- `ball_visible_rate(active_lag)`;
- `lost_ball_recovery_rate(active_lag)`。

如果 `active_lag=18` 时 Goal Rate 近似归零,不要硬训。先判断:
- `history_length=3` 是否不够,尝试 5/8 帧历史;
- MLP 是否不适合 POMDP,改 RNN/GRU 或 temporal encoder;
- gaze/ball memory 是否提供了足够的 velocity/bearing history。
`18` 是现实延迟上限候选,不是必须一次达到的硬指标。

---

### 4.5.2 深度相机失败模式(不仅是高斯噪声)

**问题:** 真实 structured-light 深度相机(D435)的失败模式远不止高斯噪声:

| 失败模式 | 频率 | 影响 | 当前覆盖 |
|---------|------|------|---------|
| z² 噪声 | 常态 | 远处精度差 | ✅ §6.3 已设计 |
| Depth dropout(整块丢失) | 5-15% 像素 | 球消失 | ❌ 必须补 |
| 边缘 flying pixels | 物体边缘 | 深度跳变 | ❌ 应补 |
| 反射面全黑 | 光滑表面 | 大面积无效 | ⚠️ 球是哑光,影响小 |
| Motion blur | 快速运动时 | 深度模糊 | ⚠️ 用 delay 近似 |

**必须补的两个(其余可选):**

**(1) Depth dropout(随机像素丢失):**
```python
def depth_dropout_noise(depth: Tensor, p_drop: float = 0.08) -> Tensor:
    """随机将 p_drop 比例的像素设为 0(无效深度)。"""
    mask = torch.rand_like(depth) < p_drop
    return depth.masked_fill(mask, 0.0)
```

**(2) 区域性 dropout(模拟反射/遮挡导致的大块丢失):**
```python
def depth_patch_dropout(depth: Tensor, p_patch: float = 0.02,
                        patch_size: int = 8) -> Tensor:
    """随机丢弃 patch_size×patch_size 的区域。"""
    B, H, W, C = depth.shape
    n_patches = int(p_patch * (H * W) / (patch_size ** 2))
    for _ in range(n_patches):
        y = torch.randint(0, H - patch_size, (B,))
        x = torch.randint(0, W - patch_size, (B,))
        # 批量 mask(实际实现用 scatter 更高效)
        depth[:, y:y+patch_size, x:x+patch_size, :] = 0.0
    return depth
```

**实现方式:** 自定义 `NoiseModelCfg` 子类,注册到 depth obs term 的 `noise` 字段。
mjlab 的 noise 系统支持自定义 callable,不需要改框架。

**课程:** dropout 概率从 0 逐步增到 0.08(Stage 3 中期开始加)。

---

### 4.5.3 球丢失处理(ball_not_visible 状态)

**问题:** 即使最终由策略主动控制 gaze,视觉输入也会出现"看不见球"的状态:
- 球被踢远出视野
- 检测失败(depth dropout 正好覆盖球)
- 球在身后(转头来不及)

**不处理的后果:** 一旦丢球 → 策略收到全零/噪声 depth → 输出随机动作 → 崩溃。

**原则:** 球丢失后的转头/搜索动作仍由**策略自己输出**。工程代码只提供
memory/visibility/confidence 这类观测和 reward/metric,不能在最终评估时用
外部 controller 直接接管 waist/gaze。

**(1) v3-core 必做: "上一次有效观测" + 衰减,作为策略输入**

```python
# 在 depth obs term 中:
def depth_with_ball_memory(env, sensor_name, memory_decay=0.95):
    depth = get_current_depth(env, sensor_name)
    ball_detected = detect_ball_in_depth(depth)  # 简单阈值
    
    # 如果检测到球,更新记忆
    env._ball_memory = torch.where(
        ball_detected.unsqueeze(-1),
        ball_bearing_from_depth(depth),
        env._ball_memory * memory_decay,  # 衰减旧记忆
    )
    env._ball_memory_confidence = torch.where(
        ball_detected,
        torch.ones_like(env._ball_memory_confidence),
        env._ball_memory_confidence * memory_decay,
    )
    return depth, env._ball_memory, env._ball_memory_confidence
```

策略观测中加入 `_ball_memory` 和 `_ball_memory_confidence`。如果当前帧看不到球,
策略应学会根据记忆方向转动 waist/gaze;如果置信度衰减到很低,策略应学会搜索。
这可以通过 `lost_ball_recovery`、`ball_visible`、`gaze_rate_l2` 等 reward/metric
引导,但最终动作仍来自 policy 的 gaze action head。

**(2) v3-core 训练项: learned search,不是 controller search**

当 `ball_memory_confidence < threshold` 时,不由代码直接扫视,而是给策略一个明确状态
并奖励"重新看到球":
```python
obs += [ball_memory, ball_memory_confidence, time_since_ball_seen]
reward += lost_ball_recovery_bonus  # 丢球后 N 秒内重新看到球
reward -= gaze_rate_l2              # 防止乱甩头
```

可以用 teacher 生成扫视/跟踪 target 做 warmup,但 teacher 权重必须衰减到 0。
验收必须报告 teacher-off 的 `lost_ball_recovery_rate`。

**(3) v3-ext: 分层搜索策略**

如果 single-policy gaze head 在 full-field goal 中恢复能力不足,再引入低频高层
search/gaze policy。高层仍然是学习策略,不是写死的 controller。否则 Goal Rate
会被感知恢复能力卡死。

**验收指标:**
- `ball_visible_rate`:按球距 `{0-1,1-2,2-3,3-5}` m 分桶;
- `lost_ball_duration`:连续看不见球的时间分布;
- `lost_ball_recovery_rate`:丢球后 N 秒内重新看到球的比例;
- `teacher_off_gap`:关闭 gaze teacher 后 Goal Rate/ball_visible_rate 的下降幅度。

---

### 4.5.4 Critic 噪声(避免 asymmetric AC 梯度消失)

**问题:** Critic 用纯 GT 状态 → value 估计极准 → actor 的 noisy 体验和
critic 的 clean 估值差距太大 → advantage 估计不稳定 → actor 学不动。

**解决方案:** 给 critic 加轻微噪声(比 actor 小 3-5 倍):

```python
# critic obs group 配置
critic_obs = ObservationGroupCfg(
    terms={
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=GaussianNoiseCfg(operation="add", std=0.003),  # actor 是 0.01
        ),
        "ball_pos": ObservationTermCfg(
            func=mdp.ball_pos_b,
            noise=GaussianNoiseCfg(operation="add", std=0.015),  # actor 是 0.05
        ),
        ...
    },
    enable_corruption=True,
)
```

**原理:** Critic 仍然比 actor 信息更好(噪声小 3-5 倍),但不再"完美" →
advantage 估计更平滑 → actor 梯度更稳定。

**替代方案:** 如果上面不够,用 observation normalization(RunningMeanStd)
让 actor 和 critic 的 obs 在同一尺度上。mjlab 是否已有 obs normalization
需要确认。

---

### 4.5.5 动作平滑约束(已有,确认覆盖)

**现状:** v2 已有 `action_rate_l2`(weight=-0.1),惩罚 `||a_t - a_{t-1}||²`。

**v3 是否需要加强?** 加了执行器延迟后,策略可能学到更激进的动作来补偿延迟,
导致真机电机过载。建议:

- Stage 1-2: 维持 weight=-0.1(和 v2 一致)
- Stage 3-4: 如果视频显示动作抖动,加大到 -0.2 或加 jerk penalty:

```python
def action_jerk_l2(env: ManagerBasedRlEnv) -> Tensor:
    """Penalize second-order action changes (jerk)."""
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    a_prev2 = env.action_manager.prev_prev_action  # 需确认是否有
    jerk = a - 2 * a_prev + a_prev2
    return torch.sum(jerk ** 2, dim=-1)
```

**注意:** 不要过度约束。真实 G1 的 PD 控制器本身就是过阻尼(ζ=2.0),
已经天然平滑了输出力矩。`action_rate_l2` 大概率够用。

---

### 4.5.6 失败恢复能力(recovery)

**问题:** 当前训练的是"正常流程"(看到球→走→踢)。但现实中会:
- 被推倒后站起来
- 错过球后重新定位
- 偏离目标后修正

**解决方案:** 不需要专门的 recovery reward。原因:

1. `push_robot` event 已经在训练中随机推机器人 → 策略被迫学会恢复平衡
2. `dribble_approach` reward 持续引导策略接近球 → 错过球后自然会重新追
3. 球被踢远后 `DribbleCommand` 会 resample → 策略学会处理各种球位置

**真正需要的:** 确保训练中有足够的"非理想初始状态":
- 球在身后(需要转身)
- 球在远处(需要长距离追)
- 机器人被推偏(需要恢复)

这些通过 `reset_base` 的随机 pose + `push_robot` 的随机外力已经覆盖。
如果不够,可以加一个 curriculum:初始球-机器人距离从近到远。

---

### 4.5.7 Reward 连续性(线性插值,不硬切)

**问题:** `reward_weights_curriculum` 用 step_threshold 硬切 → 切换点梯度突变。

**修复:** 在 stage 之间做线性插值:

```python
def reward_weights_curriculum_smooth(
    env: ManagerBasedRlEnv,
    env_ids: Sequence[int],
    stages: tuple[RewardStage, ...],
    ramp_steps: int = 50_000,  # 过渡窗口
) -> dict[str, torch.Tensor]:
    """带线性插值的 reward 权重调度。"""
    step = env.common_step_counter
    
    # 找当前所在的两个 stage
    prev_stage = stages[0]
    next_stage = stages[0]
    for i, stage in enumerate(stages):
        if step >= stage.step_threshold:
            prev_stage = stage
            next_stage = stages[min(i + 1, len(stages) - 1)]
    
    # 计算插值比例
    if prev_stage is next_stage:
        alpha = 1.0
    else:
        progress = (step - prev_stage.step_threshold) / ramp_steps
        alpha = min(1.0, max(0.0, progress))
    
    # 线性插值权重
    for name in set(prev_stage.weights) | set(next_stage.weights):
        w0 = prev_stage.weights.get(name, 0.0)
        w1 = next_stage.weights.get(name, w0)
        weight = w0 + alpha * (w1 - w0)
        env.reward_manager.get_term_cfg(name).weight = weight
    
    return {"reward_stage_alpha": torch.tensor(alpha, device=env.device)}
```

这替换 §3 中的硬切版本。

---

### 4.5.8 Metric-driven curriculum(基于指标而非步数)

**问题:** 纯 step-based curriculum 不适应训练速度差异。如果某个 stage
收敛慢,硬切到下一个 stage 会崩溃。

**更好的做法:** 基于实际训练指标决定何时推进:

```python
def metric_driven_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: Sequence[int],
    transitions: tuple[dict, ...],
) -> dict[str, torch.Tensor]:
    """当指标达标时才推进到下一 stage。
    
    transitions: ({metric: threshold, ...}, {param: value, ...})
    """
    for conditions, actions in transitions:
        all_met = all(
            env.extras.get(metric, 0.0) >= threshold
            for metric, threshold in conditions.items()
        )
        if all_met:
            for param_path, value in actions.items():
                # 设置对应参数(reward weight / noise std / delay 等)
                set_nested_param(env, param_path, value)
    return {}
```

**配置示例:**
```python
transitions = (
    # 当 Goal Rate 达标且 fall_rate < 10% 时,加规则
    ({"goal_rate": 0.4, "fall_rate_below": 0.1},
     {"penalty_weight": 0.5}),
    # 当 Goal Rate 尚可且 illegal_contact < 15% 时,加噪声
    ({"goal_rate": 0.35, "illegal_contact_below": 0.15},
     {"obs_noise_std": 0.05, "delay_max_lag": 3}),
)
```

**和 step-based 的关系:** 两者可以共存。step-based 作为"最慢保底"
(防止永远卡在某个 stage),metric-driven 作为"快速通道"(指标达标就推进)。

---

## 5. 本体真实性(硬件参数对齐)

> **平台说明:** 本节以 G1 为例描述参数。如果 v2.5 Spike G 确认 MOS92 为最终平台,
> 以下参数需要全部重新校准:
> - MOS92 用 `BuiltinMotorActuatorCfg`（力矩控制）或 `BuiltinPositionActuatorCfg`
>   （加 PD servo），取决于 Spike G-1 的选择
> - MOS92 的 effort_limit 为 36/60 Nm（vs G1 的 50-139 Nm）
> - MOS92 无 waist joints → 所有 waist 相关的 DR/delay 配置不适用
> - MOS92 的通信延迟需要实测（可能与 G1 不同）
> - 参考文件: `docs/robot_param/MOS92_urdf_0517_v3_simplified.xml`

### 5.1 已有且只需启用的

| 参数 | 现状 | v3 动作 |
|------|------|---------|
| 执行器延迟 | `ActuatorCfg.delay_*` 已实现,G1 未启用 | 设 `delay_max_lag=3` |
| 质量/惯量/COM 随机化 | `pseudo_inertia` dr 已实现,soccer 未用 | 加 EventTermCfg |
| PD 增益随机化 | `pd_gains` dr 已实现,soccer 未用 | 加 EventTermCfg ±20% |
| 关节摩擦随机化 | `joint_friction` dr 已实现,soccer 未用 | 加 EventTermCfg |
| 编码器偏差 | 已用(±0.015 rad) | 可加大到 ±0.03 |
| 外力推扰 | 已用(±0.5 m/s) | Stage 4 加大到 ±1.0 |

### 5.2 需要新建的

**执行器/动作噪声(torque noise):**

这是唯一需要新代码的本体真实性模块。真实电机有:
- 力矩纹波(torque ripple)
- 齿轮间隙(backlash)
- 温度漂移

最简模型:在 actuator `compute()` 输出的力矩上加高斯噪声。

```python
# 方案 A: 在 ActuatorCfg 加字段
torque_noise_std: float = 0.0
"""Additive Gaussian noise on output torque (Nm). 0 = disabled."""

# 在 Actuator.compute() 末尾:
if self.cfg.torque_noise_std > 0:
    noise = torch.randn_like(torque) * self.cfg.torque_noise_std
    torque = torque + noise
```

**工作量:** ~20 行改动(ActuatorCfg + Actuator base class)。

### 5.3 关键硬件参数

**G1 机器人(从机器人资料提取):**

| 参数 | 值 | 来源 | v3 用途 |
|------|-----|------|---------|
| 总质量 | ~35 kg | g1.xml | 已在模型中 |
| 足部尺寸 | capsule 阵列 | g1.xml | 踢球接触面,已在模型中 |
| 髋关节力矩 | 88-139 Nm | g1_constants.py | 已配置 |
| 踝关节力矩 | 50 Nm | g1_constants.py (2×5020) | 已配置 |
| 通信延迟 | ~10-20ms | 真机测量 | delay_max_lag=3-4 |
| PD 自然频率 | 10 Hz | g1_constants.py | 已配置 |
| PD 阻尼比 | 2.0 (过阻尼) | g1_constants.py | 已配置 |
| 头部相机 | RealSense D435 | 硬件规格 | depth sensor cfg |

**MOS92 机器人(如果 Spike G 确认为最终平台):**

| 参数 | 值 | 来源 | 与 G1 差异 |
|------|-----|------|-----------|
| 总质量 | ~16.5 kg | MOS92_*.xml | G1 的 47% |
| 站高 | 0.45m | keyframe z | G1 的 58% |
| 自由度 | 18 | xml | G1 的 62%(无 waist/wrist） |
| 腿部力矩 | 60 Nm | xml ctrlrange | G1 的 43-68% |
| 臂/踝力矩 | 36 Nm | xml ctrlrange | G1 的 72% |
| 头部控制 | neck fixed | xml | 无主动 gaze |
| 执行器类型 | 力矩电机 | xml `<motor>` | G1 是 PD position |

**RoboCup 标准球(已正确配置,不可更改):**

| 参数 | 值 | RoboCup 规则范围 | DR 范围(Stage 4) |
|------|-----|-----------------|-----------------|
| 直径 | 22 cm | 20-23 cm | ±5% |
| 质量 | 0.43 kg | 0.35-0.45 kg | 0.35-0.50 |
| 滑动摩擦 | 0.5 | — | 0.3-0.7 |
| 滚动摩擦 | 0.01 | — | 0.005-0.02 |
| 弹性(restitution) | ~0.35 | — | 0.2-0.6 |
| 颜色 | 橙色 | 单色(通常橙色) | RGB DR(Stage 4) |

---

## 6. 视觉 Pipeline 详细设计

### 6.1 为什么选 Depth 而不是 RGB

| 维度 | Depth | RGB |
|------|-------|-----|
| Sim2Real gap | 小(结构光/ToF 直接给深度) | 大(纹理/光照/反射) |
| 计算量 | 1 通道 | 3 通道 |
| 对球的区分 | 球是凸起的深度异常,天然容易检测 | 需要颜色/纹理学习 |
| mjlab 支持 | ✅ 一行配置 | ✅ 一行配置 |
| 真机可用 | ✅ D435 原生 depth | 需额外处理 |

**结论:** v3-core 可以从 depth-only 开始,因为近距离控球时 Sim2Real gap 小。
但 full-field 进球需要"远距离找球",depth-only 在低分辨率下很可能不够。
因此视觉路线必须拆成两层:
- **近距离控球/踢球**:depth-only 优先;
- **远距离找球/重新捕获**:允许 RGB-depth 或 segmentation 辅助,并单独评估
  `far_ball_acquisition_rate`。

### 6.2 CNN Encoder 设计

```
Input: [B, 1, 48, 64] depth
  → Conv2d(1→16, 5×5, stride=2) → ReLU  → [B, 16, 22, 30]
  → Conv2d(16→32, 3×3, stride=2) → ReLU → [B, 32, 10, 14]
  → Conv2d(32→64, 3×3, stride=2) → ReLU → [B, 64, 4, 6]
  → Flatten → Linear(1536→128) → ReLU
  → Linear(128→64) → latent
Output: [B, 64]
```

总参数量 ~200K,训练可承受。latent 64 维拼接到 proprioception obs。

**rsl_rl 支持:** 需确认 rsl_rl 的 ActorCritic 是否支持 multi-input
(image + vector)。如果不支持,需要:
- 方案 A: 在 obs term 里做 flatten(48×64=3072 维直接 concat 到 obs vector),
  用 MLP 处理(简单但效率低)
- 方案 B: 自定义 network class 带 CNN head(更好但需改 rsl_rl config)

**建议:** 先试方案 A(flatten depth → large MLP),验证视觉信号有用;
确认后再换方案 B(CNN encoder)优化性能。

### 6.3 Depth 噪声模型(真实 structured-light 特性)

真实深度相机的噪声不是均匀高斯,而是:
- 噪声 ∝ z²(远处噪声大)
- 边缘处有 flying pixels(深度跳变)
- 反射面/透明面有 dropout(深度=0)

v3 简化模型:
```python
def depth_realistic_noise(depth: Tensor, base_std: float = 0.005) -> Tensor:
    """z²-proportional depth noise."""
    noise_std = base_std * depth.clamp(min=0.1) ** 2
    noise = torch.randn_like(depth) * noise_std
    return depth + noise
```

这可以作为自定义 noise model 注册到 `ObservationTermCfg.noise`。

### 6.4 主动视线控制(Active Gaze — v3 核心难点)

#### 问题本质

概念上,深度相机位于 G1 头部/上身前方;在当前 XML 中必须挂到实际存在的
`parent_body`(例如 `robot/torso_link`),并通过 smoke test 确认相机位姿会随
腰部姿态变化。不要假设 `head_link` 一定是可挂载 body。相机视线主要受三个
腰部关节影响:

| 关节 | 轴 | 范围 | 功能 |
|------|-----|------|------|
| `waist_yaw_joint` | Z(垂直) | ±150° | 左右转头 |
| `waist_pitch_joint` | Y(侧向) | ±30° | 抬头/低头 |
| `waist_roll_joint` | X(前向) | ±30° | 侧倾(次要) |

这意味着:**相机的视野方向是策略动作的一部分,不是固定的。**
机器人必须学会"往哪看"才能"看到球",然后才能"走过去踢球"。

这和固定相机的 vision RL 有本质区别:
- 固定相机:球要么在视野里,要么不在,策略只需处理"看到/没看到"
- 主动视线:策略必须**同时学会两件事** — 控制视线找到球 + 控制腿脚踢球

#### 为什么这很难

1. **探索困难:** 随机动作很难同时把头转对方向 AND 把腿迈对方向
2. **奖励稀疏:** 如果球不在视野里,策略得不到任何关于球位置的梯度信号
3. **耦合冲突:** waist_yaw 同时影响视线方向和躯干朝向(影响行走稳定性)
4. **当前 pose reward 惩罚转头:** `std_walking` 里 `waist_yaw=0.3`,
   `waist_roll=0.08`, `waist_pitch=0.2` — 策略被惩罚偏离直立朝前

#### 设计方案:策略主动视线 + 课程辅助

**核心要求:** 最终策略本身必须输出 gaze/waist 动作,主动转动相机找球。
启发式 gaze 只能做 teacher/warmup,不能作为最终控制器。

**结构:** 仍然分头输出,但同一个策略学习两个 action head:

```
policy(obs) -> {
  locomotion_action: legs/arms,
  gaze_action: waist_yaw / waist_pitch / optional waist_roll
}
```

locomotion 和 gaze 可以在网络里共享视觉 encoder,但动作头、scale、reward
和日志必须分开,这样能单独诊断"看不见"还是"走不好"。

---

**方案 A(teacher,只用于 warmup): 启发式/检测器视线**

用简单规则生成 gaze target,但目标是训练策略模仿它:

```python
def gaze_teacher(ball_bearing_estimate: Tensor) -> Tensor:
    """生成 gaze imitation target,不作为最终控制器。"""
    target_yaw = ball_bearing_estimate[:, 0].clamp(-1.5, 1.5)
    target_pitch = ball_bearing_estimate[:, 1].clamp(-0.3, 0.3)
    return torch.stack([target_yaw, target_pitch], dim=-1)
```

- 训练 warmup:短时间用 GT/detector 生成 teacher target,加 `gaze_imitation`
  reward 或监督损失;
- 主训练:teacher 权重衰减到 0,策略自己输出 gaze action;
- 部署:没有 teacher,只有策略 gaze action + 视觉输入。

**验收:** teacher 关闭后,ball_visible_rate 和 Goal Rate 不能崩。

**严禁长期 teacher forcing:** 如果训练一直用 GT gaze、部署才换 depth gaze,
会形成严重分布偏移。v3 文档里的"GT gaze"只能是 bootstrap 阶段,并且必须有
ablation:teacher-on vs teacher-off 的 Goal Rate 差距。

---

**方案 B(v3-core 正式方案): 视线作为独立 action head**

把 action space 分成两组:

```python
actions = {
    "locomotion": JointPositionActionCfg(
        actuator_names=(".*hip.*", ".*knee.*", ".*ankle.*", ".*shoulder.*", ".*elbow.*", ".*wrist.*"),
        scale=0.5,
    ),
    "gaze": JointPositionActionCfg(
        actuator_names=("waist_yaw_joint", "waist_pitch_joint"),
        scale=0.3,
    ),
}
```

加一组**视线奖励/指标**引导头部朝向球并恢复丢球:

```python
def ball_visible_centered(env, sensor_name: str) -> Tensor:
    """奖励球出现在相机视野中心区域。"""
    camera: CameraSensor = env.scene[sensor_name]
    depth = camera.data.depth  # [B, H, W, 1]
    # 实际实现应基于 segmentation 或 detector 输出,不要只用最近点。
    return visible_and_centered.float()
```

推荐 gaze reward/metric:
- `ball_visible`:球在视野中;
- `ball_centered`:球在中心区域;
- `lost_ball_recovery`:球丢失后 N 秒内重新看到;
- `gaze_rate_l2`:限制腰部/相机快速抖动;
- `torso_stability`:防止 gaze 破坏平衡。

**优点:** 策略本身学会主动看球,满足 v3 最终要求。
**缺点:** 探索更难,必须用方案 A 的 teacher warmup 和 ball memory 降低难度。

---

**方案 C(v3-extended): 分层策略 + 注意力机制**

```
高层策略(低频,每 10 步): 输入=proprioception + 球记忆 → 输出=视线目标 + 移动方向
低层策略(高频,每步):    输入=proprioception + depth + 视线目标 → 输出=关节动作
```

高层维护一个"球位置记忆"(上次看到球的位置 + 时间衰减),
当球丢失时高层输出"搜索模式"(左右扫视),找到球后切换到"跟踪模式"。

**优点:** 最接近真实行为,能处理球丢失/遮挡
**缺点:** 实现复杂,需要分层 RL 或 options framework

---

#### v3 推荐路径

```
v3-core:  方案 A teacher warmup → 方案 B learned gaze action head
v3-ext:   方案 C 分层策略,处理长时间丢球/主动搜索
v4:       多机器人/对抗下的分层搜索和注意力
```

**理由:**
- 用户要求策略本身学会主动转头找球,因此启发式 gaze 不能作为最终 v3-core。
- 方案 A 只用于降低探索难度;方案 B 是 v3-core 的正式验收形态。
- 如果 learned gaze 在 full-field 中频繁丢球,再进入方案 C,但不能把"主动看球"
  推迟到 v4。

#### 对 v3 方案的具体修改

1. **Action space 拆分可行性先验证:** G1 现有 actuator/action 按 actuator names
   选择目标,waist_yaw 与髋关节同组、waist_pitch/roll 另组。必须先确认能否
   干净隔离 waist joints。不能隔离时,要实现 joint-name 级 gaze action 或重构
   actuator/action 配置;不能把最终方案退回启发式 gaze。
2. **Pose reward 修改:** 降低但不完全移除 waist_yaw/pitch 的 pose 惩罚。
   它们由策略控制后需要可动,但仍要约束躯干稳定。
3. **相机挂载确认:** G1 XML 里 `head_link` 是 `torso_link` 下的 visual geom,
   不是独立 body。`CameraSensorCfg.parent_body` 应先用实际存在的 prefixed body
   (如 `robot/torso_link`)做 smoke test,不要假设有可挂载的 `head_link` body。
4. **Gaze action head 实现:** 新增 gaze action term 或扩展 action cfg,让策略输出
   waist_yaw/pitch 目标;teacher 只提供 warmup target/reward。
5. **球方位估计:** GT 只用于短 warmup 和评估上限;主训练/部署都使用同一套
   delayed/noisy detector + ball memory。最简 detector 可从 depth/RGB-depth 中估计
   bearing/range,但必须记录 miss/false-positive。

#### 视野覆盖验证(必须做)

相机 fovy=45°,上身/头部相机位姿朝前下方约 20°:
- 垂直视野:[-42.5°, +2.5°] 相对水平(主要看地面)
- 水平视野:约 ±30°(取决于 aspect ratio 64:48 → hfov ≈ 58°)

球(直径 22cm)在不同距离的像素覆盖:

| 距离 | 球角直径 | 像素(48px 高) | 可见性 |
|------|---------|--------------|--------|
| 0.5m | 25° | ~27 px | 很清楚 |
| 1.0m | 12.6° | ~13 px | 清楚 |
| 2.0m | 6.3° | ~7 px | 可检测 |
| 3.0m | 4.2° | ~4 px | 勉强 |
| 5.0m | 2.5° | ~3 px | 困难 |

**结论:** 3m 内可能可见,5m 外低分辨率 depth 很困难。这个表只是几何估算,
不能替代渲染验证。v3 启动前必须用实际 `CameraSensorCfg` 截图确认:
- 球像素覆盖;
- depth 是否被地面/机器人遮挡污染;
- 腰部转动后视野是否符合预期;
- 远距离找球是否需要 RGB-depth 或更高分辨率。

---

## 7. 实施路径(关键路径 + 时间估算)

> 时间估算分成"实现能跑"和"训练通过验收"两层。视觉 + learned gaze 的不确定性
> 最大,不要用 smoke-test 时间替代收敛时间。

### Phase -1: v2.8 进球闭环补齐(1-2 天起,取决于现有 checkpoint)

| 步骤 | 内容 | 改动 |
|------|------|------|
| G1 | 实现/确认 goal-scoring command:目标为球门线/球门口 | dribble_command.py |
| G2 | 实现 `goal_scored`、`shot_on_goal`、`time_to_score`、`touches_before_goal` | rewards.py / terminations.py / evaluate_soccer.py |
| G3 | 从 target-dribble checkpoint fine-tune 到 full-field goal | 训练 |
| G4 | 单 env 视频确认"追球→多次触球→射门→进球" | soccer_eval/ |

**验收:** 没有 v2.8 goal checkpoint 就不进入 v3 Phase 0。target success 只能证明
会把球带到点,不能证明会进球。

### Phase 0: 硬前置验证(1 天,不训练)

| 步骤 | 内容 | 改动 |
|------|------|------|
| P0 | 用 v2-goal checkpoint 跑 goal evaluator | evaluate_soccer.py |
| P1 | 相机挂载 smoke test:0.5-5m 球像素/depth/遮挡截图 | tests/test_soccer_vision_smoke.py |
| P2 | waist/gaze action head 编译测试 | env_cfgs.py / action cfg |
| P3 | 规则代理映射表 + 阈值视频抽检 | stage_eval.md |
| P4 | delay buffer 预分配测试(active lag=0/target lag=N) | tests/test_soccer_delay_curriculum.py |

**验收:** 不通过 Phase 0 不进入训练。尤其是 camera/gaze 和 Goal Rate evaluator,
不能边训边猜。

### Phase A: 规则引入 + 本体鲁棒(v3-A minimal,2-4 天 + 训练)

| 步骤 | 内容 | 改动 |
|------|------|------|
| A1 | 新增 body↔ball ContactSensor(非法接触检测) | env_cfgs.py ~12 行 |
| A2 | 写 5 个规则 penalty reward terms | rewards.py ~80 行 |
| A3 | DribbleCommand 追踪 holding_time / ball_stuck_time | dribble_command.py ~20 行 |
| A4 | 写 reward_weights_curriculum(含 penalty_weight 调度) | curriculums.py ~60 行 |
| A5 | 挂规则 penalties 到 cfg.rewards,初始 penalty_weight=0.2 | env_cfgs.py ~25 行 |
| A6 | 启用执行器延迟 + 观测噪声 + DR | env_cfgs.py ~20 行 |
| A7 | 从 v2-goal checkpoint fine-tune,penalty_weight 0.2→1.0 | 训练 ~3000 iter |

**验收:**
- Goal Rate 不低于 v2-goal 的 70%
- Shot-on-Goal Rate 不低于 v2-goal 的 80%
- illegal_contact_rate < 10%, holding_violation < 5%
- 在 delay=3 + noise 下仍然合规

**产出:** `v3-A minimal` checkpoint。它不是最终 v3,但已经是独立有效成果:
GT 感知下合法、鲁棒、能进球。

### Phase B1: 视觉诊断基线(v3-B diagnostic,3-5 天实现 + 短训)

| 步骤 | 内容 | 改动 |
|------|------|------|
| B1 | 加 head/depth CameraSensorCfg(挂真实 parent_body,朝前下 20°) | env_cfgs.py ~15 行 |
| B2 | 写 depth/RGB-depth obs term(flatten 优先,CNN 后置) | observations.py / rl_cfg.py |
| B3 | 配置 asymmetric actor-critic obs groups | env_cfgs.py ~20 行 |
| B4 | 用 teacher/heuristic gaze 跑视觉 policy 诊断 | gaze_policy.py |
| B5 | 扫描分辨率、FOV、active delay、depth/RGB-depth 组合 | evaluate_soccer.py |

**验收:** 允许使用 teacher/heuristic gaze,但只能命名为 `v3-B diagnostic`。
必须回答三个问题:
- 视觉输入能不能稳定定位球和球门方向;
- depth-only 是否足够,还是必须 RGB-depth/segmentation;
- active lag 到 8/12/18 时 Goal Rate 如何退化。

### Phase B2: 主动视线策略(v3-B final,7-14 天调试+训练)

| 步骤 | 内容 | 改动 |
|------|------|------|
| B6 | 拆分/扩展 action space:策略输出 waist_yaw/pitch gaze action | env_cfgs.py / action term |
| B7 | 实现 gaze teacher warmup + gaze rewards(ball_visible/centered/recovery) | gaze_policy.py |
| B8 | 降低 waist_yaw/pitch pose 惩罚,增加 gaze_rate/torso_stability | env_cfgs.py/rewards.py |
| B9 | 训练:gaze imitation warmup → teacher 权重衰减 → teacher-off learned gaze | 训练 |
| B10 | privileged vector dropout → detector/vector distillation → pure vision actor | 训练 |
| B11 | teacher-on/off ablation + lost-ball recovery 评估 | evaluate_soccer.py |

**验收:** pure vision + learned gaze + 规则约束下 Goal Rate 非零且稳定;初始门槛为
Stage A/B 的 50%。必须报告 `ball_visible_rate`、`lost_ball_recovery_rate`、
`far_ball_acquisition_rate`、`teacher_off_gap`,且 teacher-off 视频能看到策略主动
转动视线找球。

**失败处理:** 如果 B2 暂时不收敛,保留 B1 作为诊断基线,但不能把 B1 宣称为最终 v3。
下一步应先定位失败原因(action head、视觉输入、延迟、reward、网络结构),而不是降低
最终要求。

### Phase C: 比赛级全随机化(1-2 天)

| 步骤 | 内容 | 改动 |
|------|------|------|
| C1 | 加 camera DR(fovy/pos/quat) | env_cfgs.py ~10 行 |
| C2 | 加球外观 DR(rgba/size) | env_cfgs.py ~10 行 ✅ 物理 DR(size/mass/friction/elasticity)已实现(Task A);rgba/纹理 DR 待 RGB obs |
| C3 | 加大执行器延迟 + 外力推扰 | env_cfgs.py ~5 行 |
| C4 | 写 torque_noise 到 ActuatorCfg | actuator.py ~20 行 |
| C5 | penalty_weight 提升到 2.0 + holding > 3s terminate | env_cfgs.py ~5 行 |
| C6 | 从 Phase B2 checkpoint fine-tune | ~3000 iter |

**验收:** 全随机化下 Goal Rate > 15%,Shot-on-Goal Rate > 25%,规则违规 < 5%。

---

## 8. 最容易失败的点(提前防范)

### 坑 1: 一开始就加视觉噪声

**症状:** RL 完全不收敛,reward 震荡
**原因:** 策略连基本控制都没学会就被噪声淹没
**防范:** Phase A 先不动视觉,只加本体噪声;Phase B 用 asymmetric AC 让 critic 有 GT

### 坑 2: Reward curriculum 不连续

**症状:** 训练在 stage 切换点崩溃,reward 断崖下跌
**原因:** 权重突变导致梯度方向突然翻转
**防范:** 在 curriculum term 里做线性插值(不是硬切),或用较长的 ramp window

### 坑 3: 深度相机分辨率过低看不到球

**症状:** 球在 3m 外时 depth image 上只有 1-2 个像素
**原因:** 64×48 分辨率 + 45° fov,3m 处球(直径 22cm)≈ 3 pixel
**防范:** 验证球在常见距离(0.5-3m)的像素覆盖;必要时提升到 96×72

### 坑 4: Asymmetric AC 的 critic 过于依赖 GT

**症状:** Actor 训练不动,因为 critic 给的 value baseline 和 actor 实际体验差太远
**原因:** Critic 用 GT 状态估值太准,但 actor 看的噪声观测和 GT 差距大
**防范:** 给 critic 也加轻微噪声(比 actor 小),或用 observation normalization

### 坑 5: 执行器延迟导致站不稳

**症状:** 加 delay_max_lag=3 后 fall_rate 暴涨
**原因:** G1 平衡依赖快速反馈,3 步延迟(15ms)可能已经很大
**防范:** 从 delay_max_lag=1 开始,用 curriculum 逐步增加到 3;
或者先只给腿部加延迟(手臂不影响平衡)

### 坑 6: Rule penalty 太大 → 策略不敢动

**症状:** 策略学会"站着不动"(零违规但零进球)
**原因:** penalty 权重过大,任何接近球的动作都有违规风险,不如不动
**防范:**
- penalty_weight 从 0.2 起步,不要一开始就 1.0
- 确保 task_reward(goal_scored / shot_on_goal / progress)远大于单次 penalty
- 监控 episode_length:如果策略开始"站着等超时",说明 penalty 过重

### 坑 7: Rule penalty 太小 → 学会违规最优

**症状:** 策略用膝盖/躯干推球,成功率很高但全是犯规
**原因:** 违规带来的 task reward 收益 > penalty 成本
**防范:**
- 监控 illegal_contact_rate:如果 > 30% 且不下降,说明 penalty 不够
- 经验法则:单次 penalty 应 ≈ 该步 task reward 的 20%~50%
- 夹球(最严重)的 penalty 要最大(-3.0),因为夹球 = 稳定高 reward

### 坑 8: 规则和感知同时加 → 双重困难

**症状:** Stage 3(加视觉)时规则违规率突然飙升
**原因:** 看不清球 → 乱碰 → 触发 illegal contact → 大量负 reward → 崩溃
**防范:** Stage 3 初期可以暂时降低 penalty_weight 到 0.5,
等视觉策略稳定后再恢复到 1.0(用 curriculum 自动调度)

---

## 9. v3 vs v2 的验收对比

| 指标 | v2 目标 | v3 目标 | 对比方式 |
|------|---------|---------|----------|
| Goal Rate (GT obs, rules) | v2-goal baseline | ≥70% of v2-goal | 主指标 |
| Shot-on-Goal Rate (GT obs, rules) | v2-goal baseline | ≥80% of v2-goal | 方向性质量 |
| Goal Rate (noisy obs + rules) | N/A | ≥70% of Stage 1 | v3 独有 |
| Goal Rate (vision + rules) | N/A | ≥50% of robust-GT baseline,初期可放宽但必须非零上升 | v3 核心 |
| Target Success | v2 target baseline | 课程指标 | 不能替代 Goal Rate |
| Fall Rate (no noise) | < 5% | < 5% | 不退化 |
| Fall Rate (full noise+delay) | N/A | < 15% | v3 独有 |
| Illegal Contact Rate | 未监控 | < 10% | v3 核心指标 |
| Holding Violation Rate | 未监控 | < 5% | v3 核心指标 |
| Ball Trapped Rate | 未监控 | < 3% | v3 核心指标 |
| Dangerous Kick Rate | 未监控 | < 5% | v3 核心指标 |
| Ball Visible Rate / Lost Duration | N/A | 按距离分桶报告 | 视觉恢复能力 |
| Far Ball Acquisition Rate | N/A | 必须报告 | full-field 进球前置 |
| Learned Gaze Teacher-off Gap | N/A | teacher off 后 Goal Rate 不崩 | 主动视线验收 |
| Robustness (10 seeds variance) | N/A | < 30% | v3 独有 |

**v3 的核心验收是三重的:**
1. **最终任务不丢**:Goal Rate / Shot-on-Goal Rate 仍可接受;
2. **感知退化下仍能工作**:pure vision actor 下 Goal Rate 非零且稳定上升;
3. **行为符合比赛规则**:所有违规率 < 10%,且 false-positive 视频抽检可接受。

---

## 10. 文件清单

| 操作 | 文件 | 改动量 |
|------|------|--------|
| 改 | `src/mjlab/tasks/velocity/config/g1/env_cfgs.py` | ~160 行(sensor+DR+obs+gaze action+rules+delay+goal eval) |
| 改 | `src/mjlab/tasks/velocity/mdp/rewards.py` | ~100 行(5 个规则 penalty terms + approach/to_target reward 重写为距离递减) |
| 改 | `src/mjlab/tasks/velocity/mdp/dribble_command.py` | ~30 行(holding/stuck time 追踪) |
| 改 | `src/mjlab/tasks/velocity/mdp/curriculums.py` | ~120 行(reward+noise+penalty+metric curriculum) |
| 改 | `src/mjlab/tasks/velocity/mdp/observations.py` | ~30 行(depth obs term) |
| 新建 | `src/mjlab/tasks/velocity/mdp/gaze_policy.py` | ~120 行(gaze teacher warmup+球记忆+主动视线 rewards/metrics) |
| 新建 | `src/mjlab/tasks/velocity/mdp/depth_noise.py` | ~50 行(dropout+patch+z² noise model) |
| 改 | `src/mjlab/actuator/actuator.py` | ~20 行(torque_noise) |
| 改 | `src/mjlab/tasks/velocity/config/g1/rl_cfg.py` | ~15 行(asymmetric AC config) |
| 改 | `src/mjlab/tasks/velocity/mdp/__init__.py` | ~10 行(导出新模块) |
| 新建 | `tests/test_soccer_vision_smoke.py` | ~50 行 |
| 新建 | `tests/test_soccer_rules.py` | ~60 行(规则 penalty 单元测试) |
| 新建 | `tests/test_gaze_policy.py` | ~60 行(动作头、teacher-off、球丢失恢复) |
| 新建 | `tests/test_depth_noise.py` | ~30 行 |
| 新建 | `tests/test_soccer_delay_curriculum.py` | ~30 行 |
| 改 | `src/mjlab/scripts/evaluate_soccer.py` | ~80 行(Goal Rate/Shot-on-Goal/far-ball metrics) |

**总新代码:** ~755 行

---

## 11. 与你的 v3 提案的对照

| 你的提案 | 落地方案 |
|---------|---------|
| "写 reward_scheduler.py" | 不需要新文件,一个 curriculum term 函数即可 |
| "写 rule_engine.py 独立模块" | 不需要独立模块,规则 = reward terms(和 kick_contact 同构) |
| "写 contact_analyzer.py" | 不需要,用两个 ContactSensor(foot↔ball + body↔ball)区分合法/非法 |
| "加 motion blur" | v3-core 不做(用 obs delay 近似),v3-extended 考虑 |
| "接入 vision pipeline" | ✅ 用 asymmetric AC + depth/RGB-depth camera + learned gaze action head |
| "加 delay + noise" | ✅ 全部已有基础设施,纯配置 |
| "robot_realism_cfg.py" | 不需要新模块,参数都在 ActuatorCfg + DR events 里 |
| "规则用 penalty 不用硬限制" | ✅ 完全一致,penalty 通过 curriculum 从弱到强 |
| "penalty ≈ task reward 的 20%~50%" | ✅ 采纳,通过 penalty_weight 0.2→2.0 实现 |
| "Stage 1-4 联动" | ✅ 重新设计为三轴联动(感知×规则×鲁棒) |

**核心差异:**
1. 不建独立模块(RuleEngine/ContactAnalyzer/RewardScheduler) — 用 mjlab 已有的
   RewardManager + ContactSensor + CurriculumManager 组合实现,零新基础设施
2. 规则引入在视觉之前(你的方案是同步) — 避免"看不清+不能碰"双重困难
3. 主动视线控制是 v3-core 要求 — 策略必须学会输出 gaze action,启发式 gaze 只能 warmup

---

## 12. 一句话总结

v3 = 在 v2 的真实物理和进球闭环基础上,用三轴课程联动(规则约束从弱到强 ×
感知从 GT 退化到 vision × 环境从理想到有噪声/延迟)训练出一个**合法、鲁棒、
视觉驱动、能主动转动视线找球且仍能进球**的足球策略。关键补丁:Goal Rate 主指标、
规则代理映射、POMDP 建模(obs history + 真实延迟链)、depth/RGB-depth 失败模式、
球丢失记忆、learned gaze action head、gaze teacher-forcing 消除、critic 加噪、
metric-driven curriculum。实现上尽量复用 mjlab 基础设施,但 camera/gaze action/
delay buffer 都必须先 smoke test。




---

# v3 最终成果总结(2026-06-07 实跑收口)

> 本节为 v3 的实跑落地结果。以 **04_e2e 为 v3 代表模型**(纯视觉定位+找球+带球进球单策略)。
> 完整实验链见 `soccer_robot_v2-3_experiment.md`;改进方案见 `soccer_robot_v4.md`。

## 交付的模型(checkpoints/v3_soccer_solo/)
| 编号 | 模型 | 能力 | 关键指标 |
|------|------|------|----------|
| 01 | selfloc_purevision/model_2800 | 纯视觉自定位 | 0.79m(**1.0m 加粗线**,有 sim2real gap) |
| 02 | findball_depth/model_1499 | 深度找球 | 纯视觉球向量 |
| 03 | dribble_goal/model_1600 | 带球进球 | goal_rate 0.43 |
| **04** | **e2e_integrated/model_1499** | **①②③ 端到端单策略(v3 代表)** | **goal_rate 0.30, fell_over 0, dribble_success 0.51** |
| 05 | selfloc_realspec_0125m/model_3400 | 0.125m 真场线+主动扫视+时序 | selfloc 5.8m,**但不踢球(见根因)** |

## v3 核心结论
1. **端到端整合干净成功**:①②③ 可融进单策略,稳定性满分(fell_over 0)。所谓"自定位退化"
   经对照实验证明是**操作几何变难**(进攻半场地标少)所致,非能力互相破坏。
2. **0.125m 真场线纯视觉定位的物理边界(已修正)**:64×48 下线亚像素消失;但高分辨率视角图证明
   **1280×960 下连角落位姿的线都清晰可辨**——之前"像素占比"gate 指标误导,真场线在足够分辨率下
   可定位。这是 v4 的关键依据。
3. **05 不踢球的根因(v4 必修)**:depth 和 RGB **共用 head_cam 传感器**,05 为自定位提分辨率
   (64×48→96×72)连带改了 depth 图分辨率,导致 depth-ball CNN 的 spatial_softmax 被 reinit
   (192→432),从 model_2800 继承的踢球能力归零。dribble_success 0.51→0.00。
   **修复方向:分离两个相机**(depth 保 64×48 完整迁移踢球 CNN,RGB 单独高分辨率做自定位)。

## v3 → v4 的交接
v4 聚焦"在真场线(0.125m)下同时保住踢球 + 提升自定位精度",三条主线:
(a) **分离相机 + RGB 高分辨率(到 1280×960 区间)**;(b) **距离自适应精度机制**(远粗近精);
(c) **更高级自定位方法**(关键点+PnP / 粒子滤波 MCL)。详见 `soccer_robot_v4.md`。
