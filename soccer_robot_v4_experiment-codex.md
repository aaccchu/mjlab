# Soccer Robot v4 Codex Experiment Ledger

> 监督版实验记录。用途不是替代训练日志，而是给 Claude Code 明确“现在该做什么”。
> 最后更新: 2026-06-08 18:31 CST。

## 正式落盘要求

- 本文件只做监督和状态机补充,正式实验结果必须回写 `soccer_robot_v4_experiment.md`。
- 正式方案变更必须回写 `soccer_robot_v4.md`。
- 较成功实验必须生成:
  - `mjlab/checkpoints/` 下的 checkpoint + params
  - `mjlab/soccer_eval/` 下的评估视频、图表、报告
- 若 Claude 只更新 `*-codex.md` 而没更新正式 md 或正式工件目录,视为交付不完整。

## 实验状态机

### `EXP1A_RUNNING`
- run: `logs/rsl_rl/mos92_velocity/2026-06-07_19-49-28_spike_v4_dualcam`
- 目标:
  验证 dualcam 是否足以恢复 real-spec 链里的踢球能力
- 假设:
  如果 `05` 的崩坏主要来自 depth CNN 被分辨率改坏，那么分离相机后应接近恢复到
  `04_e2e` 等级的 dribble
- 已确认:
  - depth CNN transfer 已修复
  - selfloc 从 `~5.8m` 改善到 `~2.64m`
  - 稳定性保持 `fell_over = 0`
- 当前问题:
  - dribble / goal 仍显著偏弱
  - 到 iter `980`:
    - dribble_success 当前 `0.0390`，best `0.2273 @ 686`
    - goal_rate 当前 `0.1500`，best `0.3571`
    - selfloc_pos_err_m 当前 `2.6371m`
- 监督判定:
  这是“第一层 bug 修复成功，但行为恢复不充分”

### `EXP1A_EXIT_RULE`
- 若 full run 结束且:
  - `best dribble_success >= 0.35`
  - `goal_rate best >= 0.20`
  则可记为“部分成功”，但仍需要同口径 e2e 对照
- 若 full run 结束且:
  - `best dribble_success < 0.30`
  则判定“dualcam selfloc 链不足以恢复 v3 代表行为”
- 若 `current dribble_success` 在后半程持续低于 `best * 0.4`
  则记录为“中后期塌陷”

### `EXP1B_NEXT`
- 名称:
  `realspec_e2e_dualcam`
- 这是当前最优先的新实验
- 目标:
  在和 `04_e2e` 尽量同口径的前提下，把真场线 + dualcam 引入 e2e 主线
- 必做原因:
  当前 user 的直观对照对象是 `04_e2e-step-0.mp4`
  当前 dualcam 还不是这个口径
- 实现要求:
  - 基于 `mos92_soccer_e2e_env_cfg`
  - field line width = `0.125m`
  - depth 相机固定 `64x48`
  - 新增 `head_cam_rgb`
  - RGB stack/stride 接入 real-spec 方案
  - bootstrap 用 `04_e2e_integrated/model_1499.pt`
- 成功门槛:
  - dribble_success `>= 0.35`
  - goal_rate `>= 0.20`
  - fell_over `= 0`
  - selfloc 优于 `05_realspec`
- 若失败:
  进入 `EXP1C_BOOTSTRAP_ABLATION`

### `EXP1C_BOOTSTRAP_ABLATION`
- 目标:
  分离“起点不对”与“env/感知链不对”
- 对照:
  - `model_2800 -> realspec_e2e_dualcam`
  - `04_e2e/model_1499 -> realspec_e2e_dualcam`
- 评估:
  哪个起点更能保住 dribble / goal

### `EXP2_HIGHRES_RGB`
- 前置:
  只有在 `EXP1B` 至少恢复基本踢球后才启动
- 目标:
  RGB 提分辨率压 selfloc 误差
- 顺序:
  - `96x72`
  - `256x192`
  - 必要时 `512x384`
- 注意:
  depth 分支分辨率不动

### `EXP3_DISTANCE_ADAPTIVE_SELFLOC`
- 前置:
  `EXP1B` 或 `EXP2` 有稳定可用的端到端行为
- 目标:
  远粗近精，减少全场统一精度约束造成的容量浪费
- 代码落点:
  `src/mjlab/tasks/velocity/mdp/rewards.py:selfloc_accuracy`
- 监督要求:
  新增 reward term，保留原 reward 作为对照，不要直接覆盖

### `EXP4_GEOMETRIC_LOC`
- 触发条件:
  高分辨率回归定位仍明显高于目标
- 路线:
  - 关键点 + PnP
  - 必要时再加 MCL

## 当前给 Claude Code 的监督提醒

1. 先把 `EXP1A` 跑完并诚实记录，不要把它包装成“主问题已解决”
2. 跑完立刻排 `EXP1B`
3. 在 `EXP1B` 之前不要直接跳到粒子滤波
4. 每个实验记录必须包含:
   - 目标
   - 为什么
   - 关键改动
   - current / best / last
   - 达标与否
   - 下一步去向

## 2026-06-08 09:42 监督检查点

- 当前活跃 run: `spike_v4_selfloc_hires`
- 最新观测: 日志仍在刷新,约 iter `1445/3800`, `selfloc_pos_err_m ~= 2.52m`
- 关键纠偏: `Curriculum/selfloc_gt_mask/mask_factor ~= 0.8806`, GT 还没撤干净,现在**不能**把 `2.5m`
  视为 Exp 5a 成功
- 判定门槛:
  - 只有在 GT fade 接近 `0` 后仍能稳定 `<3m`,才能说明"高分辨率 RGB 是关键杠杆"
  - 若 GT fade 后仍塌,按既定状态机切到几何主线:关键点 + PnP,必要时再加 MCL
- 跑完必做:
  - 结果回写 `mjlab/soccer_robot_v4_experiment.md`
  - 若结论变化,同步更新 `mjlab/soccer_robot_v4.md`
  - 若达标,补齐 `mjlab/checkpoints/` 和 `mjlab/soccer_eval/` 正式产物

## 2026-06-08 18:31 给 Claude Code 的 EXP5d/后续监督建议

### 监督定义

- 用户确认:“纯视觉”允许 onboard depth。
- 合格输入:RGB/depth 图像、相机内参、相机到 base 的标定/运动学、机器人本体状态、历史动作/运动模型。
- 不合格输入:正式策略或正式评估依赖 `robot_field_pose`、GT pose obs、外部定位、sim world pose 这类特权状态。
- 训练标签可以由 sim 几何投影自动生成;这是监督标签,不是部署输入。需要在文档中明确区分“训练监督/monitor”
  和“推理时输入”。

### 对当前几何方法的判断

- `关键点检测 + depth + Kabsch` 符合用户的纯视觉目标,方向可以继续。
- EXP5c 暴露的是 reward gaming:可见点太少时 masked error 近似免费高分。这不是几何范式失败。
- EXP5d 的 `(visible/K) * exp(-err/std^2)` 是合理的第一处修复,但只算一个质量修复点,不能因为早期 pixel err 好看就提前宣布成功。

### EXP5d 必须看的指标

- `kp_visible`:不能再次塌到接近 0。建议至少记录 current/best/last trend,并按阶段看是否稳定。
- `kp_pixel_err`:要和可见点数绑定解读。低 pixel err + 低 visible 仍是失败。
- `kp_selfloc_pos_err_m`:这是几何方案是否真正可用的主指标,目标先 `<3m`,长期目标 `<1m`。
- `goal_rate/dribble_success/fell_over`:最终目标是纯视觉踢球,不能只交付定位模块。
- 正式记录必须给出 current / best / best@iter / 后段趋势,并说明是否生成 checkpoint/eval 工件。

### 不限死实现的建议

- 如果 EXP5d 可见性修复后仍学不稳,可以尝试把关键点检测从纯 RL reward 中拆出来:
  用投影标签做监督 loss/预训练/辅助头,再冻结或半冻结接回 e2e。关键点检测本质上更像监督视觉任务。
- 如果关键点检测能准但位姿抖动或对称歧义大,再加 MCL/粒子滤波。MCL 用于多峰消歧和时序平滑,不要提前用来遮盖
  单帧检测失败。
- 如果可见点少是主要瓶颈,可以尝试 gaze/active scan、可见性奖励、关键点分组、置信度预测、RANSAC/Kabsch
  鲁棒估计,但每次实验仍要明确目标和判据。
- 如果 monitor 当前用 sim world camera pose 做中间计算,需要补一个非特权等价验证:用相机到 base 的标定/运动学
  完成 depth 点到 base frame 的变换,再 Kabsch 到地图。否则不能把结果写成“正式纯视觉”。

### 下一次需要提醒 Claude 的话

> depth 被允许,所以不要因为使用 depth 放弃几何主线;但正式成功必须证明策略和评估没有吃特权位姿。
> EXP5d 的关键不是 pixel err,而是可见点数、几何定位误差和踢球行为同时成立。

## 2026-06-08 19:35 EXP5f 路线选择监督意见

### 对 Claude 三条路的判断

Claude 提出的关键约束成立:监督 loss 需要 per-sample 的投影标签 `uv_gt/vis`,而这些标签依赖 rollout
时刻的相机外参和 env origin。默认 PPO update 只看 rollout buffer,不能在 update 时凭空恢复当时标签。

但路径 A 可以做成更稳的 **A' 本地扩展**,不必直接改 `.venv/site-packages/rsl_rl`:
- rsl_rl 的 `RolloutStorage` 存完整 observation `TensorDict`;额外 obs key 会随 rollout 进入 mini-batch。
- actor/critic 只读取 `cfg["obs_groups"]` 中列出的 groups。
- 因此可新增一个仅供训练的 observation group,如 `keypoint_label`,包含 `uv_gt/vis`,让它进 buffer,
  但不加入 actor/critic obs_groups,避免策略吃标签。
- 在本项目内实现 `KeypointAuxPPO` 或类似本地 subclass,通过 `algorithm.class_name` 指向本地类,
  在 `update()` 里从 `batch.observations["keypoint_label"]` 取标签,对 actor selfloc 输出加监督 loss。

### 推荐排序

1. **优先 A'**:本地 PPO/runner 扩展 + label obs 存储。它保留 on-policy/e2e 数据分布,对齐最终目标,
   维护风险低于直接改 pip 包。
2. **B 可做 fallback/预训练**:离线监督预训练检测器风险低,能快速证明关键点检测可学;但两阶段衔接和共享 trunk
   会变复杂,最终仍需 e2e 验证。
3. **C 暂缓**:rollout 采样循环中直接 backward 容易破坏 PPO 采样/旧策略假设和 optimizer 节奏,调试成本高。

### A' 的最低 smoke 门槛

- `keypoint_label` 出现在 rollout batch,shape 正确,包含 `uv_gt` 和 `vis` 或等价编码。
- `keypoint_label` 不在 actor/critic obs_groups,导出的策略输入不包含标签。
- aux loss 在固定小 batch 上能下降;`kp_pixel_err` 下降不能靠 `kp_visible` 变低作弊。
- 训练日志必须同时报 `aux_loss/kp_pixel_err/kp_visible/kp_selfloc_pos_err_m/goal_rate/fell_over`。
- 正式成功仍要过纯视觉边界:推理不吃 world pose 或 GT pose obs,训练标签仅作为监督信号。

## 2026-06-08 EXP5f 后续:主动扫视与多帧融合监督建议

### 核心判断

用户指出的 FOV 限制是 v4 必须面对的问题:单帧看不到足够关键点时,机器人需要转头扫描;但多张扫描图像必须被正确关联,
否则只是增加输入维度。

有效关联应是:
- 每帧图像的关键点检测结果带有置信度/可见性。
- 每帧都有对应的 head yaw/pitch、camera-to-base 外参或等价运动学信息。
- 多帧检测被投到共同坐标系或共同 pose hypothesis 下。
- 一个持续存在的 belief 负责累计、更新、消歧和遗忘旧观测。

一句话提醒 Claude:

> active scan 不是“多转头 + 堆帧”,而是用头部/相机运动把多帧视觉观测融合进同一个定位信念。

### 建议实验顺序

1. EXP5f 先完成单帧监督关键点检测。
   目标:证明 `uv_gt/vis` 辅助 loss 能稳定降低 pixel error,且不靠减少 visible 作弊。
   评估:`aux_loss`、`kp_pixel_err`、`kp_visible`、single-frame Kabsch/PnP solve rate、`kp_selfloc_pos_err_m`。

2. 建立非学习的滑窗几何融合 baseline。
   目标:验证“多帧 + head/camera 位姿”确实改善定位。
   做法可选:最近 N 帧检测 + depth + camera-to-base 变换 + RANSAC/Kabsch/PnP,或粒子滤波/MCL。
   评估:multi-frame pos/yaw err 是否低于 single-frame;是否减少抖动;是否能在单帧关键点不足时恢复定位。

3. 再做主动扫视目标。
   目标:让机器人主动寻找能降低位姿不确定性的视角。
   奖励/指标不要只看 visible 数,要看 unique landmark、空间分布、pose uncertainty reduction 和 time-to-localize。
   失败判据:只看同一簇/同一条线关键点、visible 高但 pose 仍不准、转头导致跌倒或踢球行为消失。

4. 最后再考虑 learned temporal memory。
   RNN/Transformer/attention memory 可以尝试,但应建立在确定性几何融合 baseline 已经有效之后。
   否则很难判断失败来自检测、几何、时序记忆还是踢球策略冲突。

### 监督红线

- 不要把 raw frame stack 当成已经解决时序关联。若堆帧,必须说明每帧的 head pose/time encoding 如何进入模型或融合模块。
- MCL/粒子滤波是多峰消歧和时序稳定器,不是掩盖关键点检测失败的补丁。
- 只看 `kp_visible` 不够;必须看 `kp_visible_unique` 或 landmarks 的空间多样性。看到多个几何退化点仍可能无法定位。
- 正式纯视觉边界不变:可用 RGB/depth、头部关节角、相机标定、本体状态、动作历史和地图;不可用 GT/world pose。

## 2026-06-08 多帧冲突与“场景想象”的实验监督要求

### 问题定义

用户提出的关键情形:
- 同一个关键点出现在两张图里,但两帧分别解出的 robot pose 不一致。
- 某一帧可见关键点很少,单帧定位本来就不可解或高度不稳定。
- 机器人应利用前后帧信息,在内部形成更完整的场景理解。

这里建议 Claude 把“场景想象”定义成带不确定性的 **pose/field belief**,不是让网络凭空补图:
- 地图是已知先验。
- 图像检测是有噪声的观测。
- 头部角度、相机标定、机器人运动模型提供跨帧约束。
- belief 可以是粒子集合、均值+协方差、滑窗优化状态或 learned memory,但必须能解释和评估不确定性。

### 具体实验建议

1. 构造跨帧一致性诊断。
   目标:确认同一 landmark 跨帧投影是否能被同一个 pose belief 解释。
   指标:`reprojection_residual_per_frame`、landmark ID mismatch、depth residual、head-pose residual。

2. 做低关键点场景的滑窗定位测试。
   目标:验证单帧点少时,多帧 belief 是否比单帧 PnP/Kabsch 更可靠。
   做法:按可见点数分桶评估,例如 `0-1/2-3/4+` 个点,比较 single-frame 与 multi-frame pose error。

3. 做冲突观测鲁棒性测试。
   目标:验证误检/错 ID/depth 噪声不会把位姿带崩。
   做法:引入或统计 outlier,使用 RANSAC/Huber/粒子权重等鲁棒机制。
   指标:`outlier_ratio`、pose jump、hypothesis switch count、multi-frame pos/yaw err。

4. 做主动消歧测试。
   目标:机器人在 belief 不确定或多峰时,能转头寻找最能消除歧义的视角。
   指标:`belief_entropy` 或 `pose_cov_trace` 下降、`time_to_localize`、`goal_rate`、`fell_over`。

### 对 Claude 的边界提醒

- 当单帧关键点少时,低置信度输出是正确行为;强行给一个精确 pose 反而危险。
- 当两帧 pose 冲突时,不要简单平均。应先判断是检测噪声、depth 噪声、头部标定误差、运动模型误差、关键点 ID 错误,
  还是场地对称性造成的多峰歧义。
- 一个实用系统需要“知道自己不知道”:belief 不确定时先扫视/保守行动,belief 稳定后再高速追球和射门。
- 如果 Claude 使用 neural memory,仍应对照一个几何 baseline,否则无法判断 learned memory 是否真的学到了跨帧几何。

## 2026-06-08 whole-body active perception 监督要求

### 核心补充

主动感知不应被限制为 head-only scan。头部转动只改变相机方向,不改变机器人本体位置和观察基线;在遮挡、场地对称、
关键点分布退化或一直只能看到同一小片区域时,身体 yaw、原地小步转向、侧移或后退可能是必要的信息获取动作。

### 建议实验

1. head-only active scan baseline。
   目标:确认仅靠头部能否在不破坏控球的情况下恢复定位。
   指标:`time_to_localize`、`belief_entropy`、`kp_visible_unique`、`goal_rate`、`fell_over`。

2. head + body yaw / small-step-turn 对照。
   目标:验证身体动作是否解决 head-only 无法消除的多峰歧义或几何退化。
   指标:相对 head-only 的 `multi_frame_pos/yaw_err`、`hypothesis_switch_count`、`time_to_localize`、`ball_lost_rate`。

3. 信息收益/动作代价门控。
   目标:避免策略学成无意义原地转圈或为了看线牺牲踢球。
   做法:只有在 belief 不确定、单帧/多帧定位不可解或多峰冲突明显时触发更大身体扫描动作。
   指标:单位动作代价的信息增益、速度损失、dribble continuity、kick-ready 恢复时间。

### 红线

- 不要把身体扫描奖励写成“转得越多越好”。奖励应绑定不确定性下降和最终踢球成功。
- 若加入身体/足部动作,必须证明它没有显著破坏 locomotion 稳定、控球和射门准备。
- whole-body active perception 是手段,不是最终目标;最终仍以纯视觉 selfloc + dribble/goal + fell_over 为主判据。

## 2026-06-08 全身扫描协调与稳定性监督要求

### 核心原则

全身主动感知不能让策略自由乱转。应采用“稳定动作原语 + 高层 belief 门控”:
- 低层负责稳:沿用或约束已有 locomotion/dribble 能力,只暴露有限的 yaw-rate、step-turn、side-step、back-step。
- 高层负责何时看:只有在定位不确定、关键点不足、多峰冲突或漂移明显时触发身体扫描。
- 动作逐级升级:head scan -> small body yaw -> step-turn -> side/back step,不要一开始就大幅移动。
- 每个扫描动作必须有退出条件:belief 达标、球丢失、姿态风险上升、动作超时或信息收益不足。

### 建议实验

1. 原地小幅 yaw scan 安全测试。
   目标:验证身体转向不会破坏站立/慢速控球。
   限制:yaw-rate、yaw acceleration、duration、base roll/pitch、foot slip。
   指标:`fell_over`、base tilt、action jerk、`ball_lost_rate`、`belief_entropy` 下降。

2. gait-phase-aware step-turn。
   目标:验证在行走/控球节奏中插入小步转身是否稳定。
   指标:dribble continuity、速度损失、足底接触异常、kick-ready 恢复时间。

3. recovery-gated scan。
   目标:定位不确定时进入扫视/转身,恢复后自动回到追球和射门。
   指标:`time_to_localize`、mode switch count、`goal_rate`、`fell_over`、`ball_lost_rate`。

4. head-only vs head+body 消融。
   目标:确认身体动作带来的信息收益是否超过动作代价。
   指标:multi-frame pose error、belief entropy reduction per action cost、goal/dribble 损失。

### 奖励与失败判据

- 信息收益奖励:不确定性下降、unique landmarks 增加、reprojection residual 降低。
- 稳定性代价:摔倒、base tilt、foot slip、action rate/jerk、关节速度/力矩峰值。
- 任务代价:丢球、远离球、dribble 中断、kick-ready 恢复慢。
- 失败判据:定位变好但摔倒/丢球显著增加;策略原地转圈;长时间找线不进攻;身体扫描吞掉 dribble/goal 目标。

## 2026-06-08 仿真训练路线监督建议

### 总原则

不要把“关键点检测 + 多帧 belief + 主动扫视 + 全身转动 + 踢球”一开始全部端到端混在一起训。
应使用课程化训练,每一层有明确可观测门槛,否则失败时无法判断责任归属。

### 推荐阶段

1. 单帧关键点监督。
   目标:让视觉检测本身可用。
   训练:使用 sim 投影标签 `uv_gt/vis` 做辅助 loss;标签进入训练 batch,但不进入 actor/critic 输入。
   门槛:`aux_loss` 下降、`kp_pixel_err` 下降、`kp_visible` 不塌、single-frame solve rate 达标。

2. 多帧几何融合验证。
   目标:证明历史帧能形成更稳定的 pose/field belief。
   训练/验证:先离线 replay rollout 序列测试滑窗 RANSAC/EKF/MCL,再接入在线策略。
   门槛:multi-frame pos/yaw err 优于 single-frame;低关键点分桶中能维持合理 uncertainty;冲突观测可被降权。

3. head-only 主动扫视。
   目标:低成本恢复可见关键点和降低定位不确定性。
   门槛:`belief_entropy` 下降、`time_to_localize` 缩短、`goal_rate/dribble` 不明显下降。

4. head + body 小幅转身课程。
   目标:解决 head-only 无法消除的遮挡、对称和几何退化。
   训练:先只开放小 yaw-rate / step-turn,逐步开放侧移/后退;通过 belief gate 触发。
   门槛:信息收益超过动作代价;`fell_over`、`ball_lost_rate`、base tilt、foot slip 不恶化。

5. 完整踢球集成。
   目标:把定位恢复模式和进攻模式协调起来。
   训练:belief 不确定时进入安全扫视/恢复;belief 稳定后回到追球、带球、射门。
   门槛:`kp_selfloc_pos_err_m`、`goal_rate`、`dribble_success`、`fell_over` 同时达标。

6. 随机化与鲁棒性。
   目标:防止方案只在理想 sim 相机里有效。
   随机化:光照/纹理、field line 宽度和磨损、相机内外参、头部标定、depth 噪声/dropout、帧延迟、摩擦、
   遮挡、球外观、初始位姿。
   门槛:随机化后指标不出现结构性崩溃,尤其是 belief 不确定时能保守恢复而不是乱踢。

### 实验记录要求

- 每阶段记录独立指标,不要只报最终 reward。
- 每阶段都要说明是否使用 GT:训练标签/monitor 可以用,正式策略输入和正式评估不能用。
- 每个新自由度都要有消融:无该模块、只有 head、head+body、不同融合方法。
- 阶段通过前不要堆下一层复杂度;如果单帧检测不稳,不要用 MCL 掩盖;如果步态不稳,不要继续加全身扫描奖励。

## 2026-06-08 oracle/noisy-oracle 分层诊断实验

### 目的

在接真实视觉检测器之前,先拆清楚后端和主动动作是否有上限。oracle 只能作为训练/诊断工具,不能作为正式纯视觉结果。

### 模块接口建议

- `Perception`:RGB/depth -> landmark id、uv、depth/3D、visibility、confidence。
- `BeliefEstimator`:detections + head/camera pose + motion history + map -> pose belief、uncertainty、hypotheses。
- `ActivePerceptionPolicy`:belief + task state -> head/body scan mode 或动作原语。
- `SoccerPolicy`:belief summary + ball observation + proprioception -> dribble/goal action。

### 实验阶段

1. GT landmark oracle。
   目标:验证在关键点检测完美时,belief 融合、主动扫视和踢球策略是否可行。
   限制:oracle 只能给 landmark 投影/visibility,不能把 GT robot pose 作为策略输入。
   判据:若这里都失败,问题不在 CNN,应优先修融合/动作/奖励/模式切换。

2. Noisy oracle。
   目标:测容错边界。
   噪声:pixel noise、depth noise、dropout、错 ID、延迟、遮挡、head calibration error。
   判据:噪声上升时 pose error 和 uncertainty 应平滑退化;若突然崩溃,先修鲁棒估计/置信度/多峰 belief。

3. Learned detector + frozen belief。
   目标:把真实检测误差单独接入,观察对定位和踢球的影响。
   判据:若比 noisy-oracle 同等噪声差很多,说明 detector confidence、ID 或 outlier handling 仍不合格。

4. Learned detector + active perception。
   目标:让机器人在真实视觉误差下主动补信息。
   消融:head-only、head+body yaw、step-turn、不同 fusion 方法。
   判据:信息收益必须超过动作代价,且 `goal_rate/dribble/fell_over` 不恶化。

5. 端到端微调。
   目标:最终整合。
   要求:保留 keypoint aux loss、稳定性惩罚、纯视觉输入边界;学习率和解冻范围要保守。

### 可用 teacher 信号

- sim 中可用 GT 计算候选动作的信息增益、expected visible unique landmarks、expected entropy reduction。
- 这些只能作为训练 reward/teacher/monitor,不能成为正式推理输入。
- 候选动作建议先离散化,例如 look-left/right/down、small yaw left/right、step-turn、side-step、back-step。

### 记录要求

- oracle 成功只能说明上限存在,不能写成纯视觉成功。
- noisy-oracle 给出 detector 需要达到的 pixel/depth/ID 误差预算。
- learned detector 失败时,必须和 oracle/noisy-oracle 对照定位原因。
- 每个阶段都要写清是否生成 checkpoint/eval 工件,成功阶段应落到正式 `checkpoints/` 和 `soccer_eval/`。

## 2026-06-08 20:35 EXP5f 当前训练审计:检测在学,但行为已崩

### 检查对象

- 活跃脚本:`scripts/spike_v4_e2e_keypoint.py`
- run:`logs/rsl_rl/mos92_velocity/2026-06-08_20-15-03_spike_v4_e2e_keypoint`
- 日志:`/tmp/v4_kp5f_full.log`
- 状态:Python 训练进程已不在,日志停在 `Learning iteration 433/3800`,最新 checkpoint 到 `model_400.pt`。

### 标签/GT 泄漏检查

- `agent.yaml` 中 actor obs_groups 为 `actor/camera/camera_rgb`;critic 为 `critic`;`keypoint_label` 是单独 obs group。
- RSL-RL `CNNModel/MLPModel` 会按 `obs_groups[obs_set]` 选择输入,不是拼所有 observation。
- 因此当前未发现 `keypoint_label` 直接泄漏进 actor/critic 输入。
- `keypoint_label` 进入 rollout storage 并由 `KeypointAuxPPO` 的 aux loss 使用,这符合 A' 设计。
- 仍需注意:aux loss 当前对整个 actor 反传,会更新共享 RGB CNN、MLP trunk、运动输出相关参数,这可能破坏已学行为。

### 当前趋势

关键点监督信号确实在学:
- `Loss/kp_aux`:约 `0.269 -> 0.096`
- `Loss/kp_aux_pix_err`:约 `0.629 -> 0.331`
- `Metrics/kp_pixel_err`:约 `1.57 -> 1.15`,中途最好约 `0.553`,后段反弹。

但运动/踢球行为已经严重崩溃:
- `Episode_Termination/fell_over`:约 `5.07 -> 52.75`
- `Train/mean_episode_length`:约 `56 @ iter100 -> 7.36 @ iter435`
- `Metrics/dribble/step_count`:约 `56 @ iter100 -> 7.32 @ iter435`
- `Metrics/dribble/goal_rate`:全程 `0`
- `Episode_Reward/dribble_success`:接近 `0`
- `Metrics/kp_selfloc_pos_err_m`:约 `8.07 -> 5.73`,仍远不达标,且是在大量摔倒/短 episode 下得到的指标。

### 主要漏洞/疑点

1. **checkpoint 起点与目标不匹配**
   脚本实际 `BASE_CKPT` 是 `checkpoints/v3_soccer_solo/01_selfloc_purevision/model_2800.pt`,不是用户对标的
   `04_e2e_integrated/model_1499.pt`。这不利于保住 `04_e2e` 的踢球行为。

2. **大量参数被重初始化**
   日志显示 actor 只 loaded 15, reinit 10:
   `obs_normalizer`、`distribution.std_param 24->66`、`mlp.0.weight`、`mlp.6.weight/bias`、`camera_rgb`
   均重初始化。也就是说它不是在稳定 e2e 踢球策略上小修检测头,而是在大幅改 actor 输出结构后重训。

3. **aux loss 可能冲坏共享运动策略**
   `KeypointAuxPPO._train_keypoint_aux()` 使用 PPO optimizer 对 actor 全部参数 step。由于 selfloc 输出和 motor 输出共享 trunk,
   `coef=1.0` 的监督更新可能直接改坏 locomotion/dribble 表征。

4. **当前 run 不能继续作为有效主实验**
   到 iter 300 后已经稳定进入短 episode/高摔倒区。即使关键点 loss 继续下降,也只是“摔倒数据分布上的检测学习”,
   不能代表纯视觉踢球能力提升。

5. **Claude 的监控命令可能误判运行状态**
   发现 shell waiter 中用 `pgrep -f "spike_v4_e2e_keypoint.py"`,可能匹配到包含该字符串的监控脚本自身。
   后续应匹配 `.venv/bin/python scripts/spike_v4_e2e_keypoint.py` 或直接检查训练 PID。

### 给 Claude 的建议

- 这次 run 应标记为失败/中止,不要等满 3800,也不要因为 `kp_aux` 下降而写成 EXP5f 成功。
- 下一次先做 smoke/短 run 对照:
  1. `aux_coef=0` 或禁用 aux loss,确认同样 keypoint env/66-d action/partial load 是否本身就摔。
  2. `aux_coef` 从 `0.01/0.05/0.1` 递增,不要直接 `1.0`。
  3. aux loss 先只更新 `camera_rgb` 和 selfloc/keypoint head,冻结或保护 motor trunk/已有 locomotion 参数。
  4. 若目标是保住 04_e2e 行为,应从 `04_e2e_integrated/model_1499.pt` 或等价 e2e checkpoint 做更一致的迁移。
  5. 每次短 run 必须同时看 `kp_aux/kp_pixel_err` 和 `fell_over/step_count/goal_rate/dribble_success`。
- 若只想验证检测头可学,建议先用离线/rollout 数据训练 detector 或 oracle/noisy-oracle 路线,不要把完整运动策略一起暴露给大梯度。
