# Soccer Robot v4 Codex Supervisor Notes

> 监督用途。供 Claude Code 在继续 v4 前先读。最后更新: 2026-06-08 18:31 CST。
> v3 代表模型: `checkpoints/v3_soccer_solo/04_e2e_integrated/model_1499.pt`。

## 0. 正式落盘要求

1. `soccer_robot_v4.md` 和 `soccer_robot_v4_experiment.md` 是正式文档,`*-codex.md` 不是正式总账。
2. 任何方案调整,必须同步更新 `soccer_robot_v4.md`。
3. 任何实验结果,必须同步更新 `soccer_robot_v4_experiment.md`。
4. 比较成功的实验,必须额外生成正式工件:
   - checkpoint/params -> `mjlab/checkpoints/`
   - 视频/曲线/诊断图/报告 -> `mjlab/soccer_eval/`
5. 不允许只在 supervisor note、临时日志或 `*-codex.md` 中留下结果。

## 1. 已确认的事实

1. `05_realspec` 不踢球的第一层根因已经确认:
   depth(找球/踢球)和 RGB(自定位)共用同一个 `head_cam`。`05` 把相机从 `64x48` 提到 `96x72`
   时，depth 图也一起变了，导致 depth-ball CNN 的 `spatial_softmax` 从 `192 -> 432` 被 reinit。
   这是结构性破坏，不是猜测。

2. v4 的 `dualcam` 已经把这层问题修掉了:
   `mos92_soccer_selfloc_dualcam_env_cfg` 里 `head_cam` 回到 `64x48 depth-only`，新增
   `head_cam_rgb` 做 `96x72 rgb-only`。当前全量 run 的 load 日志是:
   - `cnns.camera(depth)` reinit `0`
   - `cnns.camera_rgb` reinit `3`
   这说明 depth-ball CNN 的迁移已经恢复。

3. 但“只修相机耦合”还不够:
   当前 run `logs/rsl_rl/mos92_velocity/2026-06-07_19-49-28_spike_v4_dualcam` 到 iter `980` 时:
   - `Episode_Reward/dribble_success` 当前 `0.0390`，历史最好 `0.2273 @ iter 686`
   - `Metrics/dribble/goal_rate` 当前 `0.1500`，历史最好 `0.3571`
   - `Metrics/selfloc_pos_err_m` 当前 `2.6371m`，历史最好 `2.6329m`
   - `Episode_Termination/fell_over` 始终 `0.0`
   结论: dualcam 已明显改善自定位并保持稳定，但踢球/进球还远没恢复到 `04_e2e`
   的 `dribble_success 0.51 / goal_rate 0.30` 水平。

4. 监督视角下，`04_e2e` 和 `05_realspec` 不能只看 mp4 直接比较:
   两者不是同一训练链。
   - `04_e2e` 来自 `mos92_soccer_e2e_env_cfg` + `model_2800 -> e2e_fix`
   - `05_realspec` 来自 `mos92_soccer_selfloc_realspec_env_cfg` + `model_2800`
   几何、奖励层、课程都不同。视频差异不能全部归因到感知链。

## 2. 现在最重要的监督判断

### 判断 A
“共用相机毁了 depth CNN”是 `05` 崩掉的真实根因之一，但不是当前 v4 全部问题的充分解释。

### 判断 B
当前 dualcam run 表明第二层瓶颈仍在，优先怀疑:
- bootstrap 起点不对: 还在从 `01_selfloc_purevision/model_2800.pt` 起跑，而不是从已经会完整
  端到端踢球的 `04_e2e_integrated/model_1499.pt` 起跑
- env 定义不对: 现在修的是 `selfloc_realspec` 链，而用户拿来对标的是 `04_e2e`
- reward / geometry 不同: `04_e2e` 的 goal layer、进攻半场几何、时间目标与 `05/selfloc` 链不同

### 判断 C
当前 run 不能再被描述成“问题已解决，只等收敛”。更准确的说法是:
“第一层结构 bug 已修复，但 v4 仍未恢复到 v3 代表模型的行为水平，需要新的对照实验。”

## 3. 给 Claude Code 的强提醒

1. 不要再把 `05_realspec-step-0.mp4` 的差行为只归因到“分辨率改坏了 depth CNN”。
   那只解释了第一层问题，解释不了当前 dualcam 仍然偏弱。

2. 后续实验必须控制变量，优先做“和 `04_e2e` 同口径”的 real-spec 对照:
   - 同类 env
   - 同类 bootstrap
   - 同类评估指标

3. 不要只看最后一个 iter 的数值，也不要只看 best。
   当前 run 已经出现“best 还行，但 current 掉下去”的情况，必须记录:
   - current
   - best
   - best 出现的 iter
   - 最近 100~200 iter 是继续上升还是已经塌陷

## 4. 推荐的下一步实验顺序

### Exp 1A
让当前 `spike_v4_dualcam` run 继续跑完，但加监督门槛:
- 若到 `iter >= 1500`，`best dribble_success < 0.30`，则判定“未恢复到可接受踢球水平”
- 若 `current dribble_success` 长时间低于 `best` 的 40%，记为“中后期塌陷”
- 无论结果如何，都把这次 run 定位为“dualcam selfloc 链验证”，不要误写成“v4 主线已成功”

### Exp 1B
新增 `realspec_e2e_dualcam` 主线，优先级最高。
- 基础 env: 从 `mos92_soccer_e2e_env_cfg` 出发，不是从 `selfloc_realspec` 出发
- 改动:
  - field line width -> `0.125m`
  - depth 相机保持 `64x48`
  - 新增 `head_cam_rgb`
  - RGB stack / stride 按 realspec 方案接入
- bootstrap: 优先从 `checkpoints/v3_soccer_solo/04_e2e_integrated/model_1499.pt`
- 目标:
  - 保住 e2e 端到端行为
  - 再看真场线是否把 selfloc 拉坏
- 判据:
  - `dribble_success >= 0.35`
  - `goal_rate >= 0.20`
  - `fell_over = 0`
  - `selfloc_pos_err_m` 比 `05`/当前 dualcam 更好

这是最重要的新方法，应该明确提醒 Claude 去试。

### Exp 1C
如果 Exp 1B 仍然弱，做 bootstrap 对照:
- `model_2800 -> realspec_e2e_dualcam`
- `04_e2e/model_1499 -> realspec_e2e_dualcam`

目的:
拆出“是感知链问题”还是“起点 policy 已经缺少完整踢球/进球策略”。

### Exp 2
在 Exp 1B 有基本踢球能力后，再提 RGB 分辨率:
- `96x72 -> 256x192`
- 必要时 `512x384`
- depth 仍固定 `64x48`

### Exp 3
实现距离自适应精度 reward。
插点明确:
- 当前 reward 在 `src/mjlab/tasks/velocity/mdp/rewards.py:selfloc_accuracy`
- 当前 env 接在 `env_cfgs.py` 的 `selfloc_accuracy` / `selfloc_error_penalty`

建议实现成新 reward term，而不是原地把老 reward 改得不可对照。

### Exp 4
若高分辨率回归定位仍卡住，再上几何方法:
- 关键点 + PnP
- 多峰歧义再加 MCL / 粒子滤波

## 5. 监督红线

以下情况出现时，Claude 应被提醒立即停下复盘，而不是继续长跑:

1. 继续把不同 env 的视频直接拿来下因果结论
2. 只汇报“当前看起来不错”，不报 best/current/趋势
3. 新实验没有写清:
   - 目标是什么
   - 为什么做
   - 评估门槛是什么
   - 失败后转向什么
4. 在还没做 `realspec_e2e_dualcam + 04_e2e bootstrap` 前，就直接跳去粒子滤波

## 6. 当前监督结论摘要

当前最值得提醒 Claude 的一句话是:

> `dualcam` 已证明相机耦合 bug 被修掉，但它还没有把行为恢复到 `04_e2e` 水平；下一步主线应切到
> “`04_e2e` 同口径的 real-spec dualcam e2e 对照”，而不是继续把 `selfloc_realspec` 链当作
> `04_e2e` 的直接替身。

## 7. 2026-06-08 18:31 监督意见: depth 属于允许的纯视觉,但要守住无特权边界

用户已明确:这里的“纯视觉”允许使用 onboard depth。也就是说,`RGB/depth + 机器人本体状态/运动学`
属于可接受输入;禁止的是策略或正式评估依赖外部定位、真实位姿 obs、sim world pose 等特权信息。

因此,当前 `关键点检测 + depth 抬 3D + Kabsch/PNP + 必要时 MCL` 方向是合理的,不应因为用了 depth
而被误判为偏离“纯视觉踢球”目标。更准确的监督口径是:
- 训练标签可以用 sim 几何自动投影生成,这是监督信号,不是部署时输入。
- 策略推理和正式评估不能吃 `robot_field_pose`、GT pose obs、world-frame camera pose 等特权状态。
- 若几何链当前为了 monitor 使用 sim world pose,必须最终给出等价的非特权实现或验证:用相机到 base 的标定/运动学
  把 depth 点转到 base frame,再由地图配准求 field pose。
- 最终交付判据仍是“无 GT 拐杖的纯视觉踢球”:`kp_selfloc_pos_err_m` 稳定达标,同时保住 `goal_rate/dribble/fell_over`。

给 Claude 的建议要保留发挥空间,不要把他锁死在某一个实现细节:
- 可以继续试 `visible/K` 这类 reward 修复,但必须把 `kp_visible`、`kp_pixel_err`、`kp_selfloc_pos_err_m`、`goal_rate`
  一起看。只看 pixel err 会再次漏掉“看不见也高分”的漏洞。
- 若 RL 奖励学关键点仍不稳,优先考虑把关键点检测拆成更直接的监督学习/辅助 loss,再接回 e2e。关键点检测本质是
  dense supervised vision task,不一定适合完全靠 PPO reward 学。
- 若单帧几何定位达标但轨迹抖动或对称歧义明显,再加 MCL/粒子滤波。MCL 应作为时序消歧和稳定器,不要用它掩盖
  单帧关键点检测没有站稳的问题。
- 若关键点可见数长期低,可以开放尝试 gaze/active scan、可见性奖励、关键点分组、置信度门控、RANSAC/Kabsch
  鲁棒估计等方法,但每次只需要说明目标、评估门槛和失败后的转向。

当前监督结论:
> Claude 已经转到更有希望的范式,方向值得继续;接下来要严防两类质量问题:
> 1) 用特权状态完成了几何链却误报成纯视觉;
> 2) 关键点检测 reward 被策略继续 gaming,导致定位指标或踢球行为不真实达标。

## 8. 2026-06-08 19:35 EXP5f 监督建议:优先本地 A' 扩展,不要直接改 pip 包

Claude 对 rsl_rl 约束的判断基本正确:监督 loss 需要和每个 rollout 样本对齐的 `uv_gt/vis` 标签,
这些标签依赖当时的相机外参和 env origin,默认 PPO update 阶段拿不到“当前 env 状态”。因此标签必须在
rollout 时进入后续 batch,否则监督 loss 无法可靠对齐。

但路径 A 不必理解成“直接改 `.venv/site-packages/rsl_rl`”。本地源码确认:
- `RolloutStorage` 会把 `TensorDict` observations 的所有 key 存进 buffer。
- actor/critic 只消费 `cfg["obs_groups"]` 指定的 key。
- 因此可以新增一个标签 obs group,例如 `keypoint_label`,让它进 rollout storage,但不放进 actor/critic
  的 obs_groups,策略就不会吃标签。
- 再在本项目里写本地 `KeypointAuxPPO` / `KeypointAuxRunner` 或最小继承类,通过 `algorithm.class_name`
  指向本地类,避免修改 pip 包。

推荐顺序:
1. **A' 本地扩展优先**:标签作为非策略 obs 进入 storage + 本地 PPO subclass 在 update 里加 aux loss。
   这是和 on-policy 数据最对齐的方案,也最符合最终 e2e 联训目标。
2. **B 作为 fallback 或预训练阶段**:离线监督检测器风险低,适合快速证明检测头可学;但和 e2e actor trunk
   衔接复杂,不应替代最终 on-policy/e2e 验证。
3. **C 暂不推荐**:采样循环里在线 backward 会和 PPO 的旧策略采样/optimizer/normalizer 交互复杂,
   更容易引入难查的不稳定。

给 Claude 的自由度:
- 不限制必须用 exact A'。如果他能找到比 local PPO subclass 更小的本地 hook,可以用。
- 关键边界是:标签必须随 rollout 样本对齐进入训练 batch;标签不能进入 actor/critic 输入;不要直接污染
  site-packages;每步实现都要 smoke 验证 label shape、obs_groups 排除、aux loss 能下降。

## 9. 2026-06-08 监督提醒:主动扫视的核心是几何/信念融合,不是简单堆帧

用户指出一个关键问题:机器人视角有限,真正踢球时需要转头扫描;转头过程中看到的多张图必须能“联系起来”。
这里最重要的监督判断是:

> 多帧图像之间有效的联系,不应只是把 RGB/depth frame concat 给 CNN;而应是“每帧检测结果 + 该帧相机/头部位姿
> + 运动模型/历史动作”共同更新一个持续存在的场地位姿 belief。

给 Claude 的提醒:
- 单帧关键点检测先做准,EXP5f 的监督 loss 仍是必要地基。没有稳定的单帧检测,MCL/粒子滤波只会平滑错误。
- 多帧扫描要记录每帧的时间、head yaw/pitch、camera-to-base 外参或等价运动学信息。否则网络不知道每张图是朝哪里看的,
  raw stack 会把几何关系全交给 CNN 隐式学,风险很高。
- 融合层优先做可解释 baseline:最近 N 帧关键点检测 + depth 抬 3D + 按相机位姿变换到共同坐标系,
  再用 RANSAC/Kabsch/PnP 或粒子滤波维护 pose belief。学出来的 RNN/Transformer 可以后置,不应作为第一验证方式。
- 主动转头的奖励/策略目标应是“降低位姿不确定性、增加独立且几何分布好的关键点”,而不是单纯提高 visible 数。
  只盯着几个容易点或同一条线上的点,可能 visible 很高但定位仍不可解或高度歧义。
- 需要显式处理置信度、可见性和离群点。错误关键点必须能被 RANSAC/鲁棒权重/粒子权重压下去,不能让单个误检带崩定位。
- 允许使用 RGB/depth、头部关节角、相机标定、机器人本体状态、动作历史和场地地图;仍禁止正式策略/评估使用 GT pose、
  `robot_field_pose`、sim world pose 或外部定位。

建议增加的指标:
- `single_frame_kp_pixel_err` 和 `single_frame_pose_solve_rate`:单帧检测是否已经能在可见点足够时解位姿。
- `multi_frame_pos_err/yaw_err`:多帧融合是否真的比单帧更稳。
- `pose_uncertainty` 或粒子分布熵:主动扫视是否在减少不确定性。
- `kp_visible_unique` / landmark spatial diversity:看到的是不是独立、分散、可用于定位的点。
- `time_to_localize`:从丢失位姿到恢复到可踢球精度需要多久。
- `goal_rate/dribble/fell_over`:最终仍要回到纯视觉踢球行为,不能只交付定位 demo。

不限制 Claude 的发挥:他可以选择 MCL、EKF、滑窗 RANSAC、可微融合或 learned memory。监督边界只有一个:
多帧必须通过相机运动和共同坐标/共同 belief 建立约束,不能只说“堆了历史图像”就算解决了主动扫视。

## 10. 2026-06-08 监督提醒:跨帧冲突要形成带不确定性的场景 belief

用户进一步指出:同一个关键点在两张图里可能解出不同位姿;单帧关键点很少时,定位会非常困难。这里不能要求策略
每帧都给出一个确定答案,更合理的目标是让机器人在“脑海中”维护一个受几何约束的场景想象。

这里的“场景想象”应理解为 **belief/world model**,不是无约束 hallucination:
- 已知场地地图提供先验:边线、球门、中心线、角点等在地图中的位置是固定的。
- 每一帧只提供局部、有噪声、可能误检的观测。
- 机器人用头部角度、相机外参、动作历史/里程计把这些局部观测放进共同 belief。
- belief 需要保存不确定性和多峰假设:例如“我可能在 A 位姿,也可能在对称的 B 位姿”。
- 新帧进来时,如果和当前 belief 一致,提高该假设权重;如果冲突,降低该观测权重或触发多峰分裂,不要直接把位姿拉偏。

给 Claude 的具体提醒:
- 同一关键点跨帧解出不同位姿是正常现象,可能来自检测噪声、depth 噪声、头部标定误差、机器人运动估计误差、
  关键点 ID 混淆或场地对称性。实验记录应区分这些来源,不要把所有冲突都归为网络没学好。
- 单帧关键点少时,不要强行要求 Kabsch/PnP 给确定 pose。应该允许输出低置信度/高不确定性,由历史 belief 和后续扫视补足。
- 如果可见点少但历史 belief 强,可以用运动模型传播旧位姿作为 prior;如果历史 belief 也不确定,就应主动扫视最能消歧的位置。
- 跨帧融合可以从三种 baseline 做起:滑窗 RANSAC/加权最小二乘、EKF/UKF、MCL/粒子滤波。若 Claude 选择 learned memory,
  也应保留这些可解释 baseline 作对照。
- active scan 的策略不只是“寻找更多点”,而是寻找能最大减少 belief entropy / pose covariance 的视角。

建议补充评估:
- `reprojection_residual_per_frame`:当前 belief 回投到每帧图像后的残差,用于发现冲突帧。
- `outlier_ratio`:被 RANSAC/鲁棒权重拒绝的关键点比例。
- `belief_entropy` / `pose_cov_trace`:机器人是否真的更确定。
- `hypothesis_switch_count`:多峰假设是否频繁跳变。
- `low_keypoint_recovery`:单帧可见点少时,多帧 belief 是否仍能维持可踢球定位。

## 11. 2026-06-08 监督提醒:主动感知可能需要全身动作,不只是转头

需要提醒 Claude:视角受限时,头部扫描是最低成本动作,但不一定足够。头部只改变相机朝向,并不改变机器人本体在场地中的
观察基线;遇到遮挡、关键点分布退化、左右/前后对称、多帧都只看到同一片区域时,可能必须通过身体 yaw、原地小步转向、
侧移或后退来获得新的几何约束。

建议把 active scan 扩展为 **whole-body active perception**:
- 先用头部动作快速搜索,因为成本低、对步态影响小。
- 若 belief 仍高不确定或多峰,允许身体/足部执行低风险转向或小步位移,主动制造新的视差和场地覆盖。
- 身体动作必须和稳定/控球/射门准备共同优化,不能为了看线而破坏 dribble 或频繁摔倒。
- 可把“定位恢复动作”和“进攻动作”做成门控:belief 不确定时优先安全扫视/转身;belief 稳定后再高速追球和射门。

给 Claude 的实验建议:
- 做 head-only 与 head+body yaw / small-step-turn 的对照,看是否显著降低 `time_to_localize` 和多峰歧义。
- 记录信息收益与代价:`belief_entropy` 下降、`kp_visible_unique` 增加、`ball_lost_rate`、`fell_over`、速度损失和进球率。
- 不要把身体动作奖励写成“多转就好”。目标应是单位代价的信息增益和最终踢球成功率,否则策略可能学会无意义原地转圈。
- 若加入身体/足部扫描,必须保护已有 locomotion/dribble 能力:限制转向幅度、触发条件、恢复条件,并检查是否影响 kick-ready 姿态。

## 12. 2026-06-08 监督提醒:全身转动要做成稳定动作原语,不要让策略自由乱转

用户进一步担心:如果引入身体/足部转动,如何避免机器人步态崩溃或训练学坏。这里的监督建议是把全身主动感知做成
“稳定运动原语 + 高层门控”,而不是把身体动作完全交给一个视觉奖励去自由探索。

建议 Claude 采用的协调原则:
- 低层 locomotion/dribble 稳定性优先:全身扫描应输出受限的 yaw-rate、step-turn、side-step、back-step 等命令,
  不要直接破坏已有关节级步态策略。
- 高层 belief 门控:只有当 `pose_uncertainty` 高、关键点不足、多峰歧义明显或定位长期漂移时,才触发身体扫描。
- 分级动作:head scan 先尝试;若仍不确定,再做小幅 body yaw;最后才考虑侧移/后退。动作幅度按不确定性逐级增加。
- 相位/姿态安全:转身命令要限制角速度、角加速度、步长和持续时间,并监控 base roll/pitch、足底接触、滑移、动作幅度。
- 任务安全:控球/射门阶段不要突然大转身。需要有 search mode、dribble mode、kick-ready mode 的切换条件和恢复条件。
- 失败回退:若 tilt、slip、ball_lost 或 pose uncertainty 继续恶化,立即退出扫描,回到稳定站立/找球/恢复步态。

推荐奖励/代价形式:
- 信息收益:belief entropy 或 pose covariance 下降、unique landmarks 增加、reprojection residual 下降。
- 稳定代价:fall_over、base tilt、foot slip、action rate、关节速度/力矩峰值、步态中断。
- 任务代价:ball_lost、dribble continuity 下降、离球变远、kick-ready 恢复时间变长。
- 总目标不是“转得多”,而是“用最小动作代价获得足够定位置信度并继续踢球”。

建议实验从安全到复杂:
1. 原地稳定小幅 yaw scan:只验证不摔、不明显丢球、能降低不确定性。
2. gait-phase-aware step-turn:在行走/控球节奏中插入小步转身,看是否保持 dribble。
3. recovery-gated scan:不确定时进入扫视,置信度恢复后自动回到追球/射门。
4. 对照 head-only、head+body-yaw、head+body-yaw+small-step,比较信息收益和踢球损失。

监督红线:
- 若 `fell_over`、`ball_lost_rate`、action jerk 明显上升,即使定位指标变好也不能算成功。
- 若策略出现原地频繁转圈、一直找线不进攻、为了看关键点放弃球,说明 reward/门控失败。
- 全身主动感知应作为恢复定位的临时模式,不能吞掉最终的 dribble/goal 目标。

## 13. 2026-06-08 监督提醒:仿真训练要分层课程化,不要一锅端端到端

要达到“纯视觉 + 多帧 belief + 主动扫视/全身转动 + 踢球”的理想效果,仿真训练应分阶段解锁能力。
如果直接把所有机制塞进一个 PPO reward 里,失败时很难定位责任:可能是关键点没学会、几何融合错、belief 不稳、
身体转动破坏步态,也可能是踢球 reward 与定位 reward 冲突。

建议 Claude 采用的训练路线:

1. **单帧感知监督阶段**
   - 目标:关键点检测、可见性/置信度和 depth 使用先站稳。
   - 做法:用 sim 自动投影生成 `uv_gt/vis`,训练辅助 loss;策略推理不能吃 GT pose。
   - 门槛:`kp_pixel_err` 下降、`kp_visible` 不塌、单帧足够点时 Kabsch/PnP solve rate 达标。

2. **几何融合离线/半在线验证阶段**
   - 目标:证明多帧观测 + head/camera 位姿 + 运动模型能形成稳定 belief。
   - 做法:先用记录下来的 rollout 序列测试滑窗 RANSAC/EKF/MCL,再接入训练 loop。
   - 门槛:multi-frame pose error 低于 single-frame;低关键点场景不强行输出假精确;冲突帧能被降权或保留多峰假设。

3. **主动感知课程阶段**
   - 目标:让机器人在 belief 不确定时主动获取信息。
   - 顺序:head-only scan -> head+small body yaw -> step-turn/side-step/back-step。
   - 奖励:不确定性下降、unique landmarks 增加、time-to-localize 缩短。
   - 代价:摔倒、滑移、action jerk、丢球、远离球、dribble 中断。

4. **低层运动保护阶段**
   - 目标:身体扫描不摧毁已有 locomotion/dribble。
   - 做法:冻结或强约束低层步态/控球能力,高层只输出有限动作原语;逐步放开幅度和触发场景。
   - 门槛:`fell_over`、`ball_lost_rate`、base tilt、foot slip 不明显恶化。

5. **完整踢球集成阶段**
   - 目标:belief 稳定后回到追球、带球、射门。
   - 做法:加入 mode gate 或 learned gate:不确定时安全定位恢复,确定后进攻。
   - 门槛:定位指标、`goal_rate/dribble_success/fell_over` 同时达标,不能只交付定位 demo。

6. **鲁棒性和 sim-to-real 随机化阶段**
   - 随机化:光照/纹理、相机内外参、头部标定误差、depth noise/dropout、帧延迟、摩擦、球大小/颜色、遮挡、
     初始位姿、场地线宽/磨损。
   - 目标:让 belief 和 active perception 对真实相机误差、延迟和观测缺失不过度脆弱。

训练监督红线:
- 每阶段必须有独立成功门槛,不能用最终 goal_rate 掩盖中间模块失败。
- 训练时可以用 GT 生成标签和 monitor,但部署输入与正式评估不能依赖 GT pose/world pose。
- 若加入 learned memory,必须保留几何 baseline 对照;否则无法判断网络是否真的学会跨帧几何。
- 若加入全身动作,必须先通过安全课程,不能一开始在完整踢球 reward 里放开大幅身体动作。

## 14. 2026-06-08 监督提醒:先用 oracle/noisy-oracle 拆责任,再接真实视觉

为了避免“视觉检测、belief 融合、主动扫视、身体动作、踢球策略”互相背锅,建议 Claude 在仿真中加入
oracle/noisy-oracle 分层验证。它不是最终方案,而是训练和诊断工具。

推荐模块接口:
- `Perception`:输入 RGB/depth,输出 landmark id、2D uv、depth/3D 点、visibility、confidence。
- `BeliefEstimator`:输入当前检测、head/camera pose、动作历史/里程计、地图,输出 pose belief、uncertainty、multi-modal hypotheses。
- `ActivePerceptionPolicy`:输入 belief 状态和任务状态,输出 scan mode 或动作原语,例如 head scan、body yaw、step-turn。
- `SoccerPolicy`:输入 belief 摘要、球观测、本体状态,执行追球、带球、射门。

推荐诊断顺序:
1. **GT landmark oracle**
   - 用真实投影关键点和真实 visibility 替代检测器,但不把 robot GT pose 直接给策略。
   - 目的:验证 Kabsch/PnP/EKF/MCL、belief 更新、主动扫视和踢球控制在“感知完美”时是否可行。
   - 若 oracle 下仍不行,问题不在 CNN,而在融合/动作/奖励/任务协调。

2. **Noisy oracle**
   - 对 oracle keypoint 加入 pixel noise、depth noise、dropout、错 ID、延迟和遮挡。
   - 目的:找出 belief 融合和鲁棒估计的容错边界。
   - 门槛:噪声逐步增大时,pose error/uncertainty 应平滑退化,不能突然崩溃。

3. **Learned detector + frozen belief**
   - 用训练好的关键点检测器替换 oracle,belief/fusion 先固定。
   - 目的:单独评估视觉误差对定位和踢球的影响。
   - 若指标明显掉,优先修 detection/confidence/outlier handling,不要急着改踢球策略。

4. **Learned detector + learned/optimized active perception**
   - 逐步放开 active perception 和身体动作。
   - 目的:让机器人在真实视觉误差下学会主动补信息。
   - 仍需保留 head-only、head+body、不同 fusion 方法的消融。

5. **端到端微调**
   - 最后才把 detector、belief 摘要、active perception 和 soccer policy 联合微调。
   - 需要较小 learning rate、保留 aux loss、保留稳定性惩罚,避免新 reward 把已学好的视觉/步态冲掉。

可用的 teacher/课程信号:
- 在 sim 中用 GT 只计算训练信号:候选 head/body 动作的信息增益、expected visible unique landmarks、expected entropy reduction。
- 用这些信号训练 active perception 的 teacher 或 reward,但正式推理不能使用 GT pose。
- 候选动作可以先离散化:look-left/right/down、small yaw left/right、step-turn、side-step、back-step,降低探索难度。

关键监督点:
- oracle 阶段成功不等于纯视觉成功,只能说明后端和动作策略有上限。
- noisy-oracle 阶段能给出误差预算:检测器需要达到怎样的 pixel/depth/ID 准确率才值得接入。
- learned detector 接入后,若失败,要用 oracle 对照定位失败来源,不要盲目加复杂网络。
- 端到端微调只能作为最后整合,不能替代模块级验收。
