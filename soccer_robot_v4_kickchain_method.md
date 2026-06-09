# Soccer Robot v4 — 踢球链方法论与路线图(线B)

> 2026-06-09 创建。线A(自定位)经 EKF 重构基本解决(EXP13 ~1.03m)。本文件是线B
> (踢球链——goal_rate 上不去的真命门)的方法论 + 路线图。来源:GitHub/arXiv 调研
> + codex EXP7C8-C21 诊断 + 对我方实际 reward 代码的核实(三方交叉)。

---

## 一、根因诊断(三方交叉确认,已钉死)

### 症状
goal_rate 卡 0.03(目标>0.2),ball_to_target_error 4.7m 几乎不动,球常被粘住/卡住。
机器人能看见球、能走近(robot_to_ball 0.6m),但无法把球从"看见"输送到"近脚窗口"并踢进门。

### 排除了奖励问题(核实自家代码,推翻调研的简单猜测)
调研猜"奖励了占有/接近而非球朝目标运动"——**核实 EXP13 实际 reward,此猜测不成立**:
- 已有 `goal_progress`(+2.0)= 球速朝球门方向投影(正是 DribbleBot 范式)。
- 已有 `dribble_ball_velocity_to_target`(+0.5)= 球速朝目标投影。
- 粘球/卡球已重罚:holding_ball -2.0、ball_trapped -3.0、ball_sticking -1.0。
→ 奖励设计已成熟,吸收了 DribbleBot 核心思想。**问题不在奖励层。**

### 真根因(codex C20a 决定性发现 + 上帝视角对照)
**球到脚边时,头部相机已经看不到它**(C20a:有效触球瞬间 seg_visible=0.000,球在脚下/太近出视野)。
触球发生在 x≈0.09m, |y|≈0.025m, dist≈0.133m 的极近脚窗口,且接触时仍有前进命令(vx≈0.29)。
→ 真正触球那一刻机器人是"盲踢"的。所以:
- 视觉链进不了近脚窗口(它依赖看见球,但球到脚边就看不见)。
- 上帝视角在近脚窗口注入推球命令也只到 goal 0.018 vs 上界 0.68——因为**冻结的 04_e2e 策略
  不会在"看不见球"时用 belief 盲推**。
**结论:缺的是一个独立的"近脚盲推/踢球技能(contact primitive)",输入是短时 ball belief
(几帧前看到的球+自运动推算球现在在脚下哪),不是当前帧视觉。** 这是个需专门训练的技能,
不是调奖励/调阈值能解决的(codex C8-C21 十几轮已穷尽奖励/阈值/手写伺服,全部失败)。

---

## 二、调研给的成熟方法(arXiv/GitHub)

### 方法A:Ball-Velocity-Tracking 主轴(DribbleBot 范式)— 我们已部分实现
DribbleBot(MIT,arXiv:2304.01159,开源 gmargo11.github.io/dribblebot)主奖励=球速匹配命令速度
`exp(-|v_ball-v_cmd|²/σ)`,v_cmd=朝球门方向×期望速度。我们的 goal_progress 已是此思想。
人形版:Learning Agile Humanoid Dribbling(arXiv:2505.12679),强调 locomotion-dribbling 解耦课程。
→ 我们已有,不是瓶颈。但可检查 goal_progress 权重是否够大、是否被其他项淹没。

### 方法B:独立训练"踢球/近脚推球技能"(最对症,对应 codex contact primitive)
- 在近脚窗口(x≈0.08-0.15m,|y|<0.1m)直接 spawn 球做 episode reset,**专训"球在脚边→踢进门"**,
  先不管接近。直接把 codex 那个 0.018 的对照实验打上去。
- 课程式接触:先在"球已在脚正前方"窄初始分布训,学会再扩大分布。与近脚窗口概念吻合。
- 参考:Hierarchical RL for Vision-Guided Soccer(arXiv:2603.00948,approach/align/kick 分层)。

### 方法C:参考踢腿轨迹 + 接触时机奖励(解决"精确时机爆发踢击")
- 生物力学踢腿参考(arXiv:2407.14612):给一条踢腿参考轨迹,RL 跟踪+微调,比纯 reward 摸索快得多。
- 动态监督(arXiv:2403.14300):参考信号引导接触时机。
- Impulse 奖励:检测脚-球接触事件,奖励传给球的冲量/球速增量 Δv_ball 朝门投影(比持续距离奖励更教"何时发力")。
- obs 加"球相对脚的位置+速度"+ 足够历史帧,让策略预测接触时刻、提前蓄力。

### 方法D:Teacher-Distillation 合并专家(衔接断裂的根治,工程量最大,最后做)
DeepMind Learning Agile Soccer Skills(arXiv:2304.13653):
- 阶段1 单独训 teacher(get-up / soccer),各只解子问题。
- 阶段2 训单一 student,**按状态选 teacher 做行为正则**(摔倒态→get-up,比赛态→soccer),
  正则权重随训练退火,让 student 超过 teacher 学出衔接动作。叠加 self-play。
→ 映射:接近态(球远)正则到接近专家,近脚窗口态正则到踢球专家(方法B训出),退火让 student
  自己学"输送→踢"的衔接——这正是我们断裂的环节。

---

## 三、线B 路线图(排序,依赖关系明确)

**B1(先做,最对症)**:实现短时 ball belief(base-frame ball xy + belief age/confidence + 近几帧球速估计),
作为 contact primitive 的输入。这是 codex 明确指出的缺失件——触球时看不见球,必须靠 belief。
可借鉴线A 的 EKF 思路做一个球的小滤波器(predict 用自运动+球速衰减,update 用可见时的球检测)。

**B2**:在近脚窗口 spawn 球专训"近脚盲推/踢球技能"(方法B+C)。用 impulse 奖励 + 可选踢腿参考轨迹。
输入=B1 的 ball belief。判据:这个独立技能能否把 codex 的 0.018 打到接近上界 0.68。

**B3**:用 teacher-distillation(方法D)把"接近专家(现有)"+"踢球专家(B2)"合并成端到端策略,
按状态正则,退火。判据:全任务 goal_rate 能否破 0.2、不破坏自定位与稳定性。

**注意事项**:
- 线B 全程仍 oracle 检测(隔离变量,与线A 同纪律)。
- 每步用多指标判读(scripts/readout_v4_metrics.py),codex 教训:绝不只看单指标。
- 警惕技能干扰(codex C8 射门退化、EXP10 neck甩头破稳):B3 蒸馏正则就是防这个。

---

## 四、关键链接
- DribbleBot(开源,读 reward):gmargo11.github.io/dribblebot ;arxiv.org/abs/2304.01159
- 人形 dribbling:arxiv.org/abs/2505.12679
- 分层足球:arxiv.org/html/2603.00948v1 ;dribble-hrl.github.io(arxiv.org/abs/2504.14989)
- 生物力学踢腿:arxiv.org/html/2407.14612v1 ;动态监督 arxiv.org/html/2403.14300v1
- DeepMind 蒸馏:arxiv.org/abs/2304.13653 ;精读 summary.j3soon.com/posts/learning-agile-soccer-skills-for-a-bipedal-robot-with-deep-reinforcement-learning
- 反应式 approach/align/kick:arxiv.org/abs/2511.03996 ;striker skills arxiv.org/abs/2512.06571
- codex 诊断:mjlab_codex_v4/soccer_robot_v4_experiment.md EXP7C20a(根因)、C21a(上帝视角对照)
