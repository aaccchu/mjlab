# Soccer Robot v4 — 踢球链方法论:调研 + codex 经验 + 共同诊断

> 2026-06-09 创建。本文件汇总两条独立来源对"踢球技能(ball-at-foot → kick goalward)"的结论:
> ① 网络/学术调研(RoboCup + legged-RL 文献);② codex 在 `mjlab_codex_v4` 的实战经验(C20-C21 系列)。
> **核心价值在第三节"共同诊断"——两条独立来源指向同一组根因,可信度高。**
> 配套:自定位方法论见 `soccer_robot_v4_localization_method.md`。

---

## 一、codex 的实战经验(C20-C21,反证关键)

codex 走的是**"冻结会踢球的 teacher + 外挂 contact primitive 给纯视觉链"**的路线,与我的端到端 RL 不同。

### codex 试过的子方法(均在 mjlab_codex_v4/soccer_robot_v4_experiment.md)
| 方法 | 思路 | 离线指标 | 闭环结果 |
|---|---|---|---|
| contact_gate (C21c) | MLP 二分类"何时该接管"(20维belief特征,不喂teacher动作) | AUC 0.86/F1 0.93 | effective_contact 仅 0.0011 |
| contact_action_bc (C21d) | MLP 回归 teacher 的20维raw动作,触发时replace/blend | l2_imp 0.35 | **失败:fall暴涨0.07-0.086** |
| contact_phase (C21g) | GRU+phase分类+3维速度命令回归 | phase_acc 0.87 | 0.0015,换seed掉回 |
| contact_push_quality (C21h) | GRU,标签=未来12步内有效触球/球速增益 | AUC 0.94 | **失败:gate交集仅0.0028** |
| contact_trigger (C21j) | GRU,标签="当前帧该接管" | **AUC 0.99/F1 0.92** | **失败:闭环触发塌到0.0014** |

### codex 的失败根因链(它自己的诊断)
1. **C21d**:BC raw 动作与冻结步态相位不兼容——强了摔、弱了不踢。
2. **C21h**:"未来会触球"≠"当前该接管",标签错位。
3. **C21j/k**:**deployment distribution shift**——所有训练数据来自 GT teacher 的上帝视角,
   视觉链闭环根本到不了那个状态分布。on-policy 采12.8万行,contact_window 行=**0**。
4. **C21l(最深洞察)**:真正最高杠杆不在近脚触球瞬间,而在**上游 context→window 的横向对齐**
   (`vy=0.4*y_b` 的横向命令把 contact_window 从 0.0024 拉到 0.0308,13×)。

### codex 验证的物理事实(我必须吸收)
- **触球瞬间相机100%看不到球**(C20a:step_effective_contact_seg_visible=0.000),球在脚下
  `x_b≈0.09m, |y_b|≈0.025m, dist≈0.13m`(脚坐标系)。→ 踢球 primitive 输入必须是短时 ball belief,
  不能靠当前帧像素。与我 memory"MOS92 足下盲区"一致。
- **球接触参数**:FIFA5号,半径0.11/质量0.43/friction(0.5,0.02,0.01)/solref(0.02,1)/
  solimp(0.9,0.95,0.001,0.5,2.0),恢复系数 e≈0.35。
- codex **没试过近脚 spawn**(球始终 0.6-1.5m spawn,靠 teacher 带到脚边),所以"近脚 spawn 穿透"
  这个坑 codex 没替我踩——我 EXP14 踩到了(下界0.08m→穿透→oob升高),EXP15 已修(0.25m)。

---

## 二、网络/学术调研结论(踢球链相关)

1. **奖励塑形**:每步累加的"球速投影"奖励有已知局部最优——策略学会"持续轻推"(积分收益 >
   一次大力踢后球离脚)。解法:对球速/冲量设**门槛**,只奖超过阈值的真踢。
2. **bootstrap 抑制探索**:从保守策略(轻柔接近)bootstrap 会继承坍缩的低熵动作分布,PPO 高斯
   不会自发重新探索"爆发踢腿"。解法:bootstrap 后**重置动作 std** 重开探索,PPO 自行退火。
3. **多技能干扰/遗忘**:teacher distillation(DeepMind arXiv:2304.13653,先训单技能专家再蒸馏)、
   phase-gating、residual policy。**但这是 codex 失败的路线**——见下方共同诊断。
4. **横向对齐**:把球摆到正脚前(惩罚 |y_b|)是 context→踢击窗口的高杠杆(与 codex C21l 独立吻合)。
5. **评估纪律**:多 seed 平均 + 同时盯 fall,别被单次进球的假阳性骗(codex C21g 反复栽在这)。

---

## 三、共同诊断 ★(两条独立来源的交叉印证,最高可信)

> 这是本文件最重要的部分:codex 实战 与 网络调研 从完全不同角度得出一致结论。

### 诊断1:端到端 RL 是对的路,不该上 BC/teacher distillation
- **调研**说多技能可用 teacher distillation。
- **codex 实战反证**:它正是走了"冻结teacher+外挂BC"的完整路线,离线AUC高达0.99,**闭环零进球**,
  死在两个杀手——(a)BC动作与冻结步态不兼容、(b)teacher数据分布≠部署分布。
- **共同结论**:我的端到端 RL **天然绕开这两个杀手**(无冻结步态、无teacher数据gap)。
  EXP14 端到端已 goal_rate 0.11 > codex 全系列(0进球)。→ **继续端到端,不回头上 BC。**

### 诊断2:goal_rate 卡 0.11 的根因 = 奖励局部最优 + 探索坍缩(非感知/非架构)
- **调研**独立指出:无门槛球速奖励→温吞推球;bootstrap→动作std坍缩→探不到大力踢。
- **codex 数据独立佐证**:它的 teacher 上界 success 0.68 但 effective_contact 仅 0.017——
  说明"能踢"和"踢得好/常踢"是两个技能;且 codex 一切努力都卡在"球送不到脚边/送到了不发力"。
- **实测症状印证**:EXP14 ball_speed 卡 0.39-0.40(温吞推球的指纹),goal_rate 卡 0.11。
- **共同结论**:瓶颈是**踢的动作质量**(奖励+探索),不是看不见球(感知线A已解到0.97m)、
  也不是架构。→ EXP15 改动直击此处:**速度门槛奖励(掐轻推)+ 重置std(重开爆发探索)**。

### 诊断3:足下盲区是设计约束,不是 bug
- **codex** 用上帝视角独立证明触球瞬间相机看不到球。
- **调研**(RoboCup)说踢球靠脚部感知/短时 belief,不靠当前帧。
- **共同结论**:近脚 spawn 让策略高频暴露在盲区状态,配合 ball belief(EKF/本体相对位置)
  是优势而非缺陷。不要试图"让相机看到脚下球"。

---

## 四、EXP15 改动映射(诊断→行动)
| 诊断 | EXP15 行动 | 文件 |
|---|---|---|
| 诊断2-奖励局部最优 | kick_impulse 加 speed_threshold=0.6 | rewards.py dribble_kick_impulse |
| 诊断2-探索坍缩 | bootstrap 后重置 std_param→1.0 | spike_v4_e2e_ekf_kick.py |
| (EXP14 spawn bug) | near_foot_dist 0.08→0.25m,rear_spawn→0 | env_cfgs.py / dribble_command.py |
| 诊断1 | 维持端到端,不引入 BC/teacher | — |
| 诊断3 / 横向对齐 | 候选后续:惩罚 |y_b| 把球摆正脚前 | 待 EXP15 结果定 |

## 五、关键链接
- DeepMind 足球技能蒸馏(反例参考):arxiv.org/abs/2304.13653
- 多技能干扰理论:huggingface.co/papers/2606.02398
- codex 实战:mjlab_codex_v4/soccer_robot_v4_experiment.md(C20-C21段)、v4工作的理解.md
