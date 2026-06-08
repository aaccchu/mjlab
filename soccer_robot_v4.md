# Soccer Robot v4 — 真场线下"会踢球 + 高精度自定位"

> 2026-06-07 起。v3 代表模型 04_e2e(端到端①②③,goal_rate 0.30/零摔倒/dribble_success 0.51)。
> v4 目标:在**诚实的 0.125m 真场线**下,既**保住踢球能力**,又把**纯视觉自定位**从 5.8m 提升到
> 接近 01 模型的 0.79m,并实现"距离自适应精度"。
> 实验记录见 `soccer_robot_v4_experiment.md`(每实验:目标→理由→评估→调整,跑完自动接下一步)。

## 交付纪律
- **正式方案文档以本文件为准**。后续只要 v4 方案、优先级、技术路线有变化,都必须同步更新这里,
  不能只改 `*-codex.md` 或临时 note。
- **比较成功的实验必须产出正式工件**:
  - checkpoint/params 保存到 `mjlab/checkpoints/`
  - 视频、曲线、诊断图、对比报告保存到 `mjlab/soccer_eval/`
- **成功与失败的实验结果都必须回写** `soccer_robot_v4_experiment.md`;不能只留在日志目录、
  tensorboard、终端输出或 supervisor note 里。

---

## 一、v4 要解决的三个问题(按因果顺序)

### 问题 1(必先修):05 不踢球 —— 共用相机的分辨率耦合
**根因(已确认)**:depth(找球/踢球)和 RGB(自定位)共用 `head_cam`。05 为自定位把分辨率
64×48→96×72,连带改了 depth 图,导致 depth-ball CNN 的 `spatial_softmax`(H×W 维)被 reinit,
从 model_2800 继承的踢球能力归零(dribble_success 0.51→0.00)。
**修复**:**分离两个相机传感器** —— `head_cam_depth`(64×48,踢球,完整迁移)+ `head_cam_rgb`
(高分辨率,自定位)。这是 v4 的地基,必须最先做且验证。

### 问题 2:0.125m 真场线纯视觉定位精度不足(5.8m)
**已知**:高分辨率视角图证明 1280×960 下角落位姿的线都清晰可辨 → 信号在,缺的是分辨率。
**修复**:RGB 分支用高分辨率(逐步试 256×192 → 512×384 → 更高),配合已实现的主动扫视+时序 stride。

### 问题 3:精度需求应随距离变化(用户的核心想法)
**现状**:`selfloc_accuracy` 全场均匀固定 std,逼机器人在无用的远距离难位姿也达标,浪费容量。
**修复**:精度奖励 std/权重随"到球/到球门距离"调制——远粗近精。可能顺带解 e2e 几何张力。

---

## 二、实验序列(便宜→贵,每步依赖前步结论)

### Exp 1 —— 分离相机,先恢复踢球(地基)
- **目标**:让真场线 env 里 depth 相机保持 64×48,RGB 相机独立(先也设 96×72),从 model_2800
  bootstrap 时 **depth-ball CNN reinit 0**(踢球能力完整迁移),dribble_success 恢复到 ~0.5。
- **为什么**:问题 1 是其它一切的前提——不会踢球,自定位再准也没用。先用最小改动(分离相机)
  验证"踢球能力回来了"。
- **怎么评估**:训练后看 dribble_success(目标 >0.4,对标 04 的 0.51)、goal_rate(>0.2);
  load 日志确认 `cnns.camera.spatial_softmax` **不再 reinit**(只 RGB 分支 reinit)。
- **如何调整**:若 depth CNN 仍 reinit → 检查分辨率是否真的没变;若踢球仍弱 → 检查 RGB 分支
  reinit 是否扰动了共享 MLP(可冻结 RGB 分支前几百 iter 让踢球先稳)。

### Exp 2 —— RGB 高分辨率,压自定位误差
- **目标**:在 Exp 1(会踢球)基础上,把 RGB 相机分辨率提到 256×192(必要时 512×384),看纯视觉
  自定位能否从 5.8m 降到 ~2m 或更低。
- **为什么**:视角图已证 0.125m 线在高分辨率下可辨,分辨率是主瓶颈。这步验证"分辨率换精度"。
- **怎么评估**:diag 脚本 play=False+mask=0 测纯视觉 selfloc_pos_err_m;同时 dribble_success
  不退(确认提分辨率没再伤踢球)。显存不够就降 num_envs（已知图像 obs 涨,1024→384/256）。
- **如何调整**:若 256 不够→上 512；若显存爆→降 envs 或只在 RGB 分支高分辨率（depth 仍 64×48
  省显存）；若精度到 2m 但停滞→进 Exp 4（关键点/PnP）。

### Exp 3 —— 距离自适应精度(用户核心想法)
- **目标**:精度奖励随距离调制——远(>3m)std 大宽容、近(<1m)std 小严格;并对"相对位置/深度"
  (近球时严)和"绝对自定位"(近球门时严)分别调制。看近球门定位精度是否进一步提升。
- **为什么**:匹配任务真实需求 + 把容量从无用远距离挪到有用近距离 + 可能解 e2e 几何张力。
- **怎么评估**:分距离段统计 selfloc_pos_err_m(近球门段应显著优于远段);进球精度(goal_rate)
  应提升;远段放松后整体训练是否更稳。
- **如何调整**:调制太激进致远处"放弃定位走错向"→ 设精度地板;分段统计看拐点调 std_far/std_near。

### Exp 4(进阶,视 Exp 2/3 结果)—— 关键点检测 + PnP / 或 MCL
- **目标**:若回归式定位在高分辨率下仍达不到亚米级,改用 CNN 检测场地关键点(角点/罚球点/中圈)
  + 已知地图解 PnP(几何约束硬,精度上限高,sim 里 GT 可自动生成监督)。
- **为什么**:回归是"软"映射,几何方法精度上限更高、更可解释、更接近真机/RoboCup 做法。
- **怎么评估**:关键点检测准确率 + PnP 解出的位姿误差 vs 回归基线。
- **如何调整**:关键点太稀疏的位姿→回退到回归或融合;若对称歧义严重→上粒子滤波 MCL 表达多峰。

### 失败回退原则
- 每个 Exp 跑完先看"是否达到该步评估门槛";达不到先诊断根因(对照实验,像 e2e 那次),
  连续两次同向失败就换机制而非继续调参(纪律)。
- 任何"退化"都要用对照实验排除"几何/分辨率/条件不一致"的误归因(v3 的教训)。

---

## 三、工程注意(从 v3 踩坑总结)
- **比较必须控制条件**:play=False vs True、mask=0 是否真为 0、几何分布是否一致(v3 多次误判)。
- **诊断纯视觉**:删 selfloc_gt_mask 课程 + 硬置 GT obs term scale=0(课程会每步重算覆盖
  common_step_counter)。
- **显存**:图像 obs 随分辨率平方涨,提分辨率必同步降 num_envs;用 PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True。
- **训练用 harness 等待器**轮询完成(避免 stream timeout),不手动 sleep 轮询。
- **spatial_softmax 维度=H×W**,任何分辨率改动都会 reinit 它——这是分离相机的根本原因。

---

## 四、v4 执行中的认知修正(2026-06-08,实验驱动)

> 监督者纠正 + EXP1A/1B/1C 实跑后,对问题的理解发生重要修正。原计划(分辨率是主瓶颈)不全对。

### 修正 1:05 不踢球 = 两层问题,不是一层
- 第一层(已修):depth/RGB 共用相机,提分辨率毁 depth CNN。dualcam 分离相机后 depth reinit 0,踢球可继承。
- 第二层(更深):只分离相机不够。要恢复到 04 行为,必须**从 04 同口径 env + 04 bootstrap** 出发
  (监督者判断,EXP1B 验证:踢球+进球+稳定从 04 完整继承,goal_rate peak 0.5-0.625、零摔倒)。

### 修正 2:真场线 selfloc 的真根因 = 几何张力,不是分辨率/bootstrap
- EXP1B/1C 三个 bootstrap 起点(04 / 04+暖身 / model_2800+暖身)selfloc 撤拐杖后**全卡 ~10m**。
- 决定性对照:**同 model_2800 起点、同 selfloc head、GT 都在 obs** 时——
  v3 全场几何 = **2.80m**,v4 进攻半场几何 = **7.07m**。差 2.5 倍,唯一变量是出生几何。
- **根因坐实**:进攻半场几何(近球门、朝球门、位姿多样性低、地标可见性差)对 selfloc 是毒药。
  这是 v3 早识别的"几何张力"(③踢球要近球门聚焦 ↔ ①定位要全场多样性)在 v4 重现。
- **推论**:分辨率(1280×960 视角图证明线可辨)是次要因素;**几何才是 selfloc 的主瓶颈**。

### 修正后的实验主线(替代原 Exp2/3 顺序)
- **EXP2(几何课程,新主线)**:出生分布从全场渐变到进攻半场。先让 selfloc 在多样几何学好,
  再聚焦进球。这是解几何张力的正路。判据:selfloc<3m + goal_rate≥0.2 + fell_over=0。
- **EXP3(高分辨率)**:仅在 EXP2 让 selfloc 收敛后,再提 RGB 分辨率压残差(从次要因素入手)。
- **EXP4(距离自适应精度)**:远粗近精——可能与几何课程协同(远处松正好对应高多样性低精度区)。
- **EXP5(几何方法 PnP/MCL)**:若回归式在好几何下仍卡,再上。

---

## 五、v4 EXP1-4 完整诊断闭环(2026-06-08)+ EXP5 转向

### 已排除的假设(每个都有实验证据)
| 假设 | 实验 | 结论 |
|---|---|---|
| 共用相机毁 depth CNN | EXP1 | ✅ 真(已修:分离相机 depth reinit 0) |
| bootstrap 起点不对 | EXP1B/1C | ❌ 排除(04/04+暖身/m2800 撤GT后全塌~10m) |
| fresh RGB CNN 本身 | reinit 对比 | ❌ 排除(v3 v4 的 RGB CNN 都 fresh) |
| 几何课程能解 | EXP2 | ❌ 全场期都没复现 v3 的 2.8m |
| 梯度稀释 | EXP3 | ✅ 真(selfloc 0.8→2.5 后全场期 6.47→2.13m) |
| 几何终点太集中 | EXP4 封顶0.6 | ❌ 未阻止塌陷 |

### 最终根因(决定性,6 实验对齐)
所有低 selfloc 的 best 全在 **iter 1032-1363(GT 暖身期 GT 仍在)**,撤 GT(>2500)全塌 8-10m。
**fresh RGB CNN 在 e2e 任务下无法在撤 GT 后独立维持纯视觉定位**;GT 在时靠 GT obs 到 1.8m(假象)。
与 v3 realspec 末期反弹 6.5m 同现象。**已确认:e2e 多目标 + 真场线 + fresh RGB 是结构性瓶颈,
fade/几何/权重调参无效(连续同向失败→换机制)。**

### EXP5 转向:先判别,再上重武器
**关键未答问题**:RGB CNN 到底能否从 0.125m 真场线学到纯视觉定位?
- **EXP5a(便宜判别)**:纯 selfloc 任务(无 e2e 多目标干扰)+ 0.125m + 分离相机高分辨率 RGB,
  看撤 GT 后能否维持 <3m。若能→问题是 e2e 干扰(用预训练+冻结 RGB);若不能→信号/架构问题。
- **EXP5b(机制级,若 5a 证信号足够但 e2e 干扰)**:RGB selfloc 分支单独预训练到收敛,再冻结
  接入 e2e(踢球不再扰动已学好的定位)。
- **EXP5c(几何方法,若 5a 证回归学不出)**:关键点检测+PnP(几何约束硬,sim GT 自动监督),
  必要时 MCL 解对称歧义。这是监督者预留的最终路线。

---

## 六、分辨率路线证伪 + 转几何方法(2026-06-08 EXP5)

### EXP5a/5b 决定性判别(分辨率不是答案)
| 分辨率 | 撤GT后 min | 末值 | 像素倍数 |
|---|---|---|---|
| 96×72 | 2.05m | 6.73m | 1x |
| 160×120 | 2.36m | 6.68m | 2.78x |
| 384×288 | 3.15m | 6.51m | 16x |

跨 16x 像素,撤 GT 后**全塌到 ~6.5m,无改善**。**分辨率路线证伪**——既满足用户"中分辨率验证"要求,
也满足监督者"无效转 PnP"红线。

### 最终根因(范式级)
瓶颈不是分辨率/几何/权重/bootstrap,而是**"回归式 CNN 从 RGB 直接学连续位姿(x,y,yaw)"范式本身**:
fresh CNN 在 GT 拐杖撤光后学不到稳定的视觉→位姿映射(GT 在时靠抄答案,撤后塌)。6 个 e2e 实验 +
3 个分辨率实验一致证实。

### 转 EXP5c:关键点检测 + PnP(监督者 EXP4_GEOMETRIC_LOC)
**核心思路转变**:不再让网络直接回归位姿,而是让网络只做它擅长的——**检测图像中的场地关键点**
(2D 像素坐标),再用经典几何 PnP 从"2D 检测↔已知 3D 地图点"对应关系解算位姿。
- **优势**:几何约束硬(PnP 是确定性求解,非学习);精度上限高;sim 里关键点 3D 坐标已知,2D 投影
  可自动生成监督标签(无需 GT 拐杖渐隐这套);更接近真机 RoboCup 定位方案。
- **关键点选取**:场地角点、罚球区角、中圈与中线交点、球门柱根部等几何显著、地图坐标已知的点。
- **必要时 MCL**:场地对称(两半场镜像)致位姿多峰歧义,用粒子滤波 + 运动模型消歧。
- **判据**:撤"GT 拐杖"概念后(几何法本不需要),纯视觉位姿误差稳定 <3m(目标 <1m)。

---

## 七、EXP5c/d 几何范式:可见性修复成功,但 RL reward 学不动稠密检测(2026-06-08)

### 进展链
- EXP5c(keypoint+Kabsch):检测 reward `exp(-err/std²)` 被 gaming——可见点→0 时 masked_err→0→满分,
  policy 转头逃避(kp_visible 4→0.06),位姿 11m。**但范式有效**:离线几何链已证检测准时 ~1cm。
- EXP5d(`(visible/K)·exp(-err/std²)`):**可见性漏洞修复成功**,kp_visible 稳定 3.9-4.9 不再崩。
  但 **keypoint_detection reward 全程 0.0000**:std=0.15、pixel_err≈2 → exp(-89)≈0,**检测精度梯度消失**。
  policy 只学到 visible/K,pixel_err 卡 ~2,pos_err 卡 6-7m,goal_rate 0。

### 关键结论(范式有效,训练方式错)
几何范式本身正确(检测准→几何精确解位姿),失败在**用 PPO reward 学 23×2 维稠密回归**:
稀疏标量回报无法驱动稠密像素回归,且 exp 核易饱和。codex 独立提出同一判断:
**关键点检测本质是 dense supervised vision task,应拆出来用投影标签做监督 loss,而非靠 reward。**

### 纯视觉边界(codex 18:31 红线,已审查合规)
- depth 属允许的纯视觉输入(用户+codex 确认)。合格输入:RGB/depth/相机内参/cam→base 标定运动学/本体状态。
- ✓ 策略训练**不吃 GT 位姿**:keypoint env selfloc_gt_mask fade=[0,1],robot_field_pose obs 从 iter1 即乘 0。
- ✓ 几何链 base-frame 验证只依赖 cam→base 相对变换(真机标定可得);world cam pose 仅用于训练标签+monitor。
- 待办:正式评估补一条完全不含 world pose 的推理路径。

### 下一步 EXP5e(采纳 codex)
把关键点检测拆为监督学习:用 project_keypoints 投影标签 + 可见性掩码做 dense supervised loss
(辅助头/预训练/半冻结),再接回 e2e。先评估在 mjlab/rsl_rl 框架内加监督 loss 的最小改法。

---

## 八、EXP5f 设计:关键点检测拆为监督学习(三路对比,待定)

### 共同前提
- **范式已验证**:离线几何链(project→depth抬→base系→Kabsch)在检测准时复原位姿 ~1cm(误差4视角全中)。
- **三轮 RL 已证伪 reward 路线**:EXP5c gaming(看不见=满分)→ 5d 修复可见性但核饱和(reward 0.0000)
  → 5e std=1.0 梯度回来但抢本体梯度(fell_over 37、不踢球)。结论:**PPO 标量奖励学 46 维稠密回归不可行**。
- **监督信号**:`project_keypoints(kp_world, cam_pos, cam_mat, fovy, W, H)` 每帧给 (uv_gt(B,K,2), vis(B,K))。
  loss = 可见点上 `smooth_l1(pred_uv, uv_gt)`,按 vis 掩码。标签是 sim 几何投影(监督用,非部署输入,合规)。
- **纯视觉边界**:检测头输入仅 RGB(+depth 供 Kabsch);标签用的 cam_pos/cam_mat 仅训练期监控/标签,
  推理只需 cam→base 标定。已审查 GT 位姿 obs 从 iter1 mask=0,策略不吃特权。

### 核心技术约束(决定改动位置)
监督 loss 需"逐 rollout 样本的 uv_gt 标签",而 **PPO rollout buffer 默认只存 obs/action/reward,不存
cam 外参/origin**。所以标签必须先想办法进 buffer 或现算。三路差异即源于此。

### 路径 A:标签入 buffer + PPO 扩展(仿 rnd/symmetry)
**做法**:
1. 新增"存储用"obs term `keypoint_uv_label`,在 actor obs group 算 `project_keypoints` 的 (uv_gt, vis)
   展平成 (K*3,) 向量(uv+vis),**不喂给策略主干**(用单独 obs group 或标记 exclude from policy input)。
2. 仿 `rsl_rl.extensions.RandomNetworkDistillation` 写 `KeypointSupervision` 扩展:从 batch 取该标签 +
   actor 的 selfloc 输出,算 masked smooth_l1,在 PPO.update 里 `loss += coef * kp_loss`。
3. runner/config 加 `keypoint_cfg` 传入(仿 rnd_cfg)。
**改动点**:vendored `rsl_rl/algorithms/ppo.py`(构造器+update,~15行)、`rl/config.py`(加 cfg 字段)、
`rl/runner.py`(传 cfg)、新 obs term、env_cfg。
**风险**:改 pip 包升级易丢;buffer 存标签增显存(K*3*B*horizon);obs group 隔离需确认 rsl_rl 支持。
**验证**:smoke 确认 kp_loss 下降、actor selfloc 输出逼近 uv_gt;监控 kp_pixel_err & pos_err & fell_over 同时看。

### 路径 B:离线监督预训练检测器
**做法**:单独脚本 rollout 收集 (RGB stack, uv_gt, vis) 数据集 → 监督训练 CNN+检测头至 pixel_err 收敛 →
load 进 policy 的 RGB CNN+selfloc head → e2e 训练时冻结或低 lr 微调检测头,PPO 只学运动。
**改动点**:新数据收集脚本、新监督训练脚本、policy load 逻辑(部分权重)。不碰 rsl_rl 核心。
**风险**:检测头与 actor trunk 共享表征,难干净隔离;两阶段分布漂移(预训练分布≠e2e rollout 分布);
冻结后检测头不随策略行为更新,可见点分布变化时失准。
**验证**:预训练阶段独立验 pixel_err→几何链 pos_err;e2e 阶段验 pos_err 不退化 + 踢球。

### 路径 C:rollout 期在线监督(绕开 PPO.update)
**做法**:在 runner 采样循环每步,对 actor 的 selfloc 输出直接算监督 loss + 单独 backward(独立 optimizer
或并入主 optimizer),不进 PPO 奖励/buffer。
**改动点**:`rl/runner.py` 采样循环(中等);需小心与 PPO optimizer/grad 交互。
**风险**:在 no_grad 的 rollout 段插 grad 计算需重构;与 PPO 的 minibatch 多 epoch 更新节奏不一致。
**验证**:同 A。

### 推荐与权衡
- **A** 最贴合 rsl_rl 既有扩展模式(rnd/symmetry 已证可行)、解耦最干净,代价是改 vendored 包。
- **B** 不碰 rsl_rl,但两阶段衔接 + 冻结失准风险高,且"检测头与 trunk 共享"难隔离。
- **C** 改动集中在 mjlab 侧,但 grad 与 rollout 节奏耦合风险高。
**倾向 A**(扩展模式有先例、解耦最干净);vendored 改动可用 patch 脚本记录以防升级丢失。**待用户定。**

### EXP5f 决策(2026-06-08,用户+codex 一致):走 A' 本地扩展
- 标签作为**非策略 obs group `keypoint_label`** 进 RolloutStorage(TensorDict 所有 key 都存),
  但不列入 actor/critic 的 `obs_groups`→策略不吃标签(codex 已确认 rsl_rl 此机制)。
- 写**本地** `KeypointAuxPPO`(继承 rsl_rl PPO),update() 里加 masked smooth_l1 aux loss;
  用 `algorithm.class_name` 指向本地类,**不碰 site-packages**。
- 红线:标签随 rollout 对齐进 batch、不进 actor/critic 输入、不污染 pip 包、每步 smoke 验
  label shape + obs_groups 排除 + aux loss 下降。
- C(在线 backward)放弃;B(离线预训练)留作 fallback。
