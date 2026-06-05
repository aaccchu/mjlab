# v3b_gt_ablation —— 这个文件夹是什么

这里保存的是 **MOS92 视觉策略 "v3b" 的 GT 消融(GT-ablation)证据**:用来回答一个关键问题——
**机器人到底是靠头部深度相机 "看见" 了球,还是一直在偷看作弊用的真值(GT)球坐标?**

## 名词解释

- **GT(ground truth)球向量**:仿真器直接喂给策略的"上帝视角"球信息——`robot_to_ball`(机器人→球的向量)、
  `ball_velocity`(球速)、`ball_gaze_uv`(球在画面里的像素坐标)。这是**作弊捷径**:不用看图就知道球在哪。
- **depth CNN**:头部深度相机图像 → 卷积网络 → 特征。这是我们**真正想要**机器人依赖的、可迁移到真机的感知通路。
- **GT 消融(ablation)**:推理时把上面 3 个 GT 球向量在策略输入里**清零**,只留深度相机这一条路。
  - 如果清零后行为不变 → CNN 真的学会看了(GT 是多余的)。
  - 如果清零后行为崩溃 → GT 是拐杖,CNN 是死重(dead weight),根本没在用相机。
- **v3 warmup(model_2999)**:更早的"凝视热身"checkpoint,训练时**同时**保留 GT 和相机。
- **v3b**:在 warmup 基础上,用 `gt_mask` 课程把 GT 球向量在 1500 步内**渐进清零到 0**,
  强迫 PPO 把球信息改走相机→CNN 这条路。obs 维度始终保持 84(只是 scale→0,不删项),所以 checkpoint 能严格加载。

## 三个文件分别说明什么

### 1. `probe_baseline_vs_ablated.png` —— warmup 的诊断(坏消息)
对 **v3 warmup(model_2999)** 跑消融探针,256 envs:
- `dribble_success`:baseline(GT 完整)**4.689** → ablated(只给相机)**0.011** —— 带球能力几乎归零。
- 结论(图标题原话):**清零 GT 球向量 → 带球崩溃 → 深度 CNN 是死重**。
- 这证明 warmup 阶段策略完全靠 GT 拐杖,CNN 没学会看。**这正是做 v3b 的动机。**

### 2. `probe_three_way.png` —— 三方对比(好消息,核心结论)
把三种条件放在一起比 `dribble_success` 等指标(图标题:"v3 GT-ablation: the depth CNN learned to see (partially)"):
- **蓝 = warmup baseline(GT 拐杖)**:`dribble_success` ≈ 4.69 —— 靠作弊很高。
- **灰 = warmup ablated(CNN 死重)**:清零 GT 后 `dribble_success` ≈ 0.01 —— 崩溃,印证 CNN 没用。
- **绿 = v3b native(只给相机,训练后)**:`dribble_success` ≈ **1.11** —— 关键!
  v3b 训练后,**只靠相机**也能带球(0.01 → 1.11),`gaze_center` 0.172、`ball_visible` 0.391、`upright` 0.947 都活着。
- **一句话**:depth CNN **(部分)学会看球了**。带球能力从"去掉 GT 就崩"变成"只用相机也能做"。

### 3. `v3b_camera_only-step-0.mp4` —— v3b 的行为视频(相机-only 条件)
- 用 v3b 的 `model_2999.pt` 渲染,**每步把 GT 球向量清零**,所以视频反映的是它真实的
  "只用相机找球 → 接近 → 带球"行为,而不是带 GT 作弊的样子。
- 这是你说"表现良好"的那个视频。注意:这是**真正的 v3b**(本次会话重新渲染恢复的),
  之前一度被 v3d 的渲染覆盖,已修正。

## 一句话总结
`v3b_gt_ablation` = "**用 GT 消融实验证明 v3b 的深度相机 CNN 真的学会看球了**" 的证据包:
两张探针图(warmup 崩 vs v3b 存活)+ 一段 v3b 纯相机行为视频。它是从 warmup(靠作弊)
到纯视觉的关键里程碑,也是后续 v3d(修跳跃)和 Task C(纯视觉)的基线。

## 相关备注
- **手臂举起问题**与本文件夹无关,根因是机器人初始关键帧 `KNEES_BENT_KEYFRAME` 把
  `shoulder_roll` 设成 ±1.4 rad(±80°),`pose` 奖励持续把手臂拉向该默认位。修复需改关键帧 + 重训。
- v3d(跳跃修复)的产物在隔壁 `../v3d_jumpfix/`。
