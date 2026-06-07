# v3g: 纯视觉自身定位 + 深度测球 — 实现计划

## 目标
机器人**纯靠 RGB 视觉**判断自己在球场的位置(x, y, heading),据此朝球门方向踢球;
**球的位置继续靠深度相机**(v3b/v3e/v3f 已验证可用)。对应 v3.md 最终目标的第①块。

## 关键约束(已实测确认)
- 现有头部相机 **depth-only**,64×48,60° FOV,挂在 neck(随 neck_yaw/pitch 转)。
- **球场线/中圈/罚球点是平面贴花 geom(z≈0),深度完全看不见**;球门柱是 3D 但
  0.1m 直径 @10m 远不足 1 像素。→ **深度无法自定位**。
- **RGB 能渲染这些线(render group 2)** → 自定位必须用 RGB。和真实 RoboCup 一致。
- GT 自身位姿可得:`root_link_pos_w - env_origins` + base yaw(特权 teacher 信号)。
- CNN 是 `SpatialSoftmaxCNN`(output_channels[16,32]→64d latent,吃 (B,C,H,W))。
- obs_groups 非对称:actor=("actor","camera"),critic=("critic",);蒸馏脚手架
  `mask_obs_scale` 已存在。训练路径(bootstrap+strict load+GT mask pin)已三轮验证。

## 决策(用户已确认)
- 自定位视觉源:**加 RGB 相机**(深度仍专用于球)。
- 训练路径:**特权 → 蒸馏**(复用 mask_obs_scale,风险最低)。

## 分阶段

### Spike S1 — 证明 RGB 能看见球场(GATE,~1h,不训练)
在任何训练前,先证明信号存在(同深度探针的科学纪律):
- 写 `scripts/probe_rgb_field.py`:固定机器人在若干已知场上位姿,头部相机渲染 RGB,
  存图。人工/像素统计确认:**球场线、中圈、球门在 64×48 RGB 里可分辨**。
- 若 64×48 分辨不出线 → 提分辨率(如 96×72)或加第二个更广角的定位相机。
- **不通过不进入训练。**

### Phase A — 特权自定位闭环(~3h 训练)
- 新增 `robot_field_pose` GT obs:`[x/half_length, y/half_width, sin(yaw), cos(yaw)]`。
- **Mask 掉现有 `robot_to_target`**(它当前直接泄露球门方向)→ 强制策略从自身位姿
  推断"球门在哪个方向"。
- 从 v3f bootstrap,保留全部反作弊 penalty。
- 验收:GT 条件下"知道自己在哪→朝球门→踢"闭环成立,goal/dribble 不崩。

### Phase B — 加 RGB 相机 + CNN 分支(~1h 构建)
- `camera_rgb` obs fn(镜像 camera_depth,返回 (B,3,H,W) 归一化)。
- 头部相机 data_types 加 "rgb"(或加第二定位相机,依 S1 结论)。
- 第二个 SpatialSoftmax CNN 分支吃 RGB(深度分支保留给球);latent 拼进 actor obs。

### Phase C — 把自定位蒸馏进 RGB(~4h 训练)
- 用 `mask_obs_scale` curriculum 把 `robot_field_pose` GT 从 1→0 渐降,
  强制 RGB CNN 承载自定位。critic 保持全 GT(非对称)。从 Phase A bootstrap。
- 验收(探针):GT 自身位姿清零后,策略靠 RGB 仍能朝正确球门方向。

### Phase D — 验证 + Sim2Real
- 探针(自身位姿 GT 消融 → RGB 是否承载)、与 v3f A/B、行为视频。
- 加 RGB sim2real DR(光照/纹理/相机外参),拉伸目标。

## 验证纪律
- 每阶段有探针/A-B gate 才进下一阶段(同 v3b→v3e→v3f 做法)。
- **球的深度定位完全不动** —— v3f 已验证可用。
- 全部数字实跑,诚实报告未达标项。

## 新建/改动文件(预估)
- 新建:`scripts/probe_rgb_field.py`、`scripts/spike_v3g_*.py`、探针/eval 脚本。
- 改:`observations.py`(robot_field_pose + camera_rgb)、`env_cfgs.py`(相机+obs+curriculum)、
  `rl_cfg.py`(第二 CNN 分支)、可能 `soccer_field.py`(若 S1 需要可见性调整)。
- 记录:`soccer_robot_v2-3_experiment.md` 每阶段加 dated 条目。
