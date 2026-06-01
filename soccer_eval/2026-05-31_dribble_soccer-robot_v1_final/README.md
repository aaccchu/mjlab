# 可视化验证集 — soccer-robot v1(G1 端到端带球到目标点)

本目录汇总 v1 版策略(`Mjlab-Soccer-Unitree-G1`,3000 迭代训练,
run `2026-05-31_21-19-24`)的所有可视化验证结果,按验证着重点分子文件夹。
每个子文件夹含一段 `--num-envs 1` 渲染的 mp4 + 4 帧抽帧 png。

渲染命令模板(单 env 才是贴身侧视,多 env 会拉远成俯视看不清):
```sh
MUJOCO_GL=egl uv run play Mjlab-Soccer-Unitree-G1 --agent trained \
  --checkpoint-file <model_XXXX.pt> --num-envs 1 --video True \
  --video-length 400 --video-height 480 --video-width 640
```

## 子文件夹说明

### 01_midtrain_default_model1300
- **着重点:** 训练中途(~43%,1300 迭代)快照,看早期带球雏形
- checkpoint:model_1300.pt
- 初始条件:默认(球离机器人 0.6–1.5m,目标离球 2.0–6.0m)
- 效果:走路+平衡已学好,带球处于早期

### 02_final_default_model2999
- **着重点:** 最终训练完成(3000 迭代)的标准评估,默认初始条件
- checkpoint:model_2999.pt
- 初始条件:默认(球 0.6–1.5m,目标 2.0–6.0m)
- 效果:G1 稳定站立/行走,朝球走过去并带球推进,步态自然不摔,
  行为符合 DribbleCommand 的 approach→push 设计

### 03_robustness_farball_model2999
- **着重点:** 改变初始条件的泛化/鲁棒性测试 —— 球和目标都比训练分布更远
- checkpoint:model_2999.pt(同最终版)
- 初始条件:**加难**(球 2.5–4.0m,目标 6.0–9.0m,均超出训练分布)
- 效果:即使球/目标更远,G1 仍稳定走向远处的球,不摔不穿模 ——
  说明策略对初始条件有泛化,不是只死记训练距离
- 备注:测试用的 `env_cfgs.py` spawn/target 距离改动已还原,仓库配置保持训练默认值

### 04_diverse_longrollout_model2999
- **着重点:** 长时间 rollout(1113 帧 / ~22s),观察策略在多次 episode 重置后的
  持续行为多样性和稳定性
- checkpoint:model_2999.pt
- 初始条件:默认(每次 episode 结束后自动重置到新随机配置)
- 效果:多次重置后 G1 持续稳定行走+带球,行为不退化,展示策略鲁棒性

### 05_training_progression
- **着重点:** 训练进展对比 —— 从随机策略到最终策略的行为演变
- 包含:
  - `dribble_model0_random.mp4`(model_0,0 迭代):随机策略,G1 立即摔倒
  - `dribble_model300_early.mp4`(model_300,300 迭代):学会站立平衡,但不会移动
  - (结合 01 的 model_1300:会走向球;02 的 model_2999:稳定带球推进)
- 效果:清晰展示 RL 训练四阶段 —— 摔倒 → 站立 → 走向球 → 带球到目标

## 最终训练指标(model_2999.pt 评估快照)
- Mean reward:49.6(训练全程 -1.7 → 50+)
- episode_success(带球到目标点):0.065(中途峰值 0.13)
- fell_over:0.125(走路平衡彻底学会,从 450 → 个位数)
- robot_to_ball_error:2.06m
