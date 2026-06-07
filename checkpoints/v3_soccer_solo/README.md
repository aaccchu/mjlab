# v3 Soccer Solo — 三块能力最佳模型交付

> 12 小时自主推进的成果固化(2026-06-07)。三块能力**各自独立**训练、验证,
> 尚未整合成端到端单策略(整合是 Phase 2 工作)。每个模型附 `params/`(env.yaml +
> agent.yaml)可复现训练配置。

## 三个模型

### 01_selfloc_purevision — ① 纯视觉自定位
- `model_2800.pt` — 来自 `spike_v3g_selfloc_temporal`(叠帧 RGB + 加粗标线 + GT 渐隐蒸馏)。
- **指标(play=False 纯视觉,GT obs 已渐隐到≈0):** selfloc_pos_err_m **0.79m** / head **2.78°**。
- **判据:** GT obs 抽走后估计仍亚米级 → 估计来自 RGB CNN(纯视觉),非抄 GT。
- **保留:** 用 1.0m 加粗标线(真场 0.125m,有 sim2real gap);对渐隐 GT 槽有残余依赖
  (−GT_pose 消融 5.4m);iter3300 后过训练退化,故取 model_2800 而非最终。

### 02_findball_depth — ② 深度找球
- `model_1499.pt` — 来自 `spike_v3f_holding`(深度相机 CNN,带球基线)。
- **能力:** 从深度相机纯视觉估计球的方向/距离,无 GT 拐杖;带球到随机目标 success≈2.6。

### 03_dribble_goal — ③ 带球进球(最少步骤)
- `model_1600.pt` — 来自 `spike_v3g_goal_stable`(v3f bootstrap + 进攻半场几何 + 进球奖励 +
  upright/goal 权重平衡)。
- **指标:** goal_rate **0.43**(峰), fell_over **0.14**, dribble_success 1.71, ball_path 4.3m。
- **最少步骤:** 见 minsteps 实验,time penalty 把 time_to_goal 181→122,但需早停(固定权重后期
  压垮进球率)。本模型是稳定进球的最佳点;追求最快可用 minsteps early-stop ckpt。
- **保留:** 进球率非 100%;纯 GT goal env(无视觉),与 ①②的视觉 env 是独立线。

## 复现/加载
每个模型用对应 `params/agent.yaml` 的 runner cfg + `params/env.yaml` 的 env cfg 加载。
训练脚本在 `scripts/spike_v3g_*.py`、`scripts/spike_v3f_*.py`。
评估探针:`scripts/probe_v3g_selfloc_vision.py`(已修 play=False)、`scripts/diag_selfloc_env_gap.py`。

## 可视化
见 `soccer_eval/2026-06-07_v3_soccer_solo/`(训练曲线 + 纯视觉三消融对比 + 演示)。
完整实验链与诚实保留见 `soccer_eval/v3g_autonomous_worklog.md`(顶部有总结表)。

## 04_e2e_integrated — 端到端整合(①②③ 单策略)
- `model_1499.pt` — 来自 `spike_v3g_e2e_fix`(从 model_2800 整合 bootstrap 续训)。
  **三块能力在一个策略里**:纯视觉自定位 + 深度找球 + 带球进固定球门。
- **指标:** goal_rate **0.30**, fell_over **0**(全程零摔倒), dribble_success 0.51,
  selfloc_pos_err_m(进攻半场几何, play=False)**4.5m**。
- **关键诊断(翻转性结论):** 整合本身**干净成功**,无 ①↔③ 容量竞争。对照实验证明:整合前的
  model_2800 放到**同样进攻半场几何**上自己也只有 **4.65m**(vs 整合后 4.5m,几乎相同)。所以
  自定位 0.82→4.5m 的差异**几乎全来自操作几何变难**(进攻半场出生→位姿多样性低、地标可见性差),
  不是整合破坏定位。同几何下整合前后精度相同。
- **保留:** 端到端同时要亚米级定位 + 近球门进球,存在**任务几何张力**(定位要全场多样性,进球要近
  球门聚焦)——设定层面的真实权衡。后续可用课程从全场渐变到近球门、或扩大相机视野缓解。
