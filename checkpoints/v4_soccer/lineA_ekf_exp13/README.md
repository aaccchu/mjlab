# 线A 自定位 — EKF 里程碑权重归档(EXP13)

> 归档日期:2026-06-09。这是 v4 足球机器人**线A(纯视觉自定位)的成功里程碑**权重。
> 源训练:`logs/rsl_rl/mos92_velocity/2026-06-09_16-08-16_spike_v4_e2e_ekf`

## 一句话总结
把自定位从"点云拼接 Kabsch"(误差 2.13m)重构为**递归 SE2-EKF**(误差 **0.98m**),
首次突破 1m 目标,且步态稳定性、踢球链均未退化(反而间接改善)。

## 关键指标(末100窗口,多指标判读)
| 维度 | 指标 | EXP13 (EKF) | EXP11 (点云拼接基线) |
|---|---|---|---|
| **自定位** | pos_err | **0.98m** (median 0.98, max 1.06) | 2.13m |
| 自定位 | uniq_frac | 0.17 | 0.17 |
| **稳定性** | fell_over | **0.057** (median 0) | — |
| 踢球链 | goal_rate | 0.080 | — |
| 踢球链 | episode_success | 0.40 | — |
| 踢球链 | ball_to_target_err | 2.97m | — |

## 文件清单
- `model_1999.pt` — 最终权重(2000 iter)。md5: 28bdd74600d6f5a511ad11a94cc502ab
- `model_0.pt` — 起点权重(从 EXP11 model_1999 bootstrap 的初始态),可复现训练。
- `events.out.tfevents.*` — 完整 tensorboard 训练曲线。
- `git_snapshot/` — 训练时的代码 git 快照。

## 方法核心(为什么 EKF 胜出)
- **旧法(点云拼接)**:N 帧的地标点用里程计搬到当前帧→拼大点云→单次 Kabsch 配准。
  机器人朝同方向走时,N 帧看到同 ~4 个点,拼接不增加约束 → 天花板 ~2m。
- **新法(递归 EKF)**:维护 SE2 位姿分布 [x,y,yaw]+协方差。预测步用里程计推进,
  更新步对每个可见地标做序贯 EKF 校正——哪怕单帧只见 1 点也能更新,信息真正跨帧累积。
- 实现:`EkfPoseBelief`(src/mjlab/tasks/velocity/mdp/observations.py),
  EKF 数学先经离线验证(EXP12:同 rollout 下 EKF 1.21m vs 点云拼接 2.68m,+55%)+
  合成自检(每帧仅见 3/8 点,30 步收敛到 0.0000m)双重纪律验证。

## 复现 / 加载方法
```python
# 评估或继续训练(env 须用 EKF env,obs 88 维):
# scripts/spike_v4_e2e_ekf.py 已配置;加载权重:
ckpt = torch.load("model_1999.pt", map_location="cuda:0", weights_only=False)
runner.alg._raw_actor.load_state_dict(ckpt["actor_state_dict"], strict=False)
# env builder: mos92_soccer_e2e_dualcam_ekf_env_cfg
```
- 训练复现:`MUJOCO_GL=egl python scripts/spike_v4_e2e_ekf.py`
  (BASE_CKPT 指向 EXP11 model_1999;384 env, 2000 iter, ~68min)。

## 阶段定位与局限
- **仍是 oracle 检测阶段**(地标像素直接读真值,未接真 CNN 检测器)——隔离变量纪律。
- pos_err 0.98m 达标(<1m),理想 0.79m 仍差 0.19m;可选 R2 对称消歧进一步压(优先级低于线B)。
- 真检测器替换是后续阶段的命门(历史反复失败点)。

## 相关文档
- 方法论:`soccer_robot_v4_localization_method.md`(EKF/R1/R2/R3 设计)
- 实验流水:`soccer_robot_v4_experiment.md`(EXP11/12/13 记录)
- 评估产物:`soccer_eval/2026-06-09_v4/`
