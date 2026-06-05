# MOSAMP / MOS9 AMP 训练说明（中文）

本仓库用于 MOS9 机器人在 IsaacLab 中的 AMP 训练、数据处理与 MuJoCo sim2sim 验证。

## 1. 安装

```bash
cd mosamp
pip install -e ./source/amp_tasks
pip install mujoco onnxruntime pynput
```

- VS Code 工作区配置：`.vscode/mosamp.code-workspace`

## 2. 快速开始

### 2.1 训练

```bash
python scripts/amp_rsl_rl/train.py --task AMP_MOS9_V6_ALIAS --headless
```

### 2.2 回放与可视化

```bash
python scripts/amp_rsl_rl/play.py \
    --task AMP_MOS9_V6 \
    --target logs/rsl_rl/mos9_loco/walk_v6_0408_234/model_3000.pt \
    --commands.base_velocity.debug_vis=true \
    --num_envs 32 \
    --video \
    --length 1500
```

### 2.3 MuJoCo Sim2Sim

```bash
python scripts/mos9_amp_sim2sim_mujoco.py \
    --model logs/rsl_rl/mos9_loco/walk_v6_0409_080/exported/policy_1500.onnx \
    --cmd_hold_seconds 3.0 \
    --duration 50.0
```

## 3. 当前实验结论（经验）

- 当前你记录的最好结果目录：`logs/rsl_rl/mos9_loco/walk_v7_0409_210`
- 对应任务：`AMP_MOS9_V7_ALIAS`

> 说明：best checkpoint 仍建议以 `play` 与 sim2sim 的可复现表现综合判断。

## 4. 关键代码事实（以当前仓库为准）

### 4.1 Task 注册关系

文件：`source/amp_tasks/amp_tasks/velocity/config/mos9/__init__.py`

- `AMP_MOS9_V7 -> MOS9EnvCfgV7 + MOS9AMPRunnerCfgV6`
- `AMP_MOS9_V7_ALIAS -> MOS9EnvCfgV7 + MOS9AMPRunnerCfgV6Alias`

也就是说，当前 V7/V7_ALIAS 在 runner 层复用了 V6 runner 体系。

### 4.2 V6 控制与奖励设计

文件：`source/amp_tasks/amp_tasks/velocity/config/mos9/velocity_env_param_cfg.py`（`MOS9EnvCfgV6`）

- 命令：离散分离轴命令（单次仅激活一个轴）
  - `base_velocity.class_type = AxisSelectableDiscreteDecoupledAxisVelocityCommand`
  - `active_axes = (0, 1, 2)`
- 奖励：分离轴跟踪 + 方向/轴向惩罚
  - 跟踪：`track_lin_vel_x`、`track_lin_vel_y`、`track_ang_vel_z`
  - 惩罚：`wrong_direction_lin_vel`、`orthogonal_axis_lin_vel`、`yaw_rate_when_xy_cmd`、`lin_vel_xy_when_yaw_cmd`

这种设计目的是降低多轴混合命令带来的学习歧义，让策略更贴近数据中的动作模态。

### 4.3 为什么基座配置禁用部分 reward

文件：`MOS9VelocityAMPEnvCfg`

- 禁用了 `dof_pos_limits`、`joint_vel`、`joint_acc`、`energy` 等约束。
- 保留较轻量项（如 `action_rate`、`alive`）。

经验上可减轻“策略过度保守不想走路”的问题，尤其在 AMP 初期更明显。

## 5. AMP 分桶策略

核心逻辑文件：`source/amp_tasks/amp_tasks/amp_rsl_rl/runners/amp_on_policy_runner.py`

- 函数：`_compute_command_bucket_ids`
- 作用：根据命令方向仅从对应 motion bucket 采样，提高学习稳定性。

桶语义：

- 平移：`forward/backward/left/right`
- 旋转：`left_turn/right_turn`

`V6_ALIAS`/`V7_ALIAS` 的特点：

- 允许“横移与旋转共享 turn motion”（例如 `turn_left.npz` 同时用于 `left` 与 `left_turn`）。
- 通过 `use_command_conditioned_sampling` 控制是否启用分桶。

## 6. V7 curriculum 与 DR

文件：`source/amp_tasks/amp_tasks/velocity/config/mos9/velocity_env_param_cfg.py`

`MOS9EnvCfgV7` 采用“前弱后强”的随机策略：

- 前 2000 step：弱随机（接近固定参数）
- 2000 step 后：通过 curriculum 开启更强随机

当前实现要点：

- `MOS9DelayedRandomCurriculumCfgV7`
- `events.physics_material_reset`

说明：这是“等效 startup->reset 行为”实现，而非运行中直接改事件 `mode`。

## 7. AMP 观测组选择

文件：`source/amp_tasks/amp_tasks/velocity/mdp/amp_obs_grp.py`

- `AMPObsBaiscCfg`：仅关节项，约束弱
- `AMPObsSoft1`：`joint_pos/joint_vel/body_lin_vel_b`，当前常用且稳妥
- `AMPObsSoftTrack(Local)`：约束更强，对数据质量更敏感

## 8. 数据与脚本流程

- `scripts/mos9_fk_npz.py`
  - 将 GMR 导出的 motion 在 IsaacLab 中做 FK 回放并导出更多 link/body 信息。

- `data/motions/clip_motion.py`
  - 使用 `CLIP_TIME_BOUNDS_SEC` 对 motion 按时间区间裁剪。

- `scripts/replay_mos9_fk_motion.py`
  - GUI 回放 FK 动作；支持导出视频与 base 速度曲线。

- `scripts/mos9_amp_sim2sim_mujoco.py`
  - MuJoCo sim2sim；支持命令序列测试、日志与图像导出。

- 数据格式参考：`data/motions/README.md`

## 9. 数据质量注意事项

AMP 对数据分布非常敏感。若动作数量少、速度分布偏激、转向段过短或姿态噪声高，会直接影响收敛和最终步态质量。

建议每次改动数据后做三步检查：

1. `replay_mos9_fk_motion.py` 检查轨迹连续性
2. 检查 base 速度曲线是否与目标命令分布一致
3. 再进入 AMP 训练，避免把数据问题误判为算法问题
