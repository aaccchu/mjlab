# MOSAMP / MOS9 AMP Training Guide (English)

This repository provides MOS9 AMP training pipelines in IsaacLab, data-processing scripts, and MuJoCo sim2sim utilities.

## 1. Installation

```bash
cd mosamp
pip install -e ./source/amp_tasks
pip install mujoco onnxruntime pynput
```

- VS Code workspace config: `.vscode/mosamp.code-workspace`

## 2. Quick Start

### 2.1 Train

```bash
python scripts/amp_rsl_rl/train.py --task AMP_MOS9_V6_ALIAS --headless
```

### 2.2 Play / Visualization

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

## 3. Current Experimental Note

- Best run directory currently tracked: `logs/rsl_rl/mos9_loco/walk_v7_0409_210`
- Associated task: `AMP_MOS9_V7_ALIAS`

> Note: Final checkpoint selection should still be validated by reproducible `play` and sim2sim behavior.

## 4. Key Code Facts (Current Repository State)

### 4.1 Task registration mapping

File: `source/amp_tasks/amp_tasks/velocity/config/mos9/__init__.py`

- `AMP_MOS9_V7 -> MOS9EnvCfgV7 + MOS9AMPRunnerCfgV6`
- `AMP_MOS9_V7_ALIAS -> MOS9EnvCfgV7 + MOS9AMPRunnerCfgV6Alias`

So, at the moment, V7/V7_ALIAS reuse the V6 runner stack.

### 4.2 V6 command and reward design

File: `source/amp_tasks/amp_tasks/velocity/config/mos9/velocity_env_param_cfg.py` (`MOS9EnvCfgV6`)

- Command: discrete decoupled-axis commands (one axis active at a time)
  - `base_velocity.class_type = AxisSelectableDiscreteDecoupledAxisVelocityCommand`
  - `active_axes = (0, 1, 2)`
- Reward: axis-wise tracking + directional penalties
  - Tracking: `track_lin_vel_x`, `track_lin_vel_y`, `track_ang_vel_z`
  - Penalties: `wrong_direction_lin_vel`, `orthogonal_axis_lin_vel`, `yaw_rate_when_xy_cmd`, `lin_vel_xy_when_yaw_cmd`

This reduces ambiguity from mixed-axis commands and better matches motion modes in the dataset.

### 4.3 Why many rewards are disabled in base config

File: `MOS9VelocityAMPEnvCfg`

- Terms like `dof_pos_limits`, `joint_vel`, `joint_acc`, `energy`, etc. are disabled.
- Lightweight terms (e.g., `action_rate`, `alive`) are retained.

In practice, this helps avoid overly conservative policies in AMP early training.

## 5. AMP Bucketed Sampling

Core file: `source/amp_tasks/amp_tasks/amp_rsl_rl/runners/amp_on_policy_runner.py`

- Function: `_compute_command_bucket_ids`
- Purpose: command-conditioned motion sampling from matching buckets.

Bucket semantics:

- Translation: `forward/backward/left/right`
- Rotation: `left_turn/right_turn`

`V6_ALIAS`/`V7_ALIAS` behavior:

- Lateral and turning can share turn motions (e.g., `turn_left.npz` used by both `left` and `left_turn`).
- Controlled by `use_command_conditioned_sampling`.

## 6. V7 Curriculum and Domain Randomization

File: `source/amp_tasks/amp_tasks/velocity/config/mos9/velocity_env_param_cfg.py`

`MOS9EnvCfgV7` uses a weak-to-strong randomization schedule:

- First 2000 steps: weaker randomization (near-fixed params)
- After 2000 steps: stronger randomization enabled through curriculum

Current implementation highlights:

- `MOS9DelayedRandomCurriculumCfgV7`
- `events.physics_material_reset`

Note: this is an equivalent startup-to-reset behavior design, not an in-place runtime change of event `mode`.

## 7. AMP Observation Group Choices

File: `source/amp_tasks/amp_tasks/velocity/mdp/amp_obs_grp.py`

- `AMPObsBaiscCfg`: joint-only, weakest constraint
- `AMPObsSoft1`: `joint_pos/joint_vel/body_lin_vel_b`, currently a stable default
- `AMPObsSoftTrack(Local)`: stronger constraints, more sensitive to data quality

## 8. Data and Script Workflow

- `scripts/mos9_fk_npz.py`
  - Replays GMR motions in IsaacLab FK and exports richer link/body data.

- `data/motions/clip_motion.py`
  - Time-window clipping using `CLIP_TIME_BOUNDS_SEC`.

- `scripts/replay_mos9_fk_motion.py`
  - GUI replay for FK motion; supports video export and base speed curves.

- `scripts/mos9_amp_sim2sim_mujoco.py`
  - MuJoCo sim2sim with command sequence testing and plot export.

- Data format reference: `data/motions/README.md`

## 9. Data Quality Notes

AMP is highly sensitive to motion distribution. Limited motion count, extreme speed profiles, short turning segments, or noisy poses can directly destabilize training.

Recommended check cycle after any data change:

1. Replay trajectories with `replay_mos9_fk_motion.py`
2. Validate base speed curves against target command distributions
3. Run AMP training only after data sanity is confirmed
