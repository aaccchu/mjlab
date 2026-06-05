# MOSAMP / AMP Tasks for MOS9

This repository contains MOS9 AMP training pipelines, data processing scripts, and sim2sim utilities.

## Language 

- 中文完整文档: [`README.zh-CN.md`](README.zh-CN.md)
- English full documentation: [`README.en.md`](README.en.md)

This `README.md` serves as a bilingual quick index.

## Quick Start 

### Installation 
```bash
cd mosamp
pip install -e ./source/amp_tasks
pip install mujoco onnxruntime pynput
```

- VS Code workspace: `.vscode/mosamp.code-workspace`

### Command list
View details in `commands.md`


### Training 

```bash
python scripts/amp_rsl_rl/train.py --task AMP_MOS9_V6_ALIAS --headless
```

### Play

```bash
python scripts/amp_rsl_rl/play.py \
    --task AMP_MOS9_V6 \
    --target logs/rsl_rl/mos9_loco/walk_v6_0408_234/model_3000.pt \
    --commands.base_velocity.debug_vis=true \
    --num_envs 32 \
    --video \
    --length 1500
```

### MuJoCo Sim2Sim

```bash
python scripts/mos9_amp_sim2sim_mujoco.py \
    --model logs/rsl_rl/mos9_loco/walk_v6_0409_080/exported/policy_1500.onnx \
    --cmd_hold_seconds 3.0 \
    --duration 50.0
```

### Demo Videos

- IsaacLab Play (3000 steps):
  [`logs/rsl_rl/mos9_loco/walk_v7_0409_210/videos/play/isaaclab_6000-step-0.mp4`](logs/rsl_rl/mos9_loco/walk_v7_0409_210/videos/play/isaaclab_6000-step-0.mp4)
- MuJoCo Sim2Sim (3000 steps):
  [`logs/rsl_rl/mos9_loco/walk_v7_0409_210/videos/mujoco/mujoco_step6000.mp4`](logs/rsl_rl/mos9_loco/walk_v7_0409_210/videos/mujoco/mujoco_step6000.mp4)

## Code Facts Snapshot 

- Task registration: `source/amp_tasks/amp_tasks/velocity/config/mos9/__init__.py`
- V6/V6_ALIAS runner config (with bucket sampling):
  `source/amp_tasks/amp_tasks/velocity/config/mos9/rsl_rl_ppo_cfg.py`
- V6/V7 env config and curriculum:
  `source/amp_tasks/amp_tasks/velocity/config/mos9/velocity_env_param_cfg.py`
- Bucket mapping logic:
  `source/amp_tasks/amp_tasks/amp_rsl_rl/runners/amp_on_policy_runner.py`

## Data & Scripts 

- FK export: `scripts/mos9_fk_npz.py`
- Motion clip: `data/motions/clip_motion.py`
- FK replay + video/speed plot: `scripts/replay_mos9_fk_motion.py`
- Motion format reference: `data/motions/README.md`

For detailed training notes, design rationale, and troubleshooting,
please use the language-specific full docs above.
