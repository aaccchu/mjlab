"""v4 EXP24: STRIKE + AMP — keep the best shot policy, teach a human-like gait.

EXP23 (kick_strike, forensics shot rate 0.410) walks with a flat-footed shuffle:
the flat-terrain config deleted foot-clearance/swing-height rewards, zeroed
air_time, and variable_posture pulls the legs back to a barely-bent default — so
the policy learned to skate. Rather than hand-tune gait rewards, EXP24 adds an
Adversarial Motion Prior: a discriminator rewards the policy for moving like a
human-motion reference dataset (the MOS9 FK walk/run/turn clips, already
retargeted to this exact robot — FK-verified to 0 mm).

Architecture (mjlab.rl.amp_ppo:AMPPPO, a local PPO subclass; vendored rsl_rl
untouched): a separate discriminator network with its own optimizer emits a
per-step style reward added to the task reward. It shares NO parameters with the
actor/critic, so it cannot pollute the control trunk (cf. the v4 perception-trunk
failure). The style/task trade-off is governed solely by amp_task_reward_lerp.

Bootstrap: EXP23 model_1999 (current best). NO std reset (codex "ball-hugging
attractor" lesson). The discriminator starts fresh (not in the EXP23 ckpt); the
actor/critic load fully (same obs layout — the amp group is training-only).

amp_task_reward_lerp=0.7 (70% task / 30% style) protects the shot rate; raise it
toward 1.0 if forensics shot rate drops below the 0.37 veto threshold.

Usage: MUJOCO_GL=egl uv run python scripts/spike_v4_e2e_ekf_kick_strike_amp.py [--smoke]
"""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.mos92.env_cfgs import (
  AMP_JOINT_NAMES,
  mos92_soccer_e2e_dualcam_ekf_kick_strike_amp_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import (
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

BASE_CKPT = Path("checkpoints/v4_soccer/kick_strike_exp23/model_1999.pt")
NUM_ENVS = 384
MAX_ITER = 2000

# Walk/turn/side-step reference clips (no run/flip): the goal is a natural WALK
# with foot lift and longer strides, not athletic running. Paths are relative to
# the repo root (MotionDataset resolves them there).
_MOTION_GLOBS = (
  "docs/robot_param/MOS9-AMP-main/data/motions/mos9_fk_motion_clipped_simple/*.npz",
  "docs/robot_param/MOS9-AMP-main/data/motions/mos9_fk_motion/Walk_*.npz",
  "docs/robot_param/MOS9-AMP-main/data/motions/mos9_fk_motion/B*.npz",
)


def _motion_files() -> list[str]:
  files: list[str] = []
  for pattern in _MOTION_GLOBS:
    files.extend(sorted(glob.glob(pattern)))
  # Make paths repo-root-relative and drop run clips defensively.
  rel = []
  for f in files:
    p = f.split("mjlab/", 1)[-1] if "mjlab/" in f else f
    name = Path(p).name
    if name.startswith("C") or "run" in name.lower():
      continue
    rel.append(p)
  return sorted(set(rel))


def _partial_load_shape_match(module, ckpt_sd) -> tuple[int, int, list[str]]:
  """Load every name+shape-matching tensor; reinit the rest. EXP24 keeps the
  actor/critic obs layout identical to EXP23 (amp is a training-only group), so
  this should load everything (reinit 0)."""
  cur = module.state_dict()
  to_load, notes = {}, []
  reinit = 0
  for k, v in ckpt_sd.items():
    if k not in cur:
      notes.append(f"{k}:absent")
      continue
    if cur[k].shape == v.shape:
      to_load[k] = v
    else:
      reinit += 1
      notes.append(f"{k}:reinit{tuple(v.shape)}->{tuple(cur[k].shape)}")
  module.load_state_dict(to_load, strict=False)
  return len(to_load), reinit, notes


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  smoke = "--smoke" in sys.argv
  configure_torch_backends()
  device = "cuda:0"

  env_cfg = mos92_soccer_e2e_dualcam_ekf_kick_strike_amp_env_cfg(play=False)
  env_cfg.scene.num_envs = 32 if smoke else NUM_ENVS

  gt = env_cfg.curriculum["gt_mask"]
  gt.params["start_step"] = -1
  gt.params["end_step"] = 0
  pr = env_cfg.curriculum["penalty_ramp"]
  pr.params["start_step"] = -1
  pr.params["end_step"] = 0

  max_iter = 8 if smoke else MAX_ITER
  agent_cfg = mos92_selfloc_vision_ppo_runner_cfg()
  agent_cfg.max_iterations = max_iter
  agent_cfg.run_name = "spike_v4_e2e_ekf_kick_strike_amp"
  agent_cfg.save_interval = 100

  # AMP wiring (mirrors spike_v4_e2e_keypoint.py): declare the training-only "amp"
  # obs group so resolve_obs_groups keeps it in the rollout storage but NOT in the
  # actor/critic sets; point class_name at the in-project AMPPPO subclass; pass the
  # discriminator/dataset config via amp_cfg.
  motion_files = _motion_files()
  if not motion_files:
    raise SystemExit("no AMP motion files found — check _MOTION_GLOBS paths")
  agent_cfg.obs_groups["amp"] = ("amp",)
  agent_cfg.algorithm.class_name = "mjlab.rl.amp_ppo:AMPPPO"
  agent_cfg.algorithm.amp_cfg = {
    "obs_group": "amp",
    "joint_names": list(AMP_JOINT_NAMES),
    "amp_obs_terms": ["joint_pos", "joint_vel"],
    "motion_files": motion_files,
    "discr_hidden_dims": [256, 256],
    "amp_reward_coef": 0.2,
    "amp_task_reward_lerp": 0.7,  # 70% task / 30% style — protects the shot rate.
    "amp_grad_pen_coef": 10.0,
    "amp_lr_coef": 0.1,
    "replay_buffer_size": 1_000_000,
  }

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_v4_e2e_ekf_kick_strike_amp")
  log_dir = log_root / stamp
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)
  print(f"[INFO] AMP motion clips: {len(motion_files)} (walk/turn/side-step)")
  if BASE_CKPT.exists():
    print(f"[INFO] Partial-load from EXP23: {BASE_CKPT}")
    ckpt = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
    a_load, a_re, a_notes = _partial_load_shape_match(
      runner.alg._raw_actor, ckpt["actor_state_dict"]
    )
    c_load, c_re, c_notes = _partial_load_shape_match(
      runner.alg._raw_critic, ckpt["critic_state_dict"]
    )
    print(f"[INFO] actor : loaded {a_load}, reinit {a_re} -> {a_notes}")
    print(f"[INFO] critic: loaded {c_load}, reinit {c_re} -> {c_notes}")
    # NO std reset (ball-hugging attractor lesson). Discriminator stays fresh.
    if a_re != 0 or c_re != 0:
      print(f"[WARN] expected 0 reinit (same obs layout); got a={a_re} c={c_re}")
  else:
    print(f"[WARN] bootstrap ckpt missing ({BASE_CKPT}); training from scratch.")

  mode = "SMOKE" if smoke else "FULL"
  print(
    f"[INFO] v4 EXP24 strike+AMP [{mode}]: {max_iter} iters, "
    f"envs={env_cfg.scene.num_envs}, task_reward_lerp=0.7"
  )
  print("[INFO] watch: amp_disc loss DOWN, amp_expert_d->+1/amp_policy_d->-1 gap,")
  print("[INFO] gait (peak_height / air_time UP), and GUARD shot rate >=0.37.")
  runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)
  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
