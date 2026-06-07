"""Task v3g Phase A (explicit): self-localization head + accuracy reward.

Unlike the original Phase A (which only swapped robot_to_target -> GT
robot_field_pose in the obs and FAILED — a probe showed the policy ignored
field pose because the dribble target was a random point, so self-loc was
useless), this version:

  1. GEOMETRY FIX: goal_target_fraction=1.0 — the target is ALWAYS the goal
     mouth, so the goal bearing genuinely depends on the robot's field pose.
     This is the root-cause fix for the prior negative result.
  2. EXPLICIT ESTIMATE: a 4-d non-motor "selfloc" action head reports the
     policy's estimate of [x_n, y_n, sin(yaw), cos(yaw)]. It never drives the
     sim; it is scored by selfloc_accuracy (+) / selfloc_error_penalty (-) and
     fed back to the next obs via the existing "actions" term.

This widens BOTH actor and critic networks (the "actions" obs term sits in both
groups and grows 20 -> 24):
  - actor: 1D obs 85 -> 89, mlp.0 in 148 -> 153, output 20 -> 24, std_param
    (20,) -> (24,)
  - critic: obs 94 -> 98, mlp.0 in 94 -> 98 (output stays 1)
So NEITHER actor nor critic can strict-load. We use a generic shape-matching
partial load: keep every tensor whose shape is unchanged (deeper MLP layers,
the depth-ball CNN cnns.camera.*), reinit the rest (input layers, normalizers,
output layer, std_param). v3f showed the remapping relearns fast.

Usage:
  MUJOCO_GL=egl uv run python scripts/spike_v3g_selfloc_explicit.py          # full
  MUJOCO_GL=egl uv run python scripts/spike_v3g_selfloc_explicit.py --smoke  # 8-iter
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.mos92.env_cfgs import mos92_soccer_selfloc_env_cfg
from mjlab.tasks.velocity.config.mos92.rl_cfg import mos92_vision_ppo_runner_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

BASE_CKPT = Path(
  "logs/rsl_rl/mos92_velocity/2026-06-06_00-09-22_spike_v3f_holding/model_1499.pt"
)
NUM_ENVS = 1024
MAX_ITER = 2000


# Output tensors that GREW only by appending the 4 selfloc rows AFTER the 20
# motor rows (joint_pos action term is first, selfloc appended): the leading 20
# rows are the SAME motors as v3f, so copy them in and leave only the 4 selfloc
# rows fresh. This preserves v3f's learned kicking output map (the first full run
# reinit the whole output layer and lost dribbling). Input tensors (mlp.0,
# obs_normalizer) have reshuffled COLUMNS and genuinely must reinit.
_ROW_PREFIX_COPY = ("mlp.6.weight", "mlp.6.bias", "distribution.std_param")


def _partial_load_shape_match(module, ckpt_sd) -> tuple[int, int, list[str]]:
  """Load every name+shape-matching tensor; for output tensors that grew by
  appending selfloc rows, copy the v3f motor-row prefix into the new tensor and
  leave the appended selfloc rows at their fresh init. Reinit the genuinely
  input-coupled tensors (mlp.0, obs_normalizer). Deeper MLP + depth CNN carry
  the v3f skill over.

  Returns (loaded, prefix-copied + reinit count, notes)."""
  cur = module.state_dict()
  to_load, notes = {}, []
  prefix_copied = 0
  for k, v in ckpt_sd.items():
    if k not in cur:
      notes.append(f"{k}:absent")
      continue
    if cur[k].shape == v.shape:
      to_load[k] = v
    elif k in _ROW_PREFIX_COPY and v.shape[0] < cur[k].shape[0]:
      # Copy v3f's motor rows into the new (wider) tensor; selfloc rows stay fresh.
      new_t = cur[k].clone()
      n = v.shape[0]
      new_t[:n] = v
      to_load[k] = new_t
      prefix_copied += 1
      notes.append(f"{k}:prefix-copied[{n}/{cur[k].shape[0]}]")
    else:
      notes.append(f"{k}:reinit{tuple(v.shape)}->{tuple(cur[k].shape)}")
  module.load_state_dict(to_load, strict=False)
  return len(to_load), prefix_copied, notes


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  smoke = "--smoke" in sys.argv
  configure_torch_backends()
  device = "cuda:0"

  env_cfg = mos92_soccer_selfloc_env_cfg(play=False)
  env_cfg.scene.num_envs = 32 if smoke else NUM_ENVS

  # Pin GT mask to 0 (pure-vision ball, no crutch) — finetune counter restarts.
  gt = env_cfg.curriculum["gt_mask"]
  gt.params["start_step"] = -1
  gt.params["end_step"] = 0
  # Anti-cheat penalties: already learned in v3f, ramp fast back to full.
  pr = env_cfg.curriculum["penalty_ramp"]
  if smoke:
    pr.params["start_step"] = -1
    pr.params["end_step"] = 0
  else:
    pr.params["start_step"] = 50 * 24
    pr.params["end_step"] = 300 * 24
    pr.params["start_factor"] = 0.5

  max_iter = 8 if smoke else MAX_ITER
  agent_cfg = mos92_vision_ppo_runner_cfg()
  agent_cfg.max_iterations = max_iter
  agent_cfg.run_name = "spike_v3g_selfloc_explicit"
  agent_cfg.save_interval = 100

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_v3g_selfloc_explicit")
  log_dir = log_root / stamp
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)
  print(f"[INFO] Partial-load from v3f: {BASE_CKPT}")
  ckpt = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
  a_load, a_skip, a_keys = _partial_load_shape_match(
    runner.alg._raw_actor, ckpt["actor_state_dict"]
  )
  c_load, c_skip, c_keys = _partial_load_shape_match(
    runner.alg._raw_critic, ckpt["critic_state_dict"]
  )
  a_load, a_pfx, a_notes = _partial_load_shape_match(
    runner.alg._raw_actor, ckpt["actor_state_dict"]
  )
  c_load, c_pfx, c_notes = _partial_load_shape_match(
    runner.alg._raw_critic, ckpt["critic_state_dict"]
  )
  print(f"[INFO] actor : loaded {a_load} (prefix-copied {a_pfx}) -> {a_notes}")
  print(f"[INFO] critic: loaded {c_load} (prefix-copied {c_pfx}) -> {c_notes}")
  print(
    "[INFO] depth CNN + deeper MLP + v3f motor output map carried over; only the "
    "4 selfloc rows + input layers are fresh."
  )

  mode = "SMOKE" if smoke else "FULL"
  print(
    f"[INFO] v3g Phase A explicit self-loc [{mode}]: {max_iter} iters, "
    f"envs={env_cfg.scene.num_envs}"
  )
  print(
    "[INFO] watch Metrics/selfloc_pos_err_m (should fall) + dribble_success "
    "(goal-dribble should recover as the policy learns to aim from field pose)."
  )
  runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)
  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
