"""Task v3g Phase A: privileged self-localization closed loop.

Swaps the actor's goal-direction spoon-feed (robot_to_target, 3d) for GT
self-localization (robot_field_pose, 4d) — the policy must now derive "which
way is the goal" from knowing where it stands on the field. Goal: prove the
"know where I am -> face goal -> kick" loop closes WITH GT, before adding the
RGB vision that will carry self-loc in Phase C.

Obs width changes 84 -> 85, so this is a PARTIAL load from v3f:
  - reinit actor mlp.0 (input layer, was 148=84+64 latent, now 149=85+64) +
    obs_normalizer (per-dim stats, wrong shape) -> fresh
  - KEEP actor mlp.2/4/6 (deeper policy), the depth-ball CNN (cnns.camera.*),
    and the FULL critic (critic obs unchanged, strict load)
The deeper MLP + CNN + critic carry over the v3f skill (kicking, anti-cheat,
pure-vision ball); only the input remapping relearns. v3f showed fast recovery.

Usage:
  MUJOCO_GL=egl uv run python scripts/spike_v3g_selfloc.py            # full
  MUJOCO_GL=egl uv run python scripts/spike_v3g_selfloc.py --smoke    # 8-iter
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


def _partial_load_actor(raw_actor, ckpt_actor_sd) -> tuple[int, int]:
  """Load v3f actor weights EXCEPT the input layer + normalizer (obs width
  changed 84->85). Returns (loaded, skipped) counts. Keeps deeper MLP + CNN."""
  cur = raw_actor.state_dict()
  to_load, skipped = {}, []
  for k, v in ckpt_actor_sd.items():
    # skip input-coupled tensors: mlp.0 (input layer) and obs_normalizer stats
    if k.startswith("mlp.0.") or k.startswith("obs_normalizer."):
      skipped.append(k)
      continue
    if k in cur and cur[k].shape == v.shape:
      to_load[k] = v
    else:
      skipped.append(k)
  missing = raw_actor.load_state_dict(to_load, strict=False)
  del missing
  return len(to_load), len(skipped)


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
  agent_cfg.run_name = "spike_v3g_selfloc"
  agent_cfg.save_interval = 100

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_v3g_selfloc")
  log_dir = log_root / stamp
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)
  print(f"[INFO] Partial-load from v3f: {BASE_CKPT}")
  ckpt = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
  loaded, skipped = _partial_load_actor(runner.alg._raw_actor, ckpt["actor_state_dict"])
  # Critic obs unchanged (full GT) -> strict load is fine.
  runner.alg._raw_critic.load_state_dict(ckpt["critic_state_dict"], strict=True)
  print(f"[INFO] actor: loaded {loaded} tensors, reinit {skipped} (mlp.0+normalizer).")
  print("[INFO] critic strict-loaded; depth-ball CNN + deeper MLP carried over.")

  mode = "SMOKE" if smoke else "FULL"
  print(
    f"[INFO] v3g Phase A self-loc [{mode}]: {max_iter} iters, envs={env_cfg.scene.num_envs}"
  )
  print(
    "[INFO] watch dribble_success/goal recover (policy relearns goal dir from field pose)."
  )
  runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)
  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
