"""END-TO-END (目标①②③ in ONE policy): pure-vision self-localize + depth
find-ball + dribble the ball INTO the fixed goal, fewest steps.

This is the integration milestone. The selfloc-vision env already carried ①+②+
③-dribble; the e2e env (mos92_soccer_e2e_env_cfg) adds the goal-scoring layer
(attacking-half spawn, goal_progress/goal_scored, time-to-goal penalty) plus the
exp11 stability balance (upright 2.5) and keeps the GT pose obs faded from step 0.

Bootstrap from model_2800 (the pure-vision selfloc run). The e2e env has the
SAME obs/action structure (89-d actor, 24-d action, depth + 4-frame RGB CNNs),
so EVERY tensor shape-matches and loads — ①②③ skills transfer intact, and only
the goal-aiming behavior is newly shaped by the added rewards. No tensor reinit.

Usage:
  MUJOCO_GL=egl uv run python scripts/spike_v3g_e2e.py          # full
  MUJOCO_GL=egl uv run python scripts/spike_v3g_e2e.py --smoke  # 8-iter
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
from mjlab.tasks.velocity.config.mos92.env_cfgs import mos92_soccer_e2e_env_cfg
from mjlab.tasks.velocity.config.mos92.rl_cfg import (
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

# Pure-vision selfloc checkpoint (depth-ball CNN + RGB selfloc CNN + trunk +
# selfloc head, all trained). Same structure as the e2e env -> full load.
BASE_CKPT = Path(
  "logs/rsl_rl/mos92_velocity/2026-06-07_12-21-39_spike_v3g_e2e/model_2499.pt"
)
NUM_ENVS = 1024
MAX_ITER = 1500


def _shape_match_load(module, ckpt_sd) -> tuple[int, int, list[str]]:
  """Load every name+shape-matching tensor; reinit the rest. e2e env == selfloc
  vision structure, so expect reinit 0 (full transfer of ①②③).

  Returns (loaded, reinit-count, notes)."""
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
  fresh = [k for k in cur if k not in ckpt_sd]
  if fresh:
    notes.append(f"fresh(not in ckpt): {len(fresh)}")
  return len(to_load), reinit, notes


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  smoke = "--smoke" in sys.argv
  configure_torch_backends()
  device = "cuda:0"

  env_cfg = mos92_soccer_e2e_env_cfg(play=False)
  env_cfg.scene.num_envs = 32 if smoke else NUM_ENVS
  # e2e-fix: the first e2e run learned to score (goal_rate 0.21, fell_over 0)
  # but selfloc DEGRADED 0.82->4.76m — the goal rewards (w7 total) out-pulled
  # selfloc_accuracy (w0.8), a capacity competition. Raise selfloc 0.8->1.6 to
  # pull localization back; scoring is already in the weights so it persists.
  env_cfg.rewards["selfloc_accuracy"].weight = 1.6

  # Ball gt_mask already pinned to 0 (pure-vision ball) by the selfloc-vision
  # base; anti-cheat penalty_ramp already learned upstream — keep both at full.
  for key in ("gt_mask", "penalty_ramp"):
    if key in env_cfg.curriculum:
      env_cfg.curriculum[key].params["start_step"] = -1
      env_cfg.curriculum[key].params["end_step"] = 0

  max_iter = 8 if smoke else MAX_ITER
  agent_cfg = mos92_selfloc_vision_ppo_runner_cfg()
  agent_cfg.max_iterations = max_iter
  agent_cfg.run_name = "spike_v3g_e2e_fix"
  agent_cfg.save_interval = 100

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_v3g_e2e_fix")
  log_dir = log_root / stamp
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)
  print(f"[INFO] Full load from pure-vision selfloc: {BASE_CKPT}")
  ckpt = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
  a_load, a_re, a_notes = _shape_match_load(
    runner.alg._raw_actor, ckpt["actor_state_dict"]
  )
  c_load, c_re, c_notes = _shape_match_load(
    runner.alg._raw_critic, ckpt["critic_state_dict"]
  )
  print(f"[INFO] actor : loaded {a_load}, reinit {a_re} -> {a_notes}")
  print(f"[INFO] critic: loaded {c_load}, reinit {c_re} -> {c_notes}")
  print(
    "[INFO] ①selfloc + ②find-ball + ③dribble all transfer (expect reinit 0); "
    "only goal-aiming is newly shaped by goal_progress/goal_scored/time penalty."
  )

  mode = "SMOKE" if smoke else "FULL"
  print(f"[INFO] v3g E2E [{mode}]: {max_iter} iters, envs={env_cfg.scene.num_envs}")
  print(
    "[INFO] watch: Metrics/selfloc_pos_err_m (stay LOW=① held), "
    "dribble/goal_rate (rise=③ goal), upright (stay high=stable)."
  )
  runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)
  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
