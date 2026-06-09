"""v4 EXP19: STRENGTHENED SOFT-BOUNDARY SHAPING (EXP18 weights were too weak).

EXP17 fixed the time_out semantic bug (out_of_bounds 0.345 -> 0.262, shot rate
held at 0.30) but stalled short of <=0.20. EXP18 added dense soft-boundary
shaping yet out_of_bounds did not move: the Episode_Reward magnitudes showed
soft_boundary ~-0.03 and vel_toward_boundary ~-0.018 against goal_progress
~+0.53 — the boundary cost was ~20x smaller than the goalward pull, so it could
nudge but never override "charge goalward". Direction right, magnitude wrong.

EXP19 raises the weights to the same order as goal_progress so the policy
genuinely trades "shoot harder" against "stay in bounds", and widens the band:
  - soft_boundary_penalty: -0.3 -> -2.0 (depth into the soft band).
  - velocity_toward_boundary_penalty: -0.5 -> -3.0 (outward speed in the band).
  - soft_margin: 1.5 -> 2.0 m (earlier warning).
Everything else (time_out=False, one-shot penalty, goal corridor) is unchanged.

Bootstrap: EXP17 model_1999 (clean best-oob-fix start, NOT the half-finished
EXP18; same obs layout, no std reset). Target: out_of_bounds <= 0.20 WHILE the
real shot rate holds ~0.30 and guardrails (fell_over < 0.1, pos_err < 1.1) stay
put. Red line: if shot rate collapses or robot_to_ball climbs, the weights
overshot — back off toward EXP18.

Usage: MUJOCO_GL=egl uv run python scripts/spike_v4_e2e_ekf_kick_oob_soft2.py [--smoke]
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
from mjlab.tasks.velocity.config.mos92.env_cfgs import (
  mos92_soccer_e2e_dualcam_ekf_kick_oob_soft2_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import (
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

# Bootstrap from EXP17 (time_out fix + goal corridor already learned): inherits
# the full kick chain + the boundary-aware policy. EXP19 only adds dense soft
# shaping, so the obs layout is identical and every tensor loads cleanly.
BASE_CKPT = Path(
  "logs/rsl_rl/mos92_velocity/2026-06-10_02-49-31_spike_v4_e2e_ekf_kick_oob/model_1999.pt"
)
NUM_ENVS = 384
MAX_ITER = 2000


def _partial_load_shape_match(module, ckpt_sd) -> tuple[int, int, list[str]]:
  """Load every name+shape-matching tensor; reinit the rest. For EXP19 the obs
  layout is unchanged from EXP17 so this should load everything (reinit 0)."""
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

  env_cfg = mos92_soccer_e2e_dualcam_ekf_kick_oob_soft2_env_cfg(play=False)
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
  agent_cfg.run_name = "spike_v4_e2e_ekf_kick_oob_soft2"
  agent_cfg.save_interval = 100

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_v4_e2e_ekf_kick_oob_soft2")
  log_dir = log_root / stamp
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)
  if BASE_CKPT.exists():
    print(f"[INFO] Partial-load from EXP17: {BASE_CKPT}")
    ckpt = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
    a_load, a_re, a_notes = _partial_load_shape_match(
      runner.alg._raw_actor, ckpt["actor_state_dict"]
    )
    c_load, c_re, c_notes = _partial_load_shape_match(
      runner.alg._raw_critic, ckpt["critic_state_dict"]
    )
    print(f"[INFO] actor : loaded {a_load}, reinit {a_re} -> {a_notes}")
    print(f"[INFO] critic: loaded {c_load}, reinit {c_re} -> {c_notes}")
    # NO std reset: EXP17 has a well-tuned action std; EXP19 only adds dense
    # boundary shaping, not a new action repertoire, so keep the inherited std.
    if a_re != 0 or c_re != 0:
      print(f"[WARN] expected 0 reinit (same obs layout); got a={a_re} c={c_re}")
  else:
    print(f"[WARN] bootstrap ckpt missing ({BASE_CKPT}); training from scratch.")

  mode = "SMOKE" if smoke else "FULL"
  print(
    f"[INFO] v4 EXP19 soft-boundary [{mode}]: {max_iter} iters, "
    f"envs={env_cfg.scene.num_envs}"
  )
  print("[INFO] watch out_of_field_bounds (target <=0.20) WHILE shot rate holds ~0.30,")
  print("[INFO] fell_over<0.1, pos_err<1.1, robot_to_ball NOT climbing (red line).")
  runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)
  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
