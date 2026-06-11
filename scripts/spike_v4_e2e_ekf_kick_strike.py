"""v4 EXP23: NEAR-GOAL STRIKE GAIN (one decisive finish should outbid ten nudges).

Three rounds pinned the stubborn bottleneck: near-goal kick speed 0.47-0.52
(mid-field 0.73), kicks/episode stuck at 9, SHORT back to 40.6% once arrival was
fixed (EXP22b). Not a penalty problem (EXP21 exempted the corridor) — a skill
local optimum: nudges earn steady goal_progress risk-free; a strike risks
fall/OOB for only +0.045/contact of kick_impulse.

Single variable: near_goal_gain=3.0 on dribble_kick_impulse for ball x>9
(gap 0.045 -> 0.135/contact, comparable to risk cost). threshold stays 0.3
(EXP15: raising the gate starves the signal).

Bootstrap: EXP22b model_1999 (archived best, shot 0.40-0.42 band). NO std reset.

Criteria: near-goal kick speed >=0.65, SHORT <30%, shot rate >=0.45.
Guardrails: fell<0.1 (striking is riskier!), OOB<=0.25, pos_err<1.1,
ball_speed_peak >= 2.6 (should REVERSE the 2.82->2.65 three-round decline).

Usage: MUJOCO_GL=egl uv run python scripts/spike_v4_e2e_ekf_kick_strike.py [--smoke]
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
  mos92_soccer_e2e_dualcam_ekf_kick_strike_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import (
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

BASE_CKPT = Path("checkpoints/v4_soccer/kick_plant_exp22b/model_1999.pt")
NUM_ENVS = 384
MAX_ITER = 2000


def _partial_load_shape_match(module, ckpt_sd) -> tuple[int, int, list[str]]:
  """Load every name+shape-matching tensor; reinit the rest. EXP20 keeps the obs
  layout identical to EXP19 so this should load everything (reinit 0)."""
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

  env_cfg = mos92_soccer_e2e_dualcam_ekf_kick_strike_env_cfg(play=False)
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
  agent_cfg.run_name = "spike_v4_e2e_ekf_kick_strike"
  agent_cfg.save_interval = 100

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_v4_e2e_ekf_kick_strike")
  log_dir = log_root / stamp
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)
  if BASE_CKPT.exists():
    print(f"[INFO] Partial-load from EXP20: {BASE_CKPT}")
    ckpt = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
    a_load, a_re, a_notes = _partial_load_shape_match(
      runner.alg._raw_actor, ckpt["actor_state_dict"]
    )
    c_load, c_re, c_notes = _partial_load_shape_match(
      runner.alg._raw_critic, ckpt["critic_state_dict"]
    )
    print(f"[INFO] actor : loaded {a_load}, reinit {a_re} -> {a_notes}")
    print(f"[INFO] critic: loaded {c_load}, reinit {c_re} -> {c_notes}")
    # NO std reset (codex "ball-hugging attractor" lesson): keep EXP19's tuned
    # action std; the new incentives reshape gaze/alignment, not the repertoire.
    if a_re != 0 or c_re != 0:
      print(f"[WARN] expected 0 reinit (same obs layout); got a={a_re} c={c_re}")
  else:
    print(f"[WARN] bootstrap ckpt missing ({BASE_CKPT}); training from scratch.")

  mode = "SMOKE" if smoke else "FULL"
  print(
    f"[INFO] v4 EXP20b aim-continue [{mode}]: {max_iter} iters, "
    f"envs={env_cfg.scene.num_envs}"
  )
  print("[INFO] target: shot rate 0.32 -> >=0.40; watch Metrics/kick_lat_align UP,")
  print("[INFO] fell_over<0.1, pos_err<1.1 (look-down vs landmarks!), peak>2.8.")
  runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)
  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
