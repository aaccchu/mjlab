"""v4 EXP17: FIX THE OUT-OF-BOUNDS SEMANTIC BUG (on top of EXP16's kick policy).

EXP16 reached the best kick chain so far — real shot rate 0.32 (+45% over EXP14),
fell_over 0.044, pos_err 0.91 — but out_of_bounds rose to 0.345. Root cause, verified
three ways (code + data + literature, see v4越界处理调研.md):

  out_of_field_bounds was marked time_out=True, so PPO's truncation bootstrap added
  gamma * V(post-OOB) to the reward at the OOB step (rsl_rl ppo.py) — telling the
  policy that leaving the field is exactly as valuable as continuing. OOB went
  UNPENALIZED and grew as a free side effect of learning to strike goalward
  (out_bnd vs goal_rate correlation +0.514). fell_over correctly uses time_out=False;
  OOB was the lone mislabeled failure.

EXP17 applies the research doc's steps 1 + 2 TOGETHER (so the fix does not make the
policy timid about chasing the ball):
  1. out_of_field_bounds -> time_out=False (real termination, no bootstrap) +
     explicit one-shot out_of_bounds_penalty (weight -5, ~0.10 after dt-scaling).
  2. Goal-mouth corridor: inside |y| < 1.3 m the end-line OOB limit is pushed out by
     0.6 m so charging into the mouth to shoot is legal; side lines unchanged.

Bootstrap: EXP16 model_1999 — gait + kick + EKF localization all learned. NO std
reset (unlike EXP16, which reopened kick exploration from the EXP13 collapsed std):
the kick skill exists, we only retune the boundary incentive, so we keep the policy's
current (well-tuned) action std and let PPO adapt it.

Criteria: out_of_bounds drops toward <=0.20 WHILE real shot rate
(goal_rate/target_is_goal) holds near 0.32 and fell_over (<0.1) / pos_err (<1.1) stay
put. Failure mode to watch (research red line): if shot rate collapses or
robot_to_ball rises, the penalty is too harsh / corridor too narrow — back off weight.

Usage: MUJOCO_GL=egl uv run python scripts/spike_v4_e2e_ekf_kick_oob.py [--smoke]
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
  mos92_soccer_e2e_dualcam_ekf_kick_oob_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import (
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

# Bootstrap from EXP16 (best kick chain to date): inherits gait + approach + EKF
# localization (pos_err 0.91) + the near-foot kick skill (real shot rate 0.32).
# EXP17 only retunes the out-of-bounds incentive, so the obs layout is identical and
# every tensor loads cleanly (no reinit, no std reset).
BASE_CKPT = Path(
  "logs/rsl_rl/mos92_velocity/2026-06-09_22-19-34_spike_v4_e2e_ekf_kick/model_1999.pt"
)
NUM_ENVS = 384
MAX_ITER = 2000


def _partial_load_shape_match(module, ckpt_sd) -> tuple[int, int, list[str]]:
  """Load every name+shape-matching tensor; reinit the rest. For EXP17 the obs
  layout is unchanged from EXP16 so this should load everything (reinit 0)."""
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

  env_cfg = mos92_soccer_e2e_dualcam_ekf_kick_oob_env_cfg(play=False)
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
  agent_cfg.run_name = "spike_v4_e2e_ekf_kick_oob"
  agent_cfg.save_interval = 100

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_v4_e2e_ekf_kick_oob")
  log_dir = log_root / stamp
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)
  if BASE_CKPT.exists():
    print(f"[INFO] Partial-load from EXP16: {BASE_CKPT}")
    ckpt = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
    a_load, a_re, a_notes = _partial_load_shape_match(
      runner.alg._raw_actor, ckpt["actor_state_dict"]
    )
    c_load, c_re, c_notes = _partial_load_shape_match(
      runner.alg._raw_critic, ckpt["critic_state_dict"]
    )
    print(f"[INFO] actor : loaded {a_load}, reinit {a_re} -> {a_notes}")
    print(f"[INFO] critic: loaded {c_load}, reinit {c_re} -> {c_notes}")
    # NO std reset: EXP16 already learned the kick from a reopened std (0.85) and
    # PPO annealed it to a well-tuned value. EXP17 only changes the boundary
    # incentive, not the action repertoire, so we keep the inherited std and let
    # PPO adapt. (Reopening std here would needlessly destabilize a working gait.)
    if a_re != 0 or c_re != 0:
      print(f"[WARN] expected 0 reinit (same obs layout); got a={a_re} c={c_re}")
  else:
    print(f"[WARN] bootstrap ckpt missing ({BASE_CKPT}); training from scratch.")

  mode = "SMOKE" if smoke else "FULL"
  print(
    f"[INFO] v4 EXP17 oob-fix [{mode}]: {max_iter} iters, envs={env_cfg.scene.num_envs}"
  )
  print(
    "[INFO] watch Episode_Termination/out_of_field_bounds (should DROP toward <=0.2)"
  )
  print(
    "[INFO] WHILE goal_rate/target_is_goal holds ~0.32 and fell_over~0, pos_err<1.1."
  )
  runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)
  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
