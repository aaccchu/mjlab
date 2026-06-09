"""v4 EXP8: TEMPORAL FUSION + ACTIVE NECK SCAN (on the oracle belief).

EXP6 fixed the architecture (perception-as-obs -> fell_over 52->0) but single-frame
belief is stuck at ~4/23 keypoints (pos_err ~4m). EXP7 proved passive multi-frame
fusion barely helps (unique coverage 3.9->4.4) — a forward-walking robot keeps the
SAME keypoints in view. EXP8 offline analysis: the head cam rides neck_yaw (+-90deg)
and a neck sweep raises unique coverage 5->9-14; the FusedPoseBelief obs hit median
1.03m PASSIVELY in validation. This run unlocks the scan BEHAVIOR:
  - FusedPoseBelief obs (odometry-fused 8-frame/stride-4 window, 7-dim incl uniq_frac)
  - active_scan_coverage reward (drives neck sweeping toward 0.6 unique coverage)
  - relaxed neck_yaw pose centering (std 0.15->1.5) so the policy can sweep.

Bootstrap: EXP6 model_2999 — gait + kicking + belief-consumption already learned;
only the neck-scan behavior is new. Belief obs grew 6->7 so actor mlp.0/normalizer
reinit by +1; everything else (trunk, CNNs, mlp.6) carries over.

Criteria: scan_uniq_frac rises >0.4 AND selfloc_pos_err_m drops <2m (toward 1m) WITH
fell_over staying ~0 and goal_rate not collapsing (codex red line: scanning must not
eat kicking/stability).

Usage: MUJOCO_GL=egl uv run python scripts/spike_v4_e2e_fused_fixed.py [--smoke]
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
  mos92_soccer_e2e_dualcam_fused_scan_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import (
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

# Bootstrap from the EXP6 oracle run (gait+kick+belief-consumption learned).
BASE_CKPT = Path(
  "logs/rsl_rl/mos92_velocity/2026-06-08_22-16-29_spike_v4_e2e_oracle/model_2999.pt"
)
NUM_ENVS = 384
MAX_ITER = 2000


def _partial_load_shape_match(module, ckpt_sd) -> tuple[int, int, list[str]]:
  """Load every name+shape-matching tensor; reinit the rest. Belief 6->7 grows
  actor mlp.0 input by 1 and the obs_normalizer; trunk/CNNs/mlp.6 carry over."""
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

  env_cfg = mos92_soccer_e2e_dualcam_fused_scan_env_cfg(play=False)
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
  agent_cfg.run_name = "spike_v4_e2e_fused_fixed"
  agent_cfg.save_interval = 100

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_v4_e2e_fused_fixed")
  log_dir = log_root / stamp
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)
  if BASE_CKPT.exists():
    print(f"[INFO] Partial-load from EXP6: {BASE_CKPT}")
    ckpt = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
    a_load, a_re, a_notes = _partial_load_shape_match(
      runner.alg._raw_actor, ckpt["actor_state_dict"]
    )
    c_load, c_re, c_notes = _partial_load_shape_match(
      runner.alg._raw_critic, ckpt["critic_state_dict"]
    )
    print(f"[INFO] actor : loaded {a_load}, reinit {a_re} -> {a_notes}")
    print(f"[INFO] critic: loaded {c_load}, reinit {c_re} -> {c_notes}")
  else:
    print(f"[WARN] bootstrap ckpt missing ({BASE_CKPT}); training from scratch.")

  mode = "SMOKE" if smoke else "FULL"
  print(f"[INFO] v4 EXP8 fused-scan [{mode}]: {max_iter} iters, envs={env_cfg.scene.num_envs}")
  print("[INFO] watch Metrics/scan_uniq_frac (should RISE >0.4 as neck sweeps) +")
  print("[INFO] selfloc_pos_err_m (should drop <2m) with fell_over~=0, goal_rate kept.")
  runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)
  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
