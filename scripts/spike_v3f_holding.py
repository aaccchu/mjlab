"""Task v3f (MOS92): strengthen the HOLDING penalty, refine from v3e.

v3e cut holding_ball_frac 0.30->0.23 and fixed the arms, but holding is still
~23% — the dominant residual exploit (the v3d straddle was holding-type, not
strict 2-contact trapping). v3f refines from the v3e checkpoint with a stronger
holding penalty:
  - holding_ball weight  -2.0 -> -3.0   (now equal to trapping, the heaviest)
  - holding threshold     1.5s -> 1.0s   (stricter; still within the documented
                                          1-2s range, so legit between-kick
                                          pauses are not punished)
Everything else identical to v3e (same sensors, other penalties, pure-vision
mask pinned to 0). penalty_weight curriculum is SHORT here (ramp over iters
50-300) because the policy already kicks — we just need the stronger holding
weight to bite quickly. 1500 iters (refinement, not from-scratch).

Risk being watched: too-aggressive holding penalty could make the robot keep
the ball too far (afraid to control it) -> dribble_success drops. The A/B eval
(eval_v3d_vs_v3e.py, extended to v3f) checks holding DOWN *and* dribble HELD.

Usage:
  MUJOCO_GL=egl uv run python scripts/spike_v3f_holding.py            # full
  MUJOCO_GL=egl uv run python scripts/spike_v3f_holding.py --smoke    # 8-iter
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
  mos92_soccer_vision_ablation_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import mos92_vision_ppo_runner_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

BASE_CKPT = Path(
  "logs/rsl_rl/mos92_velocity/2026-06-05_22-43-22_spike_v3e_anticheat/model_2999.pt"
)
NUM_ENVS = 1024
MAX_ITER = 1500
_NSPE = 24


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  smoke = "--smoke" in sys.argv
  configure_torch_backends()
  device = "cuda:0"

  env_cfg = mos92_soccer_vision_ablation_env_cfg(play=False)
  env_cfg.scene.num_envs = 32 if smoke else NUM_ENVS

  # Stronger HOLDING penalty (the dominant residual exploit after v3e).
  env_cfg.rewards["holding_ball"].weight = -3.0
  env_cfg.rewards["holding_ball"].params["time_threshold"] = 1.0

  # Pin GT mask to 0 (pure vision, no crutch) — finetune step counter restarts.
  gt_mask = env_cfg.curriculum["gt_mask"]
  gt_mask.params["start_step"] = -1
  gt_mask.params["end_step"] = 0

  # Penalty ramp: policy already kicks (from v3e), so ramp fast (50->300) and
  # update the holding base weight to the new -3.0 so the curriculum scales the
  # right target (it overwrites .weight each step from base_weights).
  pr = env_cfg.curriculum["penalty_ramp"]
  pr.params["base_weights"]["holding_ball"] = -3.0
  if smoke:
    pr.params["start_step"] = -1
    pr.params["end_step"] = 0
  else:
    pr.params["start_step"] = 50 * _NSPE
    pr.params["end_step"] = 300 * _NSPE
    pr.params["start_factor"] = 0.5  # already-trained base; start mid-strength

  max_iter = 8 if smoke else MAX_ITER
  agent_cfg = mos92_vision_ppo_runner_cfg()
  agent_cfg.max_iterations = max_iter
  agent_cfg.run_name = "spike_v3f_holding"
  agent_cfg.save_interval = 100

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_v3f_holding")
  log_dir = log_root / stamp
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)
  print(f"[INFO] Bootstrapping (strict) from v3e: {BASE_CKPT}")
  ckpt = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
  runner.alg._raw_actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
  runner.alg._raw_critic.load_state_dict(ckpt["critic_state_dict"], strict=True)
  print("[INFO] Loaded v3e strict; holding penalty -3.0 @1.0s, GT mask pinned 0.")

  mode = "SMOKE" if smoke else "FULL"
  print(
    f"[INFO] Task v3f holding [{mode}]: {max_iter} iters, envs={env_cfg.scene.num_envs}"
  )
  print("[INFO] watch holding_ball_frac -> down WITHOUT dribble_success collapse.")
  runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)
  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
