"""Task v3e (MOS92): anti-cheat finetune from the v3d jump-fix checkpoint.

Continues from v3d (pure-vision, no-jump) and adds the 4 anti-cheat rule
penalties (trapping/holding/illegal-body-contact/sticking) now wired into the
env, with a penalty_weight curriculum that ramps them 0.2->1.0 over iters
100-700 (Spike-C: full strength from step 0 makes "don't touch the ball"
optimal and dribbling collapses). Also picks up the arm-keyframe fix
(shoulder_roll ±1.4->±0.15, arms down) — the policy relearns the new neutral.

Goal: kill the v3d straddling/ball-pinning exploit while KEEPING pure vision.
Success: Metrics/ball_trapped_frac & holding_ball_frac -> ~0 while
dribble_success / gaze_center hold; arms hang at sides on video.

The gt_mask curriculum is pinned to factor=0 from step 0 (start=-1, end=0) so
this finetune NEVER re-introduces the GT crutch (same safeguard as v3d).

Usage:
  MUJOCO_GL=egl uv run python scripts/spike_v3e_anticheat.py            # full
  MUJOCO_GL=egl uv run python scripts/spike_v3e_anticheat.py --smoke    # 8-iter
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
  "logs/rsl_rl/mos92_velocity/2026-06-05_20-46-37_spike_v3d_jumpfix/model_1499.pt"
)
NUM_ENVS = 1024
MAX_ITER = 3000


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  smoke = "--smoke" in sys.argv
  configure_torch_backends()
  device = "cuda:0"

  env_cfg = mos92_soccer_vision_ablation_env_cfg(play=False)
  env_cfg.scene.num_envs = 32 if smoke else NUM_ENVS

  # Pin the GT-ablation mask to factor=0 from step 0 (same safeguard as v3d):
  # the finetune step counter restarts at 0, so reusing v3b's ramp would re-
  # introduce the GT ball crutch. start_step=-1, end_step=0 => factor=0 always.
  gt_mask = env_cfg.curriculum["gt_mask"]
  gt_mask.params["start_step"] = -1
  gt_mask.params["end_step"] = 0

  # Anti-cheat penalty ramp: for a smoke run, compress to fire immediately so we
  # can see penalties bite within 8 iters; for the full run keep the designed
  # 100->700 iter ramp (Spike-C: don't slam full penalties before kicking is
  # relearned from the v3d bootstrap).
  if smoke:
    pr = env_cfg.curriculum["penalty_ramp"]
    pr.params["start_step"] = -1
    pr.params["end_step"] = 0

  max_iter = 8 if smoke else MAX_ITER
  agent_cfg = mos92_vision_ppo_runner_cfg()
  agent_cfg.max_iterations = max_iter
  agent_cfg.run_name = "spike_v3e_anticheat"
  agent_cfg.save_interval = 100

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_v3e_anticheat")
  log_dir = log_root / stamp
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)

  # Strict load: v3d shares the exact obs/actor/critic layout this env produces
  # (penalties add reward terms, not obs dims). Fresh optimizer/iteration so the
  # penalty curriculum can reshape the policy off the no-jump pure-vision base.
  print(f"[INFO] Bootstrapping (strict) from: {BASE_CKPT}")
  ckpt = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
  runner.alg._raw_actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
  runner.alg._raw_critic.load_state_dict(ckpt["critic_state_dict"], strict=True)
  print("[INFO] Actor (CNN) + critic loaded strict; GT mask pinned to 0.")

  mode = "SMOKE" if smoke else "FULL"
  print(
    f"[INFO] Task v3e anti-cheat [{mode}]: {max_iter} iters, envs={env_cfg.scene.num_envs}"
  )
  print("[INFO] watch ball_trapped_frac & holding_ball_frac -> 0 while dribble holds.")
  runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)

  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
