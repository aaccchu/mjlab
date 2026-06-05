"""Spike v3b-MOS92 (GT ablation): force the depth CNN to carry ball bearing.

Bootstraps from the v3 gaze-warmup checkpoint (model_2999) and PROGRESSIVELY
MASKS the actor's GT ball obs to zero via the `gt_mask` curriculum, so PPO is
forced to route ball information through the head depth camera -> CNN path. The
obs dimension stays 84 (mask = scale->0, terms not removed), so the warmup
checkpoint loads strict — actor (with CNN) AND critic.

Success signal: gaze_center rises off its stuck ~0.05 warmup baseline as the
`gt_mask` curriculum scalar falls 1->0, while dribble_success / upright hold.
That is the thing warmup could not prove: the CNN learned to see.

Usage:
  MUJOCO_GL=egl uv run python scripts/spike_v3b_ablation.py
"""

from __future__ import annotations

import os
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
  "logs/rsl_rl/mos92_velocity/2026-06-02_20-19-43_spike_v3_vision/model_2999.pt"
)
NUM_ENVS = 1024
MAX_ITER = 3000


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()
  device = "cuda:0"

  env_cfg = mos92_soccer_vision_ablation_env_cfg(play=False)
  env_cfg.scene.num_envs = NUM_ENVS

  agent_cfg = mos92_vision_ppo_runner_cfg()
  agent_cfg.max_iterations = MAX_ITER
  agent_cfg.run_name = "spike_v3b_ablation"
  agent_cfg.save_interval = 100

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_v3b_ablation")
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)

  # Bootstrap strict: the warmup checkpoint already has the CNN actor + GT critic
  # at the exact 84-d (actor 1D) / 94-d (critic) layout this env produces, since
  # masking keeps dims unchanged. Load actor + critic, fresh optimizer/iteration.
  print(f"[INFO] Bootstrapping (strict) from: {BASE_CKPT}")
  ckpt = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
  runner.alg._raw_actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
  runner.alg._raw_critic.load_state_dict(ckpt["critic_state_dict"], strict=True)
  print("[INFO] Actor (CNN) + critic loaded strict.")

  print(f"[INFO] Spike v3b GT-ablation: {MAX_ITER} iters, envs={NUM_ENVS}")
  print("[INFO] gt_mask ramps actor GT ball obs scale 1->0 over iters 500-1500.")
  runner.learn(num_learning_iterations=MAX_ITER, init_at_random_ep_len=True)

  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
