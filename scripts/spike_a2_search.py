"""Spike A2-MOS92: search -> lock -> approach -> kick via reward gating.

Bootstraps from the validated MOS92 dribble checkpoint (G-2 soccer) by
zero-padding the first actor/critic layer for the 3 new ball_gaze_uv obs dims
(appended last), so the learned gait+dribble is preserved while the new gaze /
search / track behavior is what actually has to emerge.

Usage:
  MUJOCO_GL=egl uv run python scripts/spike_a2_search.py
"""

from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.mos92.env_cfgs import mos92_soccer_search_env_cfg
from mjlab.tasks.velocity.config.mos92.rl_cfg import mos92_ppo_runner_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

BASE_CKPT = Path("logs/rsl_rl/mos92_velocity/2026-06-02_13-33-44/model_3998.pt")
NUM_ENVS = 4096
MAX_ITER = 3000
NUM_NEW_OBS = 3  # ball_gaze_uv = (u, v, visible), appended last.


def _pad_state_dict(sd: dict, n_new: int) -> dict:
  """Zero-pad first-layer weight + obs normalizer for n_new appended obs dims."""
  w = sd["mlp.0.weight"]  # (hidden, in_dim)
  hidden, in_dim = w.shape
  sd["mlp.0.weight"] = torch.cat([w, torch.zeros(hidden, n_new, dtype=w.dtype)], dim=1)
  mean = sd["obs_normalizer._mean"]
  pad_mean = torch.zeros(1, n_new, dtype=mean.dtype)
  pad_unit = torch.ones(1, n_new, dtype=mean.dtype)
  sd["obs_normalizer._mean"] = torch.cat([mean, pad_mean], dim=1)
  sd["obs_normalizer._var"] = torch.cat([sd["obs_normalizer._var"], pad_unit], dim=1)
  sd["obs_normalizer._std"] = torch.cat([sd["obs_normalizer._std"], pad_unit], dim=1)
  return sd


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()
  device = "cuda:0"

  env_cfg = mos92_soccer_search_env_cfg(play=False)
  env_cfg.scene.num_envs = NUM_ENVS

  agent_cfg = mos92_ppo_runner_cfg()
  agent_cfg.max_iterations = MAX_ITER
  agent_cfg.run_name = "spike_a2_search"
  agent_cfg.save_interval = 100

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_a2_search")
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)

  # Bootstrap: pad the base dribble checkpoint to the wider gaze obs, then load
  # actor + critic only (fresh optimizer / iteration counter).
  print(f"[INFO] Bootstrapping from: {BASE_CKPT}")
  ckpt = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
  _pad_state_dict(ckpt["actor_state_dict"], NUM_NEW_OBS)
  _pad_state_dict(ckpt["critic_state_dict"], NUM_NEW_OBS)
  runner.alg._raw_actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
  runner.alg._raw_critic.load_state_dict(ckpt["critic_state_dict"], strict=True)

  print(f"[INFO] Spike A2-MOS92: {MAX_ITER} iters, envs={NUM_ENVS}")
  runner.learn(num_learning_iterations=MAX_ITER, init_at_random_ep_len=True)

  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
