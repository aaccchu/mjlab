"""Spike v3-MOS92 (gaze warmup): A2 + head depth camera into the actor CNN.

Bootstraps from the validated A2 search checkpoint. The vision actor's first
MLP layer takes 148 inputs (84 GT 1D obs + 64 CNN spatial-softmax latent,
latent appended LAST per the CNN model). We load A2's learned weights into the
first 84 columns and zero-fill the trailing 64, so the gait + dribble + GT
gaze behavior is preserved while the depth->latent path starts from zero and
must earn its way in via the gaze reward. The CNN encoder keeps its fresh init.
The critic is GT-only and identical to A2, so it loads strict.

Usage:
  MUJOCO_GL=egl uv run python scripts/spike_v3_vision.py
"""

from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.mos92.env_cfgs import mos92_soccer_vision_env_cfg
from mjlab.tasks.velocity.config.mos92.rl_cfg import mos92_vision_ppo_runner_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

BASE_CKPT = Path(
  "logs/rsl_rl/mos92_velocity/2026-06-02_17-14-09_spike_a2_search/model_2999.pt"
)
NUM_ENVS = 1024
MAX_ITER = 3000
N_CNN_LATENT = 64  # spatial-softmax latent appended after the 84-d 1D obs.


def _pad_actor_for_cnn(sd: dict, n_latent: int) -> dict:
  """Append n_latent zero columns to mlp.0.weight (CNN latent goes last)."""
  w = sd["mlp.0.weight"]  # (hidden, 84)
  hidden, _ = w.shape
  sd["mlp.0.weight"] = torch.cat(
    [w, torch.zeros(hidden, n_latent, dtype=w.dtype)], dim=1
  )
  return sd


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()
  device = "cuda:0"

  env_cfg = mos92_soccer_vision_env_cfg(play=False)
  env_cfg.scene.num_envs = NUM_ENVS

  agent_cfg = mos92_vision_ppo_runner_cfg()
  agent_cfg.max_iterations = MAX_ITER
  agent_cfg.run_name = "spike_v3_vision"
  agent_cfg.save_interval = 100

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_v3_vision")
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)

  # CNN-aware bootstrap: pad actor first layer for the appended CNN latent,
  # then load A2's keys non-strict (CNN encoder keeps its fresh init).
  # Critic is identical to A2 (GT-only, 94-d), so it loads strict.
  print(f"[INFO] Bootstrapping from: {BASE_CKPT}")
  ckpt = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
  _pad_actor_for_cnn(ckpt["actor_state_dict"], N_CNN_LATENT)
  missing, unexpected = runner.alg._raw_actor.load_state_dict(
    ckpt["actor_state_dict"], strict=False
  )
  cnn_missing = [k for k in missing if "cnn" not in k]
  assert not cnn_missing, f"unexpected missing (non-CNN) actor keys: {cnn_missing}"
  assert not unexpected, f"unexpected actor keys in checkpoint: {unexpected}"
  print(f"[INFO] Actor: loaded A2, fresh CNN keys = {len(missing)}")
  runner.alg._raw_critic.load_state_dict(ckpt["critic_state_dict"], strict=True)

  print(f"[INFO] Spike v3 gaze-warmup: {MAX_ITER} iters, envs={NUM_ENVS}")
  runner.learn(num_learning_iterations=MAX_ITER, init_at_random_ep_len=True)

  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
