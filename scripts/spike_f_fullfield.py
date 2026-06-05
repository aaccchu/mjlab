"""Spike F: Full-field scale generalization.

Tests whether a two-phase reward (approach delta + dribble) enables the
policy to generalize to full-field distances (9m spawn, 10.5m target).

Three groups:
  F1: moderate expansion (spawn 0.6-4m, target 2-8m)
  F2: full-field (spawn 0.6-9m, target 2-12m)
  F3: curriculum from F1 -> F2 (not implemented here, manual)

Each fine-tunes from v2 checkpoint for 2000 iterations.

Usage:
  uv run python scripts/spike_f_fullfield.py --group f1
  uv run python scripts/spike_f_fullfield.py --group f2
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_soccer_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.mdp.dribble_command import DribbleCommandCfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

V2_CHECKPOINT = Path("logs/rsl_rl/g1_velocity/2026-06-01_11-09-25/model_4999.pt")
MAX_ITERATIONS = 2000
NUM_ENVS = 4096

GROUPS = {
  "f1": {"spawn_dist": (0.6, 4.0), "target_dist": (2.0, 8.0)},
  "f2": {"spawn_dist": (0.6, 9.0), "target_dist": (2.0, 12.0)},
}


def make_spike_f_cfg(group: str):
  cfg = unitree_g1_soccer_env_cfg()
  cfg.scene.num_envs = NUM_ENVS

  params = GROUPS[group]
  dribble_cmd = cfg.commands["dribble"]
  assert isinstance(dribble_cmd, DribbleCommandCfg)
  dribble_cmd.spawn_dist_range = params["spawn_dist"]
  dribble_cmd.target_dist_range = params["target_dist"]

  # Replace exp-based approach reward with distance-delta reward.
  del cfg.rewards["dribble_approach"]
  cfg.rewards["approach_delta"] = RewardTermCfg(
    func=mdp.approach_delta,
    weight=2.0,
    params={
      "command_name": "dribble",
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.rewards["heading_to_ball"] = RewardTermCfg(
    func=mdp.heading_to_ball,
    weight=0.5,
    params={
      "command_name": "dribble",
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  return cfg


def main():
  import os

  parser = argparse.ArgumentParser()
  parser.add_argument("--group", choices=["f1", "f2"], required=True)
  args = parser.parse_args()

  os.environ["MUJOCO_GL"] = "egl"
  configure_torch_backends()

  env_cfg = make_spike_f_cfg(args.group)
  agent_cfg = unitree_g1_ppo_runner_cfg()
  agent_cfg.max_iterations = MAX_ITERATIONS
  agent_cfg.experiment_name = "g1_velocity"
  agent_cfg.run_name = f"spike_f_{args.group}"
  agent_cfg.save_interval = 200

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  log_dir = log_root / datetime.now().strftime(
    f"%Y-%m-%d_%H-%M-%S_spike_f_{args.group}"
  )
  log_dir.mkdir(parents=True, exist_ok=True)

  device = "cuda:0"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)

  print(f"[INFO] Loading v2 checkpoint: {V2_CHECKPOINT}")
  runner.load(str(V2_CHECKPOINT))

  print(
    f"[INFO] Training Spike F group={args.group} for {MAX_ITERATIONS} iterations..."
  )
  runner.learn(num_learning_iterations=MAX_ITERATIONS, init_at_random_ep_len=True)

  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
