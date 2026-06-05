"""Spike A: Gaze feasibility — fine-tune v2 with gaze_at_ball reward.

Supports multiple experiment groups to iterate on parameters.

Usage:
  uv run python scripts/spike_a_gaze.py --exp a1   # original (high weight)
  uv run python scripts/spike_a_gaze.py --exp a2   # conservative
  uv run python scripts/spike_a_gaze.py --exp a3   # ultra-conservative
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
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

V2_CHECKPOINT = Path("logs/rsl_rl/g1_velocity/2026-06-01_11-09-25/model_4999.pt")
NUM_ENVS = 4096

EXPERIMENTS = {
  "a1": {
    "gaze_weight": 1.0,
    "gaze_std": 2.0,
    "waist_yaw_std_walk": 1.0,
    "waist_pitch_std_walk": 0.5,
    "waist_yaw_std_run": 1.5,
    "waist_pitch_std_run": 0.8,
    "max_iterations": 500,
  },
  "a2": {
    "gaze_weight": 0.3,
    "gaze_std": 2.0,
    "waist_yaw_std_walk": 0.5,
    "waist_pitch_std_walk": 0.3,
    "waist_yaw_std_run": 0.8,
    "waist_pitch_std_run": 0.5,
    "max_iterations": 1000,
  },
  "a3": {
    "gaze_weight": 0.1,
    "gaze_std": 2.0,
    "waist_yaw_std_walk": 0.35,
    "waist_pitch_std_walk": 0.2,
    "waist_yaw_std_run": 0.5,
    "waist_pitch_std_run": 0.3,
    "max_iterations": 1000,
  },
}


def make_spike_a_cfg(exp: str):
  params = EXPERIMENTS[exp]
  cfg = unitree_g1_soccer_env_cfg()
  cfg.scene.num_envs = NUM_ENVS

  cfg.rewards["pose"].params["std_walking"][r".*waist_yaw.*"] = params[
    "waist_yaw_std_walk"
  ]
  cfg.rewards["pose"].params["std_walking"][r".*waist_pitch.*"] = params[
    "waist_pitch_std_walk"
  ]
  cfg.rewards["pose"].params["std_running"][r".*waist_yaw.*"] = params[
    "waist_yaw_std_run"
  ]
  cfg.rewards["pose"].params["std_running"][r".*waist_pitch.*"] = params[
    "waist_pitch_std_run"
  ]

  cfg.rewards["gaze_at_ball"] = RewardTermCfg(
    func=mdp.gaze_at_ball,
    weight=params["gaze_weight"],
    params={
      "command_name": "dribble",
      "std": params["gaze_std"],
      "asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"]),
    },
  )
  return cfg, params["max_iterations"]


def main():
  import os

  parser = argparse.ArgumentParser()
  parser.add_argument("--exp", choices=list(EXPERIMENTS.keys()), required=True)
  args = parser.parse_args()

  os.environ["MUJOCO_GL"] = "egl"
  configure_torch_backends()

  env_cfg, max_iter = make_spike_a_cfg(args.exp)
  agent_cfg = unitree_g1_ppo_runner_cfg()
  agent_cfg.max_iterations = max_iter
  agent_cfg.experiment_name = "g1_velocity"
  agent_cfg.run_name = f"spike_a_{args.exp}"
  agent_cfg.save_interval = 100

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  log_dir = log_root / datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_spike_a_{args.exp}")
  log_dir.mkdir(parents=True, exist_ok=True)

  device = "cuda:0"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)

  print(f"[INFO] Loading v2 checkpoint: {V2_CHECKPOINT}")
  runner.load(str(V2_CHECKPOINT))

  print(f"[INFO] Spike A exp={args.exp}, {max_iter} iterations")
  print(f"[INFO] Params: {EXPERIMENTS[args.exp]}")
  runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)

  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
