"""Smoke (d): full vision env + CNN actor forward pass, no shape/NaN errors.

Builds the registered Mjlab-Velocity-Soccer-Vision-MOS92 env, wraps it,
constructs the MjlabOnPolicyRunner (which builds the spatial-softmax CNN
actor + GT-only critic), then runs one obs -> act() -> step cycle and checks
the action tensor is finite with the expected shape.

Usage:
  MUJOCO_GL=egl uv run python scripts/smoke_vision_fwd.py
"""

from __future__ import annotations

import os
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.mos92.env_cfgs import mos92_soccer_vision_env_cfg
from mjlab.tasks.velocity.config.mos92.rl_cfg import mos92_vision_ppo_runner_cfg
from mjlab.utils.torch import configure_torch_backends

NUM_ENVS = 8
DEV = "cuda:0"


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()

  env_cfg = mos92_soccer_vision_env_cfg(play=True)
  env_cfg.scene.num_envs = NUM_ENVS
  agent_cfg = mos92_vision_ppo_runner_cfg()

  env = ManagerBasedRlEnv(cfg=env_cfg, device=DEV)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  obs = env.get_observations()
  print("[obs groups]", {k: tuple(v.shape) for k, v in obs.items()})

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), "/tmp/smoke_vision_fwd", DEV)

  # Inference path: full obs TensorDict -> policy -> action.
  policy = runner.get_inference_policy(device=DEV)
  with torch.no_grad():
    act = policy(obs)
  print("[action]", tuple(act.shape), "finite=", bool(torch.isfinite(act).all()))

  # One env step with the action to confirm the full loop.
  obs2, rew, dones, extras = env.step(act)
  print(
    "[step] rew finite=",
    bool(torch.isfinite(rew).all()),
    "obs finite=",
    {k: bool(torch.isfinite(v).all()) for k, v in obs2.items()},
  )

  assert torch.isfinite(act).all(), "action has NaN/Inf"
  assert act.shape[0] == NUM_ENVS
  print("[PASS] vision env + CNN actor forward OK")
  env.close()


if __name__ == "__main__":
  main()
