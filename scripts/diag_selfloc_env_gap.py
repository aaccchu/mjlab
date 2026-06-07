"""Minimal diagnostic: measure model_2800 self-loc error in the EXACT training env
(play=False, fade curriculum active) vs what the probe (play=True) reports.

The 4.5x gap (training-time 2.09m vs probe 9.47m for the same ckpt) must come
from a run-condition difference, since est/gt use identical normalization. This
builds the play=False env, advances common_step_counter past the fade end so the
GT mask is ~0 (matching the ckpt's training condition), loads model_2800, and
measures mean |est-gt| in meters. One-shot, no probe machinery.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.mos92.env_cfgs import (
  mos92_soccer_selfloc_vision_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import (
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.utils.torch import configure_torch_backends

RUN = Path("logs/rsl_rl/mos92_velocity/2026-06-07_01-08-00_spike_v3g_selfloc_temporal")
ROBOT = SceneEntityCfg("robot")


def main():
  configure_torch_backends()
  device = "cuda:0"
  # play=False keeps the fade curriculum so the GT obs term gets masked just like
  # training. num_envs modest for speed.
  env_cfg = mos92_soccer_selfloc_vision_env_cfg(play=False)
  env_cfg.scene.num_envs = 256
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  agent_cfg = mos92_selfloc_vision_ppo_runner_cfg()
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  uenv = env.unwrapped

  # Force the GT mask to ~0 like the ckpt's iter (2800, fade [1200,3500] => the
  # curriculum needs common_step_counter ~ 2800*24). Set it directly.
  uenv.common_step_counter = 2800 * 24

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), None, device)
  ckpt = next(p for p in RUN.glob("model_2800.pt"))
  runner.load(str(ckpt))
  policy = runner.get_inference_policy(device=device)
  sl = uenv.action_manager.get_term("selfloc")
  cmd = uenv.command_manager.get_term("dribble")

  errs = []
  with torch.no_grad():
    env.reset()
    obs = env.get_observations()
    for step in range(400):
      action = policy(obs)
      obs, _, _, _ = env.step(action)
      if step >= 150:
        est = sl.raw_action
        gt = mdp.robot_field_pose(uenv, "dribble", ROBOT)
        dx = (est[:, 0] - gt[:, 0]) * cmd.cfg.half_length
        dy = (est[:, 1] - gt[:, 1]) * cmd.cfg.half_width
        errs.append(torch.sqrt(dx**2 + dy**2).mean().item())
        # also report current mask factor of the GT term
  env.close()
  mask = uenv.observation_manager.get_term_cfg("actor", "ball_to_target").scale
  print(f"[DIAG] GT-term mask scale = {mask}")
  print(f"[DIAG] play=False env, model_2800, mean selfloc_pos_err_m = "
        f"{sum(errs) / len(errs):.3f} m  (over {len(errs)} steps)")


if __name__ == "__main__":
  main()
