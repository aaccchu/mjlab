"""Render Spike v3b-MOS92 (GT-ablation) in its NATIVE camera-only condition.

v3b was trained with the actor's GT ball obs masked to zero (iters 1500-3000),
so its in-distribution input is GT=0 and the ball bearing must come from the
head depth camera -> CNN. This render zeroes the same GT slices each step (as
the ablation probe does) so the video shows the policy's TRUE camera-only
search -> approach -> dribble behavior, not the OOD GT-intact condition.

Usage:
  MUJOCO_GL=egl uv run python scripts/render_v3b_ablation.py
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
from mjlab.tasks.velocity.mdp.dribble_command import DribbleCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

RUN_DIR = Path("logs/rsl_rl/mos92_velocity/2026-06-05_13-40-53_spike_v3b_ablation")
if len(sys.argv) > 1:
  RUN_DIR = Path(sys.argv[1])
RENDER_NUM_ENVS = 16
VIDEO_LENGTH = 500
GT_BALL_TERMS = ("robot_to_ball", "ball_velocity", "ball_gaze_uv")


def _latest_ckpt() -> Path:
  ckpts = sorted(RUN_DIR.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
  if not ckpts:
    raise FileNotFoundError(f"No checkpoint in {RUN_DIR}")
  return ckpts[-1]


def _gt_slices(obs_mgr) -> list[tuple[int, int]]:
  names = obs_mgr.active_terms["actor"]
  dims = obs_mgr.group_obs_term_dim["actor"]
  slices, offset = [], 0
  for name, shape in zip(names, dims, strict=False):
    width = int(torch.tensor(shape).prod().item()) if len(shape) else 1
    if name in GT_BALL_TERMS:
      slices.append((offset, offset + width))
    offset += width
  return slices


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  # play=True: no curriculum, so GT terms are NOT auto-masked — we zero them
  # manually each step to reproduce the trained camera-only condition.
  env_cfg = mos92_soccer_vision_ablation_env_cfg(play=True)
  env_cfg.scene.num_envs = RENDER_NUM_ENVS
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)
  dribble = env_cfg.commands["dribble"]
  assert isinstance(dribble, DribbleCommandCfg)
  dribble.rear_spawn_fraction = 1.0  # force search behavior into view
  dribble.ball_init_speed_range = (0.0, 0.3)

  date = datetime.now().strftime("%Y-%m-%d")
  out_dir = Path("soccer_eval") / f"{date}_spikes" / "v3b_gt_ablation"
  out_dir.mkdir(parents=True, exist_ok=True)

  agent_cfg = mos92_vision_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env = VideoRecorder(
    env,
    video_folder=out_dir,
    step_trigger=lambda step: step == 0,
    video_length=VIDEO_LENGTH,
    name_prefix="v3b_camera_only",
    disable_logger=False,
  )
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  ckpt = _latest_ckpt()
  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  slices = _gt_slices(env.unwrapped.observation_manager)

  print(f"[INFO] Rendering v3b camera-only from {ckpt.name} (GT zeroed each step)")
  obs = env.get_observations()
  with torch.inference_mode():
    for _ in range(VIDEO_LENGTH + 5):
      a = obs["actor"]
      for s, e in slices:
        a[:, s:e] = 0.0
      obs, _, _, _ = env.step(policy(obs))

  env.close()
  print(f"[INFO] Saved v3b camera-only video to {out_dir}")


if __name__ == "__main__":
  main()
