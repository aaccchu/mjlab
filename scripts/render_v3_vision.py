"""Render Spike v3-MOS92 gaze-warmup (vision) search -> approach -> kick video.

Same scenario as the A2 render (ball forced into the rear blind sector so the
search behavior is visible), but the policy is the vision actor that also reads
the head depth camera through a CNN. Confirms the gait + dribble + head-turning
survive with the camera in the loop.

Usage:
  MUJOCO_GL=egl uv run python scripts/render_v3_vision.py
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
from mjlab.tasks.velocity.mdp.dribble_command import DribbleCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

RUN_DIR = Path("logs/rsl_rl/mos92_velocity/2026-06-02_20-19-43_spike_v3_vision")
RENDER_NUM_ENVS = 16
VIDEO_LENGTH = 500


def _latest_ckpt() -> Path:
  ckpts = sorted(RUN_DIR.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
  if not ckpts:
    raise FileNotFoundError(f"No checkpoint in {RUN_DIR}")
  return ckpts[-1]


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  env_cfg = mos92_soccer_vision_env_cfg(play=True)
  env_cfg.scene.num_envs = RENDER_NUM_ENVS
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)
  dribble = env_cfg.commands["dribble"]
  assert isinstance(dribble, DribbleCommandCfg)
  dribble.rear_spawn_fraction = 1.0
  dribble.ball_init_speed_range = (0.0, 0.3)

  ckpt = _latest_ckpt()
  date = datetime.now().strftime("%Y-%m-%d")
  out_dir = Path("soccer_eval") / f"{date}_spikes" / "v3_vision_mos92"
  out_dir.mkdir(parents=True, exist_ok=True)

  agent_cfg = mos92_vision_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env = VideoRecorder(
    env,
    video_folder=out_dir,
    step_trigger=lambda step: step == 0,
    video_length=VIDEO_LENGTH,
    name_prefix="v3_vision_mos92",
    disable_logger=False,
  )
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  print(f"[INFO] Rendering v3 vision from {ckpt.name} (rear_spawn=1.0)")
  obs = env.get_observations()
  with torch.inference_mode():
    for _ in range(VIDEO_LENGTH + 5):
      obs, _, _, _ = env.step(policy(obs))

  env.close()
  print(f"[INFO] Saved v3 vision video to {out_dir}")


if __name__ == "__main__":
  main()
