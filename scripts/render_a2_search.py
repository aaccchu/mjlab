"""Render Spike A2-MOS92 search -> approach -> kick behavior video.

Forces every episode to spawn the ball in the rear blind sector
(rear_spawn_fraction=1.0) so the clip reliably shows the robot turning its
head / body to find the ball, then walking to it and kicking — the full
search -> lock -> approach -> kick sequence the spike validates.

Usage:
  MUJOCO_GL=egl uv run python scripts/render_a2_search.py
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
from mjlab.tasks.velocity.mdp.dribble_command import DribbleCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

RUN_DIR = Path("logs/rsl_rl/mos92_velocity/2026-06-02_17-14-09_spike_a2_search")
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

  env_cfg = mos92_soccer_search_env_cfg(play=True)
  env_cfg.scene.num_envs = RENDER_NUM_ENVS
  # Clean playback + force every episode to start with the ball behind the
  # robot so the search behavior is always visible.
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)
  dribble = env_cfg.commands["dribble"]
  assert isinstance(dribble, DribbleCommandCfg)
  dribble.rear_spawn_fraction = 1.0
  dribble.ball_init_speed_range = (0.0, 0.3)  # mostly static so it stays findable

  ckpt = _latest_ckpt()
  date = datetime.now().strftime("%Y-%m-%d")
  out_dir = Path("soccer_eval") / f"{date}_spikes" / "a2_mos92_search"
  out_dir.mkdir(parents=True, exist_ok=True)

  agent_cfg = mos92_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env = VideoRecorder(
    env,
    video_folder=out_dir,
    step_trigger=lambda step: step == 0,
    video_length=VIDEO_LENGTH,
    name_prefix="a2_mos92_search",
    disable_logger=False,
  )
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  print(f"[INFO] Rendering A2 search from {ckpt.name} (rear_spawn=1.0)")
  obs = env.get_observations()
  with torch.inference_mode():
    for _ in range(VIDEO_LENGTH + 5):
      obs, _, _, _ = env.step(policy(obs))

  env.close()
  print(f"[INFO] Saved A2 search video to {out_dir}")


if __name__ == "__main__":
  main()
