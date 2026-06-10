"""EXP19 (representative deployment model) milestone video + stills.

Loads the archived EXP19 checkpoint (e2e EKF kick + out-of-bounds fix + strong
soft-boundary shaping) and renders a rollout demo into
soccer_eval/2026-06-09_v4/kick_oob_exp19/:
  - kick_oob_exp19-step-0.mp4 : rollout video (9 envs tiled)
  - frame_*.png               : preview stills sampled across the rollout

Uses play=True (no GT-pose fade curriculum — the e2e EKF policy is pure belief,
no oracle pose obs to fade). The quantitative metrics.json in this directory is
the EXP16-19 multi-metric comparison (computed from the training-log steady
state, four runs computed identically); this script only (re)generates the
visual demo, so it does NOT overwrite metrics.json.

Usage: MUJOCO_GL=egl uv run python scripts/eval_v4_kick_oob_exp19.py
"""

import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapy as media
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.mos92.env_cfgs import (
  mos92_soccer_e2e_dualcam_ekf_kick_oob_soft2_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import (
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers.video_recorder import VideoRecorder

CKPT = Path("checkpoints/v4_soccer/kick_oob_exp19/model_1999.pt")
OUT_DIR = Path("soccer_eval/2026-06-09_v4/kick_oob_exp19")
NAME_PREFIX = "kick_oob_exp19"
VIDEO_NUM_ENVS = 9
VIDEO_LEN = 500


def _extract_frames(out_dir: Path) -> None:
  mp4s = sorted(out_dir.glob("*.mp4"))
  if not mp4s:
    return
  video = media.read_video(mp4s[0])
  n = len(video)
  idxs = [0, n // 3, 2 * n // 3, n - 1]
  for j, fi in enumerate(idxs):
    media.write_image(out_dir / f"frame_{j}.png", video[fi])


def render_video(device: str) -> None:
  env_cfg = mos92_soccer_e2e_dualcam_ekf_kick_oob_soft2_env_cfg(play=True)
  env_cfg.scene.num_envs = VIDEO_NUM_ENVS
  if "actor" in env_cfg.observations:
    env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)
  agent_cfg = mos92_selfloc_vision_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env = VideoRecorder(
    env,
    video_folder=OUT_DIR,
    step_trigger=lambda step: step == 0,
    video_length=VIDEO_LEN,
    name_prefix=NAME_PREFIX,
    disable_logger=True,
  )
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(OUT_DIR), device)
  runner.load(str(CKPT), load_cfg={"actor": True}, strict=False, map_location=device)
  policy = runner.get_inference_policy(device=device)
  print(f"[INFO] rendering video from {CKPT.name}")
  obs = env.get_observations()
  for _ in range(VIDEO_LEN + 5):
    with torch.no_grad():
      actions = policy(obs)
    obs, _, _, _ = env.step(actions)
  env.close()
  _extract_frames(OUT_DIR)
  print(f"[INFO] video + frames saved -> {OUT_DIR}")


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  if not CKPT.exists():
    print(f"[ERROR] checkpoint missing: {CKPT}")
    return
  render_video(device)


if __name__ == "__main__":
  main()
