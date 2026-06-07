"""Render demo videos for the three v3 soccer-solo deliverable models.

One clip per capability, each rebuilt with its OWN training env so the recorded
behavior matches what the policy learned:
  ① selfloc  — selfloc-vision env, GT pose faded to ~0 (pure-vision condition)
  ② findball — vision-ablation env (depth camera ball-finding)
  ③ goal     — goal env (attacking half, score into the fixed goal)

Outputs mp4 + 4 preview PNGs per model into
soccer_eval/2026-06-07_v3_soccer_solo/<key>/.

Usage:
  MUJOCO_GL=egl uv run python scripts/render_v3_soccer_solo.py
  MUJOCO_GL=egl uv run python scripts/render_v3_soccer_solo.py --only goal
"""

from __future__ import annotations

import argparse
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
  mos92_soccer_e2e_env_cfg,
  mos92_soccer_goal_env_cfg,
  mos92_soccer_selfloc_vision_env_cfg,
  mos92_soccer_vision_ablation_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import (
  mos92_ppo_runner_cfg,
  mos92_selfloc_vision_ppo_runner_cfg,
  mos92_vision_ppo_runner_cfg,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers.video_recorder import VideoRecorder

CKPT_ROOT = Path("checkpoints/v3_soccer_solo")
OUT_ROOT = Path("soccer_eval/2026-06-07_v3_soccer_solo")
RENDER_NUM_ENVS = 9
VIDEO_LEN = 500

# key -> (env builder, runner cfg, ckpt path, fade-to-pure-vision?)
MODELS = {
  "01_selfloc": (
    mos92_soccer_selfloc_vision_env_cfg,
    mos92_selfloc_vision_ppo_runner_cfg,
    CKPT_ROOT / "01_selfloc_purevision" / "model_2800.pt",
    True,  # advance step counter so GT pose obs fades -> pure vision
  ),
  "02_findball": (
    mos92_soccer_vision_ablation_env_cfg,
    mos92_vision_ppo_runner_cfg,
    CKPT_ROOT / "02_findball_depth" / "model_1499.pt",
    False,
  ),
  "03_goal": (
    mos92_soccer_goal_env_cfg,
    mos92_ppo_runner_cfg,
    CKPT_ROOT / "03_dribble_goal" / "model_1600.pt",
    False,
  ),
  "04_e2e": (
    mos92_soccer_e2e_env_cfg,
    mos92_selfloc_vision_ppo_runner_cfg,
    CKPT_ROOT / "04_e2e_integrated" / "model_1499.pt",
    True,  # pure-vision selfloc + find-ball + dribble-to-goal in one policy
  ),
}


def _extract_frames(out_dir: Path, n_frames: int = 4) -> None:
  mp4s = sorted(out_dir.glob("*.mp4"))
  if not mp4s:
    print(f"[WARN] no mp4 in {out_dir}")
    return
  video = media.read_video(str(mp4s[0]))
  total = len(video)
  idxs = [int(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)]
  for j, fi in enumerate(idxs):
    media.write_image(out_dir / f"frame_{j}.png", video[fi])


def render_one(key: str) -> None:
  builder, runner_cfg_fn, ckpt_path, fade = MODELS[key]
  if not ckpt_path.exists():
    print(f"[SKIP] {key}: checkpoint missing {ckpt_path}")
    return
  out_dir = OUT_ROOT / key
  out_dir.mkdir(parents=True, exist_ok=True)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  env_cfg = builder(play=True)
  env_cfg.scene.num_envs = RENDER_NUM_ENVS
  if "actor" in env_cfg.observations:
    env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)

  agent_cfg = runner_cfg_fn()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  # For selfloc, drive the GT-pose obs mask to ~0 so the clip shows the PURE-VISION
  # condition (play=True skips the fade curriculum, so set the counter directly).
  if fade:
    env.common_step_counter = 3600 * 24
  env = VideoRecorder(
    env,
    video_folder=out_dir,
    step_trigger=lambda step: step == 0,
    video_length=VIDEO_LEN,
    name_prefix=key,
    disable_logger=True,
  )
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(out_dir), device)
  runner.load(str(ckpt_path), load_cfg={"actor": True}, strict=False, map_location=device)
  policy = runner.get_inference_policy(device=device)

  print(f"[INFO] rendering {key} from {ckpt_path.name} (fade={fade})")
  obs = env.get_observations()
  for _ in range(VIDEO_LEN + 5):
    with torch.no_grad():
      actions = policy(obs)
    obs, _, _, _ = env.step(actions)
  env.close()
  _extract_frames(out_dir)
  print(f"[INFO] saved {key} -> {out_dir}")


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()
  ap = argparse.ArgumentParser()
  ap.add_argument("--only", choices=list(MODELS.keys()), default=None)
  args = ap.parse_args()
  keys = [args.only] if args.only else list(MODELS.keys())
  for k in keys:
    try:
      render_one(k)
    except Exception as e:  # noqa: BLE001 — render is best-effort per model.
      print(f"[ERROR] {k} render failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
  main()
