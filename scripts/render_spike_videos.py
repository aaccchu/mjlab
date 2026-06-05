"""Render evaluation videos for completed spike experiments.

Rebuilds the exact training env for each spike (spawn/target distribution,
reward changes, waist std) so the recorded behavior matches what the policy
was trained on — the default registry env uses a different distribution and
would not show far-distance approach or gaze behavior.

Outputs into soccer_eval/<date>_spikes/<spike>/ for unified management.

Usage:
  uv run python scripts/render_spike_videos.py --spike a1
  uv run python scripts/render_spike_videos.py --spike a2
  uv run python scripts/render_spike_videos.py --spike f1
  uv run python scripts/render_spike_videos.py --spike f2
  uv run python scripts/render_spike_videos.py --spike all
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Allow importing sibling spike scripts when run as `python scripts/<file>.py`
# (sys.path[0] is the scripts/ dir, so the project root must be added too).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapy as media
import torch
from scripts.spike_a_gaze import make_spike_a_cfg
from scripts.spike_f_fullfield import make_spike_f_cfg

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class SpikeCfg:
  builder: Callable[[], ManagerBasedRlEnvCfg]
  desc: str
  video_length: int
  # None means resolve the latest checkpoint at runtime (training in progress).
  checkpoint: str | None = None


SPIKES: dict[str, SpikeCfg] = {
  "a1": SpikeCfg(
    checkpoint="logs/rsl_rl/g1_velocity/2026-06-01_15-54-10_spike_a/model_5498.pt",
    builder=lambda: make_spike_a_cfg("a1")[0],
    desc="gaze weight=1.0 (FAIL, fell_over 73.9%)",
    video_length=400,
  ),
  "a2": SpikeCfg(
    checkpoint=None,
    builder=lambda: make_spike_a_cfg("a2")[0],
    desc="gaze weight=0.3 (root-cause validation)",
    video_length=400,
  ),
  "f1": SpikeCfg(
    checkpoint="logs/rsl_rl/g1_velocity/2026-06-01_16-07-40_spike_f_f1/model_6998.pt",
    builder=lambda: make_spike_f_cfg("f1"),
    desc="full-field 4m (67.4% success)",
    video_length=500,
  ),
  "f2": SpikeCfg(
    checkpoint="logs/rsl_rl/g1_velocity/2026-06-01_16-57-26_spike_f_f2/model_6998.pt",
    builder=lambda: make_spike_f_cfg("f2"),
    desc="full-field 9m (18.4% success)",
    video_length=600,
  ),
}

# A small num_envs keeps rendering memory low; env 0 is the one recorded.
RENDER_NUM_ENVS = 16


def _resolve_a2_checkpoint() -> str:
  """Find the latest a2 checkpoint (training may still be running)."""
  dirs = sorted(Path("logs/rsl_rl/g1_velocity").glob("*spike_a_a2"))
  if not dirs:
    raise FileNotFoundError("No spike_a_a2 log dir found.")
  ckpts = sorted(dirs[-1].glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
  if not ckpts:
    raise FileNotFoundError(f"No checkpoint in {dirs[-1]}.")
  return str(ckpts[-1])


def render_spike(spike: str, out_root: Path) -> Path:
  info = SPIKES[spike]
  checkpoint = info.checkpoint or _resolve_a2_checkpoint()
  ckpt_path = Path(checkpoint)
  if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

  out_dir = out_root / spike
  out_dir.mkdir(parents=True, exist_ok=True)

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env_cfg = info.builder()
  env_cfg.scene.num_envs = RENDER_NUM_ENVS
  # Play-mode overrides: disable observation corruption / external pushes for
  # clean playback. auto_reset stays on so the clip runs the full length.
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)

  agent_cfg = unitree_g1_ppo_runner_cfg()

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env = VideoRecorder(
    env,
    video_folder=out_dir,
    step_trigger=lambda step: step == 0,
    video_length=info.video_length,
    name_prefix=f"spike_{spike}",
    disable_logger=False,
  )
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
  runner.load(
    str(ckpt_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  print(f"[INFO] Rendering spike {spike}: {info.desc}")
  print(f"[INFO] Checkpoint: {ckpt_path.name}")
  obs = env.get_observations()
  for _ in range(info.video_length + 5):
    with torch.no_grad():
      actions = policy(obs)
    obs, _, _, _ = env.step(actions)

  env.close()
  print(f"[INFO] Saved spike {spike} video to {out_dir}")
  _extract_frames(out_dir, n_frames=4)
  return out_dir


def _extract_frames(out_dir: Path, n_frames: int = 4) -> None:
  """Extract evenly-spaced PNG frames from the rendered mp4 for quick preview."""
  mp4s = sorted(out_dir.glob("*.mp4"))
  if not mp4s:
    print(f"[WARN] No mp4 found in {out_dir} to extract frames from.")
    return
  video = media.read_video(str(mp4s[0]))
  total = len(video)
  if total == 0:
    print(f"[WARN] Empty video: {mp4s[0]}")
    return
  idxs = [int(i * (total - 1) / max(n_frames - 1, 1)) for i in range(n_frames)]
  for j, idx in enumerate(idxs, start=1):
    media.write_image(str(out_dir / f"frame_{j:02d}.png"), video[idx])
  print(f"[INFO] Extracted {n_frames} frames to {out_dir}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--spike", choices=[*SPIKES.keys(), "all"], required=True)
  args = parser.parse_args()

  os.environ["MUJOCO_GL"] = "egl"
  configure_torch_backends()

  date = datetime.now().strftime("%Y-%m-%d")
  out_root = Path("soccer_eval") / f"{date}_spikes"
  out_root.mkdir(parents=True, exist_ok=True)

  targets = list(SPIKES.keys()) if args.spike == "all" else [args.spike]
  for spike in targets:
    render_spike(spike, out_root)

  print(f"\n[INFO] All done. Videos under: {out_root}")


if __name__ == "__main__":
  main()
