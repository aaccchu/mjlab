"""Line-A (EKF self-localization) milestone evaluation + visualization.

Loads the archived EXP13 EKF checkpoint and produces, into
soccer_eval/2026-06-09_v4/lineB_kick_exp14/:
  - lineA_ekf.mp4         : rollout video (9 envs tiled)
  - frame_*.png           : preview stills
  - metrics.json          : quantitative pos_err / kick-chain / stability over a
                            longer headless rollout (more envs, more steps)

The video uses play=True (no GT-pose fade curriculum here — line A is already pure
EKF belief, no oracle pose obs to fade). Metrics mirror scripts/readout_v4_metrics.py
dimensions so this archival number is comparable to the training-log readout.

Usage: MUJOCO_GL=egl uv run python scripts/eval_v4_lineA.py
"""

import json
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
  mos92_soccer_e2e_dualcam_ekf_kick_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import (
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers.video_recorder import VideoRecorder

CKPT = Path("checkpoints/v4_soccer/lineB_kick_exp14/model_1999.pt")
OUT_DIR = Path("soccer_eval/2026-06-09_v4/lineB_kick_exp14")
VIDEO_NUM_ENVS = 9
VIDEO_LEN = 500
METRIC_NUM_ENVS = 64
METRIC_STEPS = 400
METRIC_WARMUP = 150  # let the EKF fully converge from the wide prior (matches the
# training-log steady state, not the transient first ~50 steps after reset)


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
  env_cfg = mos92_soccer_e2e_dualcam_ekf_kick_env_cfg(play=True)
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
    name_prefix="lineA_ekf",
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


def collect_metrics(device: str) -> dict:
  env_cfg = mos92_soccer_e2e_dualcam_ekf_kick_env_cfg(play=False)
  env_cfg.scene.num_envs = METRIC_NUM_ENVS
  agent_cfg = mos92_selfloc_vision_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env_w = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(env_w, asdict(agent_cfg), str(OUT_DIR), device)
  runner.load(str(CKPT), load_cfg={"actor": True}, strict=False, map_location=device)
  policy = runner.get_inference_policy(device=device)

  cmd = env.command_manager.get_term("dribble")
  # Two metric sources: reward/monitor terms write some keys into env.extras["log"]
  # every step (selfloc_pos_err_m, fell_over); the dribble command term keeps its
  # own per-env metrics dict (goal_rate, episode_success, ...). Read both.
  log_keys = ["Metrics/selfloc_pos_err_m", "Episode_Termination/fell_over"]
  cmd_keys = ["goal_rate", "episode_success", "ball_to_target_error", "ball_speed"]
  acc = {k: [] for k in log_keys + cmd_keys}
  obs, _ = env_w.reset()
  for t in range(METRIC_STEPS):
    with torch.inference_mode():
      act = policy(obs)
    obs, _, _, _ = env_w.step(act)
    if t < METRIC_WARMUP:
      continue  # let EKF settle from the wide prior before scoring
    log = env.extras.get("log", {})
    for k in log_keys:
      if k in log:
        v = log[k]
        acc[k].append(float(v.mean()) if hasattr(v, "mean") else float(v))
    for k in cmd_keys:
      if k in cmd.metrics:
        acc[k].append(float(cmd.metrics[k].mean()))

  def stats(v):
    if not v:
      return None
    v = sorted(v)
    n = len(v)
    return {"mean": sum(v) / n, "median": v[n // 2], "min": v[0], "max": v[-1], "n": n}

  result = {k: stats(acc[k]) for k in log_keys + cmd_keys}
  result["_meta"] = {
    "ckpt": str(CKPT),
    "num_envs": METRIC_NUM_ENVS,
    "steps": METRIC_STEPS,
    "warmup": METRIC_WARMUP,
  }
  env.close()
  return result


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  if not CKPT.exists():
    print(f"[ERROR] checkpoint missing: {CKPT}")
    return
  metrics = collect_metrics(device)
  (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
  print(f"[INFO] metrics -> {OUT_DIR / 'metrics.json'}")
  print(json.dumps(metrics, indent=2))
  render_video(device)


if __name__ == "__main__":
  main()
