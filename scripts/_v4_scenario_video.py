"""Shared helper: render representative SCENARIO demo clips for a v4 soccer policy.

The VideoRecorder wrapper only films env[0], and the env's default camera already
tracks the robot's base_link (ASSET_BODY, distance 3.0) — good for a behavior
demo. Rather than dump one long undifferentiated rollout, this module runs env[0]
for many episodes (play=False, so it resets on goal/fall and produces variety),
records every frame AND a per-step trace, then auto-segments the trace into the
three phases of the core dribble/kick arc — each of which occurs EVERY episode:

  - approach : robot walking up to a (not-yet-moving) ball.
  - strike   : the kick itself — the fastest-ball moment.
  - goalward : ball driven deep toward the goal mouth (the shot outcome).

We deliberately do NOT force a "goal" or "out-of-bounds" clip: in a single-env
rollout this in-bounds policy scores only ~1/3 of episodes and essentially never
leaves the field (root stays >1 m inside the edge), so picking those would either
fail or misrepresent typical behavior. If an actual goal happens to occur, the
'goalward' clip is centered on it and flagged ``scored: true``.

Each picked clip is written as <prefix>_<scenario>.mp4 plus one mid-clip still
<prefix>_<scenario>.png. Picked windows are returned for the README / scenarios.json.

Used by scripts/eval_v4_kick_oob_exp19.py and scripts/eval_v4_kick_exp16.py.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import mediapy as media
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.mos92.rl_cfg import (
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.utils.torch import configure_torch_backends

_GOAL_HALF_WIDTH = 1.0  # goal mouth opening |y| < 1.0 m.

ROLLOUT_STEPS = 4000  # many 20 s episodes (reset on goal/fall) -> arc variety.
CLIP_HALF = 75  # +/- steps around an event -> ~3 s clips at 50 fps.
FPS = 50


def _clip_bounds(center: int, n: int) -> tuple[int, int]:
  return max(0, center - CLIP_HALF), min(n, center + CLIP_HALF)


def render_scenarios(
  env_cfg_fn: Callable[..., object],
  ckpt: Path,
  out_dir: Path,
  prefix: str,
  device: str,
) -> dict:
  """Run env[0] for many episodes, write the three dribble-arc clips."""
  # play=False gives normal 20 s episodes that reset on goal/fall (the variety we
  # segment on), but its GT-pose fade curriculum starts mid-fade. Force it to the
  # fully-deployed end state (pure EKF belief, no oracle pose) exactly as the
  # training spike scripts do, and turn off corruption/push for a clean demo.
  env_cfg = env_cfg_fn(play=False)
  env_cfg.scene.num_envs = 1
  if "actor" in env_cfg.observations:
    env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)
  if "gt_mask" in env_cfg.curriculum:
    env_cfg.curriculum["gt_mask"].params["start_step"] = -1
    env_cfg.curriculum["gt_mask"].params["end_step"] = 0
  if "penalty_ramp" in env_cfg.curriculum:
    env_cfg.curriculum["penalty_ramp"].params["start_step"] = -1
    env_cfg.curriculum["penalty_ramp"].params["end_step"] = 0
  # Bias every episode toward the goal mouth so the goalward clip is a real shot.
  dribble_cmd_cfg = env_cfg.commands.get("dribble")
  if dribble_cmd_cfg is not None and hasattr(dribble_cmd_cfg, "goal_target_fraction"):
    dribble_cmd_cfg.goal_target_fraction = 1.0

  agent_cfg = mos92_selfloc_vision_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env_w = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(env_w, asdict(agent_cfg), str(out_dir), device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=False, map_location=device)
  policy = runner.get_inference_policy(device=device)

  cmd = env.command_manager.get_term("dribble")
  origin_xy = env.scene.env_origins[0, :2]

  frames: list[np.ndarray] = []
  approach_score = np.full(ROLLOUT_STEPS, -1e9)  # robot closing on a slow ball
  strike_score = np.full(ROLLOUT_STEPS, -1e9)  # the kick: peak ball speed
  goalward_score = np.full(ROLLOUT_STEPS, -1e9)  # ball deep toward the mouth
  scored_flag = np.zeros(ROLLOUT_STEPS, dtype=bool)

  print(f"[INFO] rolling out env0 from {ckpt.name} for {ROLLOUT_STEPS} steps")
  obs = env_w.get_observations()
  prev_scored = 0.0
  for t in range(ROLLOUT_STEPS):
    with torch.no_grad():
      actions = policy(obs)
    obs, _, _, _ = env_w.step(actions)
    frame = env.render()
    if isinstance(frame, np.ndarray) and frame.ndim == 4:
      frame = frame[0]
    frames.append(np.asarray(frame))

    gs = float(cmd.goal_scored[0].item())
    scored_flag[t] = gs > prev_scored
    prev_scored = gs

    r2b = float(cmd.metrics["robot_to_ball_error"][0].item())
    bspeed = float(cmd.metrics["ball_speed"][0].item())
    ball_local = cmd.ball_pos_w[0, :2] - origin_xy
    bx = float(ball_local[0].item())
    by = float(ball_local[1].item())

    approach_score[t] = -r2b - bspeed  # near ball, ball still
    strike_score[t] = bspeed  # fastest ball = the kick
    goalward_score[t] = bx - 2.0 * max(0.0, abs(by) - _GOAL_HALF_WIDTH)

  env_w.close()
  n = len(frames)

  picked: list[dict] = []
  used_centers: list[int] = []

  def _far_enough(c: int) -> bool:
    return all(abs(c - u) > CLIP_HALF for u in used_centers)

  def _pick(score: np.ndarray, scenario: str) -> None:
    for c in np.argsort(score)[::-1]:
      c = int(c)
      if score[c] <= -1e8:
        return
      if not _far_enough(c):
        continue
      lo, hi = _clip_bounds(c, n)
      clip = frames[lo:hi]
      if not clip:
        return
      media.write_video(str(out_dir / f"{prefix}_{scenario}.mp4"), clip, fps=FPS)
      media.write_image(out_dir / f"{prefix}_{scenario}.png", clip[len(clip) // 2])
      info = {"scenario": scenario, "center_step": c, "window": [lo, hi]}
      if scenario == "goalward":
        info["scored"] = bool(scored_flag[lo:hi].any())
      picked.append(info)
      used_centers.append(c)
      return

  # Order matters: goalward first (most specific), then strike, then approach,
  # so the de-overlap keeps the clearest example of each phase.
  _pick(goalward_score, "goalward")
  _pick(strike_score, "strike")
  _pick(approach_score, "approach")

  result = {
    "ckpt": str(ckpt),
    "rollout_steps": n,
    "fps": FPS,
    "clip_len_steps": 2 * CLIP_HALF,
    "scenarios": picked,
    "num_goals_in_rollout": int(scored_flag.sum()),
    "max_ball_speed": round(float(strike_score.max()), 3),
  }
  print(
    f"[INFO] wrote {len(picked)} scenario clips -> {out_dir} "
    f"(goals in rollout: {int(scored_flag.sum())})"
  )
  return result


def run(env_cfg_fn, ckpt: Path, out_dir: Path, prefix: str) -> dict:
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()
  out_dir.mkdir(parents=True, exist_ok=True)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  if not ckpt.exists():
    raise FileNotFoundError(f"checkpoint missing: {ckpt}")
  return render_scenarios(env_cfg_fn, ckpt, out_dir, prefix, device)
