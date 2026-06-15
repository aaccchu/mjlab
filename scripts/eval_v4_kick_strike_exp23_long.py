"""EXP23 (strike, current-best) — ONE long continuous demo video.

Unlike eval_v4_kick_strike_exp23.py (which segments the rollout into three short
~3 s scenario clips), this dumps the ENTIRE multi-episode rollout as a single
unbroken mp4 — as long as you ask for. The env runs play=False so it resets on
goal/fall and keeps producing fresh dribble/kick arcs; every frame is filmed by
the base_link-tracking camera, so the whole thing reads as one continuous reel
of the robot repeatedly approaching, kicking, and shooting.

A lightweight on-frame HUD overlay (step, ball speed, robot->ball, cumulative
goals) is burned in so the viewer can follow what is happening. A goal banner
flashes for ~0.5 s on each scored frame.

  - kick_strike_exp23_long.mp4   : the full reel
  - long.json                    : rollout stats (steps, goals, fps, duration)

Length knobs (env vars):
  EXP23_LONG_STEPS  total env steps to roll (default 9000 = 180 s @ 50 fps)
  EXP23_LONG_FPS    playback fps (default 50, matches the 50 Hz control rate)

Usage:
  MUJOCO_GL=egl EXP23_LONG_STEPS=12000 uv run python \
    scripts/eval_v4_kick_strike_exp23_long.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapy as media  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.velocity.config.mos92.env_cfgs import (  # noqa: E402
  mos92_soccer_e2e_dualcam_ekf_kick_strike_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import (  # noqa: E402
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

CKPT = Path("checkpoints/v4_soccer/kick_strike_exp23/model_1999.pt")
OUT_DIR = Path("soccer_eval/2026-06-11_v4_showdown/kick_strike_exp23")
PREFIX = "kick_strike_exp23"

STEPS = int(os.environ.get("EXP23_LONG_STEPS", "9000"))
FPS = int(os.environ.get("EXP23_LONG_FPS", "50"))
GOAL_FLASH = 25  # frames (~0.5 s) to hold the GOAL banner.


def _hud(frame: np.ndarray, lines: list[str], color=(255, 255, 0)) -> np.ndarray:
  """Burn small text into the top-left corner without extra deps (block font)."""
  # Minimal 5x7 bitmap font would be heavy; instead use a translucent banner with
  # numpy-drawn ticks is overkill — keep it dependency-free via PIL if available.
  try:
    from PIL import Image, ImageDraw  # noqa: E402

    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, 230, 14 * len(lines) + 6], fill=(0, 0, 0, 120))
    for i, ln in enumerate(lines):
      d.text((4, 3 + 14 * i), ln, fill=color)
    return np.asarray(img)
  except Exception:
    return frame


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  if not CKPT.exists():
    raise FileNotFoundError(f"checkpoint missing: {CKPT}")

  # Match the scenario/POV scripts: deployed end-state (pure EKF belief, no oracle
  # pose), no corruption/push, targets biased to the goal mouth for real shots.
  env_cfg = mos92_soccer_e2e_dualcam_ekf_kick_strike_env_cfg(play=False)
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
  cmd_cfg = env_cfg.commands.get("dribble")
  if cmd_cfg is not None and hasattr(cmd_cfg, "goal_target_fraction"):
    cmd_cfg.goal_target_fraction = 1.0

  agent_cfg = mos92_selfloc_vision_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env_w = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(env_w, asdict(agent_cfg), str(OUT_DIR), device)
  runner.load(str(CKPT), load_cfg={"actor": True}, strict=False, map_location=device)
  policy = runner.get_inference_policy(device=device)

  cmd = env.command_manager.get_term("dribble")

  print(f"[INFO] rolling {STEPS} steps ({STEPS / FPS:.0f}s @ {FPS}fps) from {CKPT.name}")
  frames: list[np.ndarray] = []
  obs = env_w.get_observations()
  prev_scored = 0.0
  goals = 0
  last_goal_step = -10_000
  for t in range(STEPS):
    with torch.no_grad():
      actions = policy(obs)
    obs, _, _, _ = env_w.step(actions)
    frame = env.render()
    if isinstance(frame, np.ndarray) and frame.ndim == 4:
      frame = frame[0]
    frame = np.asarray(frame)

    gs = float(cmd.goal_scored[0].item())
    if gs > prev_scored:
      goals += 1
      last_goal_step = t
    prev_scored = gs

    bspeed = float(cmd.metrics["ball_speed"][0].item())
    r2b = float(cmd.metrics["robot_to_ball_error"][0].item())
    hud = [
      f"EXP23  step {t}/{STEPS}",
      f"ball {bspeed:4.2f} m/s   r2b {r2b:4.2f} m",
      f"goals {goals}",
    ]
    frame = _hud(frame, hud)
    if 0 <= (t - last_goal_step) < GOAL_FLASH:
      frame = _hud(frame, ["", "", "", "  GOAL!"], color=(255, 60, 60))
    frames.append(frame)

    if (t + 1) % 1000 == 0:
      print(f"[INFO] {t + 1}/{STEPS} steps, goals so far: {goals}")

  env_w.close()

  out_path = OUT_DIR / f"{PREFIX}_long.mp4"
  media.write_video(str(out_path), frames, fps=FPS)
  result = {
    "ckpt": str(CKPT),
    "video": str(out_path),
    "steps": len(frames),
    "fps": FPS,
    "duration_s": round(len(frames) / FPS, 1),
    "goals_in_rollout": goals,
  }
  (OUT_DIR / "long.json").write_text(json.dumps(result, indent=2))
  print(f"[INFO] wrote long reel -> {out_path} "
        f"({result['duration_s']}s, {goals} goals)")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
