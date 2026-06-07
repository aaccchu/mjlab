"""Experiment 5 GATE: does a downward-tilted / wider-FOV localization camera see
MORE field markings than the current forward-looking head cam?

The pure-vision self-loc failures (exp 3 single-frame, exp 4 frame-stack) trace
to a visibility root cause: the forward head cam (pitch 0, fovy 60) sees field
lines in only ~2/5 poses (3/5 had <0.5% marking pixels). Before spending another
training run, prove a better camera geometry exists. This renders RGB from each
candidate camera config at the TRAINING resolution (64x48) across the same known
poses and reports field-marking pixel fraction. No training until a config beats
the forward baseline at the LOW poses.

Usage: MUJOCO_GL=egl uv run python scripts/probe_loc_camera_gate.py
"""

from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path

import numpy as np

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.mos92.env_cfgs import mos92_soccer_vision_env_cfg

OUT = Path("soccer_eval/2026-06-06_spikes/v3g_loc_cam_gate")
POSES = [
  ("midfield_face_x", 0.0, 0.0, 0.0),
  ("own_half_face_goal", -6.0, 0.0, 0.0),
  ("corner_face_center", -8.0, 5.0, -0.5),
  ("near_opp_goal", 7.0, 0.0, 0.0),
  ("center_circle_edge", 2.0, 0.0, 0.3),
]
RES = (64, 48)  # training resolution.


def _quat_tilt_down(deg: float) -> tuple[float, float, float, float]:
  """Forward head-cam quat (w,x,y,z)=(.707,0,-.707,0) rotated `deg` downward
  about the body y-axis. Returns (w,x,y,z)."""
  base = np.array([0.70710678, 0.0, -0.70710678, 0.0])  # w,x,y,z
  half = math.radians(deg) / 2.0
  rot = np.array([math.cos(half), 0.0, math.sin(half), 0.0])  # about y
  w0, x0, y0, z0 = base
  w1, x1, y1, z1 = rot
  return (
    w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
    w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
    w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
    w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
  )


# Candidate camera configs: (label, tilt_deg, fovy).
CANDIDATES = [
  ("forward_60 (baseline)", 0.0, 60.0),
  ("down15_60", 15.0, 60.0),
  ("down30_60", 30.0, 60.0),
  ("down30_90", 30.0, 90.0),
  ("down45_90", 45.0, 90.0),
]


def _set_robot_pose(env, x, y, yaw):
  robot = env.scene["robot"]
  origin = env.scene.env_origins
  qpos = robot.data.root_link_pos_w.new_zeros((env.num_envs, 7))
  qpos[:, 0] = origin[:, 0] + x
  qpos[:, 1] = origin[:, 1] + y
  qpos[:, 2] = 0.45
  half = yaw / 2.0
  qpos[:, 3] = float(np.cos(half))
  qpos[:, 6] = float(np.sin(half))
  robot.write_root_link_pose_to_sim(qpos)
  env.sim.forward()
  env.sim.sense()


def _marking_fraction(rgb: np.ndarray) -> float:
  """Fraction of pixels that are bright field markings (not pitch-green)."""
  r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
  green_dom = (g > r + 15) & (g > b + 15)
  bright = rgb.mean(-1) > 120
  marking = bright & (~green_dom)
  return float(marking.mean())


def _eval_candidate(label, tilt, fovy, save_dir):
  """Build env with the head cam at (tilt, fovy), render each pose, return stats."""
  cfg = mos92_soccer_vision_env_cfg(play=True)
  cfg.scene.num_envs = 1
  cam = next(s for s in cfg.scene.sensors if getattr(s, "name", "") == "head_cam")
  cam = dataclasses.replace(
    cam,
    data_types=("rgb", "depth"),
    width=RES[0],
    height=RES[1],
    fovy=fovy,
    quat=_quat_tilt_down(tilt),
  )
  cfg.scene.sensors = tuple(
    cam if getattr(s, "name", "") == "head_cam" else s for s in cfg.scene.sensors
  )
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  env.reset()
  sensor = env.scene["head_cam"]
  stats = []
  try:
    from PIL import Image

    have_pil = True
  except Exception:
    have_pil = False
  for name, x, y, yaw in POSES:
    _set_robot_pose(env, x, y, yaw)
    for _ in range(3):
      env.sim.forward()
    sensor.update(0.0)
    rgb = sensor.data.rgb[0].cpu().numpy()
    stats.append((name, _marking_fraction(rgb)))
    if have_pil:
      safe = label.split()[0]
      Image.fromarray(rgb).resize((RES[0] * 4, RES[1] * 4), Image.NEAREST).save(
        save_dir / f"{safe}_{name}.png"
      )
  env.close()
  return stats


def main():
  os.environ.setdefault("MUJOCO_GL", "egl")
  OUT.mkdir(parents=True, exist_ok=True)
  print(f"[INFO] localization-camera GATE @ {RES[0]}x{RES[1]} -> {OUT}\n")
  all_stats = {}
  for label, tilt, fovy in CANDIDATES:
    stats = _eval_candidate(label, tilt, fovy, OUT)
    all_stats[label] = stats
    fracs = [f for _, f in stats]
    n_ok = sum(1 for f in fracs if f > 0.02)
    print(
      f"=== {label:24s} tilt={tilt:>4.0f} fovy={fovy:>4.0f} : "
      f"mean={np.mean(fracs) * 100:5.2f}%  OK={n_ok}/{len(fracs)} ==="
    )
    for name, f in stats:
      flag = "OK " if f > 0.02 else "LOW"
      print(f"    [{flag}] {name:22s} {f * 100:5.2f}%")
  # Verdict: which candidate maximizes the WORST-pose visibility (the bottleneck).
  print("\n[GATE] worst-pose marking fraction per candidate (higher = better):")
  best = None
  for label, stats in all_stats.items():
    worst = min(f for _, f in stats)
    print(f"    {label:24s} worst={worst * 100:5.2f}%")
    if best is None or worst > best[1]:
      best = (label, worst)
  print(f"\n[GATE] best worst-pose: {best[0]} ({best[1] * 100:.2f}%).")
  print(f"[GATE] inspect images in {OUT}. Train a loc-cam only if it beats forward.")


if __name__ == "__main__":
  main()
