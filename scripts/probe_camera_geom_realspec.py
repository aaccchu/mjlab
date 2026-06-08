"""Camera-geometry GATE at the REAL-SPEC 0.125 m line width.

Two cheap-test results constrain the search: (1) the resolution sweep showed more
pixels DON'T help (worst-pose stuck ~0.36% even at 512x384) — so the bottleneck
is angular, not pixel count; (2) the old exp5 camera gate only swept FOV WIDER
(60->90), which spreads the same distant lines over more sky/grass. The worst
poses are all DISTANT-line poses, so the untested lever is a NARROW (telephoto)
FOV that magnifies distant lines — each line subtends more pixels. This sweeps
downward tilt x narrow FOV at 0.125 m and reports worst-pose marking fraction.
A config with worst-pose >2% means real-spec self-loc is viable with NO line-
width sim-to-real gap — only a camera-intrinsics change (a real, buildable knob).

Usage: MUJOCO_GL=egl uv run python scripts/probe_camera_geom_realspec.py
"""

from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path

import numpy as np

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.mos92.env_cfgs import mos92_soccer_vision_env_cfg
from mjlab.terrains.soccer_field import SoccerFieldCfg, build_soccer_field

OUT = Path("soccer_eval/2026-06-07_v3_soccer_solo/camera_geom_realspec")
LINE_WIDTH = 0.125  # REAL SPEC — fixed.
RES = (128, 96)  # modest bump from 64x48; resolution alone proven insufficient.
POSES = [
  ("midfield_face_x", 0.0, 0.0, 0.0),
  ("own_half_face_goal", -6.0, 0.0, 0.0),
  ("corner_face_center", -8.0, 5.0, -0.5),
  ("near_opp_goal", 7.0, 0.0, 0.0),
  ("center_circle_edge", 2.0, 0.0, 0.3),
]
# (label, tilt_deg, fovy) — narrow FOV (telephoto) is the new lever.
CANDIDATES = [
  ("fwd_60_baseline", 0.0, 60.0),
  ("down20_40", 20.0, 40.0),
  ("down20_30", 20.0, 30.0),
  ("down30_40", 30.0, 40.0),
  ("down30_25", 30.0, 25.0),
  ("down10_25", 10.0, 25.0),
]


def _quat_tilt_down(deg: float):
  base = np.array([0.70710678, 0.0, -0.70710678, 0.0])  # w,x,y,z
  half = math.radians(deg) / 2.0
  rot = np.array([math.cos(half), 0.0, math.sin(half), 0.0])
  w0, x0, y0, z0 = base
  w1, x1, y1, z1 = rot
  return (
    w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
    w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
    w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
    w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
  )


def _marking_fraction(rgb: np.ndarray) -> float:
  r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
  green_dom = (g > r + 15) & (g > b + 15)
  bright = rgb.mean(-1) > 120
  return float((bright & (~green_dom)).mean())


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


def _eval(label, tilt, fovy, save_dir):
  cfg = mos92_soccer_vision_env_cfg(play=True)
  cfg.scene.num_envs = 1
  field_cfg = SoccerFieldCfg(line_width=LINE_WIDTH)
  cfg.scene.spec_fn = lambda spec: build_soccer_field(spec, field_cfg)
  cam = next(s for s in cfg.scene.sensors if getattr(s, "name", "") == "head_cam")
  cam = dataclasses.replace(
    cam, data_types=("rgb", "depth"), width=RES[0], height=RES[1],
    fovy=fovy, quat=_quat_tilt_down(tilt),
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
      Image.fromarray(rgb).resize((RES[0] * 3, RES[1] * 3), Image.NEAREST).save(
        save_dir / f"{label}_{name}.png"
      )
  env.close()
  return stats


def main():
  os.environ.setdefault("MUJOCO_GL", "egl")
  OUT.mkdir(parents=True, exist_ok=True)
  print(f"[INFO] camera-geom GATE @ lw={LINE_WIDTH}m (REAL SPEC) {RES[0]}x{RES[1]} -> {OUT}\n")
  all_stats = {}
  for label, tilt, fovy in CANDIDATES:
    stats = _eval(label, tilt, fovy, OUT)
    all_stats[label] = stats
    fracs = [f for _, f in stats]
    n_ok = sum(1 for f in fracs if f > 0.02)
    print(
      f"=== {label:18s} tilt={tilt:>4.0f} fovy={fovy:>4.0f} : "
      f"mean={np.mean(fracs) * 100:5.2f}%  OK={n_ok}/{len(fracs)} ==="
    )
    for name, f in stats:
      print(f"    [{'OK ' if f > 0.02 else 'LOW'}] {name:22s} {f * 100:5.2f}%")
  print("\n[GATE] worst-pose marking fraction per candidate (the bottleneck):")
  best = None
  for label, stats in all_stats.items():
    worst = min(f for _, f in stats)
    print(f"    {label:18s} worst={worst * 100:5.2f}%  {'PASS' if worst > 0.02 else 'fail'}")
    if best is None or worst > best[1]:
      best = (label, worst)
  print(f"\n[GATE] best worst-pose: {best[0]} ({best[1] * 100:.2f}%). "
        "Retrain real-spec self-loc here if it clears 2% — no line-width gap, only camera intrinsics.")


if __name__ == "__main__":
  main()
