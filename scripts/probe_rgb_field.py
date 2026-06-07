"""Spike S1 (v3g GATE): can the head camera SEE the field in RGB?

Before any self-localization training, prove the signal exists — the field
lines/center-circle/goals are flat painted geoms (z~0, invisible to depth) but
SHOULD render in RGB. This places a standing robot at several known field poses,
renders RGB from the head cam at a few resolutions, saves images, and reports
pixel statistics (how many non-pitch-green pixels = visible field markings).

GATE: if 64x48 RGB cannot resolve lines/goals (too few marking pixels), we bump
resolution or add a wider localization camera BEFORE training. No training until
this passes.

Usage: MUJOCO_GL=egl uv run python scripts/probe_rgb_field.py
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.mos92.env_cfgs import mos92_soccer_vision_env_cfg

OUT = Path("soccer_eval/2026-06-06_spikes/v3g_rgb_probe")
# Known field poses to render from (env-local xy in meters, heading rad).
# Field is x in [-11,11], y in [-7,7]; goals at x=+-11. Pick spots a robot must
# localize from: near own goal looking out, midfield, near opponent goal.
POSES = [
  ("midfield_face_x", 0.0, 0.0, 0.0),
  ("own_half_face_goal", -6.0, 0.0, 0.0),
  ("corner_face_center", -8.0, 5.0, -0.5),
  ("near_opp_goal", 7.0, 0.0, 0.0),
  ("center_circle_edge", 2.0, 0.0, 0.3),
]
RESOLUTIONS = [(64, 48), (96, 72), (128, 96)]


def _set_robot_pose(env, x, y, yaw):
  """Teleport the robot freejoint to env-local (x,y) with given heading, then
  re-run sim.forward() + sim.sense() so the camera actually re-renders the new
  view (sense() is the render trigger, NOT forward() alone)."""
  robot = env.scene["robot"]
  origin = env.scene.env_origins  # (N,3)
  qpos = robot.data.root_link_pos_w.new_zeros((env.num_envs, 7))
  qpos[:, 0] = origin[:, 0] + x
  qpos[:, 1] = origin[:, 1] + y
  qpos[:, 2] = 0.45
  half = yaw / 2.0
  qpos[:, 3] = float(np.cos(half))  # w
  qpos[:, 6] = float(np.sin(half))  # z
  robot.write_root_link_pose_to_sim(qpos)
  env.sim.forward()
  env.sim.sense()


def _render_at(res, save_dir):
  """Build env at resolution `res`, render RGB from each pose, save + stats."""

  cfg = mos92_soccer_vision_env_cfg(play=True)
  cfg.scene.num_envs = 1
  # rebuild head_cam with RGB enabled at this resolution
  cam = next(s for s in cfg.scene.sensors if getattr(s, "name", "") == "head_cam")
  cam = dataclasses.replace(
    cam, data_types=("rgb", "depth"), width=res[0], height=res[1]
  )
  cfg.scene.sensors = tuple(
    cam if getattr(s, "name", "") == "head_cam" else s for s in cfg.scene.sensors
  )
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  env.reset()
  sensor = env.scene["head_cam"]
  stats = []
  for name, x, y, yaw in POSES:
    _set_robot_pose(env, x, y, yaw)
    for _ in range(3):
      env.sim.forward()
    sensor.update(0.0)
    rgb = sensor.data.rgb[0].cpu().numpy()  # (H,W,3) uint8
    # field markings = bright/white-ish pixels NOT pitch-green
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1], rgb[..., 2].astype(int)
    green_dom = (g.astype(int) > r + 15) & (g.astype(int) > b + 15)
    bright = rgb.mean(-1) > 120
    marking = bright & (~green_dom)
    frac = float(marking.mean())
    stats.append((name, frac))
    try:
      from PIL import Image

      Image.fromarray(rgb).resize((res[0] * 4, res[1] * 4), Image.NEAREST).save(
        save_dir / f"{res[0]}x{res[1]}_{name}.png"
      )
    except Exception:
      np.save(save_dir / f"{res[0]}x{res[1]}_{name}.npy", rgb)
  env.close()
  return stats


def main():
  os.environ.setdefault("MUJOCO_GL", "egl")
  OUT.mkdir(parents=True, exist_ok=True)
  print(f"[INFO] RGB field-visibility probe -> {OUT}")
  for res in RESOLUTIONS:
    stats = _render_at(res, OUT)
    print(f"\n=== {res[0]}x{res[1]} : field-marking pixel fraction per pose ===")
    for name, frac in stats:
      flag = "OK" if frac > 0.02 else "LOW"
      print(f"  [{flag}] {name:24s} {frac * 100:5.2f}%")
  print("\n[GATE] >2% marking pixels at SOME resolution => RGB sees the field.")
  print(f"[GATE] inspect images in {OUT} to confirm lines/goals are discriminable.")


if __name__ == "__main__":
  main()
