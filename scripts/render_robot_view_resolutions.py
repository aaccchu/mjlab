"""Save the robot's head-camera view at INCREASING resolutions, real 0.125m lines.

The resolution gate showed worst-pose marking fraction barely moves across an 8x
resolution sweep — but a NUMBER hides what the robot actually sees. This dumps the
real RGB view at each resolution from representative poses so you can eyeball
whether higher resolution genuinely recovers the distant 0.125m lines (and where
it saturates). Saved into soccer_eval/2026-06-07_v3_soccer_solo/robot_view_res/.

Usage: MUJOCO_GL=egl uv run python scripts/render_robot_view_resolutions.py
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.mos92.env_cfgs import mos92_soccer_vision_env_cfg
from mjlab.terrains.soccer_field import SoccerFieldCfg, build_soccer_field

OUT = Path("soccer_eval/2026-06-07_v3_soccer_solo/robot_view_res")
LINE_WIDTH = 0.125  # real spec
# Increasing resolutions (4:3). High ones test where line recovery saturates.
RESOLUTIONS = [(64, 48), (128, 96), (256, 192), (512, 384), (640, 480), (960, 720), (1280, 960)]
POSES = [
  ("midfield_face_x", 0.0, 0.0, 0.0),       # sees center circle + far goal
  ("own_half_face_goal", -6.0, 0.0, 0.0),   # far goal, the hard pose
  ("corner_face_center", -8.0, 5.0, -0.5),  # corner, sparse landmarks
  ("near_opp_goal", 7.0, 0.0, 0.0),         # near goal, easy pose
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
  r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
  green_dom = (g > r + 15) & (g > b + 15)
  bright = rgb.mean(-1) > 120
  return float((bright & (~green_dom)).mean())


def _render_res(res, save_dir):
  w, h = res
  cfg = mos92_soccer_vision_env_cfg(play=True)
  cfg.scene.num_envs = 1
  field_cfg = SoccerFieldCfg(line_width=LINE_WIDTH)
  cfg.scene.spec_fn = lambda spec: build_soccer_field(spec, field_cfg)
  cam = next(s for s in cfg.scene.sensors if getattr(s, "name", "") == "head_cam")
  cam = dataclasses.replace(cam, data_types=("rgb", "depth"), width=w, height=h)
  cfg.scene.sensors = tuple(
    cam if getattr(s, "name", "") == "head_cam" else s for s in cfg.scene.sensors
  )
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  env.reset()
  sensor = env.scene["head_cam"]
  from PIL import Image

  out = []
  for name, x, y, yaw in POSES:
    _set_robot_pose(env, x, y, yaw)
    for _ in range(3):
      env.sim.forward()
    sensor.update(0.0)
    rgb = sensor.data.rgb[0].cpu().numpy()
    frac = _marking_fraction(rgb)
    # Save at native pixels (no upscale) so resolution is honest.
    Image.fromarray(rgb).save(save_dir / f"res{w:04d}x{h:04d}_{name}.png")
    out.append((name, frac))
  env.close()
  return out


def main():
  os.environ.setdefault("MUJOCO_GL", "egl")
  OUT.mkdir(parents=True, exist_ok=True)
  print(f"[INFO] robot-view @ lw={LINE_WIDTH}m, increasing res -> {OUT}\n")
  for res in RESOLUTIONS:
    try:
      stats = _render_res(res, OUT)
    except Exception as e:  # noqa: BLE001 — high res may exceed GL buffer; skip.
      print(f"[SKIP] res={res[0]}x{res[1]}: {type(e).__name__}: {e}")
      continue
    worst = min(f for _, f in stats)
    line = "  ".join(f"{n.split('_')[0]}={f * 100:.2f}%" for n, f in stats)
    print(f"res={res[0]:4d}x{res[1]:4d}  worst={worst * 100:5.2f}%  | {line}")
  print(f"\n[INFO] inspect PNGs in {OUT} — eyeball where distant 0.125m lines appear.")


if __name__ == "__main__":
  main()
