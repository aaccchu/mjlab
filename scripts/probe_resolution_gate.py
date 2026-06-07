"""Resolution GATE: at the REAL-SPEC 0.125 m line width, what camera resolution
makes the field markings visible enough for pure-vision self-loc?

The original line-width gate fixed resolution at 64x48 and swept width, concluding
0.125 m lines are sub-pixel there (only 1.0 m cleared the 2% bar). The user wants
the honest 0.125 m spec back WITHOUT losing localization — so the lever is now
RESOLUTION, not width. This fixes width at 0.125 m and sweeps resolution, reporting
worst-pose marking-pixel fraction per resolution. A resolution where worst-pose
>2% means a real-spec self-loc retrain is viable (no sim-to-real line-width gap).

Usage: MUJOCO_GL=egl uv run python scripts/probe_resolution_gate.py
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.mos92.env_cfgs import mos92_soccer_vision_env_cfg
from mjlab.terrains.soccer_field import SoccerFieldCfg, build_soccer_field

OUT = Path("soccer_eval/2026-06-07_v3_soccer_solo/resolution_gate")
POSES = [
  ("midfield_face_x", 0.0, 0.0, 0.0),
  ("own_half_face_goal", -6.0, 0.0, 0.0),
  ("corner_face_center", -8.0, 5.0, -0.5),
  ("near_opp_goal", 7.0, 0.0, 0.0),
  ("center_circle_edge", 2.0, 0.0, 0.3),
]
LINE_WIDTH = 0.125  # REAL SPEC — fixed.
# Sweep resolution (w, h), 4:3 aspect like the 64x48 baseline.
RESOLUTIONS = [(64, 48), (128, 96), (256, 192), (384, 288), (512, 384)]


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


def _eval_resolution(res, save_dir):
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
      Image.fromarray(rgb).resize((w * 2, h * 2), Image.NEAREST).save(
        save_dir / f"res{w:04d}x{h:04d}_{name}.png"
      )
  env.close()
  return stats


def main():
  os.environ.setdefault("MUJOCO_GL", "egl")
  OUT.mkdir(parents=True, exist_ok=True)
  print(f"[INFO] resolution GATE @ line_width={LINE_WIDTH}m (REAL SPEC) -> {OUT}\n")
  all_stats = {}
  for res in RESOLUTIONS:
    stats = _eval_resolution(res, OUT)
    all_stats[res] = stats
    fracs = [f for _, f in stats]
    n_ok = sum(1 for f in fracs if f > 0.02)
    print(
      f"=== res={res[0]}x{res[1]} : mean={np.mean(fracs) * 100:5.2f}%  "
      f"OK={n_ok}/{len(fracs)} ==="
    )
    for name, f in stats:
      print(f"    [{'OK ' if f > 0.02 else 'LOW'}] {name:22s} {f * 100:5.2f}%")
  print("\n[GATE] worst-pose marking fraction per resolution (the bottleneck):")
  for res, stats in all_stats.items():
    worst = min(f for _, f in stats)
    print(
      f"    res={res[0]}x{res[1]}  worst={worst * 100:5.2f}%  "
      f"{'PASS' if worst > 0.02 else 'fail'}"
    )
  print(
    f"\n[GATE] inspect {OUT}. Retrain self-loc at the SMALLEST resolution where "
    "worst-pose >2% (smaller = cheaper to train + closer to real cameras)."
  )


if __name__ == "__main__":
  main()
