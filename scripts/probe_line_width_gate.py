"""Experiment 6 GATE: does WIDENING the field lines make them visible at 64x48?

Root cause (exp 3 res-sweep + exp 5 camera-geometry sweep both failed): the
field lines are 0.125 m solid bands — sub-pixel at 64x48 from >10 m, so neither
resolution nor camera tilt/FOV recovers them. This sweeps line_width and reports
field-marking pixel fraction per pose at the training resolution. If wider lines
clear the 2% bar at the previously-LOW poses, a self-loc retrain becomes viable
(sim-only aid; real-field lines are a fixed spec, noted as a sim-to-real caveat).

Usage: MUJOCO_GL=egl uv run python scripts/probe_line_width_gate.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.mos92.env_cfgs import mos92_soccer_vision_env_cfg
from mjlab.terrains.soccer_field import SoccerFieldCfg, build_soccer_field

OUT = Path("soccer_eval/2026-06-06_spikes/v3g_line_width_gate")
POSES = [
  ("midfield_face_x", 0.0, 0.0, 0.0),
  ("own_half_face_goal", -6.0, 0.0, 0.0),
  ("corner_face_center", -8.0, 5.0, -0.5),
  ("near_opp_goal", 7.0, 0.0, 0.0),
  ("center_circle_edge", 2.0, 0.0, 0.3),
]
RES = (64, 48)
WIDTHS = [0.125, 0.25, 0.5, 1.0]  # baseline .125 -> 8x.


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


def _eval_width(lw: float, save_dir):
  import dataclasses

  cfg = mos92_soccer_vision_env_cfg(play=True)
  cfg.scene.num_envs = 1
  field_cfg = SoccerFieldCfg(line_width=lw)
  cfg.scene.spec_fn = lambda spec: build_soccer_field(spec, field_cfg)
  cam = next(s for s in cfg.scene.sensors if getattr(s, "name", "") == "head_cam")
  cam = dataclasses.replace(
    cam, data_types=("rgb", "depth"), width=RES[0], height=RES[1]
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
      Image.fromarray(rgb).resize((RES[0] * 4, RES[1] * 4), Image.NEAREST).save(
        save_dir / f"lw{int(lw * 1000):04d}_{name}.png"
      )
  env.close()
  return stats


def main():
  os.environ.setdefault("MUJOCO_GL", "egl")
  OUT.mkdir(parents=True, exist_ok=True)
  print(f"[INFO] line-width GATE @ {RES[0]}x{RES[1]} -> {OUT}\n")
  all_stats = {}
  for lw in WIDTHS:
    stats = _eval_width(lw, OUT)
    all_stats[lw] = stats
    fracs = [f for _, f in stats]
    n_ok = sum(1 for f in fracs if f > 0.02)
    print(
      f"=== line_width={lw:.3f}m ({lw / 0.125:.0f}x) : "
      f"mean={np.mean(fracs) * 100:5.2f}%  OK={n_ok}/{len(fracs)} ==="
    )
    for name, f in stats:
      print(f"    [{'OK ' if f > 0.02 else 'LOW'}] {name:22s} {f * 100:5.2f}%")
  print("\n[GATE] worst-pose marking fraction per width (the bottleneck):")
  for lw, stats in all_stats.items():
    worst = min(f for _, f in stats)
    print(
      f"    lw={lw:.3f}m  worst={worst * 100:5.2f}%  {'PASS' if worst > 0.02 else 'fail'}"
    )
  print(
    f"\n[GATE] inspect {OUT}. Retrain self-loc only at a width where worst-pose >2%."
  )


if __name__ == "__main__":
  main()
