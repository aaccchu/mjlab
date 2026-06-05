"""Spike B: Camera visibility test for soccer ball at various distances.

Renders depth + segmentation images with the ball placed at different
positions relative to the robot, and reports pixel coverage statistics.
Run as a script: uv run python tests/test_camera_visibility.py
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_soccer_env_cfg
from mjlab.utils.lab_api.math import quat_apply

OUTPUT_DIR = Path("soccer_eval/spike_b_camera_visibility")

BALL_DISTANCES = [0.5, 1.0, 2.0, 3.0, 5.0]
BALL_ANGLES = [0, 30, -30, 60]

CAMERA_CFG = CameraSensorCfg(
  name="head_depth_cam",
  parent_body="robot/torso_link",
  # Placed in front of head_collision sphere (center z=0.43, r=0.06 from
  # torso_link) to avoid self-occlusion.
  pos=(0.15, 0.0, 0.43),
  # MuJoCo camera looks along local -Z. Pitch 35deg down so the ball
  # (at ground level, ~0.8m below camera) enters the FoV within 1-5m.
  quat=(0.6124, 0.3536, -0.3536, -0.6124),
  fovy=60.0,
  width=64,
  height=48,
  data_types=("depth", "segmentation"),
  use_textures=False,
  use_shadows=False,
  enabled_geom_groups=(0, 1, 2, 3),
)


def build_env() -> ManagerBasedRlEnv:
  """Create soccer env with 1 env and inject camera sensor."""
  cfg = unitree_g1_soccer_env_cfg(play=True)
  cfg.scene.num_envs = 1
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (CAMERA_CFG,)
  cfg.sim.mujoco.timestep = 0.005
  device = "cuda" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  return env


def place_ball(env: ManagerBasedRlEnv, distance: float, angle_deg: float):
  """Place ball at (distance, angle) relative to robot facing direction."""
  robot = env.scene["robot"]
  ball = env.scene["ball"]
  robot_pos = robot.data.root_link_pos_w[0]
  robot_quat = robot.data.root_link_quat_w[0]

  angle_rad = math.radians(angle_deg)

  forward_w = quat_apply(
    robot_quat.unsqueeze(0),
    torch.tensor([[1.0, 0.0, 0.0]], device=env.device),
  )[0]
  left_w = quat_apply(
    robot_quat.unsqueeze(0),
    torch.tensor([[0.0, 1.0, 0.0]], device=env.device),
  )[0]

  offset = forward_w * distance * math.cos(angle_rad) + left_w * distance * math.sin(
    angle_rad
  )
  ball_pos = robot_pos + offset
  ball_pos[2] = 0.11  # ball radius

  pose = torch.zeros(1, 7, device=env.device)
  pose[0, :3] = ball_pos
  pose[0, 3] = 1.0  # quat w
  ball.write_root_link_pose_to_sim(pose, env_ids=torch.tensor([0], device=env.device))
  ball.write_root_link_velocity_to_sim(
    torch.zeros(1, 6, device=env.device),
    env_ids=torch.tensor([0], device=env.device),
  )


def count_ball_pixels(env: ManagerBasedRlEnv) -> tuple[int, float]:
  """Render and count ball pixels in segmentation image.

  Returns (ball_pixel_count, depth_at_ball_center).
  """
  env.sim.forward()
  env.sim.sense()
  cam_sensor = env.scene["head_depth_cam"]
  data = cam_sensor.data

  seg = data.segmentation[0].cpu()  # [H, W, 2]: (obj_id, obj_type)
  depth = data.depth[0].cpu()  # [H, W, 1]

  ball_geom_id = env.sim.mj_model.geom("ball/ball_geom").id
  geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)

  ball_mask = (seg[..., 0] == ball_geom_id) & (seg[..., 1] == geom_type)
  ball_pixels = int(ball_mask.sum().item())

  depth_at_ball = float("inf")
  if ball_pixels > 0:
    depth_at_ball = float(depth[ball_mask.unsqueeze(-1).expand_as(depth)].mean())

  return ball_pixels, depth_at_ball


def save_depth_image(depth: torch.Tensor, path: Path):
  """Save depth tensor as a grayscale PNG."""
  from PIL import Image

  d = depth[0].cpu().numpy().squeeze(-1)
  valid = d[d < 100.0]
  if len(valid) == 0:
    d_norm = (d * 0).astype("uint8")
  else:
    lo, hi = valid.min(), valid.max()
    d_clipped = d.clip(lo, hi)
    d_norm = ((d_clipped - lo) / (hi - lo + 1e-6) * 255).astype("uint8")
  Image.fromarray(d_norm).save(path)


def run():
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  print("Building soccer env with camera sensor...")
  env = build_env()
  env.reset()

  total_pixels = CAMERA_CFG.width * CAMERA_CFG.height
  results = []

  for dist in BALL_DISTANCES:
    for angle in BALL_ANGLES:
      place_ball(env, dist, angle)
      ball_px, depth_val = count_ball_pixels(env)
      coverage = ball_px / total_pixels * 100
      results.append(
        {
          "dist": dist,
          "angle": angle,
          "pixels": ball_px,
          "coverage_pct": coverage,
          "depth": depth_val,
        }
      )
      tag = f"d{dist:.1f}_a{angle}"
      cam_data = env.scene["head_depth_cam"].data
      save_depth_image(cam_data.depth, OUTPUT_DIR / f"{tag}_depth.png")
      status = "VISIBLE" if ball_px > 0 else "NOT VISIBLE"
      print(f"  {tag}: {ball_px} px ({coverage:.2f}%) {status}")

  print("\n=== Visibility Summary ===")
  print(f"{'Dist(m)':<8} {'Angle':<8} {'Pixels':<8} {'Coverage%':<10} {'Visible'}")
  print("-" * 50)
  for r in results:
    vis = "Y" if r["pixels"] > 0 else "N"
    print(
      f"{r['dist']:<8.1f} {r['angle']:<8} {r['pixels']:<8} "
      f"{r['coverage_pct']:<10.2f} {vis}"
    )

  report_path = OUTPUT_DIR / "visibility_report.txt"
  with open(report_path, "w") as f:
    f.write("Spike B: Camera Visibility Report\n")
    f.write(f"Camera: {CAMERA_CFG.name}\n")
    f.write(f"Resolution: {CAMERA_CFG.width}x{CAMERA_CFG.height}\n")
    f.write(f"FoV: {CAMERA_CFG.fovy} deg\n")
    f.write(f"Mount: {CAMERA_CFG.parent_body} @ {CAMERA_CFG.pos}\n\n")
    f.write(f"{'Dist':<6} {'Angle':<6} {'Pixels':<8} {'%':<8} {'Vis'}\n")
    for r in results:
      vis = "Y" if r["pixels"] > 0 else "N"
      f.write(
        f"{r['dist']:<6.1f} {r['angle']:<6} "
        f"{r['pixels']:<8} {r['coverage_pct']:<8.2f} {vis}\n"
      )

  env.close()
  print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
  run()
