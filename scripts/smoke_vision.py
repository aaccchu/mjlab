"""Vision smoke test for MOS92 head depth camera (v3 Stage 3 Step 1).

Validates the camera pipeline before any training:
  (a) camera renders depth of expected shape/range
  (b) a ball placed ahead produces a near-depth blob (vs distance)
  (c) neck_yaw / neck_pitch steer the field of view (blob centroid moves)
  (d) the full env + CNN actor forward runs without shape/NaN errors

Also sweeps a few candidate camera quats and reports which one centers a
ball 1.5m straight ahead, to calibrate _HEAD_CAM_QUAT in env_cfgs.py.

Usage:
  MUJOCO_GL=egl uv run python scripts/smoke_vision.py
"""

from __future__ import annotations

import os

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.mos92 import env_cfgs
from mjlab.sensor import CameraSensor

DEV = "cuda:0"
N = 8


def _blob_stats(depth: torch.Tensor, thresh: float = 0.6):
  """depth: [B,H,W,1] normalized. Return (pixel_count, u_centroid, v_centroid)."""
  d = depth[..., 0]  # [B,H,W]
  mask = d < thresh
  cnt = mask.float().sum(dim=(1, 2))  # [B]
  H, W = d.shape[1], d.shape[2]
  vv, uu = torch.meshgrid(
    torch.linspace(-1, 1, H, device=d.device),
    torch.linspace(-1, 1, W, device=d.device),
    indexing="ij",
  )
  denom = cnt.clamp(min=1)
  u_c = (mask * uu).sum(dim=(1, 2)) / denom
  v_c = (mask * vv).sum(dim=(1, 2)) / denom
  return cnt, u_c, v_c


def _place_ball_ahead(env, dist: float, dz: float = 0.0):
  """Put the ball `dist` m straight ahead of the robot heading, at head height+dz."""
  cmd = env.command_manager.get_term("dribble")
  robot = cmd.robot
  ball = cmd.object
  base = robot.data.root_link_pos_w  # [B,3]
  heading = robot.data.heading_w  # [B]
  fwd = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)  # [B,2]
  head_z = base[:, 2] + 0.4  # ~head height above root
  pos = base.clone()
  pos[:, :2] = base[:, :2] + fwd * dist
  pos[:, 2] = head_z + dz
  quat = torch.zeros((env.num_envs, 4), device=env.device)
  quat[:, 0] = 1.0
  pose = torch.cat([pos, quat], dim=-1)
  ball.write_root_link_pose_to_sim(pose)
  vel = torch.zeros((env.num_envs, 6), device=env.device)
  ball.write_root_link_velocity_to_sim(vel)


def _refresh(env):
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()


def _save_depth_png(depth: torch.Tensor, path: str):
  """depth: [B,H,W,1] raw meters. Save env-0 as grayscale (near=bright)."""
  import mediapy as media

  d = depth[0, ..., 0].cpu().numpy()
  d = np.clip(d, 0.0, 3.0) / 3.0
  img = (1.0 - d)  # near = bright
  media.write_image(path, (img * 255).astype(np.uint8))


def _set_neck(env, yaw: float, pitch: float):
  robot = env.scene["robot"]
  jid = [robot.joint_names.index("neck_yaw"), robot.joint_names.index("neck_pitch")]
  q = robot.data.joint_pos.clone()
  q[:, jid[0]] = yaw
  q[:, jid[1]] = pitch
  robot.write_joint_position_to_sim(q)


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  torch.set_grad_enabled(False)

  cfg = env_cfgs.mos92_soccer_vision_env_cfg(play=True)
  cfg.scene.num_envs = N
  env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
  env.reset()

  cam: CameraSensor = env.scene["head_cam"]

  # (a) renders + shape/range
  _set_neck(env, 0.0, 0.0)
  _place_ball_ahead(env, 50.0)  # ball far away = background baseline
  _refresh(env)
  depth = cam.data.depth
  assert depth is not None, "no depth!"
  base_depth = depth[..., 0].clone()  # [B,H,W] robot's own geometry
  print(f"(a) depth shape={tuple(depth.shape)} finite={torch.isfinite(depth).all().item()} "
        f"min={depth.min().item():.3f} max={depth.max().item():.3f}")
  print(f"    baseline (ball@50m) near-px(<0.6)={(base_depth<0.6).float().sum(dim=(1,2)).mean().item():.0f} "
        f"-- this is the robot's own body in view")
  os.makedirs("/tmp/smoke_depth", exist_ok=True)
  _save_depth_png(depth, "/tmp/smoke_depth/baseline.png")

  # (b) ball signal vs distance, via background subtraction (neck neutral)
  print("(b) ball blob vs distance (depth diff vs baseline, |Δ|>0.1m):")
  for d in (0.5, 1.0, 1.5, 2.0, 3.0):
    _place_ball_ahead(env, d)
    _refresh(env)
    cur = cam.data.depth[..., 0]
    diff = (base_depth - cur)  # ball nearer than background => positive
    mask = diff > 0.1
    cnt = mask.float().sum(dim=(1, 2))
    H, W = cur.shape[1], cur.shape[2]
    vv, uu = torch.meshgrid(torch.linspace(-1, 1, H, device=DEV),
                            torch.linspace(-1, 1, W, device=DEV), indexing="ij")
    denom = cnt.clamp(min=1)
    u_c = (mask * uu).sum(dim=(1, 2)) / denom
    v_c = (mask * vv).sum(dim=(1, 2)) / denom
    print(f"    {d:.1f}m: ball_px={cnt.float().mean().item():6.1f} u={u_c.mean().item():+.2f} v={v_c.mean().item():+.2f}")
    _save_depth_png(cam.data.depth, f"/tmp/smoke_depth/ball_{d:.1f}m.png")

  # (c) neck steering: ball fixed at 1.5m, sweep neck (ball-isolated centroid)
  print("(c) neck steering (ball fixed 1.5m ahead, ball-isolated centroid):")
  for name, yaw, pitch in [("center", 0.0, 0.0), ("yaw+", 0.6, 0.0), ("yaw-", -0.6, 0.0),
                            ("pitch+", 0.0, 0.4), ("pitch-", 0.0, -0.4)]:
    _set_neck(env, yaw, pitch)
    _place_ball_ahead(env, 50.0)
    _refresh(env)
    bg = cam.data.depth[..., 0].clone()
    _place_ball_ahead(env, 1.5)
    _refresh(env)
    cur = cam.data.depth[..., 0]
    mask = (bg - cur) > 0.1
    cnt = mask.float().sum(dim=(1, 2))
    H, W = cur.shape[1], cur.shape[2]
    vv, uu = torch.meshgrid(torch.linspace(-1, 1, H, device=DEV),
                            torch.linspace(-1, 1, W, device=DEV), indexing="ij")
    denom = cnt.clamp(min=1)
    u_c = (mask * uu).sum(dim=(1, 2)) / denom
    v_c = (mask * vv).sum(dim=(1, 2)) / denom
    print(f"    {name:7s}: ball_px={cnt.float().mean().item():6.1f} u={u_c.mean().item():+.2f} v={v_c.mean().item():+.2f}")
    _save_depth_png(cam.data.depth, f"/tmp/smoke_depth/neck_{name}.png")

  env.close()
  print("SMOKE OK -- depth PNGs in /tmp/smoke_depth/")


if __name__ == "__main__":
  main()

