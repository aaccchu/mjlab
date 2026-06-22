"""Phase A-2: record a sim rollout's per-frame self-loc observations to npz.

Drives a trained self-loc policy in the fused_scan env (same as exp12_offline_ekf)
and dumps, per timestep, the EXACT observations the portable kernel needs:
  z_xy   (T, B, K, 2)  per-landmark xy in the robot base frame (front-x/left-y, m)
  z_w    (T, B, K)     per-landmark weight (visibility * valid-depth)
  gt_xyz (T, B, 3)     GT base pose [x, y, yaw] world (odometry + error ground truth)
  map_xy (K, 2)        field-local landmark map
These are produced by FusedPoseBelief._collect_frame — the oracle-correspondence
observations. The recording lets selfloc_kernel's numpy EKF be validated OFFLINE
against the in-sim torch EKF with NO mjlab dependency, proving the port is faithful.

Usage: MUJOCO_GL=egl uv run python scripts/record_selfloc_rollout.py <ckpt> [out.npz] [--steps N]
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def main() -> None:
  ckpt = sys.argv[1]
  out = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else Path(
    "/media/server625/data/AC/Robot/selfloc_kernel/data/sim_rollout.npz"
  )
  steps = 200
  for i, a in enumerate(sys.argv):
    if a == "--steps":
      steps = int(sys.argv[i + 1])
  out.parent.mkdir(parents=True, exist_ok=True)

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.velocity.config.mos92.env_cfgs import (
    mos92_soccer_e2e_dualcam_fused_scan_env_cfg as f,
  )
  from mjlab.tasks.velocity.config.mos92.rl_cfg import (
    mos92_selfloc_vision_ppo_runner_cfg,
  )
  from mjlab.tasks.velocity.mdp.field_keypoints import field_keypoints_3d
  from mjlab.tasks.velocity.mdp.observations import FusedPoseBelief

  dev = "cuda:0"
  cfg = f(play=False)
  cfg.scene.num_envs = 64
  env = ManagerBasedRlEnv(cfg=cfg, device=dev)
  agent = mos92_selfloc_vision_ppo_runner_cfg()
  env_w = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = MjlabOnPolicyRunner(env_w, asdict(agent), "/tmp/rec_dummy", dev)
  ck = torch.load(ckpt, map_location=dev, weights_only=False)
  runner.alg._raw_actor.load_state_dict(ck["actor_state_dict"], strict=False)
  policy = runner.get_inference_policy(dev)

  fused = FusedPoseBelief(env, "dribble", "head_cam", num_frames=8, stride=4)
  map_xy = field_keypoints_3d(dev)[:, :2]
  K = map_xy.shape[0]
  B = cfg.scene.num_envs

  obs, _ = env_w.reset()
  fused.reset(slice(None))
  fused._ensure_init(env)
  Z_xy, Z_w, GT = [], [], []
  for _ in range(steps):
    with torch.inference_mode():
      act = policy(obs)
    obs, _, _, _ = env_w.step(act)
    frame = fused._collect_frame(env)  # (B, 3K+3)
    z_x, z_y = frame[:, 0:K], frame[:, K : 2 * K]
    z_w = frame[:, 2 * K : 3 * K]
    gt = frame[:, 3 * K :]
    Z_xy.append(torch.stack([z_x, z_y], dim=-1).cpu().numpy())
    Z_w.append(z_w.cpu().numpy())
    GT.append(gt.cpu().numpy())
  env.close()

  np.savez_compressed(
    out,
    z_xy=np.asarray(Z_xy, dtype=np.float32),  # (T,B,K,2)
    z_w=np.asarray(Z_w, dtype=np.float32),  # (T,B,K)
    gt_xyz=np.asarray(GT, dtype=np.float32),  # (T,B,3)
    map_xy=map_xy.cpu().numpy().astype(np.float32),  # (K,2)
  )
  print(f"[OK] wrote {out}  steps={steps} B={B} K={K}")


if __name__ == "__main__":
  main()
