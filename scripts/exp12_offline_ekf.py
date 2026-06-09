"""EXP12 offline: does a recursive SE2 EKF beat the point-cloud-pooling Kabsch
fusion (and single-frame Kabsch) on pos_err, using the SAME rollout + the SAME
per-landmark depth observations? This is the cheap gate before committing to the
EKF refactor (line A R1). RoboCup's standard answer is a recursive filter; this
script tests that claim on our actual data.

Fair-comparison design: all three methods consume the identical per-frame, per-
landmark base-frame observations that FusedPoseBelief._collect_frame already
produces. The EKF starts from a WIDE prior (field center, large covariance) and
must localize purely from landmark observations + odometry — it is NOT handed the
GT start. Odometry for the EKF predict step uses the stored GT base-pose delta
(same odometry source the pooling already uses), so the ONLY variable is the
fusion mechanism: recursive filter vs batch point-pooling vs single frame.

Usage: MUJOCO_GL=egl uv run python scripts/exp12_offline_ekf.py <ckpt>
"""
import traceback

import torch


def _ekf_step(mu, Sigma, gt_delta, z_xy, z_w, map_xy, Q, R):
  """One SE2-EKF predict+update over all envs.
  mu:(B,3) [x,y,yaw]; Sigma:(B,3,3); gt_delta:(B,3) world-frame odom delta;
  z_xy:(B,K,2) measured landmark xy in CURRENT base frame; z_w:(B,K) visibility
  weight; map_xy:(K,2) known world landmark xy. Returns updated mu,Sigma."""
  B, K, _ = z_xy.shape
  dev = mu.device
  # --- Predict: world-frame pose += GT odometry delta; inflate covariance. ---
  mu = mu + gt_delta
  mu[:, 2] = torch.atan2(torch.sin(mu[:, 2]), torch.cos(mu[:, 2]))  # wrap yaw
  Sigma = Sigma + Q.unsqueeze(0)
  # --- Update: sequential per-landmark EKF correction (standard, exact). ---
  for k in range(K):
    w = z_w[:, k]  # (B,) 0 if landmark k not visible this frame
    if float(w.max()) <= 0.0:
      continue
    x, y, yaw = mu[:, 0], mu[:, 1], mu[:, 2]
    c, s = torch.cos(yaw), torch.sin(yaw)
    dx = map_xy[k, 0] - x
    dy = map_xy[k, 1] - y
    # Predicted obs: landmark in base frame = R(-yaw) @ (map_k - mu_xy).
    hx = c * dx + s * dy
    hy = -s * dx + c * dy
    # Jacobian H (B,2,3) wrt [x,y,yaw].
    H = torch.zeros(B, 2, 3, device=dev)
    H[:, 0, 0] = -c
    H[:, 0, 1] = -s
    H[:, 0, 2] = hy
    H[:, 1, 0] = s
    H[:, 1, 1] = -c
    H[:, 1, 2] = -hx
    innov = torch.stack([z_xy[:, k, 0] - hx, z_xy[:, k, 1] - hy], dim=-1)  # (B,2)
    # Per-env measurement noise scaled by 1/weight (down-weight low-vis).
    Rk = (R / w.clamp(min=1e-3).unsqueeze(-1).unsqueeze(-1))  # (B,2,2)
    S = H @ Sigma @ H.transpose(1, 2) + Rk  # (B,2,2)
    Sinv = torch.linalg.inv(S)
    Kg = Sigma @ H.transpose(1, 2) @ Sinv  # (B,3,2)
    upd = (Kg @ innov.unsqueeze(-1)).squeeze(-1)  # (B,3)
    # Only correct envs where this landmark is visible.
    m = (w > 0).float().unsqueeze(-1)
    mu = mu + m * upd
    eye3 = torch.eye(3, device=dev).unsqueeze(0)
    Sig_new = (eye3 - Kg @ H) @ Sigma
    Sigma = Sigma + m.unsqueeze(-1) * (Sig_new - Sigma)
    mu[:, 2] = torch.atan2(torch.sin(mu[:, 2]), torch.cos(mu[:, 2]))
  return mu, Sigma


def main(ckpt):
  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.velocity.config.mos92.env_cfgs import (
    mos92_soccer_e2e_dualcam_fused_scan_env_cfg as f,
  )
  from mjlab.tasks.velocity.config.mos92.rl_cfg import (
    mos92_selfloc_vision_ppo_runner_cfg,
  )
  from mjlab.tasks.velocity.mdp.field_keypoints import field_keypoints_3d
  from mjlab.tasks.velocity.mdp.observations import (
    FusedPoseBelief,
    oracle_pose_belief,
    robot_field_pose,
  )

  dev = "cuda:0"
  cfg = f(play=False)
  cfg.scene.num_envs = 64
  env = ManagerBasedRlEnv(cfg=cfg, device=dev)
  agent = mos92_selfloc_vision_ppo_runner_cfg()
  env_w = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = MjlabOnPolicyRunner(env_w, asdict(agent), "/tmp/exp12_dummy", dev)
  ck = torch.load(ckpt, map_location=dev, weights_only=False)
  runner.alg._raw_actor.load_state_dict(ck["actor_state_dict"], strict=False)
  policy = runner.get_inference_policy(dev)

  fused = FusedPoseBelief(env, "dribble", "head_cam", num_frames=8, stride=4)
  cmd = env.command_manager.get_term("dribble")
  HL, HW = cmd.cfg.half_length, cmd.cfg.half_width
  map_xy = field_keypoints_3d(dev)[:, :2]  # (K,2) world landmark xy
  K = map_xy.shape[0]
  B = cfg.scene.num_envs

  # EKF state: wide prior at field center (NOT given GT start).
  mu = torch.zeros(B, 3, device=dev)
  Sigma = torch.diag(torch.tensor([25.0, 25.0, 9.0], device=dev)).repeat(B, 1, 1)
  Q = torch.diag(torch.tensor([0.02, 0.02, 0.02], device=dev))  # process noise
  R = torch.eye(2, device=dev) * 0.10  # base measurement noise (per visible lmk)

  obs, _ = env_w.reset()
  fused.reset(slice(None))
  fused._ensure_init(env)  # lazily inits _kp_local before _collect_frame is used
  prev_gt_world = None
  s_err, f_err, e_err = [], [], []
  for t in range(200):
    with torch.inference_mode():
      act = policy(obs)
    obs, _, _, _ = env_w.step(act)
    # Per-landmark base-frame observation + GT world pose (from collect_frame pack).
    frame = fused._collect_frame(env)  # (B, 3K+3)
    z_x, z_y = frame[:, 0:K], frame[:, K:2 * K]
    z_w = frame[:, 2 * K:3 * K]
    z_xy = torch.stack([z_x, z_y], dim=-1)  # (B,K,2) base frame
    gt_world = frame[:, 3 * K:]  # (B,3) GT base pose x,y,yaw (world)
    # Odometry delta (perfect odom from GT; same source pooling uses).
    if prev_gt_world is None:
      gt_delta = torch.zeros(B, 3, device=dev)
    else:
      gt_delta = gt_world - prev_gt_world
      gt_delta[:, 2] = torch.atan2(torch.sin(gt_delta[:, 2]), torch.cos(gt_delta[:, 2]))
    prev_gt_world = gt_world.clone()
    mu, Sigma = _ekf_step(mu, Sigma, gt_delta, z_xy, z_w, map_xy, Q, R)

    single = oracle_pose_belief(env, "dribble")  # (B,6) single-frame Kabsch
    fb = fused(env, "dribble", "head_cam", 8, 4)  # (B,7) pooling Kabsch
    gt = robot_field_pose(env, "dribble")
    gx, gy = gt[:, 0] * HL, gt[:, 1] * HW
    if t < 40:
      continue  # let window + EKF settle
    se = torch.sqrt((single[:, 0] * HL - gx) ** 2 + (single[:, 1] * HW - gy) ** 2)
    fe = torch.sqrt((fb[:, 0] * HL - gx) ** 2 + (fb[:, 1] * HW - gy) ** 2)
    ee = torch.sqrt((mu[:, 0] - gx) ** 2 + (mu[:, 1] - gy) ** 2)
    s_err.append(se)
    f_err.append(fe)
    e_err.append(ee)
  se = torch.cat(s_err)
  fe = torch.cat(f_err)
  ee = torch.cat(e_err)
  print(f"RESULT single-frame Kabsch : mean={se.mean():.3f} median={se.median():.3f}")
  print(f"RESULT pooling(8f) Kabsch  : mean={fe.mean():.3f} median={fe.median():.3f}")
  print(f"RESULT recursive EKF       : mean={ee.mean():.3f} median={ee.median():.3f}")
  print(f"RESULT EKF vs pooling      : {(1-ee.mean()/fe.mean())*100:+.1f}% (neg=EKF better)")
  print(f"RESULT EKF vs single       : {(1-ee.mean()/se.mean())*100:+.1f}%")
  env.close()


if __name__ == "__main__":
  import sys
  try:
    main(sys.argv[1])
  except Exception:
    traceback.print_exc()
