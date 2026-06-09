"""v4 EXP7 OFFLINE GATE: does multi-frame fusion beat single-frame Kabsch?

EXP6 proved the architecture (perception-as-obs -> fell_over 52->0.5) but exposed
the real bottleneck: single-frame, forward-narrow-view geometry sees only ~4/23
keypoints (belief_vis_frac 0.17) -> Kabsch underdetermined -> pos_err ~4.5m ->
goal_rate 0. codex #2: prove OFFLINE that fusing N frames via odometry (different
frames see different keypoints) lowers pos_err BELOW single-frame, before wiring
it into training.

Method: roll out the EXP6 policy; each step record per-frame visible keypoints in
the BASE frame (p_base) + weights + GT base pose. Then for each frame t compare:
  - single: Kabsch on frame t's visible keypoints only (the EXP6 belief).
  - multi:  transform frames [t-W..t] keypoints into frame t's base via odometry
            (relative base_s->base_t transform = perfect-odometry oracle), pool,
            Kabsch on the union. More unique keypoints -> better-conditioned.
GT odometry is the oracle; codex's noisy-oracle stage later adds odometry noise.

Usage: MUJOCO_GL=egl uv run python scripts/exp7_offline_multiframe.py CKPT.pt
"""

from __future__ import annotations

import sys
import traceback

import torch


def _yaw_from_quat(quat, env):
  from mjlab.utils.lab_api.math import matrix_from_quat

  m = matrix_from_quat(quat)
  return torch.atan2(m[:, 1, 0], m[:, 0, 0]), m


def main(ckpt_path: str) -> None:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.velocity.config.mos92.env_cfgs import (
    mos92_soccer_e2e_dualcam_oracle_env_cfg,
  )
  from mjlab.tasks.velocity.config.mos92.rl_cfg import (
    mos92_selfloc_vision_ppo_runner_cfg,
  )

  device = "cuda:0"
  cfg = mos92_soccer_e2e_dualcam_oracle_env_cfg(play=False)
  cfg.scene.num_envs = 64
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  print(f"[EXP7] env built, rolling out {ckpt_path}")

  agent_cfg = mos92_selfloc_vision_ppo_runner_cfg()
  agent_cfg.max_iterations = 1
  env_w = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(env_w, _to_dict(agent_cfg), "/tmp/exp7_dummy", device)
  ck = torch.load(ckpt_path, map_location=device, weights_only=False)
  runner.alg._raw_actor.load_state_dict(ck["actor_state_dict"], strict=False)
  policy = runner.get_inference_policy(device)

  from mjlab.tasks.velocity.mdp.field_keypoints import (
    field_keypoints_3d,
    lift_pixels_to_world,
    project_keypoints,
  )
  from mjlab.tasks.velocity.mdp.observations import robot_field_pose
  from mjlab.utils.lab_api.math import matrix_from_quat

  kp_local = field_keypoints_3d(device)
  Kn = kp_local.shape[0]
  cam = env.scene["head_cam"]
  W, Hh = cam.cfg.width, cam.cfg.height
  fovy = cam.cfg.fovy if cam.cfg.fovy is not None else 45.0
  robot = env.scene["robot"]
  cmd = env.command_manager.get_term("dribble")
  HL, HW = cmd.cfg.half_length, cmd.cfg.half_width

  # Rollout, recording per-frame base-frame visible keypoints + GT base pose.
  # Use the WRAPPED env (env_w) for reset/step (4-tuple API); the bare env is for
  # scene/sim/command access.
  obs, _ = env_w.reset()
  frames = []
  T = 120
  for _ in range(T):
    with torch.inference_mode():
      # rsl_rl actor takes the FULL TensorDict and selects its obs group
      # internally (see PPO.act: self.actor(obs)). Passing obs["actor"] would
      # feed a pre-sliced tensor and break group resolution.
      act = policy(obs)
    obs, _, _, _ = env_w.step(act)
    cam_idx = cam.camera_idx
    sd = env.sim.data
    cam_pos = sd.cam_xpos[:, cam_idx, :]
    cam_mat = sd.cam_xmat[:, cam_idx, :].reshape(-1, 3, 3)
    origin = env.scene.env_origins
    kp_world = kp_local.unsqueeze(0) + origin.unsqueeze(1)
    uv, vis = project_keypoints(kp_world, cam_pos, cam_mat, fovy, W, Hh)
    depth_raw = cam.data.depth
    u_pix = ((uv[..., 0] + 1.0) * 0.5 * (W - 1)).round().long().clamp(0, W - 1)
    v_pix = ((uv[..., 1] + 1.0) * 0.5 * (Hh - 1)).round().long().clamp(0, Hh - 1)
    bidx = torch.arange(uv.shape[0], device=device).unsqueeze(1).expand(-1, Kn)
    depth_at = depth_raw[bidx, v_pix, u_pix, 0]
    pw = lift_pixels_to_world(uv, depth_at, cam_pos, cam_mat, fovy, W, Hh)
    base_pos = robot.data.root_link_pos_w.clone()
    base_mat = matrix_from_quat(robot.data.root_link_quat_w)
    rel = pw - base_pos.unsqueeze(1)
    p_base = torch.einsum("bij,bkj->bki", base_mat.transpose(1, 2), rel)
    w = vis.float() * ((depth_at > 0.05) & (depth_at < 30.0)).float()
    gt = robot_field_pose(env, "dribble")
    frames.append({"p_base": p_base[..., :2].clone(), "w": w.clone(),
                   "base_pos": base_pos[:, :2].clone(),
                   "base_yaw": torch.atan2(base_mat[:, 1, 0], base_mat[:, 0, 0]).clone(),
                   "gt": gt.clone()})
  print(f"[EXP7] rolled {T} frames, {env.num_envs} envs")
  _compare(frames, kp_local, Kn, HL, HW, device)
  env.close()


def _to_dict(agent_cfg):
  from dataclasses import asdict
  return asdict(agent_cfg)


def _compare(frames, kp_local, Kn, HL, HW, device):
  """For each frame t, compare single-frame vs multi-frame (odometry-fused) Kabsch
  pos_err vs GT. Multi pools keypoints from [t-Wd..t] transformed into base_t."""
  from mjlab.tasks.velocity.mdp.field_keypoints import kabsch_se2

  map_xy = kp_local[..., :2]  # (K,2) in field frame
  Wd = 10  # fusion window (frames)
  T = len(frames)
  B = frames[0]["gt"].shape[0]

  def err_of(p_base_xy, w):
    # Kabsch base->field; pos_err vs gt (normalized -> meters).
    mxy = map_xy.unsqueeze(0).expand(p_base_xy.shape[0], -1, -1)
    yaw_r, t_r = kabsch_se2(p_base_xy, mxy, w)
    return yaw_r, t_r

  single_errs, multi_errs, single_vis, multi_vis = [], [], [], []
  uniq_single, uniq_multi = [], []
  for t in range(Wd, T):
    ft = frames[t]
    gt = ft["gt"]
    gx = gt[:, 0] * HL
    gy = gt[:, 1] * HW
    # single
    _, t_s = err_of(ft["p_base"], ft["w"])
    es = torch.sqrt((t_s[:, 0] - gx) ** 2 + (t_s[:, 1] - gy) ** 2)
    # multi: transform frames [t-Wd..t] keypoints into base_t via odometry.
    pooled_p, pooled_w = [], []
    bp_t, by_t = ft["base_pos"], ft["base_yaw"]
    for s in range(t - Wd, t + 1):
      fs = frames[s]
      # base_s point -> world -> base_t (perfect odometry oracle).
      cs, ss = torch.cos(fs["base_yaw"]), torch.sin(fs["base_yaw"])
      px, py = fs["p_base"][..., 0], fs["p_base"][..., 1]
      wx = cs.unsqueeze(1) * px - ss.unsqueeze(1) * py + fs["base_pos"][:, 0:1]
      wy = ss.unsqueeze(1) * px + cs.unsqueeze(1) * py + fs["base_pos"][:, 1:2]
      dx, dy = wx - bp_t[:, 0:1], wy - bp_t[:, 1:2]
      ct, st = torch.cos(-by_t), torch.sin(-by_t)
      bx = ct.unsqueeze(1) * dx - st.unsqueeze(1) * dy
      by = st.unsqueeze(1) * dx + ct.unsqueeze(1) * dy
      pooled_p.append(torch.stack([bx, by], dim=-1))
      pooled_w.append(fs["w"])
    pp = torch.cat(pooled_p, dim=1)
    pw = torch.cat(pooled_w, dim=1)
    # Unique map-keypoint coverage: how many DISTINCT keypoint ids were seen at
    # least once across the window (distinguishes genuine new geometry from
    # redundant re-observation of the same few points).
    ever = torch.zeros(B, Kn, device=device)
    for fw in pooled_w:
      ever = torch.maximum(ever, (fw > 0).float())
    uniq = ever.sum(1)
    uniq_single.append(ft["w"].gt(0).float().sum(1))
    uniq_multi.append(uniq)
    mm = map_xy.unsqueeze(0).expand(B, -1, -1).repeat(1, Wd + 1, 1)
    yaw_m, t_m = kabsch_se2(pp, mm, pw)
    em = torch.sqrt((t_m[:, 0] - gx) ** 2 + (t_m[:, 1] - gy) ** 2)
    single_errs.append(es)
    multi_errs.append(em)
    single_vis.append(ft["w"].sum(1))
    multi_vis.append(pw.sum(1))

  se = torch.cat(single_errs)
  me = torch.cat(multi_errs)
  print(f"RESULT single pos_err: mean={se.mean():.3f} median={se.median():.3f}")
  print(f"RESULT multi  pos_err: mean={me.mean():.3f} median={me.median():.3f}")
  print(f"RESULT improvement: {(1 - me.mean()/se.mean())*100:.1f}% lower")
  print(f"RESULT vis pts single={torch.cat(single_vis).mean():.1f} multi={torch.cat(multi_vis).mean():.1f}")
  print(f"RESULT UNIQUE kp single={torch.cat(uniq_single).mean():.1f} multi={torch.cat(uniq_multi).mean():.1f} (of {Kn})")


if __name__ == "__main__":
  try:
    main(sys.argv[1])
  except Exception:
    traceback.print_exc()
