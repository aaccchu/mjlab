from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.manipulation.mdp import camera_rgb
from mjlab.tasks.velocity.mdp.dribble_command import DribbleCommand
from mjlab.utils.buffers.circular_buffer import CircularBuffer
from mjlab.utils.lab_api.math import quat_apply, quat_inv, wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def _dribble_cmd(env: ManagerBasedRlEnv, command_name: str) -> DribbleCommand:
  return cast(DribbleCommand, env.command_manager.get_term(command_name))


def robot_to_ball(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Vector from robot root to ball, in base frame. Shape (B, 3)."""
  robot: Entity = env.scene[asset_cfg.name]
  command = _dribble_cmd(env, command_name)
  vec_w = command.ball_pos_w - robot.data.root_link_pos_w
  return quat_apply(quat_inv(robot.data.root_link_quat_w), vec_w)


def ball_to_target(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Vector from ball to target, in base frame. Shape (B, 3)."""
  robot: Entity = env.scene[asset_cfg.name]
  command = _dribble_cmd(env, command_name)
  vec_w = command.target_pos - command.ball_pos_w
  return quat_apply(quat_inv(robot.data.root_link_quat_w), vec_w)


def robot_to_target(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Vector from robot root to target, in base frame. Shape (B, 3).

  Vision-ablation variant of ``ball_to_target``: gives the policy the goal
  direction (legal for pure vision — a robot knows the goal bearing from field
  localization) WITHOUT leaking ball position. Unlike ``ball_to_target``
  (= target - ball), this never references the ball, so masking the GT ball
  terms truly removes all ground-truth ball knowledge from the actor obs.
  """
  robot: Entity = env.scene[asset_cfg.name]
  command = _dribble_cmd(env, command_name)
  vec_w = command.target_pos - robot.data.root_link_pos_w
  return quat_apply(quat_inv(robot.data.root_link_quat_w), vec_w)


def robot_field_pose(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Privileged GT self-localization: where the robot is ON THE FIELD. Shape (B,4).

  Returns [x/half_length, y/half_width, sin(yaw), cos(yaw)] in env-LOCAL field
  coords (origin subtracted). This is the v3g teacher signal: a robot that knows
  its field pose can derive the goal bearing itself, so we mask `robot_to_target`
  (which spoon-feeds goal direction) and force the policy to use THIS instead.
  In the distillation phase this GT term is curriculum-masked to 0, forcing the
  RGB CNN to recover field pose from the painted lines/goals (see v3g plan).

  Heading is the base yaw about world z (atan2 of the forward-axis xy), wrapped
  to (-pi, pi] then encoded as sin/cos so the network sees a continuous circle
  with no +-pi discontinuity.
  """
  robot: Entity = env.scene[asset_cfg.name]
  command = _dribble_cmd(env, command_name)
  origin = env.scene.env_origins  # (B, 3)
  local_xy = robot.data.root_link_pos_w[:, :2] - origin[:, :2]
  x_n = local_xy[:, 0] / command.cfg.half_length
  y_n = local_xy[:, 1] / command.cfg.half_width
  # Forward axis (body +x) in world frame -> yaw.
  fwd_w = quat_apply(
    robot.data.root_link_quat_w,
    torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3),
  )
  yaw = wrap_to_pi(torch.atan2(fwd_w[:, 1], fwd_w[:, 0]))
  return torch.stack([x_n, y_n, torch.sin(yaw), torch.cos(yaw)], dim=-1)


def ball_velocity_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Ball linear velocity in the robot base frame. Shape (B, 3)."""
  robot: Entity = env.scene[asset_cfg.name]
  command = _dribble_cmd(env, command_name)
  return quat_apply(quat_inv(robot.data.root_link_quat_w), command.ball_lin_vel_w)


# --- Spike A2-MOS92: GT gaze geometry (no camera) -------------------------
# Approximates the head-camera image coordinates of the ball from ground-truth
# geometry: horizontal angle from gaze yaw (heading + neck_yaw) and vertical
# angle from neck_pitch. Used to drive vision-centered gaze + search rewards
# before a real camera is wired in.

_FOV_H = 0.61  # half horizontal FoV (rad), ~35deg (matches Spike B finding).
_FOV_V = 0.50  # half vertical FoV (rad), ~28deg.


def _gaze_uv_visible(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return (u, v, visible): normalized image coords and visibility flag."""
  robot: Entity = env.scene[asset_cfg.name]
  command = _dribble_cmd(env, command_name)
  vec_w = command.ball_pos_w - robot.data.root_link_pos_w
  dist_xy = torch.norm(vec_w[:, :2], dim=-1).clamp(min=1e-3)

  # Gaze yaw = body heading + neck_yaw joint angle.
  ball_bearing_w = torch.atan2(vec_w[:, 1], vec_w[:, 0])
  heading = robot.data.heading_w
  neck_yaw = robot.data.joint_pos[:, asset_cfg.joint_ids][:, 0]
  gaze_yaw = heading + neck_yaw
  u_ang = wrap_to_pi(ball_bearing_w - gaze_yaw)
  u = u_ang / _FOV_H

  # Vertical: ball elevation angle relative to neck_pitch.
  elev = torch.atan2(vec_w[:, 2], dist_xy)
  neck_pitch = robot.data.joint_pos[:, asset_cfg.joint_ids][:, 1]
  v_ang = wrap_to_pi(elev - neck_pitch)
  v = v_ang / _FOV_V

  visible = ((u.abs() < 1.0) & (v.abs() < 1.0)).float()
  return u, v, visible


def ball_gaze_uv(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Ball image coords (u, v) + visibility flag. Shape (B, 3).

  When the ball is out of view, (u, v) are clamped to a sentinel on the frame
  edge (sign-preserving) so the policy still gets a bearing hint for search.
  """
  u, v, visible = _gaze_uv_visible(env, command_name, asset_cfg)
  # Clamp to [-2, 2] so out-of-view values stay bounded but keep their sign.
  u_obs = u.clamp(-2.0, 2.0)
  v_obs = v.clamp(-2.0, 2.0)
  return torch.stack([u_obs, v_obs, visible], dim=-1)


class StackedCameraRGB(ManagerTermBase):
  """N-frame RGB stack -> (B, N*C, H, W) for the self-localization CNN.

  A single front-facing RGB frame is ambiguous for self-loc (center-circle and
  side lines are left/right symmetric; the goal is a few pixels at distance).
  Stacking the last N frames gives the CNN inter-frame parallax: as the robot
  moves/turns, symmetric landmarks shift differently depending on where it
  actually is, which breaks the ambiguity.

  Stateful term: holds a per-env CircularBuffer. The obs manager calls
  ``reset(env_ids)`` on episode reset (zeroes those rows; the next append
  backfills them with the fresh post-reset frame), and detects this class as a
  reset-bearing term via ``hasattr(func, "reset")``.

  Append cadence: ``compute_group`` runs several times per env step (actor/critic
  forward), but a frame must be appended only ONCE per step. We can't see the
  manager's ``update_history`` flag from here, so we dedup on
  ``env.common_step_counter`` (incremented once per step, before obs compute) and
  append only when it advances. Repeated same-step calls return the current
  stacked buffer unchanged.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    num_frames: int,
    stride: int = 1,
  ):
    super().__init__(env)
    if num_frames < 1:
      raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if stride < 1:
      raise ValueError(f"stride must be >= 1, got {stride}")
    self._sensor_name = sensor_name
    self._n = num_frames
    # Append a frame only every `stride` env steps, so the N-frame stack spans
    # N*stride control steps. At 1 stride the stack is the last N steps (~0.08s
    # at 50Hz — too short to cover a head sweep); a larger stride lets the same
    # N frames span a ~1-2s active scan, capturing the DIFFERENT landmarks the
    # neck_yaw sweep brings into view. This is the temporal-memory lever for
    # pure-vision self-loc at the real 0.125m line spec (single frame can't see
    # enough markings from every pose; a scan integrated over time can).
    self._stride = stride
    self._buf: CircularBuffer | None = None
    self._last_step: int = -1
    self._steps_since_append: int = 0

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._buf is None:
      return
    batch_ids = None if isinstance(env_ids, slice) else env_ids
    self._buf.reset(batch_ids=batch_ids)
    # Force a re-append next call so reset rows get backfilled this step.
    self._last_step = -1
    self._steps_since_append = 0

  def __call__(
    self, env: ManagerBasedRlEnv, sensor_name: str, num_frames: int, stride: int = 1
  ) -> torch.Tensor:
    rgb = camera_rgb(env, sensor_name)  # (B, C, H, W), C=3
    if self._buf is None:
      self._buf = CircularBuffer(self._n, env.num_envs, env.device)
    step = int(env.common_step_counter)
    if step != self._last_step or not self._buf.is_initialized:
      # New env step. Append only every `stride`-th step so the N-frame stack
      # spans N*stride control steps (a head-sweep time window). Always append
      # on the first call after a reset (buffer uninitialized) so the freshly
      # reset rows get a real frame immediately.
      if (
        not self._buf.is_initialized
        or self._steps_since_append >= self._stride - 1
      ):
        self._buf.append(rgb)
        self._steps_since_append = 0
      else:
        self._steps_since_append += 1
      self._last_step = step
    buf = self._buf.buffer  # (B, N, C, H, W) chronological oldest->newest
    b, n, c, h, w = buf.shape
    return buf.reshape(b, n * c, h, w)


def keypoint_uv_label(
  env: ManagerBasedRlEnv,
  sensor_name: str = "head_cam",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """v4 EXP5f: TRAINING-ONLY supervision label for keypoint detection. (B, K*3).

  Per-frame projection of the known 3D field keypoints into the camera image:
  [uv_x(K), uv_y(K), vis(K)] flattened. Fed into a dedicated obs group
  ``keypoint_label`` that enters the rollout storage (so it aligns with each
  sampled transition) but is NOT listed in the actor/critic obs_groups — so the
  policy never sees the label, only KeypointAuxPPO's supervised loss consumes it.
  Labels are sim geometry (supervision, not a deployment input).
  """
  from mjlab.tasks.velocity.mdp.field_keypoints import (
    field_keypoints_3d,
    project_keypoints,
  )

  cam = env.scene[sensor_name]
  W, H = cam.cfg.width, cam.cfg.height
  fovy = cam.cfg.fovy if cam.cfg.fovy is not None else 45.0
  cam_idx = cam.camera_idx
  sd = env.sim.data
  cam_pos = sd.cam_xpos[:, cam_idx, :]
  cam_mat = sd.cam_xmat[:, cam_idx, :].reshape(-1, 3, 3)
  origin = env.scene.env_origins
  kp_local = field_keypoints_3d(env.device)
  kp_world = kp_local.unsqueeze(0) + origin.unsqueeze(1)
  uv, vis = project_keypoints(kp_world, cam_pos, cam_mat, fovy, W, H)  # (B,K,2),(B,K)
  # Zero invisible keypoints' uv so off-frame/behind-cam projections (can be huge)
  # never leak into the loss even if masking has an indexing slip. Layout:
  # interleaved [u0,v0,u1,v1,...] then [vis0,...] so a (-1,K,2) reshape aligns.
  vis_f = vis.float()
  uv = uv * vis_f.unsqueeze(-1)
  b, k = vis.shape
  return torch.cat([uv.reshape(b, k * 2), vis_f], dim=-1)  # (B, K*3)


def oracle_pose_belief(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str = "head_cam",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """v4 EXP6: GT-landmark ORACLE pose belief as an OBSERVATION (B, 6).

  The architectural flip vs EXP5: perception NO LONGER lives in the actor trunk as
  a 46-d action. Instead the geometry chain runs HERE and feeds the soccer policy a
  pose belief it consumes. Chain: project the known 3D field keypoints with TRUE
  visibility (the oracle — perfect DETECTION, not perfect pose), sample real metric
  depth at each visible pixel, lift to base frame (kinematic, no world pose), Kabsch
  vs the known map -> recovered (x, y, yaw). Output:
    [x_n, y_n, sin_yaw, cos_yaw, visible_frac, kabsch_residual]
  matching robot_field_pose's encoding so the policy sees a familiar pose, PLUS two
  belief-quality signals (how many points were visible, how well Kabsch fit) that a
  later active-perception policy can gate scanning on.

  Oracle boundary (codex red line): the policy never eats GT robot pose — only the
  belief that GEOMETRY recovers from GT detections, which degrades naturally when
  too few keypoints are visible. That degradation is the motivation for active
  turning. No perception gradient flows through the gait trunk (this is an obs, not
  an action), so it cannot corrupt locomotion.
  """
  from mjlab.tasks.velocity.mdp.field_keypoints import (
    field_keypoints_3d,
    project_keypoints,
  )

  cam = env.scene[sensor_name]
  W, H = cam.cfg.width, cam.cfg.height
  fovy = cam.cfg.fovy if cam.cfg.fovy is not None else 45.0
  cam_idx = cam.camera_idx
  sd = env.sim.data
  cam_pos = sd.cam_xpos[:, cam_idx, :]
  cam_mat = sd.cam_xmat[:, cam_idx, :].reshape(-1, 3, 3)
  origin = env.scene.env_origins
  kp_local = field_keypoints_3d(env.device)
  kp_world = kp_local.unsqueeze(0) + origin.unsqueeze(1)
  uv, vis = project_keypoints(kp_world, cam_pos, cam_mat, fovy, W, H)  # (B,K,2),(B,K)
  return _belief_from_uv(env, uv, vis, cam, command_name, asset_cfg, kp_local)


def _belief_from_uv(
  env: ManagerBasedRlEnv,
  uv: torch.Tensor,
  vis: torch.Tensor,
  cam,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  kp_local: torch.Tensor,
) -> torch.Tensor:
  """Shared geometry: (uv, vis) -> Kabsch pose belief (B, 6). Reused by oracle and
  (later) the learned detector. uv normalized [-1,1], vis (B,K) float/bool mask."""
  from mjlab.tasks.velocity.mdp.field_keypoints import (
    kabsch_se2,
    lift_pixels_to_world,
  )
  from mjlab.utils.lab_api.math import matrix_from_quat

  W, H = cam.cfg.width, cam.cfg.height
  fovy = cam.cfg.fovy if cam.cfg.fovy is not None else 45.0
  cam_idx = cam.camera_idx
  sd = env.sim.data
  cam_pos = sd.cam_xpos[:, cam_idx, :]
  cam_mat = sd.cam_xmat[:, cam_idx, :].reshape(-1, 3, 3)
  K = kp_local.shape[0]
  vis_f = vis.float()

  # Sample raw metric depth at each pixel (nearest).
  depth_raw = cam.data.depth  # (B, H, W, 1) meters
  u_pix = ((uv[..., 0] + 1.0) * 0.5 * (W - 1)).round().long().clamp(0, W - 1)
  v_pix = ((uv[..., 1] + 1.0) * 0.5 * (H - 1)).round().long().clamp(0, H - 1)
  bidx = torch.arange(uv.shape[0], device=env.device).unsqueeze(1).expand(-1, K)
  depth_at = depth_raw[bidx, v_pix, u_pix, 0]  # (B, K)

  pw = lift_pixels_to_world(uv, depth_at, cam_pos, cam_mat, fovy, W, H)
  robot: Entity = env.scene[asset_cfg.name]
  base_pos = robot.data.root_link_pos_w
  base_mat = matrix_from_quat(robot.data.root_link_quat_w)
  rel = pw - base_pos.unsqueeze(1)
  p_base = torch.einsum("bij,bkj->bki", base_mat.transpose(1, 2), rel)

  w = vis_f * ((depth_at > 0.05) & (depth_at < 30.0)).float()  # (B,K)
  map_xy = kp_local[..., :2].unsqueeze(0).expand(p_base.shape[0], -1, -1)
  yaw_r, t_r = kabsch_se2(p_base[..., :2], map_xy, w)

  command = _dribble_cmd(env, command_name)
  x_n = t_r[:, 0] / command.cfg.half_length
  y_n = t_r[:, 1] / command.cfg.half_width
  # Kabsch residual (mean weighted reprojection error in field meters); high when
  # few/degenerate points -> the belief is unreliable and scanning is warranted.
  cos, sin = torch.cos(yaw_r), torch.sin(yaw_r)
  Rx = cos.unsqueeze(1) * p_base[..., 0] - sin.unsqueeze(1) * p_base[..., 1]
  Ry = sin.unsqueeze(1) * p_base[..., 0] + cos.unsqueeze(1) * p_base[..., 1]
  pred_map = torch.stack([Rx + t_r[:, 0:1], Ry + t_r[:, 1:2]], dim=-1)
  resid = (torch.linalg.norm(pred_map - map_xy, dim=-1) * w).sum(1) / w.sum(1).clamp(min=1.0)
  vis_frac = w.sum(1) / float(K)
  return torch.stack([x_n, y_n, sin, cos, vis_frac, resid], dim=-1)  # (B, 6)


class FusedPoseBelief(ManagerTermBase):
  """v4 EXP8: TEMPORALLY-FUSED oracle pose belief -> (B, 7).

  EXP7 proved single-frame belief is stuck at ~4/23 visible keypoints (Kabsch
  underdetermined -> ~4m). EXP6+EXP7 diagnosis: the fix is active neck scanning
  (neck_yaw +-90deg raises coverage 5 -> 9-14) FUSED across frames via odometry so
  the scan's different views accumulate into one well-conditioned solve.

  Stateful term (mirrors StackedCameraRGB): a per-env CircularBuffer stores, for
  each of the last N appended frames, the visible keypoints in THAT frame's base
  frame plus that frame's world base pose (x, y, yaw). On call, every stored
  frame's keypoints are transformed via odometry into the CURRENT base frame,
  pooled, and Kabsch-fit against the known map. Output:
    [x_n, y_n, sin_yaw, cos_yaw, vis_frac_now, uniq_frac_fused, residual]
  uniq_frac_fused (distinct keypoints seen across the window) is the active-scan
  signal: it rises only when the neck sweep brings NEW landmarks into view.

  stride lets the N frames span a ~1-2s scan window (a single control step is too
  short to cover a head sweep). Odometry here is GT base pose (oracle); codex's
  noisy-oracle stage adds odometry noise later.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str = "head_cam",
    num_frames: int = 8,
    stride: int = 4,
  ):
    super().__init__(env)
    self._cmd = command_name
    self._sensor = sensor_name
    self._n = num_frames
    self._stride = stride
    self._buf: CircularBuffer | None = None
    self._last_step: int = -1
    # env is injected lazily (registered with env=None like StackedCameraRGB), so
    # defer all env-dependent state to the first __call__.
    self._kp_local = None
    self._K = 0
    self._last_uniq = None
    self._last_resid = None
    self._last_xy_n = None

  def _ensure_init(self, env: ManagerBasedRlEnv) -> None:
    if self._kp_local is not None:
      return
    from mjlab.tasks.velocity.mdp.field_keypoints import field_keypoints_3d

    self._kp_local = field_keypoints_3d(env.device)
    self._K = self._kp_local.shape[0]
    self._last_uniq = torch.zeros(env.num_envs, device=env.device)
    self._last_resid = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._buf is None:
      return
    # Zero ONLY the buffer rows of the envs being reset. Do NOT touch the global
    # append cadence (_last_step): it is keyed on common_step_counter divisibility
    # now, and clearing it on partial resets is exactly what stalled the buffer
    # before. The CircularBuffer.reset(batch_ids) clears just those env rows.
    batch_ids = None if isinstance(env_ids, slice) else env_ids
    self._buf.reset(batch_ids=batch_ids)

  def _collect_frame(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Current frame -> (B, K*3): per-keypoint [base_x, base_y, weight], plus the
    base pose is appended separately. Packs keypoints + pose into one tensor for
    the CircularBuffer: (B, K*3 + 3)."""
    from mjlab.tasks.velocity.mdp.field_keypoints import project_keypoints
    from mjlab.utils.lab_api.math import matrix_from_quat

    cam = env.scene[self._sensor]
    W, H = cam.cfg.width, cam.cfg.height
    fovy = cam.cfg.fovy if cam.cfg.fovy is not None else 45.0
    cam_idx = cam.camera_idx
    sd = env.sim.data
    cam_pos = sd.cam_xpos[:, cam_idx, :]
    cam_mat = sd.cam_xmat[:, cam_idx, :].reshape(-1, 3, 3)
    origin = env.scene.env_origins
    kp_world = self._kp_local.unsqueeze(0) + origin.unsqueeze(1)
    uv, vis = project_keypoints(kp_world, cam_pos, cam_mat, fovy, W, H)
    B = uv.shape[0]
    u_pix = ((uv[..., 0] + 1.0) * 0.5 * (W - 1)).round().long().clamp(0, W - 1)
    v_pix = ((uv[..., 1] + 1.0) * 0.5 * (H - 1)).round().long().clamp(0, H - 1)
    bidx = torch.arange(B, device=env.device).unsqueeze(1).expand(-1, self._K)
    depth_at = cam.data.depth[bidx, v_pix, u_pix, 0]
    from mjlab.tasks.velocity.mdp.field_keypoints import lift_pixels_to_world

    pw = lift_pixels_to_world(uv, depth_at, cam_pos, cam_mat, fovy, W, H)
    robot = env.scene["robot"]
    base_pos = robot.data.root_link_pos_w
    base_mat = matrix_from_quat(robot.data.root_link_quat_w)
    rel = pw - base_pos.unsqueeze(1)
    p_base = torch.einsum("bij,bkj->bki", base_mat.transpose(1, 2), rel)
    w = vis.float() * ((depth_at > 0.05) & (depth_at < 30.0)).float()  # (B,K)
    base_yaw = torch.atan2(base_mat[:, 1, 0], base_mat[:, 0, 0])
    kp_pack = torch.cat(
      [p_base[..., 0], p_base[..., 1], w], dim=1
    )  # (B, 3K): x[K], y[K], w[K]
    pose = torch.stack([base_pos[:, 0], base_pos[:, 1], base_yaw], dim=-1)  # (B,3)
    return torch.cat([kp_pack, pose], dim=1)  # (B, 3K+3)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str = "head_cam",
    num_frames: int = 8,
    stride: int = 4,
  ) -> torch.Tensor:
    from mjlab.tasks.velocity.mdp.field_keypoints import kabsch_se2

    self._ensure_init(env)
    frame = self._collect_frame(env)  # (B, 3K+3)
    if self._buf is None:
      self._buf = CircularBuffer(self._n, env.num_envs, env.device)
    step = int(env.common_step_counter)
    # Append cadence keyed on the GLOBAL step counter's divisibility by stride, NOT
    # a reset-clearable accumulator. Earlier the cadence used self._steps_since_append
    # which reset() zeroed on every PARTIAL env reset; with hundreds of envs some env
    # terminates almost every step, so the counter never reached stride-1 and the
    # buffer stopped appending fresh frames (fusion silently degraded to ~single
    # frame). Divisibility on common_step_counter is reset-independent, so the
    # temporal window keeps rolling regardless of per-env resets.
    if not self._buf.is_initialized:
      self._buf.append(frame)
      self._last_step = step
    elif step != self._last_step:
      if step % self._stride == 0:
        self._buf.append(frame)
      self._last_step = step
    buf = self._buf.buffer  # (B, N, 3K+3) oldest->newest
    B, N, _ = buf.shape
    K = self._K

    # Current (newest) frame's base pose is the target frame for odometry.
    cur_pose = buf[:, -1, 3 * K:]  # (B,3) x,y,yaw
    cx, cy, cyaw = cur_pose[:, 0], cur_pose[:, 1], cur_pose[:, 2]
    ct, st = torch.cos(-cyaw), torch.sin(-cyaw)

    pooled_x, pooled_y, pooled_w = [], [], []
    ever = torch.zeros(B, K, device=env.device)
    for i in range(N):
      fx = buf[:, i, 0:K]
      fy = buf[:, i, K:2 * K]
      fw = buf[:, i, 2 * K:3 * K]
      fyaw = buf[:, i, 3 * K + 2]
      fpx, fpy = buf[:, i, 3 * K], buf[:, i, 3 * K + 1]
      # frame-i base point -> world -> current base.
      cf, sf = torch.cos(fyaw), torch.sin(fyaw)
      wx = cf.unsqueeze(1) * fx - sf.unsqueeze(1) * fy + fpx.unsqueeze(1)
      wy = sf.unsqueeze(1) * fx + cf.unsqueeze(1) * fy + fpy.unsqueeze(1)
      dx, dy = wx - cx.unsqueeze(1), wy - cy.unsqueeze(1)
      bx = ct.unsqueeze(1) * dx - st.unsqueeze(1) * dy
      by = st.unsqueeze(1) * dx + ct.unsqueeze(1) * dy
      pooled_x.append(bx)
      pooled_y.append(by)
      pooled_w.append(fw)
      ever = torch.maximum(ever, (fw > 0).float())
    px = torch.cat(pooled_x, dim=1)  # (B, N*K)
    py = torch.cat(pooled_y, dim=1)
    pw = torch.cat(pooled_w, dim=1)
    p_base_xy = torch.stack([px, py], dim=-1)  # (B, N*K, 2)
    map_xy = self._kp_local[..., :2].unsqueeze(0).expand(B, -1, -1).repeat(1, N, 1)
    yaw_r, t_r = kabsch_se2(p_base_xy, map_xy, pw)

    command = _dribble_cmd(env, command_name)
    x_n = t_r[:, 0] / command.cfg.half_length
    y_n = t_r[:, 1] / command.cfg.half_width
    cosr, sinr = torch.cos(yaw_r), torch.sin(yaw_r)
    Rx = cosr.unsqueeze(1) * p_base_xy[..., 0] - sinr.unsqueeze(1) * p_base_xy[..., 1]
    Ry = sinr.unsqueeze(1) * p_base_xy[..., 0] + cosr.unsqueeze(1) * p_base_xy[..., 1]
    pred = torch.stack([Rx + t_r[:, 0:1], Ry + t_r[:, 1:2]], dim=-1)
    resid = (torch.linalg.norm(pred - map_xy, dim=-1) * pw).sum(1) / pw.sum(1).clamp(min=1.0)
    vis_now = buf[:, -1, 2 * K:3 * K].sum(1) / float(K)
    uniq_frac = ever.sum(1) / float(K)
    # Cache for the active_scan_coverage reward (read, don't recompute geometry).
    self._last_uniq = uniq_frac.detach()
    self._last_resid = resid.detach()
    self._last_xy_n = torch.stack([x_n, y_n], dim=-1).detach()  # (B,2) for monitor
    return torch.stack(
      [x_n, y_n, sinr, cosr, vis_now, uniq_frac, resid], dim=-1
    )  # (B,7)


class EkfPoseBelief(FusedPoseBelief):
  """v4 line-A R1: RECURSIVE SE2-EKF self-localization, replacing the point-cloud-
  pooling Kabsch of FusedPoseBelief. EXP12 offline (same rollout, same per-landmark
  depth obs) showed the EKF beats pooling by 55% (1.21m vs 2.68m mean) — pooling
  N frames adds no constraint when the robot walks one way (same ~4 points repeat),
  whereas the EKF accumulates each sparse per-frame landmark observation recursively
  into a pose distribution, so even underdetermined single frames converge.

  State (per env): mu (B,3) [x,y,yaw] world pose, Sigma (B,3,3) covariance.
  Predict: world-frame pose += GT odometry delta, inflate covariance by Q.
  Update: sequential per-visible-landmark EKF correction against the known map.
  Reset: the reset envs go back to a wide prior (field center, large covariance).

  Output (B,7), same layout as FusedPoseBelief so the actor obs dim is unchanged and
  EXP11 weights bootstrap directly:
    [x_n, y_n, sin_yaw, cos_yaw, vis_frac_now, uniq_frac, uncertainty]
  uniq_frac is the per-step visible fraction (active-scan signal); uncertainty is the
  normalized covariance trace (replaces the pooling residual — a real belief spread).
  The _ekf_step math is the one validated in scripts/exp12_offline_ekf.py.
  """

  def __init__(self, env, command_name, sensor_name="head_cam", num_frames=8,
               stride=4, q_pos=0.02, q_yaw=0.02, r_meas=0.10):
    super().__init__(env, command_name, sensor_name, num_frames, stride)
    self._q_pos = q_pos
    self._q_yaw = q_yaw
    self._r_meas = r_meas
    self._mu = None       # (B,3)
    self._Sigma = None    # (B,3,3)
    self._prev_gt = None  # (B,3) previous GT world base pose for odometry delta
    self._map_xy = None   # (K,2)

  def _ensure_init(self, env: ManagerBasedRlEnv) -> None:
    super()._ensure_init(env)
    if self._mu is not None:
      return
    B, dev = env.num_envs, env.device
    self._map_xy = self._kp_local[:, :2].clone()  # (K,2) world landmark xy
    self._mu = torch.zeros(B, 3, device=dev)
    self._Sigma = torch.diag(
      torch.tensor([25.0, 25.0, 9.0], device=dev)
    ).repeat(B, 1, 1)
    self._prev_gt = None
    self._Q = torch.diag(
      torch.tensor([self._q_pos, self._q_pos, self._q_yaw], device=dev)
    )
    self._R = torch.eye(2, device=dev) * self._r_meas

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._mu is None:
      return
    dev = self._mu.device
    if isinstance(env_ids, slice):
      ids = torch.arange(self._mu.shape[0], device=dev)
    else:
      ids = env_ids
    # Reset envs go back to the wide prior; clear their odometry anchor so the
    # first post-reset frame does not apply a bogus cross-episode delta.
    self._mu[ids] = 0.0
    self._Sigma[ids] = torch.diag(
      torch.tensor([25.0, 25.0, 9.0], device=dev)
    )
    if self._prev_gt is not None:
      self._prev_gt[ids] = float("nan")  # NaN marks "no valid previous pose"

  def __call__(self, env, command_name, sensor_name="head_cam", num_frames=8,
               stride=4) -> torch.Tensor:
    self._ensure_init(env)
    frame = self._collect_frame(env)  # (B, 3K+3)
    K, B, dev = self._K, env.num_envs, env.device
    z_x, z_y = frame[:, 0:K], frame[:, K:2 * K]
    z_w = frame[:, 2 * K:3 * K]               # (B,K) visibility weight
    z_xy = torch.stack([z_x, z_y], dim=-1)    # (B,K,2) base-frame landmark obs
    gt_world = frame[:, 3 * K:]               # (B,3) GT base pose (odometry source)

    # --- Predict: advance pose by GT odometry delta (NaN-safe for first frame). ---
    if self._prev_gt is None:
      self._prev_gt = gt_world.clone()
      delta = torch.zeros(B, 3, device=dev)
    else:
      delta = gt_world - self._prev_gt
      delta[:, 2] = torch.atan2(torch.sin(delta[:, 2]), torch.cos(delta[:, 2]))
      # Envs just reset have NaN prev -> zero delta (start fresh from prior).
      delta = torch.where(torch.isnan(delta), torch.zeros_like(delta), delta)
      self._prev_gt = gt_world.clone()
    self._mu = self._mu + delta
    self._mu[:, 2] = torch.atan2(torch.sin(self._mu[:, 2]), torch.cos(self._mu[:, 2]))
    self._Sigma = self._Sigma + self._Q.unsqueeze(0)

    # --- Update: sequential per-landmark EKF correction. ---
    for k in range(K):
      w = z_w[:, k]
      if float(w.max()) <= 0.0:
        continue
      x, y, yaw = self._mu[:, 0], self._mu[:, 1], self._mu[:, 2]
      c, s = torch.cos(yaw), torch.sin(yaw)
      dx = self._map_xy[k, 0] - x
      dy = self._map_xy[k, 1] - y
      hx = c * dx + s * dy
      hy = -s * dx + c * dy
      H = torch.zeros(B, 2, 3, device=dev)
      H[:, 0, 0] = -c
      H[:, 0, 1] = -s
      H[:, 0, 2] = hy
      H[:, 1, 0] = s
      H[:, 1, 1] = -c
      H[:, 1, 2] = -hx
      innov = torch.stack([z_xy[:, k, 0] - hx, z_xy[:, k, 1] - hy], dim=-1)
      Rk = self._R / w.clamp(min=1e-3).unsqueeze(-1).unsqueeze(-1)
      S = H @ self._Sigma @ H.transpose(1, 2) + Rk
      Kg = self._Sigma @ H.transpose(1, 2) @ torch.linalg.inv(S)
      upd = (Kg @ innov.unsqueeze(-1)).squeeze(-1)
      m = (w > 0).float().unsqueeze(-1)
      self._mu = self._mu + m * upd
      eye3 = torch.eye(3, device=dev).unsqueeze(0)
      Sig_new = (eye3 - Kg @ H) @ self._Sigma
      self._Sigma = self._Sigma + m.unsqueeze(-1) * (Sig_new - self._Sigma)
      self._mu[:, 2] = torch.atan2(
        torch.sin(self._mu[:, 2]), torch.cos(self._mu[:, 2])
      )

    # --- Pack output (B,7), same layout as FusedPoseBelief. ---
    command = _dribble_cmd(env, command_name)
    x_n = self._mu[:, 0] / command.cfg.half_length
    y_n = self._mu[:, 1] / command.cfg.half_width
    yaw = self._mu[:, 2]
    vis_now = z_w.sum(1) / float(K)
    uniq_frac = (z_w > 0).float().sum(1) / float(K)
    # Normalized covariance trace (x,y) as belief uncertainty, ~[0,1] after clamp.
    uncertainty = (self._Sigma[:, 0, 0] + self._Sigma[:, 1, 1]).clamp(0, 50.0) / 50.0
    self._last_uniq = uniq_frac.detach()
    self._last_resid = uncertainty.detach()
    self._last_xy_n = torch.stack([x_n, y_n], dim=-1).detach()
    return torch.stack(
      [x_n, y_n, torch.sin(yaw), torch.cos(yaw), vis_now, uniq_frac, uncertainty],
      dim=-1,
    )  # (B,7)
