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
