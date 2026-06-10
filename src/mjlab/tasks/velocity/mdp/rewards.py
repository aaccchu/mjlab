from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.dribble_command import DribbleCommand
from mjlab.tasks.velocity.mdp.observations import _gaze_uv_visible, robot_field_pose
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_NECK_YAW_CFG = SceneEntityCfg("robot", joint_names=("neck_yaw",))


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


class upright:
  """Reward for keeping the base upright.

  Without ``terrain_sensor_names``, penalizes tilt relative to world up (correct for
  flat ground).

  With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
      body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
      body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    if terrain_sensor_names is not None:
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)  # [B, 3]
      # Project terrain normal into body frame. When aligned with the terrain surface
      # this should be (0, 0, 1); XY measures tilt.
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    else:
      gravity_w = asset.data.gravity_vec_w  # [3]
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
      body_quat_w[:, :3]
    )
    body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      # Terrain normal (magenta).
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      # Body up (orange).
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.sum(dim=-1).float()


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_clearance requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights  # [B, F]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
  delta = torch.abs(foot_height - target_height)  # [B, F]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def _dribble_cmd(env: ManagerBasedRlEnv, command_name: str) -> DribbleCommand:
  return cast(DribbleCommand, env.command_manager.get_term(command_name))


def dribble_approach(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Dense reward for the robot getting close to the ball (ground-plane xy)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = _dribble_cmd(env, command_name)
  robot_xy = asset.data.root_link_pos_w[:, :2]
  ball_xy = command.ball_pos_w[:, :2]
  error = torch.sum(torch.square(robot_xy - ball_xy), dim=-1)
  return torch.exp(-error / std**2)


def dribble_ball_to_target(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Dominant task reward: ball close to the target (ground-plane xy)."""
  command = _dribble_cmd(env, command_name)
  ball_xy = command.ball_pos_w[:, :2]
  target_xy = command.target_pos[:, :2]
  error = torch.sum(torch.square(target_xy - ball_xy), dim=-1)
  return torch.exp(-error / std**2)


def dribble_ball_velocity_to_target(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Reward ball velocity projected onto the ball->target direction (>=0)."""
  command = _dribble_cmd(env, command_name)
  ball_xy = command.ball_pos_w[:, :2]
  target_xy = command.target_pos[:, :2]
  ball_vel_xy = command.ball_lin_vel_w[:, :2]
  to_target = target_xy - ball_xy
  dir_to_target = to_target / (torch.norm(to_target, dim=-1, keepdim=True) + 1e-6)
  projected = torch.sum(ball_vel_xy * dir_to_target, dim=-1)
  return projected.clamp_min(0.0)


def dribble_success_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Sparse bonus while the ball is within the success threshold of the target."""
  command = _dribble_cmd(env, command_name)
  return command.metrics["at_goal"]


def dribble_kick_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
) -> torch.Tensor:
  """Reward for foot-ball contact (encourage using feet to hit ball)."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  data = contact_sensor.data
  assert data.found is not None
  return (data.found.sum(dim=-1) > 0).float()


def dribble_kick_impulse(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  speed_threshold: float = 0.0,
) -> torch.Tensor:
  """v4 line-B B2: reward the QUALITY of a foot-ball kick, not just that contact
  happened. At a foot-ball contact step, reward the ball speed projected onto the
  ball->goal direction (clamped >=0). A gentle tap leaves the ball slow (low
  reward); a goalward burst gets the full reward — so the policy learns to strike
  hard toward the goal at the contact moment, which the binary dribble_kick_contact
  cannot teach. Goal aim point mirrors goal_progress (goal line in +x at the ball's
  clamped y).

  speed_threshold (m/s): only reward when the goalward ball speed EXCEEDS this, i.e.
  gate out "gentle pushing". A per-step projected-speed reward without a threshold
  has a known local optimum where the policy learns to keep nudging the ball slowly
  (continuous contact pays more in integral than one hard strike that leaves the
  foot). EXP14 plateaued at goal_rate~0.11 with ball_speed stuck ~0.39 — the classic
  symptom. Setting the threshold above the nudge speed forces a real strike to score.
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  data = contact_sensor.data
  assert data.found is not None
  in_contact = (data.found.sum(dim=-1) > 0).float()
  command = _dribble_cmd(env, command_name)
  ball_xy = command.ball_pos_w[:, :2]
  ball_vel_xy = command.ball_lin_vel_w[:, :2]
  center_xy = env.scene.env_origins[:, :2]
  goal_x = center_xy[:, 0] + command.cfg.goal_line_x - command.cfg.field_margin
  goal_y = center_xy[:, 1] + torch.clamp(
    ball_xy[:, 1] - center_xy[:, 1],
    -command.cfg.goal_half_width,
    command.cfg.goal_half_width,
  )
  goal_xy = torch.stack([goal_x, goal_y], dim=-1)
  to_goal = goal_xy - ball_xy
  dir_to_goal = to_goal / (torch.norm(to_goal, dim=-1, keepdim=True) + 1e-6)
  projected = torch.sum(ball_vel_xy * dir_to_goal, dim=-1)
  # Gate out gentle pushing: zero reward unless the goalward speed clears threshold.
  projected = torch.where(
    projected > speed_threshold, projected, torch.zeros_like(projected)
  )
  return in_contact * projected


def kick_lateral_alignment(
  env: ManagerBasedRlEnv,
  command_name: str,
  x_min: float = 0.15,
  x_max: float = 1.0,
  std: float = 0.15,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """v4 EXP20: reward squaring up to the ball before striking (lateral alignment).

  The 0.32 shot-rate plateau decomposes as "kicks hard (peak 3.2 m/s) but the
  ball lands scattered" — and a major scatter source is striking with the ball
  laterally offset from the foot line (|y_b| large in the base frame), which
  sends the ball off-axis. codex C21l2 measured a 13x contact-window gain from a
  single lateral-alignment term, and our kick_research doc lists "惩罚 |y_b| 把
  球摆正脚前" as an unplayed card.

  Active only when the ball is IN FRONT in the kick approach band
  (x_b in [x_min, x_max]): rewards exp(-(y_b/std)^2) so the policy lines its
  body up so the ball sits on the foot line before contact. Zero when the ball
  is behind, far, or to the side (approach/search phases are not constrained).
  """
  robot: Entity = env.scene[asset_cfg.name]
  command = _dribble_cmd(env, command_name)
  vec_w = command.ball_pos_w - robot.data.root_link_pos_w
  ball_b = quat_apply_inverse(robot.data.root_link_quat_w, vec_w)
  x_b, y_b = ball_b[:, 0], ball_b[:, 1]
  in_band = ((x_b > x_min) & (x_b < x_max)).float()
  aligned = torch.exp(-((y_b / std) ** 2))
  env.extras["log"]["Metrics/kick_lat_align"] = (in_band * aligned).mean()
  return in_band * aligned


def support_foot_plant(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  ground_sensor_name: str,
  std: float = 0.10,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """v4 EXP22: reward a PLANTED support foot next to the ball at kick contact.

  EXP21 fixed the finish (SHORT 44.6%->23.3%) but the shot rate stalled because
  each touch advances the ball too little (median 9 kicks/episode; mid-field
  kick speed even dipped). codex C21cc measured the strongest single predictor
  of an effective contact: support_foot_dist <= 0.20 m gives a 7.86x lift —
  i.e. the classic human-football fundamental "plant your support foot beside
  the ball". A planted close support foot gives the swing leg a stable base so
  the kick transfers momentum instead of nudging.

  Fires ONLY at foot-ball contact steps (same gating as dribble_kick_impulse):
  reward = exp(-(d_support/std)^2) where d_support is the xy distance from the
  ball to the OTHER foot (the one not touching the ball), provided that foot is
  on the ground. Sparse and well-gated so it cannot create a hold-the-ball
  attractor: no contact, no reward.

  asset_cfg.site_names must be (left_foot, right_foot).
  """
  robot: Entity = env.scene[asset_cfg.name]
  command = _dribble_cmd(env, command_name)
  ball_contact: ContactSensor = env.scene[sensor_name]
  ground: ContactSensor = env.scene[ground_sensor_name]
  assert ball_contact.data.found is not None
  assert ground.data.found is not None

  # Both sensors share the same primary pattern ^(Rfoot|Lfoot)$ with num_slots=1,
  # so found is [B, 2] with IDENTICAL per-foot slot order — slot i in fb and fg
  # refer to the same foot. Likewise asset_cfg.site_names=(left_foot, right_foot)
  # gives foot positions; we don't need absolute L/R, only "kicker vs support",
  # and we resolve the kicker as the ball-contacting (else nearer) foot.
  fb = ball_contact.data.found > 0  # [B, 2]
  fg = ground.data.found > 0  # [B, 2]
  any_contact = fb.any(dim=-1)

  # NOTE: sensor slot order comes from the subtree pattern match; site order from
  # site_names. Both enumerate the two feet but possibly in different L/R order.
  # We avoid depending on that: kicker selection uses ball distance (geometry),
  # ground-planted check uses the sensor's own slots reduced with the matching
  # distance-ordering trick below.
  foot_pos = robot.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, 2, 2]
  ball_xy = command.ball_pos_w[:, :2].unsqueeze(1)  # [B, 1, 2]
  d = torch.norm(foot_pos - ball_xy, dim=-1)  # [B, 2] per-site dist to ball

  # Kicker = nearer foot at a contact step (the foot touching the ball is the
  # near one); support = the farther foot. This sidesteps slot-vs-site ordering.
  nearer_is_0 = d[:, 0] <= d[:, 1]
  support_d = torch.where(nearer_is_0, d[:, 1], d[:, 0])

  # Support planted: at least one ground contact beyond the kicker's. With two
  # feet and the kicker mid-kick (off the ground or on the ball), requiring BOTH
  # would be too strict; requiring >=1 ground contact captures "planted base".
  planted = fg.any(dim=-1)

  reward = any_contact.float() * planted.float() * torch.exp(-((support_d / std) ** 2))
  env.extras["log"]["Metrics/support_plant"] = reward.mean()
  env.extras["log"]["Metrics/support_dist_at_kick"] = torch.where(
    any_contact, support_d, torch.zeros_like(support_d)
  ).sum() / any_contact.float().sum().clamp(min=1.0)
  return reward


def gaze_at_ball(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 2.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward waist_yaw aligning with ball bearing in body frame.

  Encourages the robot to rotate its torso toward the ball via waist_yaw,
  which would point a head-mounted camera at the ball.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = _dribble_cmd(env, command_name)
  vec_w = command.ball_pos_w - asset.data.root_link_pos_w
  ball_b = quat_apply_inverse(asset.data.root_link_quat_w, vec_w)
  ball_bearing = torch.atan2(ball_b[:, 1], ball_b[:, 0])
  waist_yaw = asset.data.joint_pos[:, asset_cfg.joint_ids][:, 0]
  angle_error = torch.abs(waist_yaw - ball_bearing)
  return torch.exp(-std * angle_error)


def goal_scored_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Sparse one-shot reward at the step the ball crosses into the goal mouth."""
  command = _dribble_cmd(env, command_name)
  return command.newly_scored


def out_of_bounds_penalty(
  env: ManagerBasedRlEnv,
  term_name: str = "out_of_field_bounds",
) -> torch.Tensor:
  """v4 EXP17: one-shot penalty at the step the robot is flagged out of bounds.

  Register with a NEGATIVE weight. Returns a per-env 0/1 mask of the OOB
  termination this step (read from the termination manager, which the env
  computes BEFORE rewards). Pair this with ``out_of_field_bounds`` set to
  ``time_out=False`` (a real termination, not a truncation): the original
  config marked OOB as ``time_out=True``, so PPO bootstrapped ``gamma * V`` at
  the OOB step (rsl_rl ppo.py) — telling the policy "leaving the field is as
  valuable as continuing", i.e. OOB went unpenalized. Flipping to
  ``time_out=False`` stops the bootstrap; this term adds the explicit cost so
  the policy first *feels* a price for crossing the line.

  Because rewards are dt-scaled (~0.02 s/step), the per-step value here is
  multiplied by weight*dt; size the weight so weight*dt is comparable to a
  meaningful fraction of the goal bonus (goal_scored weight 5, also a one-shot).
  """
  term = env.termination_manager.get_term(term_name)
  env.extras["log"]["Metrics/oob_penalty_frac"] = term.float().mean()
  return term.float()


def soft_boundary_penalty(
  env: ManagerBasedRlEnv,
  half_length: float = 11.0,
  half_width: float = 7.0,
  soft_margin: float = 1.5,
  goal_corridor_half_width: float = 0.0,
  goal_buffer: float = 0.0,
  corridor_exempt_x: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """v4 EXP18: dense per-step penalty for being inside the soft boundary band.

  Register with a NEGATIVE weight. EXP17 proved the ``time_out`` fix is correct
  (OOB 0.345->0.262 without collapsing the shot rate) but the one-shot
  ``out_of_bounds_penalty`` fires on almost no steps (oob_penalty_frac ~2e-4) —
  far too sparse to outweigh the ~3.2 peak-kick payoff that pulls the robot
  goalward over the line. This term gives a CONTINUOUS gradient: it returns the
  positive depth (m) by which the root has entered the soft band that starts
  ``soft_margin`` inside each hard OOB line, so the policy feels the cost growing
  every step it lingers near the edge — long before it crosses.

  The band geometry mirrors ``out_of_field_bounds`` so the soft region sits just
  inside the hard line, including the goal-mouth corridor: inside
  ``|y| < goal_corridor_half_width`` the +/-x soft line is pushed out by
  ``goal_buffer`` so shooting into the mouth is not softly penalized.

  corridor_exempt_x (v4 EXP21): the corridor merely SHIFTS the x soft band out
  by goal_buffer, so a robot finishing at x~10.3 still pays depth_x — EXP20b
  forensics showed this makes goalward pushing in the mouth a NET LOSS
  (-0.036/step vs +0.016 mid-field) and produced the dominant SHORT miss (44.6%:
  ball stops at x~10.5, 0.5 m short). With this flag, inside the corridor the x
  component is fully exempt (depth_x := 0); the y component stays, so drifting
  sideways out of the mouth is still discouraged.
  """
  asset: Entity = env.scene[asset_cfg.name]
  root_xy = asset.data.root_link_pos_w[:, :2]
  abs_x = root_xy[:, 0].abs()
  abs_y = root_xy[:, 1].abs()

  hard_x = max(0.0, half_length)
  if goal_corridor_half_width > 0.0:
    in_corridor = abs_y < goal_corridor_half_width
    hard_x_t = torch.where(
      in_corridor,
      torch.full_like(abs_x, half_length + goal_buffer),
      torch.full_like(abs_x, hard_x),
    )
  else:
    in_corridor = torch.zeros_like(abs_x, dtype=torch.bool)
    hard_x_t = torch.full_like(abs_x, hard_x)

  # Depth into the soft band (>=0); zero when comfortably inside.
  depth_x = (abs_x - (hard_x_t - soft_margin)).clamp(min=0.0)
  if corridor_exempt_x:
    depth_x = torch.where(in_corridor, torch.zeros_like(depth_x), depth_x)
  depth_y = (abs_y - (half_width - soft_margin)).clamp(min=0.0)
  depth = depth_x + depth_y
  env.extras["log"]["Metrics/soft_boundary_depth"] = depth.mean()
  return depth


def velocity_toward_boundary_penalty(
  env: ManagerBasedRlEnv,
  half_length: float = 11.0,
  half_width: float = 7.0,
  soft_margin: float = 1.5,
  goal_corridor_half_width: float = 0.0,
  goal_buffer: float = 0.0,
  corridor_exempt_x: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """v4 EXP18: penalize outward root velocity only when near a boundary.

  Register with a NEGATIVE weight. Complements ``soft_boundary_penalty``: the
  depth term punishes *being* near the edge, this one punishes *moving toward*
  it, which is exactly the "charge goalward and over-run the line after a hard
  kick" behavior (robot_to_ball rose to 1.12 in EXP17 as the policy chased
  fast-moving balls off the field). Returns the outward speed component (m/s,
  >=0) summed over x and y, gated to the soft band so mid-field motion (chasing
  the ball) is never penalized — only motion that is both near a line AND
  heading out of bounds.

  corridor_exempt_x (v4 EXP21): the corridor merely shifts the x band out by
  goal_buffer, so "sprint at the goal mouth to finish" was still billed as
  "sprint at the boundary" (EXP20b forensics: net -0.036/step while finishing).
  With this flag the x velocity component is fully exempt inside the corridor;
  the y component stays so sliding sideways out of the mouth is still penalized.
  """
  asset: Entity = env.scene[asset_cfg.name]
  root_xy = asset.data.root_link_pos_w[:, :2]
  root_vxy = asset.data.root_link_lin_vel_w[:, :2]
  abs_x = root_xy[:, 0].abs()
  abs_y = root_xy[:, 1].abs()

  hard_x = max(0.0, half_length)
  if goal_corridor_half_width > 0.0:
    in_corridor = abs_y < goal_corridor_half_width
    hard_x_t = torch.where(
      in_corridor,
      torch.full_like(abs_x, half_length + goal_buffer),
      torch.full_like(abs_x, hard_x),
    )
  else:
    in_corridor = torch.zeros_like(abs_x, dtype=torch.bool)
    hard_x_t = torch.full_like(abs_x, hard_x)

  # Outward velocity = velocity projected onto the sign of the position (so it is
  # positive when moving away from center), clamped >=0.
  v_out_x = (root_vxy[:, 0] * torch.sign(root_xy[:, 0])).clamp(min=0.0)
  v_out_y = (root_vxy[:, 1] * torch.sign(root_xy[:, 1])).clamp(min=0.0)

  near_x = abs_x > (hard_x_t - soft_margin)
  if corridor_exempt_x:
    near_x = near_x & ~in_corridor
  near_y = abs_y > (half_width - soft_margin)
  penalty = v_out_x * near_x.float() + v_out_y * near_y.float()
  env.extras["log"]["Metrics/vel_toward_boundary"] = penalty.mean()
  return penalty


def goal_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Dense reward: ball velocity projected onto the ball->goal-line direction.

  Drives the ball toward the +x goal mouth. Unlike dribble_ball_velocity (which
  targets an arbitrary point), this always points goalward so the policy learns
  to shoot/dribble toward the goal.
  """
  command = _dribble_cmd(env, command_name)
  ball_xy = command.ball_pos_w[:, :2]
  ball_vel_xy = command.ball_lin_vel_w[:, :2]
  center_xy = env.scene.env_origins[:, :2]
  # Goal aim point: goal line in +x at the ball's current y (clamped to mouth).
  goal_x = center_xy[:, 0] + command.cfg.goal_line_x - command.cfg.field_margin
  goal_y = center_xy[:, 1] + torch.clamp(
    ball_xy[:, 1] - center_xy[:, 1],
    -command.cfg.goal_half_width,
    command.cfg.goal_half_width,
  )
  goal_xy = torch.stack([goal_x, goal_y], dim=-1)
  to_goal = goal_xy - ball_xy
  dir_to_goal = to_goal / (torch.norm(to_goal, dim=-1, keepdim=True) + 1e-6)
  projected = torch.sum(ball_vel_xy * dir_to_goal, dim=-1)
  return projected.clamp_min(0.0)


def approach_delta(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward = reduction in distance to ball since last step.

  Provides constant-magnitude gradient signal at any distance, unlike
  exp(-dist²) which vanishes beyond a few meters.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = _dribble_cmd(env, command_name)
  robot_xy = asset.data.root_link_pos_w[:, :2]
  ball_xy = command.ball_pos_w[:, :2]
  dist = torch.norm(robot_xy - ball_xy, dim=-1)
  delta = command._prev_robot_to_ball_dist - dist
  command._prev_robot_to_ball_dist = dist.clone()
  return delta


def heading_to_ball(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for robot heading pointing toward the ball (cosine similarity)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = _dribble_cmd(env, command_name)
  vec_w = command.ball_pos_w[:, :2] - asset.data.root_link_pos_w[:, :2]
  dist = torch.norm(vec_w, dim=-1, keepdim=True).clamp(min=1e-3)
  dir_to_ball = vec_w / dist
  forward_w = quat_apply(
    asset.data.root_link_quat_w,
    torch.tensor([[1.0, 0.0, 0.0]], device=asset.data.root_link_quat_w.device).expand(
      asset.data.root_link_quat_w.shape[0], -1
    ),
  )[:, :2]
  return (forward_w * dir_to_ball).sum(dim=-1).clamp(min=0.0)


# --- Spike A2-MOS92: vision-centered gaze + search/track behavior ----------


def gaze_center(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward keeping the ball near image center, only while it is visible.

  Drives both neck_yaw (u) and neck_pitch (v) so the ball stays centered in
  the head camera. Zero when the ball is out of view (search handles that).
  """
  u, v, visible = _gaze_uv_visible(env, command_name, asset_cfg)
  centered = torch.exp(-(u**2 + v**2) / (std**2))
  return visible * centered


def gaze_search(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward turning the gaze toward the ball bearing while it is NOT visible.

  Encourages scanning toward where the ball actually is (last-known/true
  bearing) rather than spinning randomly. Active only when out of view, so it
  rewards reducing the gaze-to-ball angle until the ball enters the frame.
  """
  u, v, visible = _gaze_uv_visible(env, command_name, asset_cfg)
  # |u| is the horizontal gaze error in half-FoV units; reward shrinking it.
  align = torch.exp(-(u.abs() - 1.0).clamp(min=0.0))
  return (1.0 - visible) * align


def search_freeze(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalty (>=0, used with negative weight) on base linear speed while the
  ball is out of view.

  Implements stage-1 search "don't walk before you see the ball": when the
  ball is not visible, any base translation is penalized. In-place yaw is NOT
  penalized, so the policy can still turn the body to cover the rear blind
  sector (stage-2 search).
  """
  robot: Entity = env.scene[asset_cfg.name]
  _, _, visible = _gaze_uv_visible(env, command_name, asset_cfg)
  base_speed = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=-1)
  return (1.0 - visible) * base_speed


def approach_intercept(
  env: ManagerBasedRlEnv,
  command_name: str,
  tau: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward reducing distance to the ball's predicted intercept point, only
  while the ball is visible.

  Predicted point = ball_pos + ball_vel * tau, so the robot leads a rolling
  ball instead of chasing its current position. Gated by visibility so the
  robot approaches only after it has found the ball.
  """
  robot: Entity = env.scene[asset_cfg.name]
  command = _dribble_cmd(env, command_name)
  _, _, visible = _gaze_uv_visible(env, command_name, asset_cfg)
  predicted = command.ball_pos_w[:, :2] + tau * command.ball_lin_vel_w[:, :2]
  robot_xy = robot.data.root_link_pos_w[:, :2]
  dist = torch.norm(robot_xy - predicted, dim=-1)
  delta = command._prev_intercept_dist - dist
  command._prev_intercept_dist = dist.clone()
  return visible * delta


def flight_phase(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  air_time_threshold: float = 0.0,
) -> torch.Tensor:
  """Penalize a flight phase: BOTH feet off the ground simultaneously.

  A bipedal walking/dribbling gait always keeps at least one foot in contact
  (single- or double-support). A flight phase only appears when the robot hops
  or jumps — e.g. to reorient by jumping instead of stepping. Returns a per-env
  cost in [0, 1] = fraction-of-feet-aware indicator that both feet are airborne;
  use with a negative weight.

  Args:
    sensor_name: feet ground-contact sensor (``found`` field, ``[B, n_feet]``).
    air_time_threshold: only count it as flight once both feet have been airborne
      for at least this long (s). 0.0 = penalize any both-feet-off step; a small
      value (~0.05) ignores the brief double-float of a fast gait transition.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  airborne = sensor.data.found == 0  # [B, n_feet] True where that foot is off.
  if air_time_threshold > 0.0:
    assert sensor.data.current_air_time is not None
    airborne = airborne & (sensor.data.current_air_time > air_time_threshold)
  in_flight = airborne.all(dim=1).float()  # [B] 1.0 when ALL feet off.
  env.extras["log"]["Metrics/flight_phase_frac"] = in_flight.mean()
  return in_flight


def illegal_body_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize a non-foot body part touching the ball (规则1, 非法身体接触).

  `illegal = body_contact and not foot_contact`. The body_ball sensor already
  excludes feet/ankles, so any contact it reports IS illegal — thigh/knee/torso/
  arm/hand pushing the ball. Returns per-env 0/1. Use with a negative weight.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  illegal = (sensor.data.found.sum(dim=-1) > 0).float()
  env.extras["log"]["Metrics/illegal_contact_frac"] = illegal.mean()
  return illegal


def ball_trapped(
  env: ManagerBasedRlEnv,
  foot_sensor: str,
  body_sensor: str,
  command_name: str,
  speed_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize trapping/pinning the ball (规则3, 夹球 — the WORST violation).

  `trapped = foot_contact AND body_contact AND ball_speed ~= 0`: the ball is
  simultaneously held by a foot and a body part while not moving — i.e. pinned
  between legs / under the torso (the v3d straddling exploit). This is the
  reward-hacking attractor, so it carries the heaviest weight (-3.0).
  """
  foot: ContactSensor = env.scene[foot_sensor]
  body: ContactSensor = env.scene[body_sensor]
  cmd = env.command_manager.get_term(command_name)
  assert foot.data.found is not None and body.data.found is not None
  both_contact = (foot.data.found.sum(-1) > 0) & (body.data.found.sum(-1) > 0)
  ball_stuck = cmd.metrics["ball_speed"] < speed_threshold
  trapped = (both_contact & ball_stuck).float()
  env.extras["log"]["Metrics/ball_trapped_frac"] = trapped.mean()
  return trapped


def holding_ball(
  env: ManagerBasedRlEnv,
  command_name: str,
  time_threshold: float = 1.5,
) -> torch.Tensor:
  """Penalize holding the ball too long (规则2, 持球过久).

  `holding = ball near (<0.4m) AND slow (<0.2 m/s) sustained > time_threshold`.
  The holding_time accumulator (DribbleCommand) resets the moment the ball is
  released or speeds up, so this fires only on sustained possession — "抱着球走".
  Soccer is dynamic control, not occupation. Returns per-env 0/1.
  """
  cmd = env.command_manager.get_term(command_name)
  holding = (cmd.metrics["holding_time"] > time_threshold).float()
  env.extras["log"]["Metrics/holding_ball_frac"] = holding.mean()
  return holding


def ball_sticking(
  env: ManagerBasedRlEnv,
  foot_sensor: str,
  command_name: str,
  time_threshold: float = 1.0,
) -> torch.Tensor:
  """Penalize sticking to the ball without moving it (规则4, 粘球).

  `sticking = foot contact AND ball ~stationary sustained > time_threshold`.
  Distinct from holding: holding is about long POSSESSION (distance-based);
  sticking is CONTACT-WITHOUT-PROGRESS (the ball touches the foot but is not
  driven forward). Uses ball_stuck_time (speed-only accumulator). Per-env 0/1.
  """
  foot: ContactSensor = env.scene[foot_sensor]
  cmd = env.command_manager.get_term(command_name)
  assert foot.data.found is not None
  has_contact = foot.data.found.sum(-1) > 0
  stuck_long = cmd.metrics["ball_stuck_time"] > time_threshold
  sticking = (has_contact & stuck_long).float()
  env.extras["log"]["Metrics/ball_sticking_frac"] = sticking.mean()
  return sticking


def _selfloc_estimate(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """The policy's raw self-localization estimate, shape (B, 4)."""
  return env.action_manager.get_term(action_name).raw_action


def selfloc_accuracy(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  action_name: str = "selfloc",
  pos_weight: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the policy for estimating its own field pose accurately (v3g).

  The policy emits a 4-d cognitive output ``[x_n, y_n, sin(yaw), cos(yaw)]``
  (see ``SelfLocAction``); this rewards it for matching the ground-truth
  ``robot_field_pose``. Splits position vs heading error so the (normalized)
  xy term and the sin/cos heading term stay comparable. Returns an exp kernel
  in (0, 1] — high when the estimate is calibrated. Also logs the raw position
  error in METERS for monitoring.
  """
  est = _selfloc_estimate(env, action_name)  # (B, 4)
  gt = robot_field_pose(env, command_name, asset_cfg)  # (B, 4)
  pos_err = torch.sum(torch.square(est[:, :2] - gt[:, :2]), dim=-1)
  head_err = torch.sum(torch.square(est[:, 2:] - gt[:, 2:]), dim=-1)
  err = pos_weight * pos_err + head_err

  # Log calibration in meters: un-normalize x_n,y_n by the field half-extents.
  cmd = _dribble_cmd(env, command_name)
  dx_m = (est[:, 0] - gt[:, 0]) * cmd.cfg.half_length
  dy_m = (est[:, 1] - gt[:, 1]) * cmd.cfg.half_width
  pos_err_m = torch.sqrt(dx_m**2 + dy_m**2)
  env.extras["log"]["Metrics/selfloc_pos_err_m"] = pos_err_m.mean()

  return torch.exp(-err / std**2)


def selfloc_error_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  action_name: str = "selfloc",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Actively penalize a wrong self-localization estimate (v3g).

  Returns the raw L2 distance between the estimate and ground-truth field pose
  (register with a NEGATIVE weight). Complements ``selfloc_accuracy``: the exp
  kernel saturates near zero for large errors, so a being-very-wrong estimate
  is barely distinguished — this linear term keeps punishing gross errors and
  matches the user's explicit "估错给罚" requirement. Per-env, positive.
  """
  est = _selfloc_estimate(env, action_name)
  gt = robot_field_pose(env, command_name, asset_cfg)
  err = torch.sqrt(torch.sum(torch.square(est - gt), dim=-1))
  env.extras["log"]["Metrics/selfloc_err_l2"] = err.mean()
  return err


def time_to_goal_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Per-step cost while the goal is NOT yet scored (用户要求的"最少步骤").

  Returns 1.0 for every env that has not yet scored this episode, 0.0 once the
  ball has crossed into the goal (goal_scored latch). Register with a NEGATIVE
  weight: the policy is pushed to score in as few steps as possible because the
  accumulated penalty shrinks the sooner it scores. Stops penalizing after the
  goal so it does not also punish post-score standing.
  """
  command = _dribble_cmd(env, command_name)
  not_scored = 1.0 - command.goal_scored
  env.extras["log"]["Metrics/not_scored_frac"] = not_scored.mean()
  return not_scored


def _camera_pose_world(env: ManagerBasedRlEnv, sensor_name: str):
  """(cam_pos (B,3), cam_mat (B,3,3)) for the named camera, from sim data."""
  cam_idx = env.scene[sensor_name].camera_idx
  sd = env.sim.data
  cam_pos = sd.cam_xpos[:, cam_idx, :]
  cam_mat = sd.cam_xmat[:, cam_idx, :].reshape(-1, 3, 3)
  return cam_pos, cam_mat


def keypoint_detection_accuracy(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  sensor_name: str = "head_cam",
  action_name: str = "selfloc",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """v4 EXP5c: reward the policy for DETECTING field keypoints in the image.

  Paradigm shift from selfloc_accuracy (which regressed pose and collapsed once
  the GT-pose obs faded). Here the policy emits K*2 normalized pixel coords; we
  supervise against the per-frame projection of the KNOWN 3D field keypoints
  (project_keypoints), masked to visible points. This label is geometric and
  present EVERY frame — there is no crutch to withdraw — so the learned image->
  keypoint map can't decay the way the regression did. Pose itself is recovered
  geometrically (see keypoint_pose_error), not learned.

  Returns exp(-mean_visible_pixel_err / std^2) in (0,1]; only visible keypoints
  contribute. Logs the raw mean pixel error for monitoring.
  """
  from mjlab.tasks.velocity.mdp.field_keypoints import (
    field_keypoints_3d,
    project_keypoints,
  )

  pred = _selfloc_estimate(env, action_name)  # (B, K*2) in [-1,1]
  cam = env.scene[sensor_name]
  W, H = cam.cfg.width, cam.cfg.height
  fovy = cam.cfg.fovy if cam.cfg.fovy is not None else 45.0
  cam_pos, cam_mat = _camera_pose_world(env, sensor_name)
  origin = env.scene.env_origins
  kp_local = field_keypoints_3d(env.device)
  K = kp_local.shape[0]
  kp_world = kp_local.unsqueeze(0) + origin.unsqueeze(1)
  uv_gt, vis = project_keypoints(
    kp_world, cam_pos, cam_mat, fovy, W, H
  )  # (B,K,2),(B,K)
  uv_pred = pred.view(-1, K, 2)
  err = torch.linalg.norm(uv_pred - uv_gt, dim=-1)  # (B,K) normalized-pixel L2
  w = vis.float()
  denom = w.sum(dim=1).clamp(min=1.0)
  mean_err = (err * w).sum(dim=1) / denom
  env.extras["log"]["Metrics/kp_pixel_err"] = mean_err.mean()
  env.extras["log"]["Metrics/kp_visible"] = w.sum(dim=1).mean()
  # EXP5d: gate accuracy by visible fraction. Without this, the policy games the
  # reward by turning away so NO keypoints are visible -> masked err -> 0 ->
  # exp(0)=1 free reward (EXP5c: kp_visible collapsed 4->0.06, pos_err 11m while
  # pixel_err stayed low). Multiplying by (visible/K) makes "see nothing" worth 0
  # and applies a two-sided gradient: must SEE keypoints AND localize them well.
  vis_frac = w.sum(dim=1) / float(uv_gt.shape[1])
  return vis_frac * torch.exp(-mean_err / std**2)


def keypoint_pose_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str = "head_cam",
  action_name: str = "selfloc",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """v4 EXP5c: recover field pose from DETECTED keypoints via depth+Kabsch, log err.

  Monitoring term (register with weight 0): runs the full geometric chain on the
  policy's detected pixels — lift to camera-frame 3D via the RAW metric depth at
  each detected pixel, transform to the robot BASE frame (kinematic, no world
  pose), Kabsch-fit against the known map to recover (x, y). Logs the position
  error in meters vs robot_field_pose GT, directly comparable to the <3m target.
  Returns zeros (no reward) — the learnable signal is keypoint_detection_accuracy.
  """
  from mjlab.tasks.velocity.mdp.field_keypoints import (
    field_keypoints_3d,
    kabsch_se2,
    lift_pixels_to_world,
  )
  from mjlab.utils.lab_api.math import matrix_from_quat

  pred = _selfloc_estimate(env, action_name)  # (B, K*2)
  cam = env.scene[sensor_name]
  W, H = cam.cfg.width, cam.cfg.height
  fovy = cam.cfg.fovy if cam.cfg.fovy is not None else 45.0
  cam_pos, cam_mat = _camera_pose_world(env, sensor_name)
  kp_local = field_keypoints_3d(env.device)
  K = kp_local.shape[0]
  uv_pred = pred.view(-1, K, 2)

  # Sample RAW metric depth at each predicted pixel (nearest).
  depth_raw = cam.data.depth  # (B, H, W, 1) meters
  u_pix = ((uv_pred[..., 0] + 1.0) * 0.5 * (W - 1)).round().long().clamp(0, W - 1)
  v_pix = ((uv_pred[..., 1] + 1.0) * 0.5 * (H - 1)).round().long().clamp(0, H - 1)
  bidx = torch.arange(uv_pred.shape[0], device=env.device).unsqueeze(1).expand(-1, K)
  depth_at = depth_raw[bidx, v_pix, u_pix, 0]  # (B, K)

  pw = lift_pixels_to_world(uv_pred, depth_at, cam_pos, cam_mat, fovy, W, H)
  robot = env.scene[asset_cfg.name]
  base_pos = robot.data.root_link_pos_w
  base_mat = matrix_from_quat(robot.data.root_link_quat_w)
  rel = pw - base_pos.unsqueeze(1)
  p_base = torch.einsum("bij,bkj->bki", base_mat.transpose(1, 2), rel)

  valid = (depth_at > 0.05) & (depth_at < 30.0)
  yaw_r, t_r = kabsch_se2(
    p_base[..., :2],
    kp_local[..., :2].unsqueeze(0).expand(p_base.shape[0], -1, -1),
    valid.float(),
  )
  origin = env.scene.env_origins
  gt_xy = robot.data.root_link_pos_w[:, :2] - origin[:, :2]
  pos_err_m = torch.linalg.norm(t_r - gt_xy, dim=-1)
  env.extras["log"]["Metrics/kp_selfloc_pos_err_m"] = pos_err_m.mean()
  return torch.zeros(p_base.shape[0], device=env.device)


def oracle_pose_belief_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str = "head_cam",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """v4 EXP6 MONITOR (weight 0): log the oracle belief's pose error vs GT.

  oracle_pose_belief is an obs (no error logged), so this term runs the same
  geometry and logs Metrics/selfloc_pos_err_m + belief_vis_frac so the full run is
  readable. Returns zeros (no reward)."""
  from mjlab.tasks.velocity.mdp.observations import (
    oracle_pose_belief,
    robot_field_pose,
  )

  belief = oracle_pose_belief(env, command_name, sensor_name, asset_cfg)  # (B,6)
  gt = robot_field_pose(env, command_name, asset_cfg)  # (B,4)
  command = _dribble_cmd(env, command_name)
  dx = (belief[:, 0] - gt[:, 0]) * command.cfg.half_length
  dy = (belief[:, 1] - gt[:, 1]) * command.cfg.half_width
  pos_err = torch.sqrt(dx * dx + dy * dy)
  env.extras["log"]["Metrics/selfloc_pos_err_m"] = pos_err.mean()
  env.extras["log"]["Metrics/belief_vis_frac"] = belief[:, 4].mean()
  return torch.zeros(belief.shape[0], device=env.device)


def active_scan_coverage(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str = "head_cam",
  target_frac: float = 0.6,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """v4 EXP8: reward the fused belief's UNIQUE keypoint coverage (active scan).

  EXP7/8 diagnosis: pos_err is bottlenecked by how many DISTINCT keypoints the
  neck sweep brings into the fused window (uniq_frac), not by per-frame precision.
  This rewards raising uniq_frac toward target_frac (~14/23), which the policy can
  only achieve by actively sweeping neck_yaw to look at new landmarks. Saturates
  at target so it doesn't fight kicking once coverage is sufficient.

  Reads the cached uniq_frac the FusedPoseBelief term computed THIS step (it caches
  to self._last_uniq on every __call__), so no geometry is recomputed and the
  stateful frame buffer is not disturbed. Returns clamp(uniq_frac / target, 0, 1)."""
  from mjlab.tasks.velocity.mdp.observations import FusedPoseBelief

  om = env.observation_manager
  term = None
  # term instances live as cfg.func in the per-group cfg lists.
  for grp in ("actor", "critic"):
    names = om.active_terms.get(grp, [])
    if "ball_to_target" in names:
      idx = names.index("ball_to_target")
      cand = om._group_obs_term_cfgs[grp][idx].func
      if isinstance(cand, FusedPoseBelief):
        term = cand
        break
  if term is None or term._last_uniq is None:
    return torch.zeros(env.num_envs, device=env.device)
  uniq = term._last_uniq
  cov = torch.clamp(uniq / target_frac, 0.0, 1.0)
  env.extras["log"]["Metrics/scan_uniq_frac"] = uniq.mean()
  return cov


def fused_belief_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """v4 EXP8 MONITOR (weight ~0): log the FUSED belief's pose error vs GT.

  Reads the FusedPoseBelief term's cached belief xy (the SAME value the policy
  consumes), so the logged Metrics/selfloc_pos_err_m reflects the fused (not
  single-frame) belief. Returns zeros."""
  from mjlab.tasks.velocity.mdp.observations import FusedPoseBelief, robot_field_pose

  om = env.observation_manager
  term = None
  for grp in ("actor", "critic"):
    names = om.active_terms.get(grp, [])
    if "ball_to_target" in names:
      cand = om._group_obs_term_cfgs[grp][names.index("ball_to_target")].func
      if isinstance(cand, FusedPoseBelief):
        term = cand
        break
  if term is None or term._last_xy_n is None:
    return torch.zeros(env.num_envs, device=env.device)
  gt = robot_field_pose(env, command_name, asset_cfg)  # (B,4)
  command = _dribble_cmd(env, command_name)
  dx = (term._last_xy_n[:, 0] - gt[:, 0]) * command.cfg.half_length
  dy = (term._last_xy_n[:, 1] - gt[:, 1]) * command.cfg.half_width
  pos_err = torch.sqrt(dx * dx + dy * dy)
  env.extras["log"]["Metrics/selfloc_pos_err_m"] = pos_err.mean()
  env.extras["log"]["Metrics/fused_uniq_frac"] = term._last_uniq.mean()
  return torch.zeros(env.num_envs, device=env.device)


def _fused_uniq_frac(env: ManagerBasedRlEnv) -> torch.Tensor | None:
  """Read the FusedPoseBelief term's cached uniq_frac (None before first call)."""
  from mjlab.tasks.velocity.mdp.observations import FusedPoseBelief

  om = env.observation_manager
  for grp in ("actor", "critic"):
    names = om.active_terms.get(grp, [])
    if "ball_to_target" in names:
      cand = om._group_obs_term_cfgs[grp][names.index("ball_to_target")].func
      if isinstance(cand, FusedPoseBelief):
        return cand._last_uniq
  return None


def gaze_center_gated(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.5,
  certain_frac: float = 0.4,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """v4 EXP9: gaze_center GATED by belief certainty. EXP8 showed ball-staring
  (gaze_center w=1.0) out-competes active_scan for the single neck joint, so the
  policy never sweeps and uniq coverage stalls (~0.24). The fix is a time-share:
  only reward ball-centering once the belief is already well-covered
  (uniq_frac >= certain_frac). When the belief is poor the gate is ~0, so staring
  pays nothing and scanning becomes the only way to earn reward; once coverage is
  good the gate opens and the policy centers the ball to kick."""
  base = gaze_center(env, command_name, std, asset_cfg)
  uniq = _fused_uniq_frac(env)
  if uniq is None:
    return base
  gate = torch.clamp(uniq / certain_frac, 0.0, 1.0)
  return base * gate


def gaze_search_gated(
  env: ManagerBasedRlEnv,
  command_name: str,
  certain_frac: float = 0.4,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """v4 EXP9: gaze_search gated by belief certainty (same rationale as
  gaze_center_gated). Turning toward the ball is only rewarded once self-loc
  coverage is sufficient; while uncertain, active_scan drives the neck instead."""
  base = gaze_search(env, command_name, asset_cfg)
  uniq = _fused_uniq_frac(env)
  if uniq is None:
    return base
  gate = torch.clamp(uniq / certain_frac, 0.0, 1.0)
  return base * gate


def neck_scan_motion(
  env: ManagerBasedRlEnv,
  command_name: str,
  certain_frac: float = 0.4,
  vel_scale: float = 2.0,
  asset_cfg: SceneEntityCfg = _NECK_YAW_CFG,
) -> torch.Tensor:
  """v4 EXP10: DIRECTLY reward neck_yaw angular motion (not the coverage result).

  EXP9 proved the indirect coverage reward (active_scan on uniq_frac) cannot teach
  the neck to sweep — even when it's the only reward on offer, uniq_frac stayed
  flat (~0.25). The credit-assignment path from "turn neck" -> "coverage rises" ->
  "reward" is too long for PPO to discover. This gives a DIRECT, dense gradient on
  the action itself: reward |neck_yaw angular velocity|, gated to fire only while
  the belief is still uncertain (uniq_frac < certain_frac) so the head sweeps to
  gather landmarks when lost and settles (to stare/kick) once well-localized.

  reward = (1 - certainty_gate) * tanh(|neck_yaw_vel| / vel_scale), in [0,1)."""
  asset = env.scene[asset_cfg.name]
  neck_vel = asset.data.joint_vel[:, asset_cfg.joint_ids][:, 0]  # (B,)
  motion = torch.tanh(neck_vel.abs() / vel_scale)
  uniq = _fused_uniq_frac(env)
  if uniq is None:
    return motion
  uncertain = 1.0 - torch.clamp(uniq / certain_frac, 0.0, 1.0)
  return uncertain * motion
