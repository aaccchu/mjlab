from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
  from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(
  env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  """Penalize the energy used by the robot's joints."""
  asset: Articulation = env.scene[asset_cfg.name]

  qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
  env: ManagerBasedRLEnv,
  command_name: str = "base_velocity",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Articulation = env.scene[asset_cfg.name]

  reward = torch.sum(
    torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1
  )
  cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
  return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
  env: ManagerBasedRLEnv,
  desired_gravity: list[float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
  # extract the used quantities (to enable type-hinting)
  asset: RigidObject = env.scene[asset_cfg.name]

  desired_gravity = torch.tensor(desired_gravity, device=env.device)
  cos_dist = torch.sum(
    asset.data.projected_gravity_b * desired_gravity, dim=-1
  )  # cosine distance
  normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
  return torch.square(normalized)


def upward(
  env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  """Penalize z-axis base linear velocity using L2 squared kernel."""
  # extract the used quantities (to enable type-hinting)
  asset: RigidObject = env.scene[asset_cfg.name]
  reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
  return reward


def track_lin_vel_x_exp(
  env: ManagerBasedRLEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward tracking of linear velocity command in x-axis using exponential kernel."""
  asset: RigidObject = env.scene[asset_cfg.name]
  lin_vel_error = torch.square(
    env.command_manager.get_command(command_name)[:, 0]
    - asset.data.root_lin_vel_b[:, 0]
  )
  return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_y_exp(
  env: ManagerBasedRLEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward tracking of linear velocity command in y-axis using exponential kernel."""
  asset: RigidObject = env.scene[asset_cfg.name]
  lin_vel_error = torch.square(
    env.command_manager.get_command(command_name)[:, 1]
    - asset.data.root_lin_vel_b[:, 1]
  )
  return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_x_exp_nonzero_cmd(
  env: ManagerBasedRLEnv,
  std: float,
  command_name: str,
  min_abs_cmd: float = 1e-3,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Track x velocity only when x command magnitude is non-zero enough."""
  asset: RigidObject = env.scene[asset_cfg.name]
  cmd_x = env.command_manager.get_command(command_name)[:, 0]
  lin_vel_error = torch.square(cmd_x - asset.data.root_lin_vel_b[:, 0])
  reward = torch.exp(-lin_vel_error / std**2)
  return reward * (torch.abs(cmd_x) > min_abs_cmd).float()


def track_lin_vel_y_exp_nonzero_cmd(
  env: ManagerBasedRLEnv,
  std: float,
  command_name: str,
  min_abs_cmd: float = 1e-3,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Track y velocity only when y command magnitude is non-zero enough."""
  asset: RigidObject = env.scene[asset_cfg.name]
  cmd_y = env.command_manager.get_command(command_name)[:, 1]
  lin_vel_error = torch.square(cmd_y - asset.data.root_lin_vel_b[:, 1])
  reward = torch.exp(-lin_vel_error / std**2)
  return reward * (torch.abs(cmd_y) > min_abs_cmd).float()


def track_ang_vel_z_exp_nonzero_cmd(
  env: ManagerBasedRLEnv,
  std: float,
  command_name: str,
  min_abs_cmd: float = 1e-3,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Track yaw angular velocity only when yaw command magnitude is non-zero enough."""
  asset: RigidObject = env.scene[asset_cfg.name]
  cmd_z = env.command_manager.get_command(command_name)[:, 2]
  ang_vel_error = torch.square(cmd_z - asset.data.root_ang_vel_b[:, 2])
  reward = torch.exp(-ang_vel_error / std**2)
  return reward * (torch.abs(cmd_z) > min_abs_cmd).float()


def track_lin_vel_x_exp_nonzero_cmd_deadzone(
  env: ManagerBasedRLEnv,
  std: float,
  command_name: str,
  min_abs_cmd: float = 1e-3,
  deadzone: float = 0.0,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: RigidObject = env.scene[asset_cfg.name]
  cmd_x = env.command_manager.get_command(command_name)[:, 0]
  abs_error = torch.abs(cmd_x - asset.data.root_lin_vel_b[:, 0])
  effective_error = torch.clamp_min(abs_error - deadzone, 0.0)
  reward = torch.exp(-torch.square(effective_error) / std**2)
  return reward * (torch.abs(cmd_x) > min_abs_cmd).float()


def track_lin_vel_y_exp_nonzero_cmd_deadzone(
  env: ManagerBasedRLEnv,
  std: float,
  command_name: str,
  min_abs_cmd: float = 1e-3,
  deadzone: float = 0.0,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: RigidObject = env.scene[asset_cfg.name]
  cmd_y = env.command_manager.get_command(command_name)[:, 1]
  abs_error = torch.abs(cmd_y - asset.data.root_lin_vel_b[:, 1])
  effective_error = torch.clamp_min(abs_error - deadzone, 0.0)
  reward = torch.exp(-torch.square(effective_error) / std**2)
  return reward * (torch.abs(cmd_y) > min_abs_cmd).float()


def track_ang_vel_z_exp_nonzero_cmd_deadzone(
  env: ManagerBasedRLEnv,
  std: float,
  command_name: str,
  min_abs_cmd: float = 1e-3,
  deadzone: float = 0.0,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: RigidObject = env.scene[asset_cfg.name]
  cmd_z = env.command_manager.get_command(command_name)[:, 2]
  abs_error = torch.abs(cmd_z - asset.data.root_ang_vel_b[:, 2])
  effective_error = torch.clamp_min(abs_error - deadzone, 0.0)
  reward = torch.exp(-torch.square(effective_error) / std**2)
  return reward * (torch.abs(cmd_z) > min_abs_cmd).float()


def wrong_direction_lin_vel_penalty(
  env: ManagerBasedRLEnv,
  command_name: str,
  min_abs_cmd: float = 1e-3,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize velocity moving in opposite direction of the commanded active axis."""
  asset: RigidObject = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  vel_x = asset.data.root_lin_vel_b[:, 0]
  vel_y = asset.data.root_lin_vel_b[:, 1]

  cmd_x = cmd[:, 0]
  cmd_y = cmd[:, 1]
  use_x = torch.abs(cmd_x) >= torch.abs(cmd_y)

  wrong_x = torch.clamp_min(-cmd_x * vel_x, 0.0)
  wrong_y = torch.clamp_min(-cmd_y * vel_y, 0.0)
  penalty = torch.where(use_x, wrong_x, wrong_y)

  active = (torch.abs(cmd_x) > min_abs_cmd) | (torch.abs(cmd_y) > min_abs_cmd)
  return penalty * active.float()


def orthogonal_axis_lin_vel_penalty(
  env: ManagerBasedRLEnv,
  command_name: str,
  min_abs_cmd: float = 1e-3,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize orthogonal-axis linear velocity under decoupled XY command."""
  asset: RigidObject = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  vel_x = torch.abs(asset.data.root_lin_vel_b[:, 0])
  vel_y = torch.abs(asset.data.root_lin_vel_b[:, 1])

  cmd_x = cmd[:, 0]
  cmd_y = cmd[:, 1]
  use_x = torch.abs(cmd_x) >= torch.abs(cmd_y)
  penalty = torch.where(use_x, vel_y, vel_x)

  active = (torch.abs(cmd_x) > min_abs_cmd) | (torch.abs(cmd_y) > min_abs_cmd)
  return penalty * active.float()


def yaw_rate_penalty_when_xy_cmd(
  env: ManagerBasedRLEnv,
  command_name: str,
  min_abs_cmd: float = 1e-3,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize yaw rate when XY command is active to reduce turn-then-walk solutions."""
  asset: RigidObject = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  active = torch.linalg.norm(cmd[:, :2], dim=1) > min_abs_cmd
  return torch.abs(asset.data.root_ang_vel_b[:, 2]) * active.float()


def lin_vel_xy_penalty_when_yaw_cmd(
  env: ManagerBasedRLEnv,
  command_name: str,
  min_abs_cmd: float = 1e-3,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize XY linear speed when yaw command is active to reduce turn-with-sidewalk behavior."""
  asset: RigidObject = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  active = torch.abs(cmd[:, 2]) > min_abs_cmd
  lin_xy = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
  return lin_xy * active.float()


def joint_position_penalty(
  env: ManagerBasedRLEnv,
  asset_cfg: SceneEntityCfg,
  stand_still_scale: float,
  velocity_threshold: float,
) -> torch.Tensor:
  """Penalize joint position error from default on the articulation."""
  # extract the used quantities (to enable type-hinting)
  asset: Articulation = env.scene[asset_cfg.name]
  cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
  body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
  reward = torch.linalg.norm(
    (asset.data.joint_pos - asset.data.default_joint_pos), dim=1
  )
  return torch.where(
    torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
    reward,
    stand_still_scale * reward,
  )


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
  # extract the used quantities (to enable type-hinting)
  contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
  forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
  forces_xy = torch.linalg.norm(
    contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2
  )
  # Penalize feet hitting vertical surfaces
  reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
  return reward


def feet_height_body(
  env: ManagerBasedRLEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  target_height: float,
  tanh_mult: float,
) -> torch.Tensor:
  """Reward the swinging feet for clearing a specified height off the ground"""
  asset: RigidObject = env.scene[asset_cfg.name]
  cur_footpos_translated = asset.data.body_pos_w[
    :, asset_cfg.body_ids, :
  ] - asset.data.root_pos_w[:, :].unsqueeze(1)
  footpos_in_body_frame = torch.zeros(
    env.num_envs, len(asset_cfg.body_ids), 3, device=env.device
  )
  cur_footvel_translated = asset.data.body_lin_vel_w[
    :, asset_cfg.body_ids, :
  ] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
  footvel_in_body_frame = torch.zeros(
    env.num_envs, len(asset_cfg.body_ids), 3, device=env.device
  )
  for i in range(len(asset_cfg.body_ids)):
    footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
      asset.data.root_quat_w, cur_footpos_translated[:, i, :]
    )
    footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
      asset.data.root_quat_w, cur_footvel_translated[:, i, :]
    )
  foot_z_target_error = torch.square(
    footpos_in_body_frame[:, :, 2] - target_height
  ).view(env.num_envs, -1)
  foot_velocity_tanh = torch.tanh(
    tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2)
  )
  reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
  reward *= (
    torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
  )
  reward *= (
    torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
  )
  return reward


def foot_clearance_reward(
  env: ManagerBasedRLEnv,
  asset_cfg: SceneEntityCfg,
  target_height: float,
  std: float,
  tanh_mult: float,
) -> torch.Tensor:
  """Reward the swinging feet for clearing a specified height off the ground"""
  asset: RigidObject = env.scene[asset_cfg.name]
  foot_z_target_error = torch.square(
    asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height
  )
  foot_velocity_tanh = torch.tanh(
    tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
  )
  reward = foot_z_target_error * foot_velocity_tanh
  return torch.exp(-torch.sum(reward, dim=1) / std)


def foot_clearance_air_time_biped(
  env: ManagerBasedRLEnv,
  asset_cfg: SceneEntityCfg,
  sensor_cfg: SceneEntityCfg,
  target_height: float,
  min_height: float,
  air_time_cap: float,
  command_name: str = "base_velocity",
) -> torch.Tensor:
  """Reward higher swing-foot clearance using contact-state and air-time gating for bipeds."""
  asset: RigidObject = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

  foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
  in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
  in_air = ~in_contact

  # Height score in [0, 1]: no reward below min_height, saturated near target_height.
  height_score = (foot_height - min_height) / (target_height - min_height + 1e-6)
  height_score = torch.clamp(height_score, min=0.0, max=1.0)

  # Air-time score in [0, 1] to reward sustained swing instead of transient hops.
  air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
  air_time_score = torch.clamp(air_time / (air_time_cap + 1e-6), min=0.0, max=1.0)

  reward_per_foot = in_air.float() * height_score * air_time_score
  reward = torch.sum(reward_per_foot, dim=1)

  command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
  reward *= command_norm > 0.1
  return reward


def feet_flat_orientation_l2(
  env: ManagerBasedRLEnv,
  asset_cfg: SceneEntityCfg,
  std: float,
  desired_up: list[float] | tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
  """Reward feet for keeping their local +Y axis aligned with world +Z."""
  asset: Articulation = env.scene[asset_cfg.name]
  body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
  num_feet = body_quat_w.shape[1]

  world_up = torch.tensor(desired_up, device=env.device, dtype=body_quat_w.dtype)
  world_up = world_up.reshape(1, 1, 3).expand(env.num_envs, num_feet, 3).reshape(-1, 3)
  body_quat_flat = body_quat_w.reshape(-1, 4)

  up_in_body = math_utils.quat_apply_inverse(body_quat_flat, world_up).reshape(
    env.num_envs, num_feet, 3
  )
  cos_align = up_in_body[:, :, 1]
  error = 1.0 - cos_align
  reward = torch.exp(-torch.square(error) / (std**2))
  return torch.mean(reward, dim=1)


def feet_too_near(
  env: ManagerBasedRLEnv,
  threshold: float = 0.2,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Articulation = env.scene[asset_cfg.name]
  feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
  distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
  return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
  env: ManagerBasedRLEnv,
  sensor_cfg: SceneEntityCfg,
  command_name: str = "base_velocity",
) -> torch.Tensor:
  """
  Reward for feet contact when the command is zero.
  """
  # asset: Articulation = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
  is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

  command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
  reward = torch.sum(is_contact, dim=-1).float()
  return reward * (command_norm < 0.1)


def air_time_variance_penalty(
  env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
  # extract the used quantities (to enable type-hinting)
  contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
  if contact_sensor.cfg.track_air_time is False:
    raise RuntimeError("Activate ContactSensor's track_air_time!")
  # compute the reward
  last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
  last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
  return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
    torch.clip(last_contact_time, max=0.5), dim=1
  )


"""
Feet Gait rewards.
"""


def feet_gait(
  env: ManagerBasedRLEnv,
  period: float,
  offset: list[float],
  sensor_cfg: SceneEntityCfg,
  threshold: float = 0.5,
  command_name=None,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
  is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

  global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
  phases = []
  for offset_ in offset:
    phase = (global_phase + offset_) % 1.0
    phases.append(phase)
  leg_phase = torch.cat(phases, dim=-1)

  reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
  for i in range(len(sensor_cfg.body_ids)):
    is_stance = leg_phase[:, i] < threshold
    reward += ~(is_stance ^ is_contact[:, i])

  if command_name is not None:
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward *= cmd_norm > 0.1
  return reward


"""
Other rewards.
"""


def joint_mirror(
  env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]
) -> torch.Tensor:
  # extract the used quantities (to enable type-hinting)
  asset: Articulation = env.scene[asset_cfg.name]
  if (
    not hasattr(env, "joint_mirror_joints_cache")
    or env.joint_mirror_joints_cache is None
  ):
    # Cache joint positions for all pairs
    env.joint_mirror_joints_cache = [
      [asset.find_joints(joint_name) for joint_name in joint_pair]
      for joint_pair in mirror_joints
    ]
  reward = torch.zeros(env.num_envs, device=env.device)
  # Iterate over all joint pairs
  for joint_pair in env.joint_mirror_joints_cache:
    # Calculate the difference for each pair and add to the total reward
    reward += torch.sum(
      torch.square(
        asset.data.joint_pos[:, joint_pair[0][0]]
        - asset.data.joint_pos[:, joint_pair[1][0]]
      ),
      dim=-1,
    )
  reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
  return reward
