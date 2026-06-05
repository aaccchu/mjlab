from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.dribble_command import DribbleCommand
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

