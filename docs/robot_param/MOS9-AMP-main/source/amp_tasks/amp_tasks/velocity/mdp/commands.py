from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class DecoupledAxisVelocityCommand(UniformVelocityCommand):
  """Velocity command where x/y/yaw are sampled in a decoupled one-axis-at-a-time manner.

  For each resampling event, only one among (lin_vel_x, lin_vel_y, ang_vel_z) is non-zero.
  """

  def __init__(self, cfg: UniformVelocityCommandCfg, env):
    super().__init__(cfg, env)
    self.metrics["error_vel_x"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_y"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["action_rate_l2"] = torch.zeros(self.num_envs, device=self.device)

  def _update_metrics(self):
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt

    error_x = torch.abs(self.vel_command_b[:, 0] - self.robot.data.root_lin_vel_b[:, 0])
    error_y = torch.abs(self.vel_command_b[:, 1] - self.robot.data.root_lin_vel_b[:, 1])
    error_yaw = torch.abs(
      self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]
    )
    action_rate_l2 = torch.sum(
      torch.square(
        self._env.action_manager.action - self._env.action_manager.prev_action
      ),
      dim=1,
    )

    self.metrics["error_vel_x"] += error_x / max_command_step
    self.metrics["error_vel_y"] += error_y / max_command_step
    self.metrics["error_vel_yaw"] += error_yaw / max_command_step
    self.metrics["action_rate_l2"] += action_rate_l2 / max_command_step

    self.metrics["error_vel_xy"] += (
      torch.sqrt(error_x**2 + error_y**2) / max_command_step
    )

  def _resample_command(self, env_ids: Sequence[int]):
    num_ids = len(env_ids)
    if num_ids == 0:
      return

    self.vel_command_b[env_ids, :] = 0.0

    axis_ids = torch.randint(0, 3, (num_ids,), device=self.device)

    mask_x = axis_ids == 0
    if torch.any(mask_x):
      env_ids_x = env_ids[mask_x]
      self.vel_command_b[env_ids_x, 0] = torch.empty(
        len(env_ids_x), device=self.device
      ).uniform_(*self.cfg.ranges.lin_vel_x)

    mask_y = axis_ids == 1
    if torch.any(mask_y):
      env_ids_y = env_ids[mask_y]
      self.vel_command_b[env_ids_y, 1] = torch.empty(
        len(env_ids_y), device=self.device
      ).uniform_(*self.cfg.ranges.lin_vel_y)

    mask_yaw = axis_ids == 2
    if torch.any(mask_yaw):
      env_ids_yaw = env_ids[mask_yaw]
      self.vel_command_b[env_ids_yaw, 2] = torch.empty(
        len(env_ids_yaw), device=self.device
      ).uniform_(*self.cfg.ranges.ang_vel_z)

    if self.cfg.heading_command:
      sampled_heading = torch.empty(num_ids, device=self.device).uniform_(
        *self.cfg.ranges.heading
      )
      self.heading_target[env_ids] = sampled_heading
      self.is_heading_env[env_ids] = (
        torch.empty(num_ids, device=self.device).uniform_(0.0, 1.0)
        <= self.cfg.rel_heading_envs
      )

    self.is_standing_env[env_ids] = (
      torch.empty(num_ids, device=self.device).uniform_(0.0, 1.0)
      <= self.cfg.rel_standing_envs
    )


class DiscreteDecoupledAxisVelocityCommand(DecoupledAxisVelocityCommand):
  """Decoupled command where each active axis is sampled from user-defined discrete values."""

  def _sample_from_values(self, values: tuple[float, ...], count: int) -> torch.Tensor:
    value_tensor = torch.tensor(values, device=self.device, dtype=torch.float32)
    indices = torch.randint(0, value_tensor.shape[0], (count,), device=self.device)
    return value_tensor[indices]

  def _resample_command(self, env_ids: Sequence[int]):
    num_ids = len(env_ids)
    if num_ids == 0:
      return

    self.vel_command_b[env_ids, :] = 0.0
    axis_ids = torch.randint(0, 3, (num_ids,), device=self.device)

    mask_x = axis_ids == 0
    if torch.any(mask_x):
      env_ids_x = env_ids[mask_x]
      self.vel_command_b[env_ids_x, 0] = self._sample_from_values(
        self.cfg.discrete_lin_vel_x, len(env_ids_x)
      )

    mask_y = axis_ids == 1
    if torch.any(mask_y):
      env_ids_y = env_ids[mask_y]
      self.vel_command_b[env_ids_y, 1] = self._sample_from_values(
        self.cfg.discrete_lin_vel_y, len(env_ids_y)
      )

    mask_yaw = axis_ids == 2
    if torch.any(mask_yaw):
      env_ids_yaw = env_ids[mask_yaw]
      self.vel_command_b[env_ids_yaw, 2] = self._sample_from_values(
        self.cfg.discrete_ang_vel_z, len(env_ids_yaw)
      )

    if self.cfg.heading_command:
      sampled_heading = torch.empty(num_ids, device=self.device).uniform_(
        *self.cfg.ranges.heading
      )
      self.heading_target[env_ids] = sampled_heading
      self.is_heading_env[env_ids] = (
        torch.empty(num_ids, device=self.device).uniform_(0.0, 1.0)
        <= self.cfg.rel_heading_envs
      )

    self.is_standing_env[env_ids] = (
      torch.empty(num_ids, device=self.device).uniform_(0.0, 1.0)
      <= self.cfg.rel_standing_envs
    )


class AxisSelectableDiscreteDecoupledAxisVelocityCommand(
  DiscreteDecoupledAxisVelocityCommand
):
  """Discrete decoupled command with configurable active axes.

  Axis mapping: 0 -> x, 1 -> y, 2 -> yaw.
  """

  def _resample_command(self, env_ids: Sequence[int]):
    num_ids = len(env_ids)
    if num_ids == 0:
      return

    self.vel_command_b[env_ids, :] = 0.0
    active_axes = torch.tensor(
      self.cfg.active_axes, device=self.device, dtype=torch.long
    )
    sampled_idx = torch.randint(0, active_axes.shape[0], (num_ids,), device=self.device)
    axis_ids = active_axes[sampled_idx]

    mask_x = axis_ids == 0
    if torch.any(mask_x):
      env_ids_x = env_ids[mask_x]
      self.vel_command_b[env_ids_x, 0] = self._sample_from_values(
        self.cfg.discrete_lin_vel_x, len(env_ids_x)
      )

    mask_y = axis_ids == 1
    if torch.any(mask_y):
      env_ids_y = env_ids[mask_y]
      self.vel_command_b[env_ids_y, 1] = self._sample_from_values(
        self.cfg.discrete_lin_vel_y, len(env_ids_y)
      )

    mask_yaw = axis_ids == 2
    if torch.any(mask_yaw):
      env_ids_yaw = env_ids[mask_yaw]
      self.vel_command_b[env_ids_yaw, 2] = self._sample_from_values(
        self.cfg.discrete_ang_vel_z, len(env_ids_yaw)
      )

    if self.cfg.heading_command:
      sampled_heading = torch.empty(num_ids, device=self.device).uniform_(
        *self.cfg.ranges.heading
      )
      self.heading_target[env_ids] = sampled_heading
      self.is_heading_env[env_ids] = (
        torch.empty(num_ids, device=self.device).uniform_(0.0, 1.0)
        <= self.cfg.rel_heading_envs
      )

    self.is_standing_env[env_ids] = (
      torch.empty(num_ids, device=self.device).uniform_(0.0, 1.0)
      <= self.cfg.rel_standing_envs
    )


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
  class_type: type = UniformVelocityCommand
  limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


@configclass
class DecoupledLevelVelocityCommandCfg(UniformVelocityCommandCfg):
  class_type: type = DecoupledAxisVelocityCommand
  limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


@configclass
class DiscreteDecoupledLevelVelocityCommandCfg(UniformVelocityCommandCfg):
  class_type: type = DiscreteDecoupledAxisVelocityCommand
  limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
  discrete_lin_vel_x: tuple[float, ...] = MISSING
  discrete_lin_vel_y: tuple[float, ...] = MISSING
  discrete_ang_vel_z: tuple[float, ...] = MISSING


@configclass
class AxisSelectableDiscreteDecoupledLevelVelocityCommandCfg(UniformVelocityCommandCfg):
  class_type: type = AxisSelectableDiscreteDecoupledAxisVelocityCommand
  limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
  discrete_lin_vel_x: tuple[float, ...] = MISSING
  discrete_lin_vel_y: tuple[float, ...] = MISSING
  discrete_ang_vel_z: tuple[float, ...] = MISSING
  active_axes: tuple[int, ...] = (0, 1, 2)
