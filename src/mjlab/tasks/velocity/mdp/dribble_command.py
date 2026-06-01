"""Dribble command for the soccer-field task.

Generates the goal for an end-to-end "dribble the ball to a target" policy and,
critically, derives a base-frame twist ``(vx, vy, wz)`` that the existing
velocity-task gait rewards consume unchanged. The derived twist is only a goal
encoding (never applied as a control): it points the robot toward the ball and
then pushes the ball toward the target. The single policy still learns every
joint torque to walk, balance, and make foot-ball contact from scratch.

State ownership mirrors ``LiftingCommand`` (manipulation task): the command
holds the ball entity and the target position, repositions the ball on reset,
and tracks success metrics. Rewards/observations reach the ball/target by
casting ``command_manager.get_term(name)`` to ``DribbleCommand``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import sample_uniform, wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class DribbleCommand(CommandTerm):
  cfg: DribbleCommandCfg

  def __init__(self, cfg: DribbleCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.object: Entity = env.scene[cfg.entity_name]
    self.robot: Entity = env.scene[cfg.robot_name]

    # World-frame goal for the ball (z fixed at ball radius).
    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    # Derived base-frame twist consumed by gait rewards: [vx, vy, wz].
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)

    self.metrics["ball_to_target_error"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["robot_to_ball_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["time_to_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["possession"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["ball_path_length"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["step_count"] = torch.zeros(self.num_envs, device=self.device)
    self._prev_ball_xy = torch.zeros(self.num_envs, 2, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """Derived base-frame twist [vx, vy, wz] for gait-reward gating."""
    return self.vel_command_b

  @property
  def ball_pos_w(self) -> torch.Tensor:
    return self.object.data.root_link_pos_w

  @property
  def ball_lin_vel_w(self) -> torch.Tensor:
    return self.object.data.root_link_lin_vel_w

  def _update_metrics(self) -> None:
    ball_xy = self.ball_pos_w[:, :2]
    robot_xy = self.robot.data.root_link_pos_w[:, :2]
    ball_to_target = torch.norm(self.target_pos[:, :2] - ball_xy, dim=-1)
    robot_to_ball = torch.norm(ball_xy - robot_xy, dim=-1)
    at_goal = (ball_to_target < self.cfg.success_threshold).float()
    self.episode_success = torch.maximum(self.episode_success, at_goal)

    self.metrics["ball_to_target_error"] = ball_to_target
    self.metrics["robot_to_ball_error"] = robot_to_ball
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success

    self.metrics["step_count"] += 1.0
    first_success = (at_goal > 0) & (self.metrics["time_to_goal"] == 0)
    self.metrics["time_to_goal"] = torch.where(
      first_success, self.metrics["step_count"], self.metrics["time_to_goal"]
    )
    self.metrics["possession"] += (robot_to_ball < 0.5).float()
    ball_disp = torch.norm(ball_xy - self._prev_ball_xy, dim=-1)
    self.metrics["ball_path_length"] += ball_disp
    self._prev_ball_xy = ball_xy.clone()

  def compute_success(self) -> torch.Tensor:
    return self.metrics["ball_to_target_error"] < self.cfg.success_threshold

  def _field_clamp(self, xy: torch.Tensor, center_xy: torch.Tensor) -> torch.Tensor:
    """Clamp xy points into the field interior (outer line minus margin)."""
    lim_x = self.cfg.half_length - self.cfg.field_margin
    lim_y = self.cfg.half_width - self.cfg.field_margin
    x = torch.clamp(xy[:, 0] - center_xy[:, 0], -lim_x, lim_x) + center_xy[:, 0]
    y = torch.clamp(xy[:, 1] - center_xy[:, 1], -lim_y, lim_y) + center_xy[:, 1]
    return torch.stack([x, y], dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0
    self.vel_command_b[env_ids] = 0.0
    self.metrics["time_to_goal"][env_ids] = 0.0
    self.metrics["possession"][env_ids] = 0.0
    self.metrics["ball_path_length"][env_ids] = 0.0
    self.metrics["step_count"][env_ids] = 0.0

    # Robot spawn pose lives in fresh qpos at reset time; xpos (root_link_pos_w)
    # is still stale here because no sim.forward() runs between the reset event
    # and command resampling. Read the freejoint qpos directly.
    q_adr = self.robot.data.indexing.free_joint_q_adr
    robot_xy = self.robot.data.data.qpos[:, q_adr[:3]][env_ids][:, :2]

    center_xy = self._env.scene.env_origins[env_ids][:, :2]

    # Place the ball at a random offset from the robot so it never spawns inside
    # the robot (which would explode contacts), then clamp into the field.
    theta = sample_uniform(-math.pi, math.pi, (n,), device=self.device)
    dist = sample_uniform(
      self.cfg.spawn_dist_range[0],
      self.cfg.spawn_dist_range[1],
      (n,),
      device=self.device,
    )
    ball_offset = torch.stack(
      [dist * torch.cos(theta), dist * torch.sin(theta)], dim=-1
    )
    ball_xy = self._field_clamp(robot_xy + ball_offset, center_xy)

    # Target a non-trivial distance from the ball (directional, then clamped),
    # guaranteeing a meaningful dribble with a usable reward gradient.
    theta_t = sample_uniform(-math.pi, math.pi, (n,), device=self.device)
    dist_t = sample_uniform(
      self.cfg.target_dist_range[0],
      self.cfg.target_dist_range[1],
      (n,),
      device=self.device,
    )
    target_offset = torch.stack(
      [dist_t * torch.cos(theta_t), dist_t * torch.sin(theta_t)], dim=-1
    )
    target_xy = self._field_clamp(ball_xy + target_offset, center_xy)

    z = torch.full((n,), self.cfg.ball_radius, device=self.device)
    self.target_pos[env_ids] = torch.cat([target_xy, z.unsqueeze(-1)], dim=-1)

    # Write the ball to its new resting pose with zero velocity.
    quat = torch.zeros(n, 4, device=self.device)
    quat[:, 0] = 1.0
    pose = torch.cat([ball_xy, z.unsqueeze(-1), quat], dim=-1)
    self.object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    self.object.write_root_link_velocity_to_sim(
      torch.zeros(n, 6, device=self.device), env_ids=env_ids
    )
    self._prev_ball_xy[env_ids] = ball_xy

  def _update_command(self) -> None:
    # Runs each step after sim.forward(), so xpos-derived reads are fresh.
    robot_xy = self.robot.data.root_link_pos_w[:, :2]
    ball_xy = self.ball_pos_w[:, :2]
    target_xy = self.target_pos[:, :2]
    eps = 1e-6

    to_ball = ball_xy - robot_xy
    d_ball = torch.norm(to_ball, dim=-1)
    ball_to_target = target_xy - ball_xy
    d_bt = torch.norm(ball_to_target, dim=-1)
    dir_bt = ball_to_target / (d_bt.unsqueeze(-1) + eps)

    # Approach phase: aim for a point just behind the ball (opposite the target)
    # so the robot ends up positioned to push the ball goalward.
    behind = ball_xy - self.cfg.approach_offset * dir_bt
    to_behind = behind - robot_xy
    d_behind = torch.norm(to_behind, dim=-1)
    approach_dir = to_behind / (d_behind.unsqueeze(-1) + eps)
    approach_speed = torch.clamp(d_behind, max=self.cfg.max_speed)

    # Push phase: drive through the ball toward the target; ease off near goal so
    # the standing-posture regime engages (avoids jitter at success).
    push_dir = dir_bt
    push_speed = self.cfg.max_speed * torch.clamp(
      d_bt / self.cfg.arrive_decay_radius, max=1.0
    )

    near_ball = (d_ball < self.cfg.approach_radius).unsqueeze(-1)
    world_dir = torch.where(near_ball, push_dir, approach_dir)
    speed = torch.where(near_ball.squeeze(-1), push_speed, approach_speed)
    vel_w = world_dir * speed.unsqueeze(-1)

    # Rotate desired world velocity into the base frame (matches the world->body
    # convention in UniformVelocityCommand).
    heading = self.robot.data.heading_w
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    vx_w, vy_w = vel_w[:, 0], vel_w[:, 1]
    self.vel_command_b[:, 0] = cos_h * vx_w + sin_h * vy_w
    self.vel_command_b[:, 1] = -sin_h * vx_w + cos_h * vy_w

    # Yaw command: face the direction of motion.
    desired_heading = torch.atan2(vy_w, vx_w)
    heading_error = wrap_to_pi(desired_heading - heading)
    wz = torch.clamp(
      self.cfg.heading_control_stiffness * heading_error,
      min=-self.cfg.ang_vel_clip,
      max=self.cfg.ang_vel_clip,
    )
    # Only steer when there is meaningful desired motion.
    moving = (speed > self.cfg.standing_speed_threshold).float()
    self.vel_command_b[:, 2] = wz * moving

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    target = self.target_pos.cpu().numpy()
    ball = self.ball_pos_w.cpu().numpy()
    for batch in env_indices:
      visualizer.add_sphere(
        center=target[batch],
        radius=0.12,
        color=self.cfg.viz.target_color,
        label=f"dribble_target_{batch}",
      )
      marker = ball[batch].copy()
      marker[2] += 0.15
      visualizer.add_sphere(
        center=marker,
        radius=0.03,
        color=self.cfg.viz.ball_marker_color,
        label=f"dribble_ball_marker_{batch}",
      )


@dataclass(kw_only=True)
class DribbleCommandCfg(CommandTermCfg):
  entity_name: str = "ball"
  robot_name: str = "robot"

  ball_radius: float = 0.11
  success_threshold: float = 0.5  # Ball within this xy distance of target.

  # Field interior bounds (outer line minus field_margin) for ball/target.
  half_length: float = 11.0
  half_width: float = 7.0
  field_margin: float = 0.5

  # Ball spawn offset from the robot (m); >0 avoids spawning inside the robot.
  spawn_dist_range: tuple[float, float] = (0.6, 1.5)
  # Target offset from the ball (m); ensures a non-trivial dribble.
  target_dist_range: tuple[float, float] = (2.0, 6.0)

  # Derived-twist shaping.
  max_speed: float = 1.0
  approach_radius: float = 0.6  # Switch to push phase within this of the ball.
  approach_offset: float = 0.3  # Aim point behind the ball when approaching.
  arrive_decay_radius: float = 1.5  # Ease push speed within this of the target.
  heading_control_stiffness: float = 0.5
  ang_vel_clip: float = 1.0
  standing_speed_threshold: float = 0.05

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (1.0, 0.4, 0.0, 0.3)
    ball_marker_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> DribbleCommand:
    return DribbleCommand(self, env)
