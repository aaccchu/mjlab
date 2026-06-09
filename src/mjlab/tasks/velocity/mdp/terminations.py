from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def out_of_terrain_bounds(
  env: ManagerBasedRlEnv,
  margin: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Truncate if robot leaves the generated terrain footprint.

  Returns all-false for non-generator terrains (e.g. plane).
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(
      (env.num_envs,),
      device=env.device,
      dtype=torch.bool,
    )

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None or terrain.terrain_origins is None:
    return torch.zeros(
      (env.num_envs,),
      device=env.device,
      dtype=torch.bool,
    )

  asset: Entity = env.scene[asset_cfg.name]
  root_xy_w = asset.data.root_link_pos_w[:, :2]

  # Use the generated grid shape (curriculum mode overrides cfg.num_cols with
  # len(sub_terrains)), and include the flat border around the patch grid.
  num_rows, num_cols = terrain.terrain_origins.shape[:2]
  half_x = 0.5 * (num_rows * terrain_generator.size[0]) + terrain_generator.border_width
  half_y = 0.5 * (num_cols * terrain_generator.size[1]) + terrain_generator.border_width
  limit_x = max(0.0, half_x - margin)
  limit_y = max(0.0, half_y - margin)

  return (root_xy_w[:, 0].abs() > limit_x) | (root_xy_w[:, 1].abs() > limit_y)


def out_of_field_bounds(
  env: ManagerBasedRlEnv,
  half_length: float = 11.0,
  half_width: float = 7.0,
  margin: float = 0.3,
  goal_corridor_half_width: float = 0.0,
  goal_buffer: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Out-of-bounds detector for the soccer-field task.

  The field spans ``x in [-half_length, half_length]`` and
  ``y in [-half_width, half_width]`` (outer-edge basis). The robot is flagged
  when its root link crosses the bound shrunk by ``margin``, so it learns to
  stay inside the side/goal lines.

  Goal corridor (EXP17, ``goal_corridor_half_width > 0``): a robot that strikes
  the ball goalward naturally over-runs the goal line by its body length, and
  with ``goal_line_x == half_length`` the OOB line at ``half_length - margin``
  sits *inside* the field, penalizing legitimate shots. When the root is within
  the goal mouth corridor (``|y| < goal_corridor_half_width``), the +/-x limit
  is relaxed to ``half_length + goal_buffer`` so charging into the mouth to
  shoot is not treated as leaving the field. Side lines (``y``) and the
  off-corridor end lines are unchanged. ``goal_corridor_half_width = 0`` (the
  default) disables the corridor and recovers the plain rectangular bound.
  """
  asset: Entity = env.scene[asset_cfg.name]
  root_xy_w = asset.data.root_link_pos_w[:, :2]
  limit_y = max(0.0, half_width - margin)
  out_y = root_xy_w[:, 1].abs() > limit_y

  base_limit_x = max(0.0, half_length - margin)
  if goal_corridor_half_width > 0.0:
    # In the goal mouth corridor the end-line limit is pushed out by the buffer;
    # elsewhere it stays at the standard inset bound.
    in_corridor = root_xy_w[:, 1].abs() < goal_corridor_half_width
    corridor_limit_x = max(0.0, half_length + goal_buffer)
    limit_x = torch.where(
      in_corridor,
      torch.full_like(root_xy_w[:, 0], corridor_limit_x),
      torch.full_like(root_xy_w[:, 0], base_limit_x),
    )
    out_x = root_xy_w[:, 0].abs() > limit_x
  else:
    out_x = root_xy_w[:, 0].abs() > base_limit_x

  return out_x | out_y


def terrain_edge_reached(
  env: ManagerBasedRlEnv,
  threshold_fraction: float = 0.95,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when robot displacement from spawn exceeds sub-terrain size.

  Intended as ``time_out=True`` (successful traversal, not penalized). Skips the first
  2 steps after reset to avoid stale-position triggers.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  asset: Entity = env.scene[asset_cfg.name]
  displacement = (
    asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  ).abs()

  half_x = terrain_generator.size[0] / 2.0 * threshold_fraction
  half_y = terrain_generator.size[1] / 2.0 * threshold_fraction

  at_edge = (displacement[:, 0] > half_x) | (displacement[:, 1] > half_y)

  # Don't fire on the first 2 steps after reset (position may be stale).
  at_edge &= env.episode_length_buf > 2

  return at_edge
