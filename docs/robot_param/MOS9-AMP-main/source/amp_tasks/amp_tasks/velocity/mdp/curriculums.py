from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
  from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def lin_vel_cmd_levels(
  env: ManagerBasedRLEnv,
  env_ids: Sequence[int],
  reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
  command_term = env.command_manager.get_term("base_velocity")
  ranges = command_term.cfg.ranges
  limit_ranges = command_term.cfg.limit_ranges

  reward_term = env.reward_manager.get_term_cfg(reward_term_name)
  reward = (
    torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
    / env.max_episode_length_s
  )

  if env.common_step_counter % env.max_episode_length == 0:
    if reward > reward_term.weight * 0.80:
      delta_command = torch.tensor([-0.1, 0.1], device=env.device)
      ranges.lin_vel_x = torch.clamp(
        torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
        limit_ranges.lin_vel_x[0],
        limit_ranges.lin_vel_x[1],
      ).tolist()
      ranges.lin_vel_y = torch.clamp(
        torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
        limit_ranges.lin_vel_y[0],
        limit_ranges.lin_vel_y[1],
      ).tolist()

  return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
  env: ManagerBasedRLEnv,
  env_ids: Sequence[int],
  reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
  command_term = env.command_manager.get_term("base_velocity")
  ranges = command_term.cfg.ranges
  limit_ranges = command_term.cfg.limit_ranges

  reward_term = env.reward_manager.get_term_cfg(reward_term_name)
  reward = (
    torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
    / env.max_episode_length_s
  )

  if env.common_step_counter % env.max_episode_length == 0:
    if reward > reward_term.weight * 0.8:
      delta_command = torch.tensor([-0.1, 0.1], device=env.device)
      ranges.ang_vel_z = torch.clamp(
        torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
        limit_ranges.ang_vel_z[0],
        limit_ranges.ang_vel_z[1],
      ).tolist()

  return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def set_value_after_step(
  env: ManagerBasedRLEnv,
  env_ids: Sequence[int],
  data,
  value,
  num_steps: int,
):
  if env.common_step_counter > num_steps:
    return value
  return data


def randomize_rigid_body_com_safe(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor | None,
  com_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg,
):
  asset: Articulation = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.scene.num_envs, device="cpu")
  else:
    env_ids = env_ids.cpu()

  if asset_cfg.body_ids == slice(None):
    body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
  else:
    body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

  range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
  ranges = torch.tensor(range_list, device="cpu")
  rand_samples = math_utils.sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu"
  ).unsqueeze(1)

  coms = asset.root_physx_view.get_coms().clone()
  coms[env_ids[:, None], body_ids, :3] += rand_samples
  asset.root_physx_view.set_coms(coms, env_ids)


def randomize_rigid_body_material_reset_safe(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor | None,
  static_friction_range: tuple[float, float],
  dynamic_friction_range: tuple[float, float],
  restitution_range: tuple[float, float],
  num_buckets: int,
  asset_cfg: SceneEntityCfg,
  make_consistent: bool = False,
):
  asset: Articulation = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.scene.num_envs, device="cpu")
  else:
    env_ids = env_ids.cpu()

  range_list = [static_friction_range, dynamic_friction_range, restitution_range]
  ranges = torch.tensor(range_list, device="cpu")
  material_buckets = math_utils.sample_uniform(
    ranges[:, 0], ranges[:, 1], (int(num_buckets), 3), device="cpu"
  )

  if make_consistent:
    material_buckets[:, 1] = torch.min(material_buckets[:, 0], material_buckets[:, 1])

  total_num_shapes = asset.root_physx_view.max_shapes
  bucket_ids = torch.randint(
    0, int(num_buckets), (len(env_ids), total_num_shapes), device="cpu"
  )
  material_samples = material_buckets[bucket_ids]

  materials = asset.root_physx_view.get_material_properties()
  materials[env_ids] = material_samples
  asset.root_physx_view.set_material_properties(materials, env_ids)


def terrain_levels_vel_v11(
  env: ManagerBasedRLEnv,
  env_ids: Sequence[int],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Terrain curriculum for MOS9 v11 with easier progression and slower demotion.

  This keeps the same idea as terrain_levels_vel but lowers the move-up threshold
  and makes move-down condition less aggressive for low-speed commands.
  """
  asset: Articulation = env.scene[asset_cfg.name]
  terrain: TerrainImporter = env.scene.terrain
  command = env.command_manager.get_command("base_velocity")

  distance = torch.norm(
    asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Easier promotion than the default implementation.
  move_up = distance > terrain.cfg.terrain_generator.size[0] * 0.35

  # Reduce demotion pressure when command magnitudes are small.
  cmd_xy = torch.norm(command[env_ids, :2], dim=1)
  required_distance = torch.clamp(cmd_xy, min=0.05) * env.max_episode_length_s * 0.3
  move_down = distance < required_distance
  move_down *= ~move_up

  terrain.update_env_origins(env_ids, move_up, move_down)
  return torch.mean(terrain.terrain_levels.float())
