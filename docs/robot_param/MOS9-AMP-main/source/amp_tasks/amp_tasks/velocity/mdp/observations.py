from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
  from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
  if not hasattr(env, "episode_length_buf"):
    env.episode_length_buf = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )

  global_phase = (env.episode_length_buf * env.step_dt) % period / period

  phase = torch.zeros(env.num_envs, 2, device=env.device)
  phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
  phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
  return phase


def body_quat_w(
  env: ManagerBasedRLEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Articulation = env.scene[asset_cfg.name]
  body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]
  return body_quat.reshape(env.num_envs, -1)


def body_lin_vel_w(
  env: ManagerBasedRLEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Articulation = env.scene[asset_cfg.name]
  body_lin_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
  return body_lin_vel.reshape(env.num_envs, -1)


def body_ang_vel_w(
  env: ManagerBasedRLEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Articulation = env.scene[asset_cfg.name]
  body_ang_vel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids]
  return body_ang_vel.reshape(env.num_envs, -1)


def body_quat_b(
  env: ManagerBasedRLEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Articulation = env.scene[asset_cfg.name]
  body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
  root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, body_quat_w.shape[1], -1)
  body_quat_b = math_utils.quat_mul(
    math_utils.quat_conjugate(root_quat_w.reshape(-1, 4)),
    body_quat_w.reshape(-1, 4),
  ).reshape_as(body_quat_w)
  return body_quat_b.reshape(env.num_envs, -1)


def body_lin_vel_b(
  env: ManagerBasedRLEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Articulation = env.scene[asset_cfg.name]
  body_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
  root_lin_vel_w = asset.data.root_lin_vel_w.unsqueeze(1)
  root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(
    -1, body_lin_vel_w.shape[1], -1
  )
  body_lin_vel_b = math_utils.quat_apply_inverse(
    root_quat_w.reshape(-1, 4),
    (body_lin_vel_w - root_lin_vel_w).reshape(-1, 3),
  ).reshape_as(body_lin_vel_w)
  return body_lin_vel_b.reshape(env.num_envs, -1)


def body_ang_vel_b(
  env: ManagerBasedRLEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Articulation = env.scene[asset_cfg.name]
  body_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids]
  root_ang_vel_w = asset.data.root_ang_vel_w.unsqueeze(1)
  root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(
    -1, body_ang_vel_w.shape[1], -1
  )
  body_ang_vel_b = math_utils.quat_apply_inverse(
    root_quat_w.reshape(-1, 4),
    (body_ang_vel_w - root_ang_vel_w).reshape(-1, 3),
  ).reshape_as(body_ang_vel_w)
  return body_ang_vel_b.reshape(env.num_envs, -1)


def root_lin_vel_b(
  env: ManagerBasedRLEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Articulation = env.scene[asset_cfg.name]
  return math_utils.quat_apply_inverse(
    asset.data.root_quat_w, asset.data.root_lin_vel_w
  )


def root_ang_vel_b(
  env: ManagerBasedRLEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Articulation = env.scene[asset_cfg.name]
  return math_utils.quat_apply_inverse(
    asset.data.root_quat_w, asset.data.root_ang_vel_w
  )


def body_pose_w(
  env: ManagerBasedRLEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Articulation = env.scene[asset_cfg.name]
  body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
  return body_pos.reshape(env.num_envs, -1)


def amp_obs_body_displacement(
  env: ManagerBasedRLEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  return body_pose_w(env, asset_cfg)
