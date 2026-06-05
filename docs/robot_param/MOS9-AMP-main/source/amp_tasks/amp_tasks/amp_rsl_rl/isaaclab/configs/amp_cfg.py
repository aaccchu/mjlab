from dataclasses import MISSING
from typing import List

from isaaclab.utils import configclass

from amp_tasks.amp_rsl_rl.runners.amp_on_policy_runner import AMPOnPolicyRunner

from .rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class AMPPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name = "AMPPPO"
  amp_replay_buffer_size: int = 100000


@configclass
class AMPDataCfg:
  asset_name: str = "robot"
  motion_files: List[str] = MISSING
  motion_buckets: dict[str, List[str]] | None = None
  use_command_conditioned_sampling: bool = False
  bucket_names: List[str] = ["forward", "backward", "left", "right"]
  command_name: str = "base_velocity"
  strict_bucket_check: bool = False
  command_axis_threshold: float = 0.02
  body_names: List[str] = MISSING
  joint_names: List[str] | None = None
  base_body_name: str = "base_link"
  amp_obs_terms: List[str] = ["joint_pos", "joint_vel"]


@configclass
class AMPRunnerCfg(RslRlOnPolicyRunnerCfg):
  runner_type: type[AMPOnPolicyRunner] = AMPOnPolicyRunner
  amp_data: AMPDataCfg = MISSING
  amp_reward_coef: float = MISSING
  amp_discr_hidden_dims: List[int] = MISSING
  amp_task_reward_lerp: float = 0.9
  amp_min_normalized_std: float = 0.0
  amp_grad_pen_coef: float = 10.0
