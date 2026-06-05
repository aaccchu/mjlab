# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from amp_tasks.amp_rsl_rl.isaaclab.configs.amp_cfg import AMPDataCfg

from ...mdp.amp_obs_grp import AMPObsBaiscTerms
from ..rsl_rl_ppo_cfg import BaseAMPRunnerCfg, BasePPORunnerCfg
from .config import g1_key_body_names


@configclass
class G1BasePPORunnerCfg(BasePPORunnerCfg):
  experiment_name = "g1_loco"  # same as task name
  run_name = "cliped_with_lin"


@configclass
class G1FlatAMPRunnerCfg(BaseAMPRunnerCfg):
  experiment_name = "g1_loco"
  run_name = "amp"
  amp_data = AMPDataCfg(
    motion_files=[
      "data/datasets/MocapG1Full/LAFAN/walk1_subject1.npz",
      "data/datasets/MocapG1Full/LAFAN/walk1_subject2.npz",
      "data/datasets/MocapG1Full/LAFAN/walk1_subject5.npz",
      "data/datasets/MocapG1Full/LAFAN/walk2_subject1.npz",
      "data/datasets/MocapG1Full/LAFAN/walk2_subject3.npz",
      "data/datasets/MocapG1Full/LAFAN/walk2_subject4.npz",
      "data/datasets/MocapG1Full/LAFAN/walk3_subject1.npz",
      "data/datasets/MocapG1Full/LAFAN/walk3_subject2.npz",
      "data/datasets/MocapG1Full/LAFAN/walk3_subject3.npz",
      "data/datasets/MocapG1Full/LAFAN/walk3_subject4.npz",
      "data/datasets/MocapG1Full/LAFAN/walk3_subject5.npz",
      "data/datasets/MocapG1Full/LAFAN/walk4_subject1.npz",
    ],
    body_names=g1_key_body_names,
    amp_obs_terms=AMPObsBaiscTerms,
  )
  amp_discr_hidden_dims = [256, 256]
  amp_reward_coef = 0.5
  amp_task_reward_lerp = 0.3


BasePPORunnerCfg = G1BasePPORunnerCfg
