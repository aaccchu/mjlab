from isaaclab.utils import configclass

from amp_tasks.amp_rsl_rl.isaaclab.configs.amp_cfg import AMPDataCfg

from ...mdp.amp_obs_grp import (
  AMPObsBaiscTerms,
  AMPObsSoft1BaseVelBTerms,
  AMPObsSoft1Terms,
)
from ..rsl_rl_ppo_cfg import BaseAMPRunnerCfg
from .velocity_env_base_cfg import MOS9_JOINT_NAMES, MOS9_KEY_BODY_NAMES


@configclass
class MOS9VelocityAMPRunnerCfg(BaseAMPRunnerCfg):
  def __post_init__(self):
    self.amp_data = AMPDataCfg()
    self.max_iterations = 6000

    self.amp_discr_hidden_dims = [256, 256]
    self.amp_task_reward_lerp = 0.7
    self.amp_lr_coef = 0.1

    self.amp_reward_coef = 0.2
    self.amp_grad_pen_coef = 10.0

    self.policy.init_noise_std = 1.0

    self.experiment_name = "mos9_loco"
    self.run_name = "amp"
    self.amp_data.body_names = MOS9_KEY_BODY_NAMES
    self.amp_data.joint_names = MOS9_JOINT_NAMES
    self.amp_data.base_body_name = "base_link"
    self.amp_data.amp_obs_terms = AMPObsBaiscTerms
    self.amp_data.motion_files = [
      "data/motions/mos9_fk_motion/B10_-__Walk_turn_left_45_stageii.npz",
      "data/motions/mos9_fk_motion/B11_-__Walk_turn_left_135_stageii.npz",
      "data/motions/mos9_fk_motion/B13_-__Walk_turn_right_90_stageii.npz",
      "data/motions/mos9_fk_motion/B14_-__Walk_turn_right_45_t2_stageii.npz",
      "data/motions/mos9_fk_motion/B15_-__Walk_turn_around_stageii.npz",
      "data/motions/mos9_fk_motion/B22_-__side_step_left_stageii.npz",
      "data/motions/mos9_fk_motion/B23_-__side_step_right_stageii.npz",
      "data/motions/mos9_fk_motion/B4_-_Stand_to_Walk_backwards_stageii.npz",
      "data/motions/mos9_fk_motion/B9_-__Walk_turn_left_90_stageii.npz",
      "data/motions/mos9_fk_motion/C11_-_run_turn_left_90_stageii.npz",
      "data/motions/mos9_fk_motion/C12_-_run_turn_left_45_stageii.npz",
      "data/motions/mos9_fk_motion/C13_-_run_turn_left_135_stageii.npz",
      "data/motions/mos9_fk_motion/C14_-_run_turn_right_90_stageii.npz",
      "data/motions/mos9_fk_motion/C15_-_run_turn_right_45_stageii.npz",
      "data/motions/mos9_fk_motion/C16_-_run_turn_right_135_stageii.npz",
      "data/motions/mos9_fk_motion/C17_-_run_change_direction_stageii.npz",
      "data/motions/mos9_fk_motion/C1_-_stand_to_run_stageii.npz",
      "data/motions/mos9_fk_motion/C3_-_run_stageii.npz",
      "data/motions/mos9_fk_motion/C4_-_run_to_walk_a_stageii.npz",
      "data/motions/mos9_fk_motion/C5_-_walk_to_run_stageii.npz",
      "data/motions/mos9_fk_motion/C6_-_stand_to_run_backwards_stageii.npz",
      "data/motions/mos9_fk_motion/C8_-_run_backwards_to_stand_stageii.npz",
      "data/motions/mos9_fk_motion/C9_-_run_backwards_turn_run_forward_stageii.npz",
      "data/motions/mos9_fk_motion/Walk_B10_-_Walk_turn_left_45_stageii.npz",
      "data/motions/mos9_fk_motion/Walk_B13_-_Walk_turn_right_45_stageii.npz",
      "data/motions/mos9_fk_motion/Walk_B15_-_Walk_turn_around_stageii.npz",
      "data/motions/mos9_fk_motion/Walk_B16_-_Walk_turn_change_stageii.npz",
      "data/motions/mos9_fk_motion/Walk_B22_-_Side_step_left_stageii.npz",
      "data/motions/mos9_fk_motion/Walk_B23_-_Side_step_right_stageii.npz",
      "data/motions/mos9_fk_motion/Walk_B4_-_Stand_to_Walk_Back_stageii.npz",
    ]


@configclass
class MOS9VelocityAMPLessMotionRunnerCfg(MOS9VelocityAMPRunnerCfg):
  def __post_init__(self):
    super().__post_init__()
    self.run_name = "amp_less_motion"

    self.amp_data.motion_files = [
      "data/motions/mos9_fk_motion/B10_-__Walk_turn_left_45_stageii.npz",
      "data/motions/mos9_fk_motion/B11_-__Walk_turn_left_135_stageii.npz",
      "data/motions/mos9_fk_motion/B13_-__Walk_turn_right_90_stageii.npz",
      "data/motions/mos9_fk_motion/B14_-__Walk_turn_right_45_t2_stageii.npz",
      "data/motions/mos9_fk_motion/B15_-__Walk_turn_around_stageii.npz",
      "data/motions/mos9_fk_motion/B22_-__side_step_left_stageii.npz",
      "data/motions/mos9_fk_motion/B23_-__side_step_right_stageii.npz",
      "data/motions/mos9_fk_motion/B4_-_Stand_to_Walk_backwards_stageii.npz",
      "data/motions/mos9_fk_motion/B9_-__Walk_turn_left_90_stageii.npz",
      "data/motions/mos9_fk_motion/Walk_B10_-_Walk_turn_left_45_stageii.npz",
      "data/motions/mos9_fk_motion/Walk_B13_-_Walk_turn_right_45_stageii.npz",
      "data/motions/mos9_fk_motion/Walk_B15_-_Walk_turn_around_stageii.npz",
      "data/motions/mos9_fk_motion/Walk_B16_-_Walk_turn_change_stageii.npz",
      "data/motions/mos9_fk_motion/Walk_B22_-_Side_step_left_stageii.npz",
      "data/motions/mos9_fk_motion/Walk_B23_-_Side_step_right_stageii.npz",
      "data/motions/mos9_fk_motion/Walk_B4_-_Stand_to_Walk_Back_stageii.npz",
    ]


@configclass
class MOS9VelocityAMPModifiedMotionRunnerCfg(MOS9VelocityAMPRunnerCfg):
  def __post_init__(self):
    super().__post_init__()
    self.run_name = "amp_modified_motion"

    self.amp_data.motion_files = [
      "data/motions/mos9_fk_motion_clipped/B10_-__Walk_turn_left_45_stageii.npz",
      "data/motions/mos9_fk_motion_clipped/B11_-__Walk_turn_left_135_stageii.npz",
      "data/motions/mos9_fk_motion_clipped/B13_-__Walk_turn_right_90_stageii.npz",
      "data/motions/mos9_fk_motion_clipped/B14_-__Walk_turn_right_45_t2_stageii.npz",
      "data/motions/mos9_fk_motion_clipped/B15_-__Walk_turn_around_stageii.npz",
      "data/motions/mos9_fk_motion_clipped/B9_-__Walk_turn_left_90_stageii.npz",
      "data/motions/mos9_fk_motion_clipped/Walk_B10_-_Walk_turn_left_45_stageii.npz",
      "data/motions/mos9_fk_motion_clipped/Walk_B13_-_Walk_turn_right_45_stageii.npz",
      "data/motions/mos9_fk_motion_clipped/Walk_B15_-_Walk_turn_around_stageii.npz",
      "data/motions/mos9_fk_motion_clipped/Walk_B16_-_Walk_turn_change_stageii.npz",
      "data/motions/mos9_fk_motion_clipped/B4_-_Stand_to_Walk_backwards_stageii.npz",
      "data/motions/mos9_fk_motion_clipped/Walk_B4_-_Stand_to_Walk_Back_stageii.npz",
      "data/motions/mos9_fk_motion_clipped/B22_-__side_step_left_stageii.npz",
      "data/motions/mos9_fk_motion_clipped/B23_-__side_step_right_stageii.npz",
      "data/motions/mos9_fk_motion_clipped/Walk_B22_-_Side_step_left_stageii.npz",
      "data/motions/mos9_fk_motion_clipped/Walk_B23_-_Side_step_right_stageii.npz",
      "data/motions/mos9_fk_motion_clipped/B22_-__side_step_left_stageii_copy1.npz",
      "data/motions/mos9_fk_motion_clipped/B23_-__side_step_right_stageii_copy1.npz",
      "data/motions/mos9_fk_motion_clipped/Walk_B22_-_Side_step_left_stageii_copy1.npz",
      "data/motions/mos9_fk_motion_clipped/Walk_B23_-_Side_step_right_stageii_copy1.npz",
      "data/motions/mos9_fk_motion_clipped/B22_-__side_step_left_stageii_copy2.npz",
      "data/motions/mos9_fk_motion_clipped/B23_-__side_step_right_stageii_copy2.npz",
      "data/motions/mos9_fk_motion_clipped/Walk_B22_-_Side_step_left_stageii_copy2.npz",
      "data/motions/mos9_fk_motion_clipped/Walk_B23_-_Side_step_right_stageii_copy2.npz",
    ]


@configclass
class MOS9AMPRunnerCfgV6(MOS9VelocityAMPRunnerCfg):
  def __post_init__(self):
    super().__post_init__()
    self.run_name = "amp_v6_rotate"
    self.amp_data.amp_obs_terms = AMPObsSoft1Terms

    self.amp_data.motion_files = [
      "data/motions/mos9_fk_motion_clipped_simple/step_left1.npz",
      "data/motions/mos9_fk_motion_clipped_simple/step_right1.npz",
      "data/motions/mos9_fk_motion_clipped_simple/step_left2.npz",
      "data/motions/mos9_fk_motion_clipped_simple/step_right2.npz",
      "data/motions/mos9_fk_motion_clipped_simple/walk_backwards.npz",
      "data/motions/mos9_fk_motion_clipped_simple/walk_straight1.npz",
      "data/motions/mos9_fk_motion_clipped_simple/walk_straight2.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_left.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_right.npz",
      "data/motions/mos9_fk_motion_clipped_simple/step_left1_copy1.npz",
      "data/motions/mos9_fk_motion_clipped_simple/step_left1_copy2.npz",
      "data/motions/mos9_fk_motion_clipped_simple/step_right1_copy1.npz",
      "data/motions/mos9_fk_motion_clipped_simple/step_right1_copy2.npz",
    ]

    self.amp_data.use_command_conditioned_sampling = True
    self.amp_data.strict_bucket_check = True
    self.amp_data.command_axis_threshold = 0.05
    self.amp_data.command_name = "base_velocity"
    self.amp_data.bucket_names = [
      "forward",
      "backward",
      "left",
      "right",
      "left_turn",
      "right_turn",
    ]
    self.amp_data.motion_buckets = {
      "forward": [
        "data/motions/mos9_fk_motion_clipped_simple/walk_straight1.npz",
        "data/motions/mos9_fk_motion_clipped_simple/walk_straight2.npz",
      ],
      "backward": [
        "data/motions/mos9_fk_motion_clipped_simple/walk_backwards.npz",
      ],
      "left": [
        "data/motions/mos9_fk_motion_clipped_simple/step_left1.npz",
        "data/motions/mos9_fk_motion_clipped_simple/step_left2.npz",
        "data/motions/mos9_fk_motion_clipped_simple/step_left1_copy1.npz",
        "data/motions/mos9_fk_motion_clipped_simple/step_left1_copy2.npz",
      ],
      "right": [
        "data/motions/mos9_fk_motion_clipped_simple/step_right1.npz",
        "data/motions/mos9_fk_motion_clipped_simple/step_right2.npz",
        "data/motions/mos9_fk_motion_clipped_simple/step_right1_copy1.npz",
        "data/motions/mos9_fk_motion_clipped_simple/step_right1_copy2.npz",
      ],
      "left_turn": [
        "data/motions/mos9_fk_rotate_motion_clipped/turn_left.npz",
      ],
      "right_turn": [
        "data/motions/mos9_fk_rotate_motion_clipped/turn_right.npz",
      ],
    }


@configclass
class MOS9AMPRunnerCfgV6Alias(MOS9AMPRunnerCfgV6):
  def __post_init__(self):
    super().__post_init__()
    self.run_name = "amp_v6_alias_turn_as_lr"
    self.amp_data.strict_bucket_check = False

    self.amp_data.motion_files = [
      "data/motions/mos9_fk_motion_clipped_simple/walk_straight1.npz",
      "data/motions/mos9_fk_motion_clipped_simple/walk_straight2.npz",
      "data/motions/mos9_fk_motion_clipped_simple/walk_backwards.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_left.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_right.npz",
    ]

    self.amp_data.motion_buckets = {
      "forward": [
        "data/motions/mos9_fk_motion_clipped_simple/walk_straight1.npz",
        "data/motions/mos9_fk_motion_clipped_simple/walk_straight2.npz",
      ],
      "backward": [
        "data/motions/mos9_fk_motion_clipped_simple/walk_backwards.npz",
      ],
      "left": [
        "data/motions/mos9_fk_rotate_motion_clipped/turn_left.npz",
      ],
      "right": [
        "data/motions/mos9_fk_rotate_motion_clipped/turn_right.npz",
      ],
      "left_turn": [
        "data/motions/mos9_fk_rotate_motion_clipped/turn_left.npz",
      ],
      "right_turn": [
        "data/motions/mos9_fk_rotate_motion_clipped/turn_right.npz",
      ],
    }


@configclass
class MOS9AMPRunnerCfgV9Alias(MOS9AMPRunnerCfgV6):
  def __post_init__(self):
    super().__post_init__()
    self.run_name = "amp_v6_alias_turn_as_lr"
    self.amp_data.strict_bucket_check = False

    self.amp_data.motion_files = [
      "data/motions/mos9_fk_motion_clipped_simple/walk_straight1.npz",
      "data/motions/mos9_fk_motion_clipped_simple/walk_straight2.npz",
      "data/motions/mos9_fk_motion_clipped_simple/walk_backwards.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_left.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_right.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_left_slow.npz",
    ]

    self.amp_data.motion_buckets = {
      "forward": [
        "data/motions/mos9_fk_motion_clipped_simple/walk_straight1.npz",
        "data/motions/mos9_fk_motion_clipped_simple/walk_straight2.npz",
      ],
      "backward": [
        "data/motions/mos9_fk_motion_clipped_simple/walk_backwards.npz",
      ],
      "left": [
        "data/motions/mos9_fk_rotate_motion_clipped/turn_left.npz",
        "data/motions/mos9_fk_rotate_motion_clipped/turn_left_slow.npz",
      ],
      "right": [
        "data/motions/mos9_fk_rotate_motion_clipped/turn_right.npz",
      ],
      "left_turn": [
        "data/motions/mos9_fk_rotate_motion_clipped/turn_left.npz",
        "data/motions/mos9_fk_rotate_motion_clipped/turn_left_slow.npz",
      ],
      "right_turn": [
        "data/motions/mos9_fk_rotate_motion_clipped/turn_right.npz",
      ],
    }


@configclass
class MOS9AMPRunnerCfgV9B(MOS9AMPRunnerCfgV6):
  def __post_init__(self):
    super().__post_init__()
    self.run_name = "amp_v6_alias_turn_as_lr"
    self.amp_data.strict_bucket_check = False

    self.amp_data.motion_files = [
      "data/motions/mos9_fk_motion_clipped_simple/walk_straight1.npz",
      "data/motions/mos9_fk_motion_clipped_simple/walk_straight2.npz",
      "data/motions/mos9_fk_motion_clipped_simple/walk_backwards.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_left_seg1.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_left_seg2.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_left_seg3.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_left_slow_seg1.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_left_slow_seg2.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_right_seg1.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_right_seg2.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_right_seg3.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_right_slow_seg1.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_right_slow_seg2.npz",
    ]

    self.amp_data.motion_buckets = {
      "forward": [
        "data/motions/mos9_fk_motion_clipped_simple/walk_straight1.npz",
        "data/motions/mos9_fk_motion_clipped_simple/walk_straight2.npz",
      ],
      "backward": [
        "data/motions/mos9_fk_motion_clipped_simple/walk_backwards.npz",
      ],
      "left": [
        "data/motions/mos9_fk_rotate_motion_clipped/turn_left_seg1.npz",
        "data/motions/mos9_fk_rotate_motion_clipped/turn_left_seg2.npz",
        "data/motions/mos9_fk_rotate_motion_clipped/turn_left_seg3.npz",
        "data/motions/mos9_fk_rotate_motion_clipped/turn_left_slow_seg1.npz",
        "data/motions/mos9_fk_rotate_motion_clipped/turn_left_slow_seg2.npz",
      ],
      "right": [
        "data/motions/mos9_fk_rotate_motion_clipped/turn_right_seg1.npz",
        "data/motions/mos9_fk_rotate_motion_clipped/turn_right_seg2.npz",
        "data/motions/mos9_fk_rotate_motion_clipped/turn_right_seg3.npz",
        "data/motions/mos9_fk_rotate_motion_clipped/turn_right_slow_seg1.npz",
        "data/motions/mos9_fk_rotate_motion_clipped/turn_right_slow_seg2.npz",
      ],
      "left_turn": [
        "data/motions/mos9_fk_rotate_motion_clipped/turn_left_seg1.npz",
        "data/motions/mos9_fk_rotate_motion_clipped/turn_left_seg2.npz",
        "data/motions/mos9_fk_rotate_motion_clipped/turn_left_seg3.npz",
        "data/motions/mos9_fk_rotate_motion_clipped/turn_left_slow_seg1.npz",
        "data/motions/mos9_fk_rotate_motion_clipped/turn_left_slow_seg2.npz",
      ],
      "right_turn": [
        "data/motions/mos9_fk_rotate_motion_clipped/turn_right_seg1.npz",
        "data/motions/mos9_fk_rotate_motion_clipped/turn_right_seg2.npz",
        "data/motions/mos9_fk_rotate_motion_clipped/turn_right_seg3.npz",
        "data/motions/mos9_fk_rotate_motion_clipped/turn_right_slow_seg1.npz",
        "data/motions/mos9_fk_rotate_motion_clipped/turn_right_slow_seg2.npz",
      ],
    }


@configclass
class MOS9AMPRunnerCfgV11(MOS9AMPRunnerCfgV6):
  def __post_init__(self):
    super().__post_init__()
    self.run_name = "amp_v11_clean_bucket"
    self.amp_data.amp_obs_terms = AMPObsSoft1BaseVelBTerms

    self.amp_data.use_command_conditioned_sampling = True
    self.amp_data.strict_bucket_check = True
    self.amp_data.command_axis_threshold = 0.05
    self.amp_data.command_name = "base_velocity"
    self.amp_data.bucket_names = [
      "forward",
      "backward",
      "left",
      "right",
      "left_turn",
      "right_turn",
    ]

    self.amp_data.motion_files = [
      "data/motions/mos9_fk_motion_clipped_simple/step_left1.npz",
      "data/motions/mos9_fk_motion_clipped_simple/step_right1.npz",
      "data/motions/mos9_fk_motion_clipped_simple/step_left2.npz",
      "data/motions/mos9_fk_motion_clipped_simple/step_right2.npz",
      "data/motions/mos9_fk_motion_clipped_simple/walk_backwards.npz",
      "data/motions/mos9_fk_motion_clipped_simple/walk_straight1.npz",
      "data/motions/mos9_fk_motion_clipped_simple/walk_straight2.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_left.npz",
      "data/motions/mos9_fk_rotate_motion_clipped/turn_right.npz",
      "data/motions/mos9_fk_motion_clipped_simple/step_left1_copy1.npz",
      "data/motions/mos9_fk_motion_clipped_simple/step_left1_copy2.npz",
      "data/motions/mos9_fk_motion_clipped_simple/step_right1_copy1.npz",
      "data/motions/mos9_fk_motion_clipped_simple/step_right1_copy2.npz",
    ]

    self.amp_data.motion_buckets = {
      "forward": [
        "data/motions/mos9_fk_motion_clipped_simple/walk_straight1.npz",
        "data/motions/mos9_fk_motion_clipped_simple/walk_straight2.npz",
      ],
      "backward": [
        "data/motions/mos9_fk_motion_clipped_simple/walk_backwards.npz",
      ],
      "left": [
        "data/motions/mos9_fk_motion_clipped_simple/step_left1.npz",
        "data/motions/mos9_fk_motion_clipped_simple/step_left2.npz",
        "data/motions/mos9_fk_motion_clipped_simple/step_left1_copy1.npz",
        "data/motions/mos9_fk_motion_clipped_simple/step_left1_copy2.npz",
      ],
      "right": [
        "data/motions/mos9_fk_motion_clipped_simple/step_right1.npz",
        "data/motions/mos9_fk_motion_clipped_simple/step_right2.npz",
        "data/motions/mos9_fk_motion_clipped_simple/step_right1_copy1.npz",
        "data/motions/mos9_fk_motion_clipped_simple/step_right1_copy2.npz",
      ],
      "left_turn": [
        "data/motions/mos9_fk_rotate_motion_clipped/turn_left.npz",
      ],
      "right_turn": [
        "data/motions/mos9_fk_rotate_motion_clipped/turn_right.npz",
      ],
    }
