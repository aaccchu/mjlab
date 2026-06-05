from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ....robots.MOS9_newusd import MOS9_ACTION_SCALE_NEW, MOS9_CYLINDER_CFG_NEW
from ... import mdp
from ...mdp.amp_obs_grp import (
  AMPObsBaiscCfg,
  AMPObsSoft1,
  AMPObsSoft1BaseVelBCfg,
)
from .velocity_env_base_cfg import MOS9_JOINT_NAMES, MOS9_KEY_BODY_NAMES
from .velocity_env_base_cfg import RobotEnvCfg as MOS9BaseRobotEnvCfg


@configclass
class MOS9DelayedRandomCurriculumCfgV7:
  randomize_joint_position_range = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "events.reset_robot_joints.params.position_range",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": (0.95, 1.05), "num_steps": 2000},
    },
  )
  randomize_actuator_stiffness = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "events.actuator_gains.params.stiffness_distribution_params",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": (1.0, 0.05), "num_steps": 2000},
    },
  )
  randomize_actuator_damping = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "events.actuator_gains.params.damping_distribution_params",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": (1.0, 0.05), "num_steps": 2000},
    },
  )
  randomize_physics_static_friction = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "events.physics_material.params.static_friction_range",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": (0.3, 1.6), "num_steps": 2000},
    },
  )
  randomize_physics_dynamic_friction = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "events.physics_material.params.dynamic_friction_range",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": (0.3, 1.2), "num_steps": 2000},
    },
  )
  randomize_physics_restitution = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "events.physics_material.params.restitution_range",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": (0.0, 0.5), "num_steps": 2000},
    },
  )
  randomize_base_com = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "events.base_com.params.com_range",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {
        "value": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
        "num_steps": 2000,
      },
    },
  )
  randomize_waist_mass = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "events.add_waist_mass.params.mass_distribution_params",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": (0.8, 1.2), "num_steps": 2000},
    },
  )
  randomize_physics_static_friction = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "events.physics_material_reset.params.static_friction_range",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": (0.3, 1.6), "num_steps": 2000},
    },
  )
  randomize_physics_dynamic_friction = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "events.physics_material_reset.params.dynamic_friction_range",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": (0.3, 1.2), "num_steps": 2000},
    },
  )
  randomize_physics_restitution = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "events.physics_material_reset.params.restitution_range",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": (0.0, 0.5), "num_steps": 2000},
    },
  )


@configclass
class MOS9DelayedRandomCurriculumCfgV9(MOS9DelayedRandomCurriculumCfgV7):
  randomize_physics_material_reset_mode = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "events.physics_material_reset.mode",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": "reset", "num_steps": 2000},
    },
  )
  restore_yaw_rate_when_xy_cmd_weight = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "rewards.yaw_rate_when_xy_cmd.weight",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": -1.0, "num_steps": 2000},
    },
  )
  restore_lin_vel_xy_when_yaw_cmd_weight = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "rewards.lin_vel_xy_when_yaw_cmd.weight",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": -1.0, "num_steps": 2000},
    },
  )

  restore_action_rate_weight_early = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "rewards.action_rate.weight",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": -0.12, "num_steps": 1000},
    },
  )
  restore_action_rate_weight_late = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "rewards.action_rate.weight",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": -0.15, "num_steps": 3000},
    },
  )


@configclass
class MOS9DelayedRandomCurriculumCfgV10(MOS9DelayedRandomCurriculumCfgV9):
  restore_yaw_rate_when_xy_cmd_weight_v10 = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "rewards.yaw_rate_when_xy_cmd.weight",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": -0.6, "num_steps": 2500},
    },
  )
  restore_lin_vel_xy_when_yaw_cmd_weight_v10 = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "rewards.lin_vel_xy_when_yaw_cmd.weight",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": -0.6, "num_steps": 2500},
    },
  )
  restore_action_rate_weight_to_nominal_v10 = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "rewards.action_rate.weight",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": -0.10, "num_steps": 3500},
    },
  )
  tighten_track_ang_vel_z_std_v10 = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "rewards.track_ang_vel_z.params.std",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": 0.35, "num_steps": 4000},
    },
  )
  tighten_track_ang_vel_z_deadzone_v10 = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
      "address": "rewards.track_ang_vel_z.params.deadzone",
      "modify_fn": mdp.set_value_after_step,
      "modify_params": {"value": 0.08, "num_steps": 4000},
    },
  )


@configclass
class MOS9TerrainOnlyCurriculumCfg:
  terrain_levels = CurrTerm(func=mdp.terrain_levels_vel_v11)


@configclass
class MOS9VelocityAMPEnvCfg(MOS9BaseRobotEnvCfg):
  def __post_init__(self):
    super().__post_init__()
    self.enable_rsi = True
    self.observations.amp = AMPObsBaiscCfg().adjust_key_joint_and_body_indexes(
      ["joint_pos", "joint_vel"], [], MOS9_JOINT_NAMES, []
    )

    self.rewards.track_lin_vel_xy.weight = 1.5
    self.rewards.track_ang_vel_z.weight = 1.0

    self.rewards.dof_pos_limits = None
    self.rewards.joint_vel = None
    self.rewards.joint_acc = None
    self.rewards.energy = None
    self.rewards.undesired_contacts = None
    self.rewards.base_angular_velocity = None
    self.rewards.base_linear_velocity = None

    self.rewards.action_rate.weight = -0.001
    self.rewards.alive.weight = 0.001

    self.commands.base_velocity.debug_vis = False

    self.commands.base_velocity.limit_ranges.lin_vel_x = (-0.3, 0.6)
    self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.2, 0.2)

    self.commands.base_velocity.ranges.lin_vel_x = (-0.1, 0.1)
    self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
    self.commands.base_velocity.ranges.ang_vel_z = (-0.1, 0.1)

    self.rewards.track_lin_vel_xy.params["std"] = 0.35
    self.rewards.track_ang_vel_z.params["std"] = 0.35


@configclass
class MOS9VelocityAMPEnvCfgNoCurriculum(MOS9VelocityAMPEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.curriculum = None  # curriculum会导致机器人行走速度慢，更趋向于静止，可能因为ref motion速度都很快

    self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.6)
    self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
    self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)


@configclass
class MOS9EnvCfgV6(MOS9VelocityAMPEnvCfgNoCurriculum):
  def __post_init__(self):
    super().__post_init__()

    self.observations.amp = AMPObsSoft1().adjust_key_joint_and_body_indexes(
      ["joint_pos", "joint_vel"],
      ["body_lin_vel_b"],
      MOS9_JOINT_NAMES,
      MOS9_KEY_BODY_NAMES,
    )

    # ------------------------------ Command configuration ------------------------------
    self.commands.base_velocity.class_type = (
      mdp.AxisSelectableDiscreteDecoupledAxisVelocityCommand
    )
    self.commands.base_velocity.rel_standing_envs = 0.0
    self.commands.base_velocity.active_axes = (0, 1, 2)
    # self.commands.base_velocity.discrete_lin_vel_x = (-0.4, -0.3, -0.2, 0.2, 0.3, 0.4)
    self.commands.base_velocity.discrete_lin_vel_x = (
      -0.4,
      -0.3,
      -0.2,
      0.25,
      0.35,
      0.45,
    )
    self.commands.base_velocity.discrete_lin_vel_y = (-0.4, -0.3, -0.2, 0.2, 0.3, 0.4)
    self.commands.base_velocity.discrete_ang_vel_z = (-0.6, -0.4, -0.2, 0.2, 0.4, 0.6)

    # ------------------------------ Reward configuration ------------------------------
    self.rewards.track_lin_vel_xy = None
    self.rewards.track_lin_vel_x = RewTerm(
      func=mdp.track_lin_vel_x_exp_nonzero_cmd,
      weight=1.0,
      params={"command_name": "base_velocity", "std": 0.15, "min_abs_cmd": 0.05},
    )
    self.rewards.track_lin_vel_y = RewTerm(
      func=mdp.track_lin_vel_y_exp_nonzero_cmd,
      weight=1.0,
      params={"command_name": "base_velocity", "std": 0.15, "min_abs_cmd": 0.05},
    )
    self.rewards.track_ang_vel_z = RewTerm(
      func=mdp.track_ang_vel_z_exp_nonzero_cmd,
      weight=1.0,
      params={"command_name": "base_velocity", "std": 0.20, "min_abs_cmd": 0.05},
    )

    self.rewards.wrong_direction_lin_vel = RewTerm(
      func=mdp.wrong_direction_lin_vel_penalty,
      weight=-2.0,
      params={"command_name": "base_velocity", "min_abs_cmd": 0.05},
    )
    self.rewards.orthogonal_axis_lin_vel = RewTerm(
      func=mdp.orthogonal_axis_lin_vel_penalty,
      weight=-1.0,
      params={"command_name": "base_velocity", "min_abs_cmd": 0.05},
    )
    self.rewards.yaw_rate_when_xy_cmd = RewTerm(
      func=mdp.yaw_rate_penalty_when_xy_cmd,
      weight=-1.0,
      params={"command_name": "base_velocity", "min_abs_cmd": 0.05},
    )
    self.rewards.lin_vel_xy_when_yaw_cmd = RewTerm(
      func=mdp.lin_vel_xy_penalty_when_yaw_cmd,
      weight=-1.0,
      params={"command_name": "base_velocity", "min_abs_cmd": 0.05},
    )

    # ------------------------------ Domain randomization configuration ------------------------------
    self.events.base_external_force_torque = None
    self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
    self.events.actuator_gains = EventTerm(
      func=mdp.randomize_actuator_gains,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "stiffness_distribution_params": (1.0, 0.05),
        "damping_distribution_params": (1.0, 0.05),
        "operation": "scale",
        "distribution": "gaussian",
      },
    )
    self.events.physics_material = EventTerm(
      func=mdp.randomize_rigid_body_material,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "static_friction_range": (0.3, 1.6),
        "dynamic_friction_range": (0.3, 1.2),
        "restitution_range": (0.0, 0.5),
        "num_buckets": 64,
      },
    )
    self.events.base_com = EventTerm(
      func=mdp.randomize_rigid_body_com,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
      },
    )
    self.events.add_waist_mass = EventTerm(
      func=mdp.randomize_rigid_body_mass,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", body_names=r"^(?!contact)(?!base_link$).+$"
        ),
        "mass_distribution_params": (0.8, 1.2),
        "operation": "scale",
      },
    )


@configclass
class MOS9EnvCfgV7(MOS9EnvCfgV6):
  def __post_init__(self):
    super().__post_init__()
    self.scene.robot = MOS9_CYLINDER_CFG_NEW.replace(prim_path="{ENV_REGEX_NS}/Robot")

    self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    self.events.actuator_gains.params["stiffness_distribution_params"] = (1.0, 0.0)
    self.events.actuator_gains.params["damping_distribution_params"] = (1.0, 0.0)

    self.events.physics_material_reset = EventTerm(
      func=mdp.randomize_rigid_body_material_reset_safe,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "static_friction_range": (1.0, 1.0),
        "dynamic_friction_range": (1.0, 1.0),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 64,
      },
    )

    self.events.add_waist_mass.params["mass_distribution_params"] = (1.0, 1.0)

    self.curriculum = MOS9DelayedRandomCurriculumCfgV7()


@configclass
class MOS9EnvCfgV9(MOS9EnvCfgV7):
  def __post_init__(self):
    super().__post_init__()

    self.events.physics_material_reset = EventTerm(
      func=mdp.randomize_rigid_body_material_reset_safe,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "static_friction_range": (0.8, 1.2),
        "dynamic_friction_range": (0.8, 1.1),
        "restitution_range": (0.0, 0.1),
        "num_buckets": 64,
      },
    )

    self.rewards.yaw_rate_when_xy_cmd.weight = -0.5
    self.rewards.lin_vel_xy_when_yaw_cmd.weight = -0.5
    self.rewards.action_rate.weight = -0.1

    self.commands.base_velocity.discrete_ang_vel_z = (
      -0.8,
      -0.6,
      -0.4,
      -0.2,
      0.2,
      0.4,
      0.6,
      0.8,
    )
    self.rewards.track_lin_vel_x = RewTerm(
      func=mdp.track_lin_vel_x_exp_nonzero_cmd_deadzone,
      weight=2.0,
      params={
        "command_name": "base_velocity",
        "std": 0.18,
        "min_abs_cmd": 0.05,
        "deadzone": 0.08,
      },
    )
    self.rewards.track_lin_vel_y = RewTerm(
      func=mdp.track_lin_vel_y_exp_nonzero_cmd_deadzone,
      weight=2.0,
      params={
        "command_name": "base_velocity",
        "std": 0.18,
        "min_abs_cmd": 0.05,
        "deadzone": 0.08,
      },
    )
    self.rewards.track_ang_vel_z = RewTerm(
      func=mdp.track_ang_vel_z_exp_nonzero_cmd_deadzone,
      weight=1.5,
      params={
        "command_name": "base_velocity",
        "std": 0.25,
        "min_abs_cmd": 0.05,
        "deadzone": 0.60,
      },
    )

    self.curriculum = MOS9DelayedRandomCurriculumCfgV9()


@configclass
class MOS9EnvCfgV10(MOS9EnvCfgV9):
  def __post_init__(self):
    super().__post_init__()

    self.rewards.yaw_rate_when_xy_cmd.weight = -0.2
    self.rewards.lin_vel_xy_when_yaw_cmd.weight = -0.2
    self.rewards.action_rate.weight = -0.04

    self.commands.base_velocity.discrete_ang_vel_z = (
      -1.0,
      -0.8,
      -0.6,
      -0.4,
      -0.2,
      0.2,
      0.4,
      0.6,
      0.8,
      1.0,
    )

    self.rewards.track_ang_vel_z = RewTerm(
      func=mdp.track_ang_vel_z_exp_nonzero_cmd_deadzone,
      weight=1.5,
      params={
        "command_name": "base_velocity",
        "std": 0.45,
        "min_abs_cmd": 0.05,
        "deadzone": 0.15,
      },
    )

    self.curriculum = MOS9DelayedRandomCurriculumCfgV10()


@configclass
class MOS9EnvCfgV11(MOS9EnvCfgV7):
  def __post_init__(self):
    super().__post_init__()
    # Inherited base_com targets torso_link, which is absent in the current MOS9 body names.
    self.events.base_com = None
    self.observations.amp = AMPObsSoft1BaseVelBCfg().adjust_key_joint_and_body_indexes(
      ["joint_pos", "joint_vel"],
      ["body_lin_vel_b"],
      MOS9_JOINT_NAMES,
      MOS9_KEY_BODY_NAMES,
    )
    self.scene.robot = MOS9_CYLINDER_CFG_NEW.replace(prim_path="{ENV_REGEX_NS}/Robot")
    self.actions.JointPositionAction.scale = MOS9_ACTION_SCALE_NEW

    # Keep only terrain curriculum and ensure terrain generator curriculum is enabled.
    self.curriculum = MOS9TerrainOnlyCurriculumCfg()
    if self.scene.terrain.terrain_generator is not None:
      self.scene.terrain.terrain_generator.curriculum = True

    # Enable domain randomization from the beginning (no delayed curriculum).
    self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
    self.events.physics_material_reset = EventTerm(
      func=mdp.randomize_rigid_body_material_reset_safe,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "static_friction_range": (0.3, 1.6),
        "dynamic_friction_range": (0.3, 1.5),
        "restitution_range": (0.0, 0.5),
        "num_buckets": 64,
      },
    )
    # keep mass scale strictly positive to avoid invalid inertia tensors
    self.events.add_waist_mass.params["mass_distribution_params"] = (0.8, 1.2)
    # avoid negative additive mass on base link
    self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)

    self.events.actuator_gains = EventTerm(
      func=mdp.randomize_actuator_gains,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "stiffness_distribution_params": (1.0, 0.05),
        "damping_distribution_params": (1.0, 0.05),
        "operation": "scale",
        "distribution": "gaussian",
      },
    )

    self.rewards.foot_height = RewTerm(
      func=mdp.foot_clearance_reward,
      weight=1.0,
      params={
        "std": 0.06,
        "tanh_mult": 2.5,
        "target_height": 0.15,
        "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
      },
    )
    self.rewards.foot_flat = RewTerm(
      func=mdp.feet_flat_orientation_l2,
      weight=0.2,
      params={
        "std": 0.15,
        "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
      },
    )
    self.rewards.joint_mirror = RewTerm(
      func=mdp.joint_mirror,
      weight=-0.2,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "mirror_joints": [
          ["right_shoulder_pitch", "left_shoulder_pitch"],
          ["right_shoulder_roll", "left_shoulder_roll"],
          ["right_elbow", "left_elbow"],
          ["right_hip_pitch", "left_hip_pitch"],
          ["right_hip_roll", "left_hip_roll"],
          ["right_hip_yaw", "left_hip_yaw"],
          ["right_knee", "left_knee"],
          ["right_ankle_pitch", "left_ankle_pitch"],
          ["right_ankle_roll", "left_ankle_roll"],
        ],
      },
    )

    # Reduce all velocity commands by about 20%.
    self.commands.base_velocity.discrete_lin_vel_x = (-0.4, -0.3, -0.2, 0.2, 0.3, 0.4)
    self.commands.base_velocity.discrete_lin_vel_y = (
      -0.32,
      -0.24,
      -0.16,
      0.16,
      0.24,
      0.32,
    )
    self.commands.base_velocity.discrete_ang_vel_z = (
      -0.48,
      -0.32,
      -0.16,
      0.16,
      0.32,
      0.48,
    )

    self.events.push_robot.params["velocity_range"] = {
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
      "z": (-0.2, 0.2),
      "roll": (-0.52, 0.52),
      "pitch": (-0.52, 0.52),
      "yaw": (-0.78, 0.78),
    }
    self.events.push_robot.interval_range_s = (1.0, 2.0)

    self.events.base_link_com = EventTerm(
      func=mdp.randomize_rigid_body_com,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        "com_range": {
          "x": (-0.05, 0.05),
          "y": (-0.05, 0.05),
          "z": (-0.05, 0.05),
        },
      },
    )
