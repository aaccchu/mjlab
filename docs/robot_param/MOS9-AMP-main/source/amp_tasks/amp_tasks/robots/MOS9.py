from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ASSET_DIR = str(Path(__file__).resolve().parents[4] / "data" / "assets")

# ARMATURE_4310 = 0.0235872
# ARMATURE_6408 = 0.03890875

# NATURAL_FREQ = 8 * 2.0 * 3.1415926535
# DAMPING_RATIO = 2.0

# STIFFNESS_4310 = ARMATURE_4310 * NATURAL_FREQ**2
# STIFFNESS_6408 = ARMATURE_6408 * NATURAL_FREQ**2

# DAMPING_4310 = 2.0 * DAMPING_RATIO * ARMATURE_4310 * NATURAL_FREQ
# DAMPING_6408 = 2.0 * DAMPING_RATIO * ARMATURE_6408 * NATURAL_FREQ

TORQUE_LIMIT_4310 = 36.0
TORQUE_LIMIT_6408 = 60.0

SPEED_LIMIT_4310 = 9.32
SPEED_LIMIT_6408 = 15.60


ARMATURE_4310 = 0.0282528
ARMATURE_6408 = 0.0478125
STIFFNESS_4310 = 47.177610
STIFFNESS_6408 = 105.193621
DAMPING_4310 = 1.782347
DAMPING_6408 = 2.629726


MOS9_CYLINDER_CFG = ArticulationCfg(
  spawn=sim_utils.UsdFileCfg(
    usd_path=f"{ASSET_DIR}/MOS/MOS92_urdf_0308/usd/MOS92_urdf_0308_simplified/MOS92_urdf_0308_simplified.usd",
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
      disable_gravity=False,
      retain_accelerations=False,
      linear_damping=0.0,
      angular_damping=0.0,
      max_linear_velocity=1000.0,
      max_angular_velocity=1000.0,
      max_depenetration_velocity=1.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
      fix_root_link=False,
      enabled_self_collisions=False,
      solver_position_iteration_count=4,
      solver_velocity_iteration_count=4,
    ),
  ),
  init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.56),
    joint_pos={
      "left_shoulder_roll": 1.4,
      "right_shoulder_roll": -1.4,
    },
    joint_vel={".*": 0.0},
  ),
  soft_joint_pos_limit_factor=0.9,
  actuators={
    "legs": ImplicitActuatorCfg(
      joint_names_expr=[
        ".*_hip_yaw",
        ".*_hip_roll",
        ".*_hip_pitch",
        ".*_knee",
        ".*_ankle_pitch",
        ".*_ankle_roll",
      ],
      effort_limit_sim={
        ".*_hip_yaw": TORQUE_LIMIT_6408,
        ".*_hip_roll": TORQUE_LIMIT_6408,
        ".*_hip_pitch": TORQUE_LIMIT_6408,
        ".*_knee": TORQUE_LIMIT_6408,
        ".*_ankle_pitch": TORQUE_LIMIT_6408,
        ".*_ankle_roll": TORQUE_LIMIT_4310,
      },
      velocity_limit_sim={
        ".*_hip_yaw": SPEED_LIMIT_6408,
        ".*_hip_roll": SPEED_LIMIT_6408,
        ".*_hip_pitch": SPEED_LIMIT_6408,
        ".*_knee": SPEED_LIMIT_6408,
        ".*_ankle_pitch": SPEED_LIMIT_6408,
        ".*_ankle_roll": SPEED_LIMIT_4310,
      },
      stiffness={
        ".*_hip_pitch": STIFFNESS_6408,
        ".*_hip_roll": STIFFNESS_6408,
        ".*_hip_yaw": STIFFNESS_6408,
        ".*_knee": STIFFNESS_6408,
        ".*_ankle_pitch": STIFFNESS_6408,
        ".*_ankle_roll": STIFFNESS_4310,
      },
      damping={
        ".*_hip_pitch": DAMPING_6408,
        ".*_hip_roll": DAMPING_6408,
        ".*_hip_yaw": DAMPING_6408,
        ".*_knee": DAMPING_6408,
        ".*_ankle_pitch": DAMPING_6408,
        ".*_ankle_roll": DAMPING_4310,
      },
      armature={
        ".*_hip_pitch": ARMATURE_6408,
        ".*_hip_roll": ARMATURE_6408,
        ".*_hip_yaw": ARMATURE_6408,
        ".*_knee": ARMATURE_6408,
        ".*_ankle_pitch": ARMATURE_6408,
        ".*_ankle_roll": ARMATURE_4310,
      },
    ),
    "arms": ImplicitActuatorCfg(
      joint_names_expr=[
        ".*_shoulder_pitch",
        ".*_shoulder_roll",
        ".*_elbow",
      ],
      effort_limit_sim={
        ".*_shoulder_pitch": TORQUE_LIMIT_4310,
        ".*_shoulder_roll": TORQUE_LIMIT_4310,
        ".*_elbow": TORQUE_LIMIT_6408,
      },
      velocity_limit_sim={
        ".*_shoulder_pitch": SPEED_LIMIT_4310,
        ".*_shoulder_roll": SPEED_LIMIT_4310,
        ".*_elbow": SPEED_LIMIT_6408,
      },
      stiffness={
        ".*_shoulder_pitch": STIFFNESS_4310,
        ".*_shoulder_roll": STIFFNESS_4310,
        ".*_elbow": STIFFNESS_6408,
      },
      damping={
        ".*_shoulder_pitch": DAMPING_4310,
        ".*_shoulder_roll": DAMPING_4310,
        ".*_elbow": DAMPING_6408,
      },
      armature={
        ".*_shoulder_pitch": ARMATURE_4310,
        ".*_shoulder_roll": ARMATURE_4310,
        ".*_elbow": ARMATURE_6408,
      },
    ),
  },
)


MOS9_ACTION_SCALE = {}
for actuator in MOS9_CYLINDER_CFG.actuators.values():
  effort = actuator.effort_limit_sim
  stiffness = actuator.stiffness
  names = actuator.joint_names_expr
  if not isinstance(effort, dict):
    effort = {name: effort for name in names}
  if not isinstance(stiffness, dict):
    stiffness = {name: stiffness for name in names}
  for name in names:
    if name in effort and name in stiffness and stiffness[name]:
      MOS9_ACTION_SCALE[name] = 0.25 * effort[name] / stiffness[name]
