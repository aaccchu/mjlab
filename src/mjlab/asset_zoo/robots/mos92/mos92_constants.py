"""MOS92 humanoid robot constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

MOS92_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "mos92" / "xmls" / "mos92.xml"
)
assert MOS92_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(MOS92_XML))


##
# Actuator config.
##

# MOS92 has two motor types:
#   - 4310: shoulders, ankle_roll (effort 36 Nm)
#   - 6408: elbows, hips, knees, ankle_pitch (effort 60 Nm)
# Parameters below are from MOS9-AMP-main (tuned for IsaacLab AMP training).
ARMATURE_4310 = 0.0282528
ARMATURE_6408 = 0.0478125
STIFFNESS_4310 = 47.177610
STIFFNESS_6408 = 105.193621
DAMPING_4310 = 1.782347
DAMPING_6408 = 2.629726

EFFORT_36 = 36.0
EFFORT_60 = 60.0

MOS92_ACTUATOR_SHOULDER = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "left_shoulder_pitch",
    "left_shoulder_roll",
  ),
  stiffness=STIFFNESS_4310,
  damping=DAMPING_4310,
  effort_limit=EFFORT_36,
  armature=ARMATURE_4310,
)

MOS92_ACTUATOR_ELBOW = BuiltinPositionActuatorCfg(
  target_names_expr=("right_elbow", "left_elbow"),
  stiffness=STIFFNESS_6408,
  damping=DAMPING_6408,
  effort_limit=EFFORT_60,
  armature=ARMATURE_6408,
)

MOS92_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
  ),
  stiffness=STIFFNESS_6408,
  damping=DAMPING_6408,
  effort_limit=EFFORT_60,
  armature=ARMATURE_6408,
)

MOS92_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=("right_knee", "left_knee"),
  stiffness=STIFFNESS_6408,
  damping=DAMPING_6408,
  effort_limit=EFFORT_60,
  armature=ARMATURE_6408,
)

MOS92_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("right_ankle_pitch", "left_ankle_pitch"),
  stiffness=STIFFNESS_6408,
  damping=DAMPING_6408,
  effort_limit=EFFORT_60,
  armature=ARMATURE_6408,
)

MOS92_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("right_ankle_roll", "left_ankle_roll"),
  stiffness=STIFFNESS_4310,
  damping=DAMPING_4310,
  effort_limit=EFFORT_36,
  armature=ARMATURE_4310,
)

MOS92_ACTUATOR_NECK = BuiltinPositionActuatorCfg(
  target_names_expr=("neck_yaw", "neck_pitch"),
  stiffness=STIFFNESS_4310,
  damping=DAMPING_4310,
  effort_limit=EFFORT_36,
  armature=ARMATURE_4310,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.45),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.43),
  joint_pos={
    ".*hip_pitch": -0.2,
    ".*knee": 0.4,
    ".*ankle_pitch": -0.2,
    # Arms hang naturally at the sides. Small outward abduction (~9deg) keeps the
    # hands clear of the thighs during leg swing (avoids hand<->thigh self-
    # collision) while reading as "arms down", not the prior ±1.4rad (±80deg)
    # T-pose that the pose reward was rewarding. See mos92_arms_out_keyframe note.
    "left_shoulder_roll": 0.15,
    "right_shoulder_roll": -0.15,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FEET_AND_BODY_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_col_\d+",),
  contype=0,
  conaffinity=1,
  condim={r"^[RL]foot_col_\d+$": 3, r".*_col_\d+": 1},
  priority={r"^[RL]foot_col_\d+$": 1},
  friction={r"^[RL]foot_col_\d+$": (0.6,)},
)

##
# Final config.
##

MOS92_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    MOS92_ACTUATOR_SHOULDER,
    MOS92_ACTUATOR_ELBOW,
    MOS92_ACTUATOR_HIP,
    MOS92_ACTUATOR_KNEE,
    MOS92_ACTUATOR_ANKLE_PITCH,
    MOS92_ACTUATOR_ANKLE_ROLL,
    MOS92_ACTUATOR_NECK,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_mos92_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FEET_AND_BODY_COLLISION,),
    spec_fn=get_spec,
    articulation=MOS92_ARTICULATION,
  )


MOS92_ACTION_SCALE: dict[str, float] = {}
for _a in MOS92_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  assert _e is not None
  for _n in _a.target_names_expr:
    MOS92_ACTION_SCALE[_n] = 0.25 * _e / _s
