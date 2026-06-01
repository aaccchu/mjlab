"""Unitree G1 velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.dribble_command import DribbleCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.soccer_field import (
  SoccerBallCfg,
  SoccerFieldCfg,
  build_soccer_field,
  get_soccer_ball_spec,
)


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def unitree_g1_soccer_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 soccer-field dribbling configuration.

  A mid-size robot soccer field (22 x 14 m) is painted onto a flat ground plane
  via ``SceneCfg.spec_fn``, and a FIFA size-5 ball is added as a movable entity.
  The end-to-end policy learns to walk to the ball and dribble it to a random
  target point. The ``DribbleCommand`` owns the ball/target state and derives a
  base-frame twist that the inherited gait rewards consume unchanged, so all
  walking/balance shaping keeps working while the dribble rewards layer on top.
  A soft boundary truncates the episode if the robot leaves the field.
  """
  cfg = unitree_g1_flat_env_cfg(play=play)

  field_cfg = SoccerFieldCfg()
  ball_cfg = SoccerBallCfg()

  # Paint the field into the scene right before compilation.
  cfg.scene.spec_fn = lambda spec: build_soccer_field(spec, field_cfg)

  # Add the ball as a standalone movable entity (own MjSpec + freejoint), not
  # painted into the shared field worldbody, so it can be reset per-env.
  assert cfg.scene.entities is not None
  cfg.scene.entities = {
    **cfg.scene.entities,
    "ball": EntityCfg(
      spec_fn=lambda: get_soccer_ball_spec(ball_cfg),
      init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, ball_cfg.radius)),
    ),
  }

  # All parallel worlds share one field at the origin, so spawn every robot at
  # the field center (env_spacing=0) and scatter within the field via the reset
  # pose range. Without this, grid-spaced origins would place robots far outside
  # the field and trip the soft boundary on every reset.
  cfg.scene.env_spacing = 0.0
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.env_spacing = 0.0

  # Foot↔ball contact sensor (for kick reward, not added to observations).
  foot_ball_cfg = ContactSensorCfg(
    name="foot_ball_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="geom", pattern="ball_geom", entity="ball"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (foot_ball_cfg,)
  cfg.sim.nconmax = 100
  cfg.sim.contact_sensor_maxmatch = 128

  # Scatter spawns inside the field (well within the lines), random heading.
  spawn_x = field_cfg.half_length - 2.0
  spawn_y = field_cfg.half_width - 2.0
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-spawn_x, spawn_x),
    "y": (-spawn_y, spawn_y),
    "z": (0.01, 0.05),
    "yaw": (-3.14, 3.14),
  }

  # Replace the random velocity command with the dribble command. It resamples
  # only at reset (large resampling time) so the ball never teleports mid-dribble.
  cfg.commands = {
    "dribble": DribbleCommandCfg(
      entity_name="ball",
      robot_name="robot",
      resampling_time_range=(1.0e6, 1.0e6),
      debug_vis=True,
      ball_radius=ball_cfg.radius,
      half_length=field_cfg.half_length,
      half_width=field_cfg.half_width,
    ),
  }

  # Repoint every gait reward/observation that gated on the old "twist" command
  # to the derived "dribble" twist.
  for reward in cfg.rewards.values():
    if reward.params.get("command_name") == "twist":
      reward.params["command_name"] = "dribble"
  for group in ("actor", "critic"):
    cmd_term = cfg.observations[group].terms.get("command")
    if cmd_term is not None:
      cmd_term.params["command_name"] = "dribble"

  # Add ball/target observations to both groups (base-frame relative vectors).
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    terms["robot_to_ball"] = ObservationTermCfg(
      func=mdp.robot_to_ball,
      params={"command_name": "dribble", "asset_cfg": SceneEntityCfg("robot")},
    )
    terms["ball_to_target"] = ObservationTermCfg(
      func=mdp.ball_to_target,
      params={"command_name": "dribble", "asset_cfg": SceneEntityCfg("robot")},
    )
    terms["ball_velocity"] = ObservationTermCfg(
      func=mdp.ball_velocity_b,
      params={"command_name": "dribble", "asset_cfg": SceneEntityCfg("robot")},
    )

  # Dribble task rewards (layered on top of the inherited gait rewards).
  cfg.rewards["dribble_approach"] = RewardTermCfg(
    func=mdp.dribble_approach,
    weight=1.0,
    params={"command_name": "dribble", "std": 1.0},
  )
  cfg.rewards["dribble_to_target"] = RewardTermCfg(
    func=mdp.dribble_ball_to_target,
    weight=3.0,
    params={"command_name": "dribble", "std": 2.0},
  )
  cfg.rewards["dribble_ball_velocity"] = RewardTermCfg(
    func=mdp.dribble_ball_velocity_to_target,
    weight=0.5,
    params={"command_name": "dribble"},
  )
  cfg.rewards["dribble_success"] = RewardTermCfg(
    func=mdp.dribble_success_bonus,
    weight=5.0,
    params={"command_name": "dribble"},
  )
  cfg.rewards["kick_contact"] = RewardTermCfg(
    func=mdp.dribble_kick_contact,
    weight=0.3,
    params={"sensor_name": "foot_ball_contact", "command_name": "dribble"},
  )

  # The velocity-range curriculum mutated the now-removed "twist" command.
  cfg.curriculum.pop("command_vel", None)

  # Soft field boundary (truncation, not penalized).
  cfg.terminations["out_of_field_bounds"] = TerminationTermCfg(
    func=mdp.out_of_field_bounds,
    time_out=True,
    params={
      "half_length": field_cfg.half_length,
      "half_width": field_cfg.half_width,
      "margin": 0.3,
    },
  )

  return cfg
