"""MOS92 velocity task environment configurations."""

from mjlab.asset_zoo.robots import MOS92_ACTION_SCALE, get_mos92_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import CameraSensorCfg, ContactMatch, ContactSensorCfg
from mjlab.tasks.manipulation.mdp import camera_depth
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


def mos92_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create MOS92 flat terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_mos92_robot_cfg()}

  # MOS92 has no pelvis body; the base is "base_link".
  # Remove terrain scan (no pelvis frame) — flat terrain doesn't need it.
  cfg.scene.sensors = tuple(
    s
    for s in (cfg.scene.sensors or ())
    if s.name not in ("terrain_scan", "foot_height_scan")
  )

  # Remove terrain-related observations that reference removed sensors.
  _excluded_obs = ("terrain_scan", "foot_height_scan", "height_scan", "foot_height")
  actor_obs = cfg.observations["actor"]
  actor_obs.terms = {k: v for k, v in actor_obs.terms.items() if k not in _excluded_obs}
  # Also filter critic obs if present.
  if "critic" in cfg.observations:
    critic_obs = cfg.observations["critic"]
    critic_obs.terms = {
      k: v for k, v in critic_obs.terms.items() if k not in _excluded_obs
    }

  site_names = ("left_foot", "right_foot")
  geom_names = (
    "Rfoot_col_1",
    "Rfoot_col_2",
    "Rfoot_col_3",
    "Lfoot_col_1",
    "Lfoot_col_2",
    "Lfoot_col_3",
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(Rfoot|Lfoot)$",
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
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  # Flat terrain — keep terrain entity (provides "terrain" body for contact sensor)
  # but disable rough terrain generation.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = MOS92_ACTION_SCALE

  cfg.viewer.body_name = "base_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.7

  # Foot friction randomization.
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  # Base COM randomization — use base_link (the torso).
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # Pose reward std — MOS92 has neck joints but no waist or wrist.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch": 0.3,
    r".*hip_roll": 0.15,
    r".*hip_yaw": 0.15,
    r".*knee": 0.35,
    r".*ankle_pitch": 0.25,
    r".*ankle_roll": 0.1,
    r".*shoulder_pitch": 0.15,
    r".*shoulder_roll": 0.15,
    r".*elbow": 0.15,
    r"neck_yaw": 0.1,
    r"neck_pitch": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch": 0.5,
    r".*hip_roll": 0.2,
    r".*hip_yaw": 0.2,
    r".*knee": 0.6,
    r".*ankle_pitch": 0.35,
    r".*ankle_roll": 0.15,
    r".*shoulder_pitch": 0.5,
    r".*shoulder_roll": 0.2,
    r".*elbow": 0.35,
    r"neck_yaw": 0.15,
    r"neck_pitch": 0.15,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

  # foot_slip still works (no height sensor needed), set its site_names.
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  # foot_clearance and foot_swing_height need foot_height_scan which we removed.
  del cfg.rewards["foot_clearance"]
  del cfg.rewards["foot_swing_height"]

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Flight-phase penalty: forbid BOTH feet leaving the ground at once. A biped
  # walk/dribble gait is always in single- or double-support, so a flight phase
  # only appears when the robot hops to reorient instead of stepping. Penalizing
  # it (with track_angular_velocity already rewarding commanded yaw) makes turns
  # happen by stepping. air_time_threshold ignores the brief double-float of a
  # fast gait transition; tune up if a legitimate quick gait gets suppressed.
  cfg.rewards["flight_phase"] = RewardTermCfg(
    func=mdp.flight_phase,
    weight=-1.0,
    params={"sensor_name": feet_ground_cfg.name, "air_time_threshold": 0.05},
  )

  # Remove terrain-related rewards/terminations/curriculum.
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum = {}

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg


def mos92_soccer_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create MOS92 soccer-field dribbling configuration."""
  cfg = mos92_flat_env_cfg(play=play)

  field_cfg = SoccerFieldCfg()
  ball_cfg = SoccerBallCfg()

  cfg.scene.spec_fn = lambda spec: build_soccer_field(spec, field_cfg)

  assert cfg.scene.entities is not None
  cfg.scene.entities = {
    **cfg.scene.entities,
    "ball": EntityCfg(
      spec_fn=lambda: get_soccer_ball_spec(ball_cfg),
      init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, ball_cfg.radius)),
    ),
  }

  cfg.scene.env_spacing = 0.0
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.env_spacing = 0.0

  foot_ball_cfg = ContactSensorCfg(
    name="foot_ball_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(Rfoot|Lfoot)$",
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

  # Ball domain randomization (Sim2Real). Startup mode = sampled once per env at
  # model construction, so each parallel env sees a different ball.
  #   - radius +-5% via geom_size scale (recomputes broadphase bounds).
  #   - mass +-15% via pseudo_inertia (scales mass AND inertia consistently;
  #     mass ~ e^(2*alpha), so +-15% -> alpha in (ln0.85/2, ln1.15/2)).
  #   - slide friction in [0.3, 0.7] (turf grip variation).
  #   - elasticity (restitution) via geom_solref stiffness. CALIBRATED at the real
  #     timestep (scripts/calibrate_ball_restitution.py): stiffness in [-9000,
  #     -1050] (damping -32) -> e in ~[0.58, 0.20], covering the target [0.2, 0.6].
  ball_geom_cfg = SceneEntityCfg("ball", geom_names=("ball_geom",))
  ball_body_cfg = SceneEntityCfg("ball", body_names=("ball",))
  cfg.events["ball_radius"] = EventTermCfg(
    mode="startup",
    func=dr.geom_size,
    params={
      "asset_cfg": ball_geom_cfg,
      "operation": "scale",
      "ranges": (0.95, 1.05),
      "axes": [0],  # Sphere: only the radius component matters.
      "shared_random": True,
    },
  )
  cfg.events["ball_mass"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": ball_body_cfg,
      "alpha_range": (-0.0813, 0.0699),  # +-15% mass (and inertia) consistently.
    },
  )
  cfg.events["ball_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": ball_geom_cfg,
      "operation": "abs",
      "ranges": (0.3, 0.7),  # Slide friction only (axis 0).
      "shared_random": True,
    },
  )
  cfg.events["ball_elasticity"] = EventTermCfg(
    mode="startup",
    func=dr.geom_solref,
    params={
      "asset_cfg": ball_geom_cfg,
      "operation": "abs",
      "ranges": (-9000.0, -1050.0),  # Stiffness (axis 0) -> e in ~[0.2, 0.6].
      "axes": [0],
      "shared_random": True,
    },
  )

  spawn_x = field_cfg.half_length - 2.0
  spawn_y = field_cfg.half_width - 2.0
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-spawn_x, spawn_x),
    "y": (-spawn_y, spawn_y),
    "z": (0.01, 0.05),
    "yaw": (-3.14, 3.14),
  }

  cfg.commands = {
    "dribble": DribbleCommandCfg(
      entity_name="ball",
      robot_name="robot",
      resampling_time_range=(1.0e6, 1.0e6),
      debug_vis=True,
      ball_radius=ball_cfg.radius,
      half_length=field_cfg.half_length,
      half_width=field_cfg.half_width,
      spawn_dist_range=(0.3, 0.8),
      approach_radius=0.25,
      approach_offset=0.12,
      max_speed=0.6,
    ),
  }

  for reward in cfg.rewards.values():
    if reward.params.get("command_name") == "twist":
      reward.params["command_name"] = "dribble"
  for group in ("actor", "critic"):
    cmd_term = cfg.observations[group].terms.get("command")
    if cmd_term is not None:
      cmd_term.params["command_name"] = "dribble"

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

  cfg.rewards["dribble_approach"] = RewardTermCfg(
    func=mdp.dribble_approach,
    weight=1.5,
    params={"command_name": "dribble", "std": 0.5},
  )
  cfg.rewards["dribble_to_target"] = RewardTermCfg(
    func=mdp.dribble_ball_to_target,
    weight=3.0,
    params={"command_name": "dribble", "std": 1.5},
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
    weight=1.0,
    params={"sensor_name": "foot_ball_contact", "command_name": "dribble"},
  )

  cfg.curriculum.pop("command_vel", None)

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


def mos92_soccer_gaze_env_cfg(
  play: bool = False,
  gaze_weight: float = 0.3,
  neck_pose_std_val: float = 10.0,
) -> ManagerBasedRlEnvCfg:
  """MOS92 soccer env + neck gaze reward for Spike A-MOS92."""
  cfg = mos92_soccer_env_cfg(play=play)

  cfg.rewards["gaze_at_ball"] = RewardTermCfg(
    func=mdp.gaze_at_ball,
    weight=gaze_weight,
    params={
      "command_name": "dribble",
      "std": 2.0,
      "asset_cfg": SceneEntityCfg("robot", joint_names=("neck_yaw",)),
    },
  )

  neck_pose_std = cfg.rewards["pose"].params.get("std", {})
  if isinstance(neck_pose_std, dict):
    neck_pose_std["neck_yaw"] = neck_pose_std_val
    neck_pose_std["neck_pitch"] = neck_pose_std_val

  return cfg


def mos92_soccer_goal_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """MOS92 soccer env + goal-scoring rewards for Spike E-MOS92.

  Half the episodes target the +x goal mouth; the policy is rewarded for
  driving the ball through the goal line within the opening. Ball spawns near
  the goal so near-range shooting is learnable from the G-2b checkpoint.
  """
  cfg = mos92_soccer_env_cfg(play=play)

  dribble = cfg.commands["dribble"]
  assert isinstance(dribble, DribbleCommandCfg)
  dribble.goal_target_fraction = 0.5
  dribble.goal_line_x = dribble.half_length
  dribble.goal_half_width = 1.0
  # Keep dribble distances short so the goal mouth is reachable from spawn.
  dribble.target_dist_range = (1.0, 3.0)

  # Spawn the robot in the attacking half, facing +x toward the goal, so the
  # ball (spawned just ahead) sits between robot and goal — a near-range shot.
  spawn_x_lo = dribble.half_length - 5.0
  spawn_x_hi = dribble.half_length - 2.5
  spawn_y = dribble.half_width - 2.0
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (spawn_x_lo, spawn_x_hi),
    "y": (-spawn_y, spawn_y),
    "z": (0.01, 0.05),
    "yaw": (-0.6, 0.6),  # Roughly face +x (the goal).
  }

  cfg.rewards["goal_progress"] = RewardTermCfg(
    func=mdp.goal_progress,
    weight=2.0,
    params={"command_name": "dribble"},
  )
  cfg.rewards["goal_scored"] = RewardTermCfg(
    func=mdp.goal_scored_bonus,
    weight=10.0,
    params={"command_name": "dribble"},
  )

  return cfg


def mos92_soccer_search_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """MOS92 soccer env + vision-centered gaze & search/track rewards (Spike A2).

  Validates whether the search -> lock -> approach -> kick behavior sequence
  emerges from reward gating (no hardcoded FSM). Uses GT ball geometry (no
  camera). Half the episodes spawn the ball in the rear blind sector and the
  ball is given a random initial roll velocity so the policy must search for,
  then track, a moving target.
  """
  cfg = mos92_soccer_env_cfg(play=play)

  neck_cfg = SceneEntityCfg("robot", joint_names=("neck_yaw", "neck_pitch"))

  dribble = cfg.commands["dribble"]
  assert isinstance(dribble, DribbleCommandCfg)
  dribble.rear_spawn_fraction = 0.5
  dribble.ball_init_speed_range = (0.0, 0.6)
  # Wider spawn so search is non-trivial (ball not always right at the feet).
  dribble.spawn_dist_range = (0.8, 2.5)

  for group in ("actor", "critic"):
    cfg.observations[group].terms["ball_gaze_uv"] = ObservationTermCfg(
      func=mdp.ball_gaze_uv,
      params={"command_name": "dribble", "asset_cfg": neck_cfg},
    )

  # Gaze: keep ball centered when visible; scan toward it when not.
  cfg.rewards["gaze_center"] = RewardTermCfg(
    func=mdp.gaze_center,
    weight=1.0,
    params={"command_name": "dribble", "std": 0.5, "asset_cfg": neck_cfg},
  )
  cfg.rewards["gaze_search"] = RewardTermCfg(
    func=mdp.gaze_search,
    weight=0.5,
    params={"command_name": "dribble", "asset_cfg": neck_cfg},
  )

  # Stage-1 search: penalize walking while the ball is out of view (feet still,
  # but in-place body yaw is allowed to cover the rear blind sector).
  cfg.rewards["search_freeze"] = RewardTermCfg(
    func=mdp.search_freeze,
    weight=-0.5,
    params={"command_name": "dribble", "asset_cfg": SceneEntityCfg("robot")},
  )
  # Approach the ball's predicted intercept point while visible (leads a roll).
  cfg.rewards["approach_intercept"] = RewardTermCfg(
    func=mdp.approach_intercept,
    weight=2.0,
    params={
      "command_name": "dribble",
      "tau": 0.5,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  return cfg


# Camera pose relative to the head body frame. The head frame (head body quat
# = +90deg about x) has local axes x=forward, y=up, z=right in world at neutral.
# A MuJoCo camera looks along its local -z, so to look forward with world-up the
# camera-to-head rotation is -90deg about y => quat (0.7071, 0, -0.7071, 0).
# Verified in scripts/smoke_vision.py: forward-looking (clean ground->horizon
# depth gradient, no self-occlusion), ball centered at neutral, neck_yaw/pitch
# steer the ball across the frame. Downtilt for close/ground balls comes from
# neck_pitch (range +-28.6deg), matching the foot-zone geometry finding.
_HEAD_CAM_POS = (0.08, 0.03, 0.0)
_HEAD_CAM_QUAT = (0.70710678, 0.0, -0.70710678, 0.0)


def mos92_soccer_vision_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """MOS92 soccer env + head depth camera (v3 Stage 3: vision smoke + gaze warmup).

  Bolts a depth camera onto the head body (which follows neck_yaw + neck_pitch)
  on top of the GT-based search env. Actor KEEPS all GT ball observations
  (gaze warmup, not pure vision) and additionally sees the depth image via a
  CNN branch; critic stays GT-only (asymmetric). Trains the policy to use the
  camera to find/track the ball while the GT vector keeps dribbling working.
  """
  cfg = mos92_soccer_search_env_cfg(play=play)

  head_cam = CameraSensorCfg(
    name="head_cam",
    parent_body="robot/head",
    pos=_HEAD_CAM_POS,
    quat=_HEAD_CAM_QUAT,
    fovy=60.0,
    width=64,
    height=48,
    data_types=("depth",),
    use_textures=True,
    use_shadows=False,
    enabled_geom_groups=(0, 1, 2),
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (head_cam,)

  cfg.observations["camera"] = ObservationGroupCfg(
    terms={
      "head_cam_depth": ObservationTermCfg(
        func=camera_depth,
        params={
          "sensor_name": "head_cam",
          "cutoff_distance": 3.0,
          "min_depth": 0.05,
        },
      )
    },
    enable_corruption=False,
    concatenate_terms=True,
    concatenate_dim=0,
  )

  return cfg


# Step <-> iteration mapping: common_step_counter = iteration * num_steps_per_env
# (num_steps_per_env=24 for the mos92 vision runner). The GT-ablation mask holds
# scale=1 for ~500 iters (restabilize the bootstrap), ramps 1->0 over iters
# 500-1500, then holds 0 for iters 1500-3000 (pure-CNN-for-ball training).
_MASK_NSPE = 24
_MASK_START_ITER = 500
_MASK_END_ITER = 1500


def mos92_soccer_vision_ablation_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """MOS92 vision env with the actor's GT ball obs progressively MASKED (v3b).

  Minimal GT-ablation: forces the depth CNN to carry fine ball bearing by
  ramping the actor's GT ball terms (robot_to_ball, ball_velocity, ball_gaze_uv)
  to zero via a curriculum, while keeping the obs DIMENSION at 84 so the warmup
  checkpoint loads strict. The actor's ball_to_target (= target - ball, leaks
  ball pos) is swapped for robot_to_target (legal goal direction, no ball pos).
  command + track_* gait scaffolding are KEPT (accepted coarse-ball-pos caveat).
  Critic stays full-GT (asymmetric) — only the ACTOR group is masked/swapped.
  """
  cfg = mos92_soccer_vision_env_cfg(play=play)

  # Swap ONLY the actor's ball_to_target -> robot_to_target (critic keeps GT).
  cfg.observations["actor"].terms["ball_to_target"] = ObservationTermCfg(
    func=mdp.robot_to_target,
    params={"command_name": "dribble", "asset_cfg": SceneEntityCfg("robot")},
  )

  if not play:
    cfg.curriculum["gt_mask"] = CurriculumTermCfg(
      func=mdp.mask_obs_scale,
      params={
        "group_name": "actor",
        "term_names": ["robot_to_ball", "ball_velocity", "ball_gaze_uv"],
        "start_step": _MASK_START_ITER * _MASK_NSPE,
        "end_step": _MASK_END_ITER * _MASK_NSPE,
      },
    )

  return cfg

