"""MOS92 velocity task environment configurations."""

from mjlab.asset_zoo.robots import MOS92_ACTION_SCALE, get_mos92_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg, SelfLocActionCfg
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

  # Cap shoulder_roll so the hands can never rise above shoulder height, while
  # leaving shoulder_pitch (fore/aft swing) and elbow fully free for balance.
  # Empirically calibrated (single-joint sweep, hand-vs-shoulder z + hand-thigh
  # clearance): roll RAISES the arm (pitch only swings it fore/aft, ~0 z change).
  #   right arm: roll default -0.15 == shoulder height; >-0.15 lifts above. Floor
  #              -0.6 keeps the hand ~0.29 m off the thigh (no self-collision).
  #   left arm:  mirror (default +0.15 == shoulder height; <+0.15 lifts above).
  # clip applies to the absolute joint-position target (radians), since
  # use_default_offset=True. Pitch/elbow are intentionally left unclipped so the
  # policy retains arm motion for dynamic balance (the user's explicit ask).
  joint_pos_action.clip = {
    "right_shoulder_roll": (-0.6, -0.15),
    "left_shoulder_roll": (0.15, 0.6),
  }

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

  # Anti-cheat: NON-foot body parts touching the ball = illegal contact (规则1).
  # `illegal = body_contact AND NOT foot_contact`. mode="body" matches each body
  # individually; exclude the feet AND ankles (ankle contact is essentially a low
  # foot-kick, excluding avoids false-positives on legit kicks). Everything else —
  # torso/thigh/knee/shin/arm/hand/shoulder/head — touching the ball is a foul.
  body_ball_cfg = ContactSensorCfg(
    name="body_ball_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r".*",
      entity="robot",
      exclude=("Rfoot", "Lfoot", "Rankle", "Lankle"),
    ),
    secondary=ContactMatch(mode="geom", pattern="ball_geom", entity="ball"),
    fields=("found",),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = cfg.scene.sensors + (body_ball_cfg,)

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

  # Anti-cheat rule penalties (规则1-4). Base weights here are the FULL strength;
  # a penalty_weight curriculum scales them up from 0.2 over training so the
  # policy first learns to kick before penalties bite (avoids "don't touch the
  # ball = safest"). Trapping (-3.0) is heaviest — it is the v3d straddling
  # exploit and the strongest reward-hacking attractor.
  cfg.rewards["ball_trapped"] = RewardTermCfg(
    func=mdp.ball_trapped,
    weight=-3.0,
    params={
      "foot_sensor": "foot_ball_contact",
      "body_sensor": "body_ball_contact",
      "command_name": "dribble",
    },
  )
  cfg.rewards["holding_ball"] = RewardTermCfg(
    func=mdp.holding_ball,
    weight=-2.0,
    params={"command_name": "dribble", "time_threshold": 1.5},
  )
  cfg.rewards["dangerous_high_kick"] = RewardTermCfg(
    func=mdp.illegal_body_contact,
    weight=-1.0,
    params={"sensor_name": "body_ball_contact"},
  )
  cfg.rewards["ball_sticking"] = RewardTermCfg(
    func=mdp.ball_sticking,
    weight=-1.0,
    params={
      "foot_sensor": "foot_ball_contact",
      "command_name": "dribble",
      "time_threshold": 1.0,
    },
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
    # Anti-cheat penalty ramp (Spike-C). Hold penalties weak (0.2x) while the
    # policy relearns to kick from the v3d bootstrap, then ramp to full strength
    # over iters 100->700 so trapping/holding/illegal/sticking bite once kicking
    # is established. base_weights MUST match the RewardTermCfg weights above.
    _PENALTY_TERMS = {
      "ball_trapped": -3.0,
      "holding_ball": -2.0,
      "dangerous_high_kick": -1.0,
      "ball_sticking": -1.0,
    }
    cfg.curriculum["penalty_ramp"] = CurriculumTermCfg(
      func=mdp.penalty_weight_curriculum,
      params={
        "term_names": list(_PENALTY_TERMS.keys()),
        "base_weights": _PENALTY_TERMS,
        "start_step": 100 * _MASK_NSPE,
        "end_step": 700 * _MASK_NSPE,
        "start_factor": 0.2,
      },
    )

  return cfg


# v3g self-localization step <-> iteration mapping (num_steps_per_env=24).
_SELFLOC_NSPE = 24
_SELFLOC_MASK_START_ITER = 400
_SELFLOC_MASK_END_ITER = 1200

# v3g temporal (frame-stacked RGB) self-loc. N frames stacked for inter-frame
# parallax; GT fade is SLOWER than the failed single-frame run ([800,2500] vs
# [400,1200]) so the estimate converges with GT present before the crutch goes.
_NUM_RGB_FRAMES = 4
# exp8: RGB-CNN warmup. Start GT fade LATER (1200 vs 800) so the fresh RGB branch
# converges before the crutch is pulled, and fade SLOWER (end 3500 vs 2500). The
# 4 prior pure-vision failures showed the fresh dual-CNN branch is a noise source
# that pollutes the estimate during fade; a longer warmup gives it time to settle.
_SELFLOC_VIS_MASK_START_ITER = 1200
_SELFLOC_VIS_MASK_END_ITER = 3500

# v3g exp7: widen field lines so they are VISIBLE at the 64x48 training res. The
# line-width GATE (probe_line_width_gate.py) proved the 0.125 m spec lines are
# sub-pixel from >10 m (worst-pose marking <0.5%, 2/5 poses visible); at 1.0 m
# all 5/5 poses clear the 2% bar. This is a SIM-ONLY localization aid (real-field
# lines are a fixed 0.125 m spec -> sim-to-real gap), used to test whether a CNN
# can learn pure-vision self-loc once the signal genuinely exists.
_SELFLOC_LINE_WIDTH = 1.0

# v3g real-spec active-scan temporal self-loc. The line-width / resolution /
# camera-geometry gates all proved a SINGLE frame at the real 0.125m spec can't
# see enough markings from every pose (worst-pose <1% even with narrow-FOV tilt).
# The real-robot answer (RoboCup): actively scan (neck_yaw is policy-controlled)
# and INTEGRATE landmarks seen across the sweep over time. So: real 0.125m lines,
# a modest resolution bump, and a long temporal window — more frames at a stride
# so the stack spans a ~1.5s head sweep instead of 4 consecutive 50Hz steps.
_REALSPEC_LINE_WIDTH = 0.125  # honest real-field spec — no sim-to-real gap.
_REALSPEC_RGB_FRAMES = 6
_REALSPEC_RGB_STRIDE = 6  # 6 frames x 6 steps = 36 control steps (~0.7s @ 50Hz).
_REALSPEC_CAM_RES = (96, 72)  # modest bump from 64x48 (resolution alone proven
# insufficient, but it helps once paired with the temporal scan).

# v4 Exp1+: SEPARATED cameras. Root cause of 05 not kicking: depth (ball) and RGB
# (self-loc) shared ONE head_cam, so bumping RGB resolution forced the depth image
# resolution to change, which reinit-ed the depth-ball CNN's spatial_softmax
# (H*W-dim) and ZEROED the inherited kicking skill (dribble_success 0.51 -> 0.00).
# Fix: a SECOND camera for RGB only. depth stays 64x48 (kick CNN transfers, reinit
# 0); the RGB cam gets its own (high) resolution for self-loc.
_DUALCAM_DEPTH_RES = (64, 48)  # UNCHANGED — keeps the validated ball/kick CNN.
_DUALCAM_RGB_RES = (96, 72)  # RGB-only self-loc cam; raised in later Exps (Exp2).


def mos92_soccer_selfloc_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v3g Phase A/C: replace the goal-direction spoon-feed with self-localization.

  Builds on the vision-ablation env (pure-vision ball, all anti-cheat penalties).
  SWAPS the actor's ``robot_to_target`` (which hands the policy the goal bearing
  for free) for ``robot_field_pose`` GT [x/L, y/W, sin(yaw), cos(yaw)] — so the
  policy must derive "which way is the goal" from knowing where it stands. This
  changes the actor 1D obs width 84 -> 85, so bootstraps need a PARTIAL load
  (reinit mlp.0 + normalizer; keep depth-ball CNN + deeper MLP).

  Phase A (this fn, no extra curriculum): robot_field_pose stays GT — prove the
  "know where I am -> face goal -> kick" closed loop works before adding vision.
  Phase C (spike script adds a mask curriculum on robot_field_pose): force the
  RGB CNN to recover field pose. Critic keeps full GT (asymmetric) throughout.
  """
  cfg = mos92_soccer_vision_ablation_env_cfg(play=play)

  # --- Geometry fix (root cause of the prior Phase-A failure) -----------------
  # The selfloc chain never set goal_target_fraction (default 0.0), so the dribble
  # target was a RANDOM point 2-6m from the ball. Knowing your field pose tells you
  # nothing about a random point's bearing — only a FIXED landmark (the goal at
  # x=+half_length) makes self-localization useful. Pin the target to the goal mouth
  # so the goal bearing genuinely depends on the robot's field pose. Wide initial-pose
  # randomization is inherited (base reset_base yaw +-pi, full-field xy) — DO NOT
  # narrow it; self-loc is only meaningful when the spawn pose varies.
  dribble = cfg.commands["dribble"]
  assert isinstance(dribble, DribbleCommandCfg)
  dribble.goal_target_fraction = 1.0
  dribble.goal_line_x = dribble.half_length
  dribble.goal_half_width = 1.0
  dribble.target_dist_range = (1.0, 3.0)  # keep the goal reachable from spawn.

  # Swap the actor's goal-direction leak for GT self-localization.
  cfg.observations["actor"].terms["ball_to_target"] = ObservationTermCfg(
    func=mdp.robot_field_pose,
    params={"command_name": "dribble", "asset_cfg": SceneEntityCfg("robot")},
  )

  # --- Explicit self-localization: cognitive output + accuracy reward/penalty -
  # Add a 4-d non-motor action term: the policy REPORTS its estimate of its own
  # field pose [x_n, y_n, sin(yaw), cos(yaw)]. The value never drives the sim; it
  # is scored against GT (selfloc_accuracy / selfloc_error_penalty) and fed back to
  # the next obs via the existing "actions" (last_action) term, which auto-widens
  # 20 -> 24. The motor "joint_pos" term stays first so its slice is unchanged.
  cfg.actions["selfloc"] = SelfLocActionCfg(entity_name="robot", dim=4)

  # Weights kept LOW so self-loc is an auxiliary shaping signal, not a rival to
  # the dribble task. The first full run used accuracy=1.5 + upright≈0.87, which
  # the policy could bank by just standing still and reporting its pose
  # accurately (~2 reward) instead of dribbling to the far fixed goal (hard +
  # sparse) — a textbook reward imbalance that killed dribbling. Dropping to 0.3
  # makes "stand and localize" worth far less than the ~10 dribble potential.
  cfg.rewards["selfloc_accuracy"] = RewardTermCfg(
    func=mdp.selfloc_accuracy,
    weight=0.3,
    params={"command_name": "dribble", "std": 0.5, "action_name": "selfloc"},
  )
  cfg.rewards["selfloc_error_penalty"] = RewardTermCfg(
    func=mdp.selfloc_error_penalty,
    weight=-0.3,
    params={"command_name": "dribble", "action_name": "selfloc"},
  )
  return cfg


def mos92_soccer_selfloc_vision_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v3g temporal: distill self-localization into an RGB CNN with frame-stacking.

  Builds on the explicit-selfloc env (4-d cognitive estimate head + accuracy
  reward). VISION WIRING: enable RGB on the head camera and add a "camera_rgb"
  image obs group fed to a SECOND CNN branch (the depth-ball CNN is untouched —
  depth stays for the ball, RGB carries self-loc from the painted field lines/
  goal).

  TEMPORAL MEMORY: the RGB obs is an N-frame STACK (StackedCameraRGB ->
  (B, N*3, H, W)). A single front view is ambiguous (center-circle / side lines
  are left/right symmetric, the goal is a few distant pixels); stacking frames
  gives the CNN inter-frame parallax to disambiguate as the robot moves/turns.
  This is the fix after the single-frame Phase B+C run failed (estimate could
  not localize from one frame; error grew as GT faded).

  DISTILLATION: a curriculum ramps the GT robot_field_pose obs (held under the
  "ball_to_target" key) from scale 1 -> 0. The selfloc_accuracy reward compares
  the estimate against robot_field_pose computed FRESH from true state each step
  (independent of the obs term), so the teacher survives the obs mask. As GT
  fades, the policy must read the RGB CNN to stay accurate. The fade is SLOWER
  than the failed single-frame run (iters [800, 2500] vs [400, 1200]) so the
  estimate converges with GT present before the crutch is removed. Critic keeps
  full GT (asymmetric).
  """
  cfg = mos92_soccer_selfloc_env_cfg(play=play)

  # Exp7 visibility fix: rebuild the field with WIDENED lines so the painted
  # markings are resolvable at the 64x48 training resolution (GATE-verified: at
  # 1.0 m all 5/5 probe poses clear the 2% marking-pixel bar; the 0.125 m spec
  # lines are sub-pixel from >10 m). Both training and the probe build from this
  # cfg, so they stay consistent. Sim-only aid (sim-to-real caveat noted above).
  _wide_field = SoccerFieldCfg(line_width=_SELFLOC_LINE_WIDTH)
  cfg.scene.spec_fn = lambda spec: build_soccer_field(spec, _wide_field)

  # Vision wiring: enable RGB on the head camera + add the stacked-RGB obs group.
  for sensor in cfg.scene.sensors or ():
    if isinstance(sensor, CameraSensorCfg) and sensor.name == "head_cam":
      sensor.data_types = ("depth", "rgb")
  cfg.observations["camera_rgb"] = ObservationGroupCfg(
    terms={
      "head_cam_rgb": ObservationTermCfg(
        # Stateful term instance (holds a per-env frame ring buffer). The obs
        # manager calls .reset(env_ids) on episode reset and detects it via
        # hasattr(func, "reset").
        func=mdp.StackedCameraRGB(
          None,  # type: ignore[arg-type]  # env injected lazily on first call
          sensor_name="head_cam",
          num_frames=_NUM_RGB_FRAMES,
        ),
        params={"sensor_name": "head_cam", "num_frames": _NUM_RGB_FRAMES},
      )
    },
    enable_corruption=False,
    concatenate_terms=True,
    concatenate_dim=0,
  )

  # Distillation: fade the GT self-pose obs to 0 so the RGB CNN must carry
  # self-loc. The reward's GT teacher (fresh robot_field_pose) is unaffected.
  if not play:
    cfg.curriculum["selfloc_gt_mask"] = CurriculumTermCfg(
      func=mdp.mask_obs_scale,
      params={
        "group_name": "actor",
        "term_names": ["ball_to_target"],  # holds GT robot_field_pose (4-d).
        "start_step": _SELFLOC_VIS_MASK_START_ITER * _SELFLOC_NSPE,
        "end_step": _SELFLOC_VIS_MASK_END_ITER * _SELFLOC_NSPE,
      },
    )

  # exp8: raise the self-loc accuracy weight 0.3 -> 0.8. The 4 prior pure-vision
  # failures traced (via cross-run probe comparison) to the dual-CNN branch
  # SCATTERING the self-loc head: same weight 0.3 gave 1.5 m without vision but
  # 9-11 m once the fresh RGB CNN was added. exp1 proved the head reaches 0.57 m
  # at weight 1.5 — so 0.3 was too weak to hold the mapping against the noisy
  # fresh branch. 0.8 restores gradient pressure on the head without the 1.5
  # imbalance that killed dribbling. Only on the vision env (explicit env stays 0.3).
  cfg.rewards["selfloc_accuracy"].weight = 0.8

  return cfg


def mos92_soccer_selfloc_realspec_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """REAL-SPEC pure-vision self-loc via ACTIVE SCANNING + temporal memory.

  Drops the 1.0m sim-only widened lines back to the honest 0.125m field spec
  (no sim-to-real line-width gap). The line-width / resolution / camera-geometry
  gates all proved a SINGLE frame can't see enough 0.125m markings from every
  pose. The real-robot fix (RoboCup): the neck is policy-controlled (neck_yaw +
  neck_pitch are in the 20-motor action), so the policy can SCAN the field, and
  a long temporal RGB window INTEGRATES the different landmarks the sweep brings
  into view. Changes vs the widened-line vision env:
    - field lines 1.0m -> 0.125m (real spec)
    - RGB stack 4 frames -> 6 frames at stride 6 (spans ~0.7s of head sweep,
      not 4 consecutive 50Hz steps)
    - head cam 64x48 -> 96x72 (modest; resolution alone proven insufficient but
      helps once paired with the scan)

  Bootstrap from model_2800 (find-ball + dribble + partial single-frame self-loc
  carry over); the RGB CNN's mlp.0 input grows (6*3 vs 4*3 channels) so that
  layer reinit-s, the rest transfers. Active scanning emerges from the existing
  selfloc_accuracy reward: scanning that lowers localization error is rewarded.
  """
  cfg = mos92_soccer_selfloc_vision_env_cfg(play=play)

  # Honest real-field line width — removes the sim-to-real gap.
  _real_field = SoccerFieldCfg(line_width=_REALSPEC_LINE_WIDTH)
  cfg.scene.spec_fn = lambda spec: build_soccer_field(spec, _real_field)

  # Bump head-cam resolution (depth + rgb) for finer markings.
  for sensor in cfg.scene.sensors or ():
    if isinstance(sensor, CameraSensorCfg) and sensor.name == "head_cam":
      sensor.width = _REALSPEC_CAM_RES[0]
      sensor.height = _REALSPEC_CAM_RES[1]

  # Rebuild the stacked-RGB obs with MORE frames at a STRIDE (temporal window).
  cfg.observations["camera_rgb"] = ObservationGroupCfg(
    terms={
      "head_cam_rgb": ObservationTermCfg(
        func=mdp.StackedCameraRGB(
          None,  # type: ignore[arg-type]  # env injected lazily on first call
          sensor_name="head_cam",
          num_frames=_REALSPEC_RGB_FRAMES,
          stride=_REALSPEC_RGB_STRIDE,
        ),
        params={
          "sensor_name": "head_cam",
          "num_frames": _REALSPEC_RGB_FRAMES,
          "stride": _REALSPEC_RGB_STRIDE,
        },
      )
    },
    enable_corruption=False,
    concatenate_terms=True,
    concatenate_dim=0,
  )

  return cfg


def mos92_soccer_selfloc_dualcam_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v4 Exp1: SEPARATED depth + RGB cameras (fixes 05 not kicking).

  05 stopped kicking because depth (ball) and RGB (self-loc) shared one head_cam:
  bumping RGB resolution forced the depth image resolution to change, reinit-ing
  the depth-ball CNN's spatial_softmax and zeroing the inherited kick skill
  (dribble_success 0.51 -> 0.00). Fix: keep the original head_cam at 64x48 for
  DEPTH ONLY (kick CNN transfers cleanly, reinit 0) and add a SECOND camera
  `head_cam_rgb` at its own resolution for the stacked-RGB self-loc branch.

  Starts from the real-spec self-loc env (0.125m lines + temporal scan) but undoes
  its resolution bump on the depth cam. Bootstrap from model_2800: depth CNN +
  MLP trunk + selfloc head transfer; only the RGB CNN reinit-s (fresh branch).
  """
  cfg = mos92_soccer_selfloc_realspec_env_cfg(play=play)

  # Revert the depth cam to its validated 64x48 (realspec had bumped it to 96x72).
  for sensor in cfg.scene.sensors or ():
    if isinstance(sensor, CameraSensorCfg) and sensor.name == "head_cam":
      sensor.width = _DUALCAM_DEPTH_RES[0]
      sensor.height = _DUALCAM_DEPTH_RES[1]
      sensor.data_types = ("depth",)  # depth ONLY now — RGB moves to its own cam.

  # Add a SECOND camera, co-located on the head, for RGB self-loc at its own res.
  head_cam_rgb = CameraSensorCfg(
    name="head_cam_rgb",
    parent_body="robot/head",
    pos=_HEAD_CAM_POS,
    quat=_HEAD_CAM_QUAT,
    fovy=60.0,
    width=_DUALCAM_RGB_RES[0],
    height=_DUALCAM_RGB_RES[1],
    data_types=("rgb",),
    use_textures=True,
    use_shadows=False,
    enabled_geom_groups=(0, 1, 2),
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (head_cam_rgb,)

  # Re-point the stacked-RGB obs at the NEW rgb-only camera.
  cfg.observations["camera_rgb"] = ObservationGroupCfg(
    terms={
      "head_cam_rgb": ObservationTermCfg(
        func=mdp.StackedCameraRGB(
          None,  # type: ignore[arg-type]  # env injected lazily on first call
          sensor_name="head_cam_rgb",
          num_frames=_REALSPEC_RGB_FRAMES,
          stride=_REALSPEC_RGB_STRIDE,
        ),
        params={
          "sensor_name": "head_cam_rgb",
          "num_frames": _REALSPEC_RGB_FRAMES,
          "stride": _REALSPEC_RGB_STRIDE,
        },
      )
    },
    enable_corruption=False,
    concatenate_terms=True,
    concatenate_dim=0,
  )

  return cfg


# Bootstrap-from-model_2800 ramp: GT pose obs starts ALREADY faded (≈0). The
# selfloc-vision policy we bootstrap from learned pure-vision at mask≈0, so
# re-introducing full-scale GT would be OOD (the exp8 lesson). start=0,end=1
# steps holds scale at ≈0 from the first step — no crutch, in-distribution.
_E2E_FADE_START_ITER = 0
_E2E_FADE_END_ITER = 1


def mos92_soccer_e2e_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """END-TO-END (目标①②③ in ONE policy): pure-vision self-localize + depth
  find-ball + dribble the ball INTO the fixed goal, fewest steps.

  Builds on the selfloc-vision env, which already integrates ① (selfloc head +
  accuracy reward + GT-fade distillation), ② (depth + RGB dual-CNN, ball-finding
  obs), and ③-dribble (full dribble reward stack). This adds the GOAL-SCORING
  layer ported from the goal env, plus the stability/min-steps lessons:

    - goal geometry: half the episodes aim the fixed +x goal; robot spawns in
      the attacking half facing the goal (but WIDER than the goal env so the
      policy still exercises self-localization across the field).
    - goal_progress (w2) + goal_scored (w5, the exp11-balanced value, NOT 10).
    - upright 1.0 -> 2.5: exp11 proved this breaks the "score-or-stay-upright"
      trade-off (fell_over 0.33 -> 0.14 with goal_rate UP).
    - time_to_goal_penalty (w-0.02): the "fewest steps" objective; small so it
      shapes speed without the late-collapse a heavier weight caused (exp12/13).
    - GT pose obs starts already faded (≈0): bootstrap is from model_2800 which
      localizes from vision, so we keep it in-distribution rather than re-adding
      the GT crutch.

  Trained via scripts/spike_v3g_e2e.py, which bootstraps the selfloc-vision
  policy (model_2800: depth-ball CNN + RGB selfloc CNN + trunk + selfloc head)
  so ①②③ skills carry over and only the goal-aiming behavior is newly learned.
  """
  cfg = mos92_soccer_selfloc_vision_env_cfg(play=play)

  dribble = cfg.commands["dribble"]
  assert isinstance(dribble, DribbleCommandCfg)
  dribble.goal_target_fraction = 0.5
  dribble.goal_line_x = dribble.half_length
  dribble.goal_half_width = 1.0
  dribble.target_dist_range = (1.0, 3.0)

  # Attacking-half spawn facing the goal, but WIDER than the goal env (x from
  # -2 to +8.5, full yaw spread on the non-goal half) so the policy keeps
  # self-localizing across the field, not just near-goal shooting.
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (dribble.half_length - 13.0, dribble.half_length - 2.5),
    "y": (-(dribble.half_width - 2.0), dribble.half_width - 2.0),
    "z": (0.01, 0.05),
    "yaw": (-1.2, 1.2),
  }

  cfg.rewards["goal_progress"] = RewardTermCfg(
    func=mdp.goal_progress, weight=2.0, params={"command_name": "dribble"}
  )
  cfg.rewards["goal_scored"] = RewardTermCfg(
    func=mdp.goal_scored_bonus, weight=5.0, params={"command_name": "dribble"}
  )
  cfg.rewards["time_to_goal_penalty"] = RewardTermCfg(
    func=mdp.time_to_goal_penalty, weight=-0.02, params={"command_name": "dribble"}
  )

  # exp11 stability balance: keep the robot on its feet while scoring.
  cfg.rewards["upright"].weight = 2.5

  # Keep GT pose obs faded from the start (bootstrap is already pure-vision).
  if not play:
    cfg.curriculum["selfloc_gt_mask"] = CurriculumTermCfg(
      func=mdp.mask_obs_scale,
      params={
        "group_name": "actor",
        "term_names": ["ball_to_target"],
        "start_step": _E2E_FADE_START_ITER * _SELFLOC_NSPE,
        "end_step": _E2E_FADE_END_ITER * _SELFLOC_NSPE,
      },
    )

  return cfg


def mos92_soccer_e2e_dualcam_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v4 EXP1B: 04_e2e SAME-PIPELINE + real 0.125m lines + SEPARATED cameras.

  The supervisor's key correction: 05/dualcam were the `selfloc_realspec` chain,
  NOT the same pipeline as the 04_e2e model the user benchmarks against (different
  geometry, reward layer, curriculum, bootstrap). So this builds on the ACTUAL
  e2e env (attacking-half spawn + goal-scoring rewards + e2e fade schedule) and
  only changes what v4 needs:
    - field lines 1.0m -> 0.125m (honest real spec)
    - depth cam stays 64x48 depth-only (kick CNN transfers, reinit 0)
    - add head_cam_rgb (96x72 rgb-only) for the stacked-RGB self-loc branch
  Bootstrap target: 04_e2e_integrated/model_1499 (already does full end-to-end
  kick+goal), so kicking is INHERITED, not relearned — we only test whether
  real-spec lines + separated RGB hurt self-loc. Supervisor's highest-priority
  EXP1B and the correct same-pipeline comparison to 04.
  """
  cfg = mos92_soccer_e2e_env_cfg(play=play)

  # Honest real-field line width (the whole point of v4).
  _real_field = SoccerFieldCfg(line_width=_REALSPEC_LINE_WIDTH)
  cfg.scene.spec_fn = lambda spec: build_soccer_field(spec, _real_field)

  # depth cam: keep validated 64x48 depth-only (kick CNN transfers cleanly).
  for sensor in cfg.scene.sensors or ():
    if isinstance(sensor, CameraSensorCfg) and sensor.name == "head_cam":
      sensor.width = _DUALCAM_DEPTH_RES[0]
      sensor.height = _DUALCAM_DEPTH_RES[1]
      sensor.data_types = ("depth",)

  # Add a SECOND, RGB-only camera for the self-loc branch at its own resolution.
  head_cam_rgb = CameraSensorCfg(
    name="head_cam_rgb",
    parent_body="robot/head",
    pos=_HEAD_CAM_POS,
    quat=_HEAD_CAM_QUAT,
    fovy=60.0,
    width=_DUALCAM_RGB_RES[0],
    height=_DUALCAM_RGB_RES[1],
    data_types=("rgb",),
    use_textures=True,
    use_shadows=False,
    enabled_geom_groups=(0, 1, 2),
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (head_cam_rgb,)

  # Re-point the stacked-RGB obs at the new rgb-only cam, with the temporal stride.
  cfg.observations["camera_rgb"] = ObservationGroupCfg(
    terms={
      "head_cam_rgb": ObservationTermCfg(
        func=mdp.StackedCameraRGB(
          None,  # type: ignore[arg-type]  # env injected lazily on first call
          sensor_name="head_cam_rgb",
          num_frames=_REALSPEC_RGB_FRAMES,
          stride=_REALSPEC_RGB_STRIDE,
        ),
        params={
          "sensor_name": "head_cam_rgb",
          "num_frames": _REALSPEC_RGB_FRAMES,
          "stride": _REALSPEC_RGB_STRIDE,
        },
      )
    },
    enable_corruption=False,
    concatenate_terms=True,
    concatenate_dim=0,
  )

  return cfg


def mos92_soccer_e2e_dualcam_geomcurric_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """v4 EXP2: e2e_dualcam + SPAWN-GEOMETRY CURRICULUM (full-field -> attacking).

  EXP1C decisively showed the self-loc bottleneck is geometric, not bootstrap /
  warmup / resolution: same model_2800 + GT present gave 2.80m on full-field
  geometry vs 7.07m on the attacking-half spawn. So this keeps everything from
  e2e_dualcam (real 0.125m lines, separated cams, goal-scoring layer) but morphs
  the spawn geometry over training: FULL-FIELD early (self-loc learns across
  diverse poses, the easy regime) -> ATTACKING-half late (goal-scoring regime),
  synced with the GT-fade warmup so localization is learned on easy geometry
  BEFORE both the crutch is pulled and the geometry tightens.

  Bootstrap target: model_2800 (in-distribution with GT warmup).
  """
  cfg = mos92_soccer_e2e_dualcam_env_cfg(play=play)

  if not play:
    full_field = {
      "x": (-9.0, 9.0),
      "y": (-5.0, 5.0),
      "z": (0.01, 0.05),
      "yaw": (-3.14159, 3.14159),
    }
    attacking = {
      "x": (-2.0, 8.5),  # 11 - 13, 11 - 2.5; matches e2e_dualcam
      "y": (-5.0, 5.0),
      "z": (0.01, 0.05),
      "yaw": (-1.2, 1.2),
    }
    cfg.events["reset_base"].params["pose_range"] = full_field
    cfg.curriculum["spawn_geometry"] = CurriculumTermCfg(
      func=mdp.spawn_geometry_curriculum,
      params={
        "event_name": "reset_base",
        "start_step": 800 * _SELFLOC_NSPE,
        "end_step": 2500 * _SELFLOC_NSPE,
        "full_field_range": full_field,
        "attacking_range": attacking,
      },
    )

  return cfg


def mos92_soccer_e2e_dualcam_keypoint_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """v4 EXP5c: e2e_dualcam + KEYPOINT-DETECTION self-loc (depth+Kabsch geometry).

  Paradigm shift after EXP1-5b proved regression-from-RGB collapses once the
  GT-pose crutch fades (all runs: ~2m in warmup -> ~6.7m after GT off, across
  16x resolution). Here the policy no longer regresses pose; it DETECTS K=23
  field keypoints as pixel coords (selfloc action 4 -> 46). Supervision is the
  per-frame projection of the known 3D field map (keypoint_detection_accuracy),
  which is present every frame and never masked — no crutch to lose. Pose is
  recovered geometrically (keypoint_pose_error: depth-lift -> base frame ->
  Kabsch vs map), validated to ~1cm when detections are correct.

  Bootstrap: model_2800. The RGB CNN + a fresh 46-d head learn keypoint pixels.
  """
  cfg = mos92_soccer_e2e_dualcam_env_cfg(play=play)

  # K*2 = 46-d detection head (23 field keypoints, normalized pixel coords).
  cfg.actions["selfloc"] = SelfLocActionCfg(entity_name="robot", dim=46)

  # Drop the old pose-regression rewards; install keypoint detection + monitor.
  for k in ("selfloc_accuracy", "selfloc_error_penalty"):
    cfg.rewards.pop(k, None)
  cfg.rewards["keypoint_detection"] = RewardTermCfg(
    func=mdp.keypoint_detection_accuracy,
    weight=2.5,  # EXP3 lesson: needs real gradient pressure in the e2e mix.
    params={
      "command_name": "dribble",
      "std": 1.0,  # EXP5e: std=0.15 saturated the kernel (err~2 -> exp(-89)~0,
      # zero gradient, keypoint_detection reward stuck at 0.0000). std=1.0 puts
      # the gradient ramp across err in [0,1.5] so PPO can actually descend.
      # Cheap falsification: if PPO STILL can't learn 46-d dense regression with
      # a healthy gradient, that proves a supervised aux-loss is required (EXP5f).
      "sensor_name": "head_cam",
      "action_name": "selfloc",
    },
  )
  cfg.rewards["keypoint_pose_error"] = RewardTermCfg(
    func=mdp.keypoint_pose_error,
    weight=1e-8,  # ~zero (term returns zeros) but NON-zero so the manager runs
    # it: weight==0 terms are SKIPPED (reward_manager.py:122), which would
    # suppress the kp_selfloc_pos_err_m geometry monitor.
    params={
      "command_name": "dribble",
      "sensor_name": "head_cam",
      "action_name": "selfloc",
    },
  )

  # Geometry needs GT pose removed from obs entirely — keypoints don't use it.
  # The old selfloc_gt_mask curriculum (if present) is now irrelevant; the
  # detection label comes from projection, not a faded GT obs term.

  # EXP5f A': training-only label group. project_keypoints -> (uv_x|uv_y|vis, K
  # each). Enters rollout storage (so it aligns with each sampled transition) but
  # is NOT added to the actor/critic obs_groups (see rl_cfg) — the policy never
  # sees it; only KeypointAuxPPO's supervised aux loss consumes it.
  cfg.observations["keypoint_label"] = ObservationGroupCfg(
    terms={
      "kp_uv": ObservationTermCfg(
        func=mdp.keypoint_uv_label,
        params={"sensor_name": "head_cam"},
      )
    },
    enable_corruption=False,
    concatenate_terms=True,
    concatenate_dim=0,
  )
  return cfg


def mos92_soccer_e2e_dualcam_oracle_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """v4 EXP6: GT-landmark ORACLE pose belief as obs — perception OUT of the trunk.

  EXP5 paradigm finding: the actor is a single shared MLP trunk emitting both
  joint_pos and the selfloc detection slice, so ANY perception learning signal
  (reward OR supervised loss) backprops through the trunk and corrupts the gait
  (fell_over climbed 1 -> 37 -> 52 with gradient strength). The fix is structural:
  perception must not be an action of the control policy.

  Here the geometry chain runs in an OBSERVATION (oracle_pose_belief): project the
  known field keypoints with TRUE visibility (oracle = perfect detection, NOT
  perfect pose), depth-lift, Kabsch vs map -> a pose belief [x,y,sin,cos] + quality
  signals [visible_frac, residual]. The soccer policy CONSUMES this belief; it has
  no selfloc action and no keypoint reward, so no perception gradient touches the
  gait trunk. This is codex diagnosis step #1: prove the back-end (belief -> kick)
  is viable under perfect detection, AND remove the fell_over pollution source.

  Derives from the dualcam keypoint base for the same kicking inheritance, but
  strips the keypoint head/reward and swaps the GT-pose obs slot for the belief.
  """
  cfg = mos92_soccer_e2e_dualcam_keypoint_env_cfg(play=play)

  # 1. Remove the keypoint detection head (action) and its reward — perception is
  #    no longer an action of the policy.
  cfg.actions.pop("selfloc", None)
  for k in ("keypoint_detection", "keypoint_pose_error"):
    cfg.rewards.pop(k, None)
  cfg.observations.pop("keypoint_label", None)

  # 2. Swap the actor's pose slot (held under the "ball_to_target" key, currently
  #    robot_field_pose GT) for the geometry-recovered ORACLE belief (6-dim).
  cfg.observations["actor"].terms["ball_to_target"] = ObservationTermCfg(
    func=mdp.oracle_pose_belief,
    params={"command_name": "dribble", "sensor_name": "head_cam"},
  )

  # 3. Remove any curriculum that fades/masks that slot — the belief is not GT, so
  #    there is nothing to fade out (it degrades naturally when keypoints aren't
  #    visible, which is the honest signal).
  for cur_key in list(cfg.curriculum.keys()):
    cur = cfg.curriculum[cur_key]
    names = getattr(cur, "params", {}).get("term_names", [])
    if "ball_to_target" in names or cur_key == "selfloc_gt_mask":
      cfg.curriculum.pop(cur_key, None)

  # 4. Monitor (weight 0): log the belief's pose error vs GT so the run is readable
  #    (oracle_pose_belief is an obs and logs nothing on its own).
  cfg.rewards["oracle_pose_belief_error"] = RewardTermCfg(
    func=mdp.oracle_pose_belief_error,
    weight=1e-8,  # ~zero (term returns zeros) but NON-zero so the manager RUNS it:
    # weight==0 terms are SKIPPED (reward_manager.py:122), which would suppress the
    # selfloc_pos_err_m / belief_vis_frac geometry monitor.
    params={"command_name": "dribble", "sensor_name": "head_cam"},
  )

  return cfg


def mos92_soccer_e2e_dualcam_fused_scan_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """v4 EXP8: TEMPORAL FUSION + ACTIVE NECK SCAN on top of the oracle belief.

  EXP6 fixed the architecture (perception-as-obs, fell_over 52->0). EXP7 proved
  single-frame belief is stuck at ~4/23 keypoints (pos_err ~4m) and passive
  multi-frame fusion barely helps (unique coverage 3.9->4.4) because a forward-
  walking robot keeps the SAME few keypoints in view. EXP8 offline analysis showed
  the head cam rides the neck (neck_yaw +-90deg), and a neck sweep raises unique
  coverage 5 -> 9-14; the FusedPoseBelief validation hit median 1.03m PASSIVELY.

  This env unlocks the scan behavior:
    1. swap the single-frame oracle belief for FusedPoseBelief (odometry-fused
       N-frame window -> 7-dim belief incl. uniq_frac);
    2. add active_scan_coverage reward (rewards raising uniq_frac toward 0.6, only
       achievable by sweeping neck_yaw to new landmarks);
    3. relax the neck pose-reward pull so the policy is free to sweep instead of
       being dragged back to neck-centered.
  Still oracle DETECTION (isolate the detector variable, per codex's stage order).
  """
  cfg = mos92_soccer_e2e_dualcam_oracle_env_cfg(play=play)

  # 1. Swap single-frame belief -> temporally-fused belief (7-dim). Registered as
  #    a stateful instance with env=None (lazily injected on first call, like
  #    StackedCameraRGB).
  cfg.observations["actor"].terms["ball_to_target"] = ObservationTermCfg(
    func=mdp.FusedPoseBelief(
      None,  # type: ignore[arg-type]  # env injected lazily on first call
      command_name="dribble",
      sensor_name="head_cam",
      num_frames=8,
      stride=4,
    ),
    params={
      "command_name": "dribble",
      "sensor_name": "head_cam",
      "num_frames": 8,
      "stride": 4,
    },
  )

  # 2. Active-scan coverage reward (drives neck sweeping toward more unique kp).
  cfg.rewards["active_scan"] = RewardTermCfg(
    func=mdp.active_scan_coverage,
    weight=0.5,
    params={"command_name": "dribble", "sensor_name": "head_cam", "target_frac": 0.6},
  )

  # 3. Relax the neck pose-reward pull (was std 0.1/0.15 = strong centering) so the
  #    policy can sweep neck_yaw freely. Larger std = weaker pull.
  for key in ("std", "std_running"):
    d = cfg.rewards["pose"].params.get(key)
    if isinstance(d, dict):
      for jk in list(d.keys()):
        if "neck_yaw" in jk:
          d[jk] = 1.5  # very weak centering on the scan joint

  # 4. Replace the single-frame belief monitor with the FUSED-belief monitor so
  #    Metrics/selfloc_pos_err_m reflects what the policy actually consumes.
  cfg.rewards.pop("oracle_pose_belief_error", None)
  cfg.rewards["fused_belief_error"] = RewardTermCfg(
    func=mdp.fused_belief_error,
    weight=1e-8,  # ~0 but non-zero so the manager runs the monitor.
    params={"command_name": "dribble"},
  )

  return cfg


def mos92_soccer_e2e_dualcam_gated_scan_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """v4 EXP9: CERTAINTY-GATED active scan. EXP8 showed ball-staring (gaze_center
  w=1.0) out-competes active_scan (0.5) for the single neck joint, so the policy
  never sweeps (uniq_frac stalled ~0.24, pos_err stuck 1.8m from passive fusion
  alone). Fix = time-share by belief certainty: gate the gaze (ball-staring)
  rewards by uniq_frac so staring pays nothing while the belief is poor, making
  scanning the only way to earn reward until coverage is good; then gaze reopens
  for kicking. Also bump active_scan 0.5 -> 1.0 to strengthen the sweep gradient.
  """
  cfg = mos92_soccer_e2e_dualcam_fused_scan_env_cfg(play=play)

  # Replace gaze rewards with certainty-gated versions (keep weights).
  if "gaze_center" in cfg.rewards:
    w = cfg.rewards["gaze_center"].weight
    cfg.rewards["gaze_center"] = RewardTermCfg(
      func=mdp.gaze_center_gated,
      weight=w,
      params={"command_name": "dribble", "std": 0.5, "certain_frac": 0.4},
    )
  if "gaze_search" in cfg.rewards:
    w = cfg.rewards["gaze_search"].weight
    cfg.rewards["gaze_search"] = RewardTermCfg(
      func=mdp.gaze_search_gated,
      weight=w,
      params={"command_name": "dribble", "certain_frac": 0.4},
    )

  # Strengthen the scan gradient (0.5 -> 1.0) now that gaze no longer dominates.
  if "active_scan" in cfg.rewards:
    cfg.rewards["active_scan"].weight = 1.0

  return cfg


def mos92_soccer_e2e_dualcam_neck_motion_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """v4 EXP10: DIRECT neck-motion reward. EXP9 proved the indirect coverage reward
  (active_scan on uniq_frac) can't teach the neck to sweep even when it's the only
  reward — the credit path turn->coverage->reward is too long. EXP10 adds a dense
  reward DIRECTLY on |neck_yaw angular velocity|, gated to fire while the belief is
  uncertain, so the policy first learns the sweep ACTION; the coverage/gaze rewards
  then shape WHERE to look. Built on the EXP9 gated env (keeps gated gaze so staring
  doesn't dominate once coverage is good)."""
  cfg = mos92_soccer_e2e_dualcam_gated_scan_env_cfg(play=play)

  cfg.rewards["neck_scan_motion"] = RewardTermCfg(
    func=mdp.neck_scan_motion,
    weight=0.8,
    params={"command_name": "dribble", "certain_frac": 0.4, "vel_scale": 2.0},
  )
  return cfg


def mos92_soccer_e2e_dualcam_ekf_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """v4 line-A R1: RECURSIVE EKF self-localization. EXP12 offline proved a recursive
  SE2-EKF beats the point-cloud-pooling Kabsch of FusedPoseBelief by 55% (1.21m vs
  2.68m mean pos_err) on the SAME rollout — pooling adds no constraint when the robot
  walks one way, the EKF accumulates each sparse landmark observation recursively.

  Derives from fused_scan (keeps active_scan reward + relaxed neck so scanning is
  still incentivized) and swaps FusedPoseBelief -> EkfPoseBelief. Output stays 7-dim
  with the same layout, so the actor obs dim is unchanged and EXP11 weights bootstrap
  directly. The fused_belief_error monitor matches EkfPoseBelief too (it subclasses
  FusedPoseBelief), so Metrics/selfloc_pos_err_m logs the EKF belief automatically.
  Still oracle DETECTION (isolate the detector variable, per codex's stage order).
  """
  cfg = mos92_soccer_e2e_dualcam_fused_scan_env_cfg(play=play)

  cfg.observations["actor"].terms["ball_to_target"] = ObservationTermCfg(
    func=mdp.EkfPoseBelief(
      None,  # type: ignore[arg-type]  # env injected lazily on first call
      command_name="dribble",
      sensor_name="head_cam",
      num_frames=8,
      stride=4,
    ),
    params={
      "command_name": "dribble",
      "sensor_name": "head_cam",
      "num_frames": 8,
      "stride": 4,
    },
  )
  return cfg


def mos92_soccer_e2e_dualcam_ekf_kick_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """v4 line-B B2: train the near-foot KICK SKILL on top of the EKF env (line A).

  EXP13 (EKF) hit pos_err 0.98m and goal_rate 0.08 — localization is solved but the
  final "ball-at-foot -> kick goalward" action is missing (codex upper bound 0.68 vs
  0.018 even with god-view command injection => it's an action-skill gap, not a
  perception or reward-axis gap). This env attacks that gap directly:
    1. near_foot_spawn_fraction=0.5: half the episodes spawn the ball already in the
       kick window (~0.1m in front of the heading) so the policy DENSELY practices
       striking from the foot; the other half keep the full approach+dribble task so
       the approach/locomotion/localization skills are not forgotten.
    2. dribble_kick_impulse reward: rewards ball speed projected goalward AT the
       contact step (kick quality), which binary kick_contact cannot teach.
  Bootstraps from EXP13 (same 88-dim obs), so gait+approach+EKF-localization carry
  over. Still oracle DETECTION (isolate the action-skill variable).
  """
  cfg = mos92_soccer_e2e_dualcam_ekf_env_cfg(play=play)

  dribble = cfg.commands["dribble"]
  dribble.near_foot_spawn_fraction = 0.5
  # near_foot_dist is measured from root_link (pelvis), NOT the foot. The toes sit
  # ~0.10-0.13m in front of root and the ball radius is ~0.11m, so a center-to-root
  # distance below ~0.23m would spawn the ball INSIDE the foot/shin geometry ->
  # reset penetration -> the ball gets flung out on step 1 (false-kick reward noise +
  # out-of-bounds). EXP14 used (0.08,0.20) and out_of_bounds rose 0.21->0.28. Lower
  # bound raised to clear the foot; the window still starts the policy near the ball.
  dribble.near_foot_dist_range = (0.25, 0.40)
  # The EKF env inherits rear_spawn_fraction=0.5 from fused_scan (line-A scan
  # training). The kick skill does not need rear-blind search, and rear/near-foot
  # are not mutually exclusive in the sampler (near-foot silently wins on overlap,
  # scrambling the intended distribution). Disable rear spawn for the kick env.
  dribble.rear_spawn_fraction = 0.0

  cfg.rewards["kick_impulse"] = RewardTermCfg(
    func=mdp.dribble_kick_impulse,
    weight=1.5,
    params={
      "sensor_name": "foot_ball_contact",
      "command_name": "dribble",
      # Gate out gentle pushing: EXP14 stalled with ball_speed~0.39, so only reward
      # goalward strikes above 0.6 m/s to push the policy from "nudge" to "kick".
      "speed_threshold": 0.6,
    },
  )
  return cfg
