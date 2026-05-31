"""Soccer field geometry builder for mjlab.

Builds a mid-size robot soccer field (22 x 14 m, outer-edge basis) directly
into a ``mujoco.MjSpec`` via the ``SceneCfg.spec_fn`` hook. All coordinates use
the field-center origin convention from the spec:

  - origin at field geometric center
  - x: length (left->right), y: width (bottom->top), z: up
  - all painted lines are solid bands of width ``line_width`` (default 0.125 m)
  - every dimension is measured to the OUTER edge of the line

Lines and the green pitch are thin, collision-free visual geoms layered just
above the physical ground plane. Goal posts/crossbars are collidable cylinders;
the net is a translucent, collision-free visual.

MuJoCo has no ring/arc primitive, so the center circle and corner arcs are
approximated by a fan of short box segments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco

# Render group for all field geoms. Kept out of group 0 so the velocity task's
# foot/terrain height scanners (which include only group 0) ignore the thin
# visual carpet and lines and read the physical ground plane instead.
_FIELD_GROUP = 2

# Colors (RGBA, 0-1).
_GREEN = (0.0, 130.0 / 255.0, 0.0, 1.0)
_WHITE = (1.0, 1.0, 1.0, 1.0)
_GOAL_LEFT = (0.10, 0.25, 0.90, 1.0)  # Blue.
_GOAL_RIGHT = (0.95, 0.82, 0.10, 1.0)  # Yellow.
_NET = (0.85, 0.85, 0.85, 0.30)  # Translucent.

# Z-layering (meters). Pitch and lines float just above the ground plane (z=0)
# to avoid z-fighting; the robot's feet rest on the physical plane.
_PITCH_TOP = 0.005
_LINE_TOP = 0.010
_PITCH_HALF_Z = 0.0025
_LINE_HALF_Z = 0.0025


@dataclass(kw_only=True)
class SoccerBallCfg:
  """FIFA size-5 / RoboCup MSL soccer ball (meters, kg).

  A floating, collidable sphere. Spawned as a standalone entity with its own
  freejoint (see ``get_soccer_ball_spec``) so the robot's feet can push it and
  it can be repositioned per-env on reset.
  """

  radius: float = 0.11  # Diameter 0.22 m.
  mass: float = 0.43
  rgba: tuple[float, float, float, float] = (1.0, 0.4, 0.0, 1.0)  # High-contrast.
  # MuJoCo geom friction (slide, spin, roll). Low spin/roll keeps it rolling.
  friction: tuple[float, float, float] = (0.5, 0.02, 0.01)
  solref: tuple[float, float] = (0.02, 1.0)
  # solimp: (dmin, dmax, width, midpoint, power) — MuJoCo expects 5 values.
  solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0)


def get_soccer_ball_spec(cfg: SoccerBallCfg | None = None) -> mujoco.MjSpec:
  """Build a standalone ``MjSpec`` for a soccer ball entity.

  Intended for use as an ``EntityCfg.spec_fn``. The ball is a single collidable
  sphere on a freejoint. Render group is left at 0 so it participates in
  collisions; the field carpet/lines live in group 2 and are collision-free.
  """
  if cfg is None:
    cfg = SoccerBallCfg()

  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball", pos=(0.0, 0.0, cfg.radius))
  body.add_freejoint(name="ball_joint")
  body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(cfg.radius, 0.0, 0.0),
    mass=cfg.mass,
    rgba=cfg.rgba,
    condim=3,
    friction=cfg.friction,
    solref=cfg.solref,
    solimp=cfg.solimp,
    name="ball_geom",
  )
  return spec


@dataclass(kw_only=True)
class SoccerFieldCfg:
  """Mid-size robot soccer field dimensions (meters).

  Defaults match the provided spec: 22 x 14 m field, outer-edge basis.
  """

  half_length: float = 11.0  # L/2, x in [-11, 11].
  half_width: float = 7.0  # W/2, y in [-7, 7].
  line_width: float = 0.125  # t.

  center_circle_radius: float = 2.0  # R_c (to line centerline).
  center_mark_radius: float = 0.0625

  # Goal area (small box): outer edge at |x| = half_length - goal_area_depth.
  goal_area_depth: float = 0.75
  goal_area_half_width: float = 1.95

  # Penalty area (big box).
  penalty_area_depth: float = 2.25  # x in [8.75, 11].
  penalty_area_half_width: float = 3.45

  penalty_mark_dist: float = 7.4  # |x| of the penalty mark.
  penalty_mark_radius: float = 0.10

  corner_arc_radius: float = 1.0

  # Goal 3D structure.
  goal_inner_half_width: float = 1.0  # Opening y in [-1, 1].
  goal_height: float = 1.0  # Opening z in [0, 1].
  goal_depth: float = 0.5  # Outward from the goal line.
  goal_post_radius: float = 0.05  # 0.10 m diameter.

  # Curve tessellation.
  circle_segments: int = 64
  arc_segments: int = 16

  add_goals: bool = True
  add_net: bool = True


def build_soccer_field(spec: mujoco.MjSpec, cfg: SoccerFieldCfg | None = None) -> None:
  """Add soccer field geometry to ``spec`` under a ``soccer_field`` body.

  Intended for use as a ``SceneCfg.spec_fn`` callback.
  """
  if cfg is None:
    cfg = SoccerFieldCfg()

  body = spec.worldbody.add_body(name="soccer_field")
  t = cfg.line_width
  ht = t / 2.0
  counter = [0]

  def _name(prefix: str) -> str:
    counter[0] += 1
    return f"field_{prefix}_{counter[0]}"

  def _visual_box(
    cx: float,
    cy: float,
    half_x: float,
    half_y: float,
    rgba: tuple[float, float, float, float],
    *,
    z_top: float = _LINE_TOP,
    half_z: float = _LINE_HALF_Z,
    yaw: float = 0.0,
    prefix: str = "box",
  ) -> None:
    quat = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(max(half_x, 1e-5), max(half_y, 1e-5), half_z),
      pos=(cx, cy, z_top - half_z),
      quat=quat,
      rgba=rgba,
      group=_FIELD_GROUP,
      contype=0,
      conaffinity=0,
      mass=0.0,
      name=_name(prefix),
    )

  def _visual_cylinder(
    cx: float,
    cy: float,
    radius: float,
    rgba: tuple[float, float, float, float],
    *,
    z_top: float = _LINE_TOP,
    half_z: float = _LINE_HALF_Z,
    prefix: str = "disc",
  ) -> None:
    body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=(radius, half_z, 0.0),
      pos=(cx, cy, z_top - half_z),
      rgba=rgba,
      group=_FIELD_GROUP,
      contype=0,
      conaffinity=0,
      mass=0.0,
      name=_name(prefix),
    )

  def _arc(
    cx: float,
    cy: float,
    radius: float,
    angle_start: float,
    angle_end: float,
    num_segments: int,
    prefix: str,
  ) -> None:
    da = (angle_end - angle_start) / num_segments
    # Slight overlap so segments form a continuous band.
    seg_half = radius * abs(da) / 2.0 * 1.08
    for i in range(num_segments):
      mid = angle_start + (i + 0.5) * da
      px = cx + radius * math.cos(mid)
      py = cy + radius * math.sin(mid)
      _visual_box(px, py, seg_half, ht, _WHITE, yaw=mid + math.pi / 2.0, prefix=prefix)

  # --- 1. Green pitch (visual carpet over the physical plane). ---
  _visual_box(
    0.0,
    0.0,
    cfg.half_length,
    cfg.half_width,
    _GREEN,
    z_top=_PITCH_TOP,
    half_z=_PITCH_HALF_Z,
    prefix="pitch",
  )

  # --- 2. Boundary lines (4 full rectangles, outer edge at field bound). ---
  L, W = cfg.half_length, cfg.half_width
  # Left / right side lines (run along y, thickness along x).
  _visual_box(-L + ht, 0.0, ht, W, _WHITE, prefix="sideline_l")
  _visual_box(L - ht, 0.0, ht, W, _WHITE, prefix="sideline_r")
  # Bottom / top goal lines (run along x, thickness along y).
  _visual_box(0.0, -W + ht, L, ht, _WHITE, prefix="endline_b")
  _visual_box(0.0, W - ht, L, ht, _WHITE, prefix="endline_t")

  # --- 3. Halfway line + center mark + center circle. ---
  _visual_box(0.0, 0.0, ht, W, _WHITE, prefix="halfway")
  _visual_cylinder(0.0, 0.0, cfg.center_mark_radius, _WHITE, prefix="center_mark")
  _arc(
    0.0,
    0.0,
    cfg.center_circle_radius,
    0.0,
    2.0 * math.pi,
    cfg.circle_segments,
    "center_circle",
  )

  # --- 4. Goal areas, penalty areas, penalty marks (both sides). ---
  def _box_area(front_dist: float, area_half_w: float, prefix: str) -> None:
    """Draw a three-sided line box (front + two sides), open at the goal line."""
    for sign in (+1.0, -1.0):
      goal_line_x = sign * L
      front_outer = sign * front_dist
      front_inner = front_outer - sign * t  # Toward field center.
      # Front line (parallel to goal line, thickness along x).
      _visual_box(
        front_outer - sign * ht,
        0.0,
        ht,
        area_half_w + ht,
        _WHITE,
        prefix=f"{prefix}_front",
      )
      # Two side lines (run along x from front to goal line, thickness along y).
      side_cx = (front_inner + goal_line_x) / 2.0
      side_half_x = abs(goal_line_x - front_inner) / 2.0
      for sy in (+1.0, -1.0):
        _visual_box(
          side_cx,
          sy * area_half_w,
          side_half_x,
          ht,
          _WHITE,
          prefix=f"{prefix}_side",
        )

  _box_area(
    cfg.half_length - cfg.goal_area_depth, cfg.goal_area_half_width, "goal_area"
  )
  _box_area(
    cfg.half_length - cfg.penalty_area_depth,
    cfg.penalty_area_half_width,
    "penalty_area",
  )
  for sign in (+1.0, -1.0):
    _visual_cylinder(
      sign * cfg.penalty_mark_dist,
      0.0,
      cfg.penalty_mark_radius,
      _WHITE,
      prefix="penalty_mark",
    )

  # --- 5. Corner arcs (quarter circle inside each corner). ---
  # (sx, sy) -> arc start angle (sweep +pi/2 CCW into the field).
  corners = [
    (+1.0, +1.0, math.pi),
    (+1.0, -1.0, math.pi / 2.0),
    (-1.0, +1.0, 3.0 * math.pi / 2.0),
    (-1.0, -1.0, 0.0),
  ]
  for sx, sy, a0 in corners:
    _arc(
      sx * L,
      sy * W,
      cfg.corner_arc_radius,
      a0,
      a0 + math.pi / 2.0,
      cfg.arc_segments,
      "corner_arc",
    )

  # --- 6. Goals (3D collidable frame + translucent net). ---
  if cfg.add_goals:
    _build_goals(body, cfg, _name)


def _build_goals(body: mujoco.MjsBody, cfg: SoccerFieldCfg, name_fn) -> None:
  r = cfg.goal_post_radius
  hw = cfg.goal_inner_half_width
  h = cfg.goal_height
  depth = cfg.goal_depth

  def _post(
    pos: tuple[float, float, float],
    half_len: float,
    quat: tuple[float, float, float, float],
    rgba: tuple[float, float, float, float],
  ) -> None:
    body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=(r, half_len, 0.0),
      pos=pos,
      quat=quat,
      rgba=rgba,
      group=_FIELD_GROUP,
      contype=1,
      conaffinity=1,
      condim=3,
      mass=0.0,
      name=name_fn("goal_post"),
    )

  def _net_panel(
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
  ) -> None:
    body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=size,
      pos=pos,
      rgba=_NET,
      group=_FIELD_GROUP,
      contype=0,
      conaffinity=0,
      mass=0.0,
      name=name_fn("goal_net"),
    )

  # Quaternions: vertical = identity; along-y = rotate 90deg about x;
  # along-x = rotate 90deg about y.
  q_vert = (1.0, 0.0, 0.0, 0.0)
  c, s = math.cos(math.pi / 4.0), math.sin(math.pi / 4.0)
  q_along_y = (c, s, 0.0, 0.0)
  q_along_x = (c, 0.0, s, 0.0)

  for sign, rgba in ((+1.0, _GOAL_RIGHT), (-1.0, _GOAL_LEFT)):
    gl = sign * cfg.half_length  # Goal line plane.
    back = gl + sign * depth  # Back plane (outside the field).
    # Front uprights (on the goal line) + back uprights.
    for x in (gl, back):
      for sy in (+1.0, -1.0):
        _post((x, sy * hw, h / 2.0), h / 2.0, q_vert, rgba)
    # Crossbars (front + back), along y at top.
    for x in (gl, back):
      _post((x, 0.0, h), hw, q_along_y, rgba)
    # Top side rails along x (front->back) at y = +/-hw.
    rail_cx = (gl + back) / 2.0
    for sy in (+1.0, -1.0):
      _post((rail_cx, sy * hw, h), depth / 2.0, q_along_x, rgba)

    if cfg.add_net:
      net_x = back - sign * r  # Just inside the back plane.
      _net_panel((net_x, 0.0, h / 2.0), (0.005, hw, h / 2.0))  # Back.
      _net_panel((rail_cx, 0.0, h), (depth / 2.0, hw, 0.005))  # Top.
      for sy in (+1.0, -1.0):  # Sides.
        _net_panel((rail_cx, sy * hw, h / 2.0), (depth / 2.0, 0.005, h / 2.0))
