"""Test that soccer ball rolling friction is effective with condim=6.

Validates that the ball decelerates and stops within a realistic distance
when given an initial velocity, confirming that the rolling friction
parameter (friction[2]) is active.
"""

from __future__ import annotations

import mujoco
import pytest

from mjlab.terrains.soccer_field import SoccerBallCfg, get_soccer_ball_spec


def _roll_distance(
  initial_speed: float,
  cfg: SoccerBallCfg | None = None,
  dt: float = 0.002,
  t_max: float = 30.0,
) -> tuple[float, float]:
  """Simulate ball rolling on a flat plane and return (distance, final_speed).

  The ball starts in pure rolling (v = omega * R) to isolate rolling friction.
  """
  if cfg is None:
    cfg = SoccerBallCfg()

  ball_spec = get_soccer_ball_spec(cfg)
  ground_body = ball_spec.worldbody.add_body(name="ground_holder")
  ground_body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(50.0, 50.0, 0.1),
    name="ground",
    condim=3,
    friction=(cfg.friction[0], cfg.friction[1], cfg.friction[2]),
  )

  model = ball_spec.compile()
  model.opt.timestep = dt
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  # Set initial linear velocity along +x and matching angular velocity.
  data.qvel[0] = initial_speed
  data.qvel[4] = -initial_speed / cfg.radius  # omega_y for rolling in +x

  x0 = float(data.qpos[0])
  steps = int(t_max / dt)
  for _ in range(steps):
    mujoco.mj_step(model, data)
    if abs(data.qvel[0]) < 0.01:
      break

  distance = float(data.qpos[0]) - x0
  final_speed = float(abs(data.qvel[0]))
  return distance, final_speed


def test_ball_stops_within_realistic_distance():
  """Ball at 6 m/s should stop within 5-20m (RoboCup range)."""
  dist, vf = _roll_distance(6.0)
  assert vf < 0.05, f"Ball did not stop: final speed {vf:.3f} m/s"
  assert 5.0 < dist < 20.0, f"Roll distance {dist:.1f}m outside 5-20m"


def test_rolling_friction_has_effect():
  """Changing friction[2] should change the rolling distance."""
  cfg_low = SoccerBallCfg(friction=(0.5, 0.02, 0.005))
  cfg_high = SoccerBallCfg(friction=(0.5, 0.02, 0.05))

  dist_low, _ = _roll_distance(6.0, cfg_low)
  dist_high, _ = _roll_distance(6.0, cfg_high)

  assert dist_low > dist_high * 1.3, (
    f"Expected low friction to roll further: low={dist_low:.1f}m, high={dist_high:.1f}m"
  )


@pytest.mark.parametrize("speed", [2.0, 4.0, 6.0, 8.0])
def test_ball_always_stops(speed: float):
  """Ball should eventually stop at any reasonable kick speed."""
  dist, vf = _roll_distance(speed)
  assert vf < 0.05, f"Ball at {speed}m/s didn't stop: vf={vf:.3f}"
  assert dist > 0.5, f"Ball barely moved: {dist:.2f}m"
