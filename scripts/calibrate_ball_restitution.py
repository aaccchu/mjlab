"""Empirically calibrate MuJoCo solref dampratio -> coefficient of restitution.

MuJoCo has no direct restitution coefficient; bounce emerges from the contact
solref = (timeconst, dampratio). dampratio=1 is critically damped (no bounce),
dampratio<1 is underdamped (bounces). The map dampratio->e is monotonic but not
closed-form, so we measure it: drop the RoboCup ball (r=0.11, m=0.43) from a
fixed height onto a plane and read e = sqrt(h_rebound / h_drop).

Goal: find the dampratio giving e ~= 0.35 (RoboCup-like), and the range giving
e in [0.2, 0.6] for domain randomization.

Usage: MUJOCO_GL=egl uv run python scripts/calibrate_ball_restitution.py
"""

from __future__ import annotations

import mujoco
import numpy as np

RADIUS = 0.11
MASS = 0.43
DROP_H = 1.0  # Ball center starts here; contact at z=RADIUS.
TIMECONST = 0.02


def _model(dampratio: float) -> mujoco.MjModel:
  xml = f"""
  <mujoco>
    <option timestep="0.002" gravity="0 0 -9.81"/>
    <worldbody>
      <geom name="floor" type="plane" size="5 5 0.1" condim="6"
            friction="0.5 0.02 0.01" solref="0.02 1.0"/>
      <body name="ball" pos="0 0 {DROP_H}">
        <freejoint/>
        <geom name="ball_geom" type="sphere" size="{RADIUS}" mass="{MASS}"
              condim="6" friction="0.5 0.02 0.01"
              solref="{TIMECONST} {dampratio}" solimp="0.9 0.95 0.001 0.5 2.0"/>
      </body>
    </worldbody>
  </mujoco>
  """
  return mujoco.MjModel.from_xml_string(xml)


def measure_restitution(dampratio: float) -> float:
  """Drop the ball, return e = sqrt(h_rebound / h_drop)."""
  model = _model(dampratio)
  data = mujoco.MjData(model)
  data.qpos[2] = DROP_H
  mujoco.mj_forward(model, data)

  drop_clearance = DROP_H - RADIUS  # Fall distance to first contact.
  contacted = False
  max_rebound_z = RADIUS
  prev_vz = 0.0

  for _ in range(4000):  # 8 s.
    mujoco.mj_step(model, data)
    z = float(data.qpos[2])
    vz = float(data.qvel[2])
    if z <= RADIUS + 1e-3:
      contacted = True
    # After first contact, track the apex (vz crosses + -> -).
    if contacted and prev_vz > 0.0 and vz <= 0.0:
      max_rebound_z = max(max_rebound_z, z)
    prev_vz = vz

  rebound_clearance = max(max_rebound_z - RADIUS, 0.0)
  return float(np.sqrt(rebound_clearance / drop_clearance))


def measure_restitution_tc(timeconst: float, dampratio: float) -> float:
  global TIMECONST
  saved = TIMECONST
  TIMECONST = timeconst
  try:
    return measure_restitution(dampratio)
  finally:
    TIMECONST = saved


def main() -> None:
  print(f"Ball r={RADIUS} m={MASS}, drop from {DROP_H} m")
  print("Sweep timeconst x dampratio -> restitution e\n")
  header = "tc\\dr  " + "".join(f"{dr:>8.2f}" for dr in [0.3, 0.2, 0.1, 0.05, 0.02])
  print(header)
  print("-" * len(header))
  grid = {}
  for tc in [0.02, 0.01, 0.005, 0.003, 0.002]:
    row = f"{tc:<7.3f}"
    for dr in [0.3, 0.2, 0.1, 0.05, 0.02]:
      e = measure_restitution_tc(tc, dr)
      grid[(tc, dr)] = e
      row += f"{e:>8.3f}"
    print(row)

  target = min(grid.items(), key=lambda kv: abs(kv[1] - 0.35))
  print("-" * len(header))
  (tc, dr), e = target
  print(f"closest to e=0.35: timeconst={tc}, dampratio={dr} (e={e:.3f})")
  lo = min(grid.items(), key=lambda kv: abs(kv[1] - 0.2))
  hi = min(grid.items(), key=lambda kv: abs(kv[1] - 0.6))
  print(f"e~=0.2: {lo[0]} (e={lo[1]:.3f});  e~=0.6: {hi[0]} (e={hi[1]:.3f})")


if __name__ == "__main__":
  main()
