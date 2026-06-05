import argparse
import time
from pathlib import Path

import matplotlib
import mujoco
import mujoco.viewer
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MANUAL_ARBITRARY_ROTATIONS = [
  {"axis": [1.0, 1.0, 0.0], "deg": 90.0, "duration": 2.0},
  {"axis": [1.0, 1.0, 0.0], "deg": -90.0, "duration": 1.0},
  {"axis": [1.0, 0.0, 1.0], "deg": -90.0, "duration": 1.8},
  {"axis": [1.0, 0.0, 1.0], "deg": 90.0, "duration": 1.8},
  {"axis": [0.0, 1.0, 1.0], "deg": 90.0, "duration": 2.2},
  {"axis": [0.0, 1.0, 1.0], "deg": -90.0, "duration": 2.0},
]


def _quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
  return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
  w1, x1, y1, z1 = q1
  w2, x2, y2, z2 = q2
  return np.array(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    dtype=np.float64,
  )


def _quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
  w, x, y, z = q
  return np.array(
    [
      [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
      [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
      [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ],
    dtype=np.float64,
  )


def _euler_xyz_to_quat_wxyz(euler_xyz: np.ndarray) -> np.ndarray:
  roll, pitch, yaw = euler_xyz
  cr = np.cos(roll * 0.5)
  sr = np.sin(roll * 0.5)
  cp = np.cos(pitch * 0.5)
  sp = np.sin(pitch * 0.5)
  cy = np.cos(yaw * 0.5)
  sy = np.sin(yaw * 0.5)

  w = cr * cp * cy + sr * sp * sy
  x = sr * cp * cy - cr * sp * sy
  y = cr * sp * cy + sr * cp * sy
  z = cr * cp * sy - sr * sp * cy
  q = np.array([w, x, y, z], dtype=np.float64)
  return q / np.linalg.norm(q)


def _quat_to_euler_xyz_wxyz(q: np.ndarray) -> np.ndarray:
  w, x, y, z = q
  sinr_cosp = 2.0 * (w * x + y * z)
  cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
  roll = np.arctan2(sinr_cosp, cosr_cosp)

  sinp = 2.0 * (w * y - z * x)
  if np.abs(sinp) >= 1.0:
    pitch = np.sign(sinp) * (np.pi / 2.0)
  else:
    pitch = np.arcsin(sinp)

  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  yaw = np.arctan2(siny_cosp, cosy_cosp)
  return np.array([roll, pitch, yaw], dtype=np.float64)


def _quat_delta_to_omega_body(
  prev_q: np.ndarray, curr_q: np.ndarray, dt: float
) -> np.ndarray:
  if dt <= 0.0:
    return np.zeros(3, dtype=np.float64)
  q_rel = _quat_mul_wxyz(_quat_conj_wxyz(prev_q), curr_q)
  q_rel /= np.linalg.norm(q_rel)
  if q_rel[0] < 0.0:
    q_rel = -q_rel

  angle = 2.0 * np.arccos(np.clip(q_rel[0], -1.0, 1.0))
  sin_half = np.sqrt(max(1.0 - q_rel[0] * q_rel[0], 0.0))
  if sin_half < 1e-8:
    axis = np.zeros(3, dtype=np.float64)
  else:
    axis = q_rel[1:] / sin_half
  return axis * angle / dt


def _draw_frame(
  viewer,
  origin: np.ndarray,
  quat_wxyz: np.ndarray,
  axis_length: float,
  axis_radius: float,
):
  if not hasattr(viewer, "user_scn"):
    return
  user_scn = viewer.user_scn
  if user_scn is None or not hasattr(user_scn, "geoms"):
    return
  if hasattr(user_scn, "maxgeom") and user_scn.maxgeom < 3:
    return

  rot = _quat_to_rotmat_wxyz(quat_wxyz)
  axis_colors = (
    np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
    np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    np.array([0.0, 0.3, 1.0, 1.0], dtype=np.float32),
  )

  def _connector_ok(
    geom, start_xyz: np.ndarray, end_xyz: np.ndarray, radius: float
  ) -> bool:
    if hasattr(mujoco, "mjv_connector"):
      try:
        mujoco.mjv_connector(
          geom,
          mujoco.mjtGeom.mjGEOM_ARROW,
          radius,
          float(start_xyz[0]),
          float(start_xyz[1]),
          float(start_xyz[2]),
          float(end_xyz[0]),
          float(end_xyz[1]),
          float(end_xyz[2]),
        )
        return True
      except TypeError:
        try:
          mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            radius,
            start_xyz.astype(np.float64),
            end_xyz.astype(np.float64),
          )
          return True
        except Exception:
          return False
      except Exception:
        return False
    if hasattr(mujoco, "mjv_makeConnector"):
      try:
        mujoco.mjv_makeConnector(
          geom,
          mujoco.mjtGeom.mjGEOM_ARROW,
          radius,
          float(start_xyz[0]),
          float(start_xyz[1]),
          float(start_xyz[2]),
          float(end_xyz[0]),
          float(end_xyz[1]),
          float(end_xyz[2]),
        )
        return True
      except Exception:
        return False
    return False

  user_scn.ngeom = 0
  for axis_idx in range(3):
    axis_world = rot @ np.eye(3, dtype=np.float64)[axis_idx]
    end = origin + axis_length * axis_world

    geom = user_scn.geoms[user_scn.ngeom]
    mujoco.mjv_initGeom(
      geom,
      mujoco.mjtGeom.mjGEOM_ARROW,
      np.zeros(3, dtype=np.float64),
      np.zeros(3, dtype=np.float64),
      np.eye(3, dtype=np.float64).reshape(-1),
      axis_colors[axis_idx],
    )
    if not _connector_ok(geom, origin, end, axis_radius):
      return
    user_scn.ngeom += 1


def _build_rotation_segments(
  move_duration: float,
  hold_duration: float,
  manual_arbitrary_rotations: list[dict],
):
  targets = [
    (0, +90.0),
    (0, -90.0),
    (0, 0.0),
    (1, +90.0),
    (1, -90.0),
    (1, 0.0),
    (2, +90.0),
    (2, -90.0),
    (2, 0.0),
  ]

  segments = []
  current = np.zeros(3, dtype=np.float64)
  for axis_idx, deg in targets:
    nxt = current.copy()
    nxt[axis_idx] = np.deg2rad(deg)
    segments.append(
      {"start": current.copy(), "end": nxt.copy(), "duration": move_duration}
    )
    if hold_duration > 0.0:
      segments.append(
        {"start": nxt.copy(), "end": nxt.copy(), "duration": hold_duration}
      )
    current = nxt

  for rot in manual_arbitrary_rotations:
    axis = np.asarray(rot.get("axis", [0.0, 0.0, 0.0]), dtype=np.float64)
    deg = float(rot.get("deg", 0.0))
    duration = float(rot.get("duration", 0.0))
    norm = np.linalg.norm(axis)
    if norm < 1e-8 or abs(deg) <= 1e-8 or duration <= 0.0:
      continue
    axis_unit = axis / norm
    delta = axis_unit * np.deg2rad(deg)
    nxt = current + delta
    segments.append({"start": current.copy(), "end": nxt.copy(), "duration": duration})
    if hold_duration > 0.0:
      segments.append(
        {"start": nxt.copy(), "end": nxt.copy(), "duration": hold_duration}
      )
    current = nxt
  return segments


def _sample_orientation(segments, t: float):
  elapsed = 0.0
  for seg in segments:
    dur = seg["duration"]
    if t <= elapsed + dur:
      tau = (t - elapsed) / max(dur, 1e-8)
      tau = np.clip(tau, 0.0, 1.0)
      s = 0.5 - 0.5 * np.cos(np.pi * tau)
      dsdt = 0.5 * np.pi * np.sin(np.pi * tau) / max(dur, 1e-8)
      euler = seg["start"] + s * (seg["end"] - seg["start"])
      euler_dot = dsdt * (seg["end"] - seg["start"])
      return euler, euler_dot
    elapsed += dur
  return segments[-1]["end"].copy(), np.zeros(3, dtype=np.float64)


def main():
  parser = argparse.ArgumentParser(description="IMU alignment simulation in MuJoCo")
  parser.add_argument("--sim_dt", type=float, default=0.002, help="Simulation dt")
  parser.add_argument(
    "--move_duration",
    type=float,
    default=2.0,
    help="Rotation move duration per segment",
  )
  parser.add_argument(
    "--hold_duration", type=float, default=0.5, help="Hold duration after each move"
  )
  parser.add_argument("--height", type=float, default=0.5, help="Body floating height")
  parser.add_argument(
    "--axis_length", type=float, default=0.42, help="Visualized axis length"
  )
  parser.add_argument(
    "--axis_radius", type=float, default=0.012, help="Visualized axis radius"
  )
  parser.add_argument(
    "--out_dir", type=str, default="logs/imu_sim", help="Output plot directory"
  )
  parser.add_argument(
    "--realtime", action="store_true", default=True, help="Run near real-time"
  )
  args = parser.parse_args()

  xml_str = f"""
<mujoco model="imu_sim">
  <option timestep="{args.sim_dt}" gravity="0 0 0"/>
  <asset>
	<texture type="2d" name="groundplane" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2" width="512" height="512"/>
	<material name="groundplane" texture="groundplane" texuniform="true" texrepeat="1 1" reflectance="0.2"/>
  </asset>
  <worldbody>
	<light pos="0 0 3" diffuse="0.8 0.8 0.8"/>
	<geom name="floor" type="plane" size="0 0 0.01" condim="3" friction="1.0 0.005 0.0001" material="groundplane"/>
	<body name="imu_body" pos="0 0 {args.height}">
	  <freejoint/>
	  <geom name="body" type="box" size="0.08 0.12 0.04" rgba="0.2 0.7 0.9 1" mass="1.0"/>
	  <site name="imu_site" pos="0 0 0" size="0.01"/>
	</body>
  </worldbody>
  <sensor>
	<framequat name="imu_orientation" objtype="site" objname="imu_site"/>
	<gyro name="imu_gyro" site="imu_site"/>
  </sensor>
</mujoco>
"""

  model = mujoco.MjModel.from_xml_string(xml_str)
  data = mujoco.MjData(model)

  site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_site")
  quat_sensor_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_orientation"
  )
  gyro_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
  quat_adr = model.sensor_adr[quat_sensor_id]
  gyro_adr = model.sensor_adr[gyro_sensor_id]

  segments = _build_rotation_segments(
    args.move_duration,
    args.hold_duration,
    MANUAL_ARBITRARY_ROTATIONS,
  )
  frame_align_quat = _euler_xyz_to_quat_wxyz(
    np.array([0.0, 0.0, -0.5 * np.pi], dtype=np.float64)
  )
  total_duration = float(sum(seg["duration"] for seg in segments))
  max_steps = int(np.ceil(total_duration / args.sim_dt))

  time_log = []
  imu_euler_log = []
  imu_gyro_log = []

  prev_quat = frame_align_quat.copy()
  t0 = time.time()

  with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
  ) as viewer:
    for step in range(max_steps):
      if not viewer.is_running():
        break

      t_sim = step * args.sim_dt
      euler_cmd, _ = _sample_orientation(segments, t_sim)
      quat_cmd_local = _euler_xyz_to_quat_wxyz(euler_cmd)
      quat_cmd = _quat_mul_wxyz(frame_align_quat, quat_cmd_local)
      quat_cmd /= np.linalg.norm(quat_cmd)
      omega_body = _quat_delta_to_omega_body(prev_quat, quat_cmd, args.sim_dt)

      data.qpos[:] = 0.0
      data.qvel[:] = 0.0
      data.qpos[0:3] = np.array([0.0, 0.0, args.height], dtype=np.float64)
      data.qpos[3:7] = quat_cmd
      data.qvel[3:6] = omega_body

      mujoco.mj_forward(model, data)

      imu_quat = data.sensordata[quat_adr : quat_adr + 4].copy()
      imu_gyro = data.sensordata[gyro_adr : gyro_adr + 3].copy()
      imu_euler = _quat_to_euler_xyz_wxyz(imu_quat)

      time_log.append(t_sim)
      imu_euler_log.append(imu_euler)
      imu_gyro_log.append(imu_gyro)

      if hasattr(viewer, "lock"):
        with viewer.lock():
          _draw_frame(
            viewer,
            data.site_xpos[site_id].copy(),
            imu_quat,
            axis_length=args.axis_length,
            axis_radius=args.axis_radius,
          )
          viewer.sync()
      else:
        _draw_frame(
          viewer,
          data.site_xpos[site_id].copy(),
          imu_quat,
          axis_length=args.axis_length,
          axis_radius=args.axis_radius,
        )
        viewer.sync()

      mujoco.mj_step(model, data)
      prev_quat = quat_cmd

      if args.realtime:
        sleep_t = (step + 1) * args.sim_dt - (time.time() - t0)
        if sleep_t > 0.0:
          time.sleep(sleep_t)

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  time_arr = np.asarray(time_log, dtype=np.float64)
  euler_arr = np.asarray(imu_euler_log, dtype=np.float64)
  gyro_arr = np.asarray(imu_gyro_log, dtype=np.float64)

  fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
  axes = np.asarray(axes).reshape(-1)
  series = [
    (euler_arr[:, 0], "roll", "rad"),
    (euler_arr[:, 1], "pitch", "rad"),
    (euler_arr[:, 2], "yaw", "rad"),
    (gyro_arr[:, 0], "wx", "rad/s"),
    (gyro_arr[:, 1], "wy", "rad/s"),
    (gyro_arr[:, 2], "wz", "rad/s"),
  ]
  for idx, (values, title, unit) in enumerate(series):
    ax = axes[idx]
    ax.plot(time_arr, values, linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.grid(True, alpha=0.3)
  axes[4].set_xlabel("time (s)")
  axes[5].set_xlabel("time (s)")
  fig.suptitle("IMU Orientation and Angular Velocity", fontsize=14)
  fig.tight_layout(rect=[0, 0, 1, 0.97])
  fig.savefig(out_dir / "imu_orientation_and_gyro.png", dpi=160)
  plt.close(fig)

  print(f"[INFO] total_duration: {total_duration:.2f}s")
  print(f"[INFO] plots saved to: {out_dir}")


if __name__ == "__main__":
  main()
