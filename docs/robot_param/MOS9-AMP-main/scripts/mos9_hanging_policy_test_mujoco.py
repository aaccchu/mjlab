#!/usr/bin/env python3
"""
MOS9 hanging test in MuJoCo with ONNX policy control.
Base translation is locked to simulate a hanging setup.
"""

import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib
import mujoco
import mujoco.viewer
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MOS9_JOINT_NAMES = [
  "right_shoulder_pitch",
  "right_shoulder_roll",
  "right_elbow",
  "left_shoulder_pitch",
  "left_shoulder_roll",
  "left_elbow",
  "right_hip_pitch",
  "right_hip_roll",
  "right_hip_yaw",
  "right_knee",
  "right_ankle_pitch",
  "right_ankle_roll",
  "left_hip_pitch",
  "left_hip_roll",
  "left_hip_yaw",
  "left_knee",
  "left_ankle_pitch",
  "left_ankle_roll",
]

TORQUE_LIMIT_4310 = 36.0
TORQUE_LIMIT_6408 = 60.0
SPEED_LIMIT_4310 = 9.32
SPEED_LIMIT_6408 = 15.60


ARMATURE_4310 = 0.0235872
ARMATURE_6408 = 0.03890875


NATURAL_FREQ = 8 * 2.0 * np.pi
DAMPING_RATIO = 2.0
DAMPING_RATIO = 0.707
STIFFNESS_4310 = ARMATURE_4310 * NATURAL_FREQ**2
STIFFNESS_6408 = ARMATURE_6408 * NATURAL_FREQ**2
DAMPING_4310 = 2.0 * DAMPING_RATIO * ARMATURE_4310 * NATURAL_FREQ
DAMPING_6408 = 2.0 * DAMPING_RATIO * ARMATURE_6408 * NATURAL_FREQ

# ARMATURE_4310 = 0.0282528
# ARMATURE_6408 = 0.0478125
STIFFNESS_4310 = 47.177610  # * 1.05
STIFFNESS_6408 = 105.193621  # * 1.05
DAMPING_4310 = 1.782347  # * 1.5
DAMPING_6408 = 2.629726  # * 1.5


def _joint_limits_by_name(joint_name: str) -> tuple[float, float]:
  if (
    "ankle_roll" in joint_name
    or "shoulder_pitch" in joint_name
    or "shoulder_roll" in joint_name
  ):
    return TORQUE_LIMIT_4310, SPEED_LIMIT_4310
  return TORQUE_LIMIT_6408, SPEED_LIMIT_6408


def _action_scale_by_name(joint_name: str) -> float:
  torque_limit, _ = _joint_limits_by_name(joint_name)
  if torque_limit == TORQUE_LIMIT_4310:
    return 0.25 * TORQUE_LIMIT_4310 / STIFFNESS_4310
  return 0.25 * TORQUE_LIMIT_6408 / STIFFNESS_6408


def _pd_gains_by_name(joint_name: str) -> tuple[float, float]:
  torque_limit, _ = _joint_limits_by_name(joint_name)
  if torque_limit == TORQUE_LIMIT_4310:
    return STIFFNESS_4310, DAMPING_4310
  return STIFFNESS_6408, DAMPING_6408


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


def _resolve_out_dir(out_dir: str, xml_path: Path, model_path: Path) -> Path:
  base_dir = Path(out_dir)
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  return base_dir / xml_path.stem / model_path.stem / ts


def _save_plots(
  out_dir: Path,
  time_s: np.ndarray,
  desired_qpos: np.ndarray,
  actual_qpos: np.ndarray,
  qvel: np.ndarray,
  torque: np.ndarray,
  imu_rpy: np.ndarray,
  imu_ang_vel: np.ndarray,
  cmd: np.ndarray,
  actual_cmd: np.ndarray,
  joint_names: list[str],
  vel_limits: np.ndarray,
  torque_limits: np.ndarray,
):
  out_dir.mkdir(parents=True, exist_ok=True)

  n_joints = len(joint_names)
  ncols = 3
  nrows = int(np.ceil(n_joints / ncols))

  fig1, axes1 = plt.subplots(nrows, ncols, figsize=(18, 3.2 * nrows), sharex=True)
  axes1 = np.asarray(axes1).reshape(-1)
  for idx, jn in enumerate(joint_names):
    ax = axes1[idx]
    ax.plot(time_s, desired_qpos[:, idx], label="policy_target", linewidth=1.2)
    ax.plot(time_s, actual_qpos[:, idx], label="actual", linewidth=1.0)
    ax.set_title(jn, fontsize=9)
    ax.grid(True, alpha=0.3)
  for idx in range(n_joints, len(axes1)):
    axes1[idx].axis("off")
  axes1[0].legend(loc="upper right", fontsize=8)
  fig1.suptitle("Joint Position: Policy Target vs Actual", fontsize=14)
  fig1.tight_layout(rect=[0, 0, 1, 0.97])
  fig1.savefig(out_dir / "01_joint_position_target_vs_actual.png", dpi=160)
  plt.close(fig1)

  fig2, axes2 = plt.subplots(nrows, ncols, figsize=(18, 3.2 * nrows), sharex=True)
  axes2 = np.asarray(axes2).reshape(-1)
  for idx, jn in enumerate(joint_names):
    ax = axes2[idx]
    ax.plot(time_s, qvel[:, idx], label="qvel", linewidth=1.0)
    ax.axhline(vel_limits[idx], color="r", linestyle="--", linewidth=0.8)
    ax.axhline(-vel_limits[idx], color="r", linestyle="--", linewidth=0.8)
    ax.set_title(jn, fontsize=9)
    ax.grid(True, alpha=0.3)
  for idx in range(n_joints, len(axes2)):
    axes2[idx].axis("off")
  fig2.suptitle("Joint Velocity", fontsize=14)
  fig2.tight_layout(rect=[0, 0, 1, 0.97])
  fig2.savefig(out_dir / "02_joint_velocity.png", dpi=160)
  plt.close(fig2)

  fig3, axes3 = plt.subplots(nrows, ncols, figsize=(18, 3.2 * nrows), sharex=True)
  axes3 = np.asarray(axes3).reshape(-1)
  for idx, jn in enumerate(joint_names):
    ax = axes3[idx]
    ax.plot(time_s, torque[:, idx], label="torque", linewidth=1.0)
    ax.axhline(torque_limits[idx], color="r", linestyle="--", linewidth=0.8)
    ax.axhline(-torque_limits[idx], color="r", linestyle="--", linewidth=0.8)
    ax.set_title(jn, fontsize=9)
    ax.grid(True, alpha=0.3)
  for idx in range(n_joints, len(axes3)):
    axes3[idx].axis("off")
  fig3.suptitle("Joint Torque", fontsize=14)
  fig3.tight_layout(rect=[0, 0, 1, 0.97])
  fig3.savefig(out_dir / "03_joint_torque.png", dpi=160)
  plt.close(fig3)

  fig4, axes4 = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
  axes4 = np.asarray(axes4).reshape(-1)

  imu_series = [
    (imu_rpy[:, 0], "roll", "rad"),
    (imu_rpy[:, 1], "pitch", "rad"),
    (imu_rpy[:, 2], "yaw", "rad"),
    (imu_ang_vel[:, 0], "wx", "rad/s"),
    (imu_ang_vel[:, 1], "wy", "rad/s"),
    (imu_ang_vel[:, 2], "wz", "rad/s"),
  ]
  for idx, (series, title, unit) in enumerate(imu_series):
    ax = axes4[idx]
    ax.plot(time_s, series, linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.grid(True, alpha=0.3)

  axes4[4].set_xlabel("time (s)")
  axes4[5].set_xlabel("time (s)")
  fig4.suptitle("IMU Orientation and Angular Velocity", fontsize=14)
  fig4.tight_layout(rect=[0, 0, 1, 0.97])
  fig4.savefig(out_dir / "04_imu_orientation_and_angular_velocity.png", dpi=160)
  plt.close(fig4)

  fig5, axes5 = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
  cmd_labels = ["vx", "vy", "wz"]
  cmd_units = ["m/s", "m/s", "rad/s"]
  for idx in range(3):
    axes5[idx].plot(time_s, cmd[:, idx], label=f"cmd_{cmd_labels[idx]}", linewidth=1.2)
    axes5[idx].plot(
      time_s,
      actual_cmd[:, idx],
      label=f"actual_{cmd_labels[idx]}",
      linestyle="--",
      linewidth=1.0,
    )
    axes5[idx].set_title(f"Command {cmd_labels[idx]}")
    axes5[idx].set_ylabel(cmd_units[idx])
    axes5[idx].grid(True, alpha=0.3)
    axes5[idx].legend(loc="upper right")
  axes5[2].set_xlabel("time (s)")
  fig5.tight_layout()
  fig5.savefig(out_dir / "05_command.png", dpi=160)
  plt.close(fig5)


def _save_plot_data_npz(
  out_dir: Path,
  time_s: np.ndarray,
  desired_qpos: np.ndarray,
  actual_qpos: np.ndarray,
  qvel: np.ndarray,
  torque: np.ndarray,
  imu_rpy: np.ndarray,
  imu_ang_vel: np.ndarray,
  cmd: np.ndarray,
  actual_cmd: np.ndarray,
  action: np.ndarray,
  joint_names: list[str],
  vel_limits: np.ndarray,
  torque_limits: np.ndarray,
  action_scales: np.ndarray,
) -> Path:
  out_dir.mkdir(parents=True, exist_ok=True)
  npz_path = out_dir / "hanging_policy_plot_data.npz"
  np.savez_compressed(
    npz_path,
    time_s=time_s,
    desired_qpos=desired_qpos,
    actual_qpos=actual_qpos,
    qvel=qvel,
    torque=torque,
    imu_rpy=imu_rpy,
    imu_ang_vel=imu_ang_vel,
    cmd=cmd,
    actual_cmd=actual_cmd,
    action=action,
    joint_names=np.asarray(joint_names, dtype=str),
    vel_limits=vel_limits,
    torque_limits=torque_limits,
    action_scales=action_scales,
  )
  return npz_path


class OnnxPolicy:
  def __init__(self, model_path: str):
    import onnxruntime as ort

    self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    self.input_name = self.session.get_inputs()[0].name

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    y = self.session.run(None, {self.input_name: obs.astype(np.float32)[None, :]})[0]
    return y[0]


def main():
  parser = argparse.ArgumentParser(description="MOS9 hanging test in MuJoCo (policy)")
  parser.add_argument(
    "--xml",
    type=str,
    default="data/assets/MOS/MOS92_urdf_0308/xml/MOS92_urdf_0308_simplified.xml",
    help="MuJoCo XML path",
  )
  parser.add_argument("--model", type=str, required=True, help="ONNX policy path")
  parser.add_argument("--sim_dt", type=float, default=0.001, help="Physics dt")
  parser.add_argument(
    "--control_decimation", type=int, default=20, help="Control decimation"
  )
  parser.add_argument(
    "--duration", type=float, default=60.0, help="Duration in seconds"
  )
  parser.add_argument(
    "--realtime", action="store_true", default=True, help="Sleep to near real-time"
  )
  parser.add_argument(
    "--base_height", type=float, default=1.0, help="Hanging base z in world frame"
  )
  parser.add_argument(
    "--cmd_hold_seconds",
    type=float,
    default=5.0,
    help="Seconds to hold each auto command",
  )
  parser.add_argument(
    "--out_dir", type=str, required=True, help="Output base directory for plots and npz"
  )
  args = parser.parse_args()

  if args.control_decimation <= 0:
    raise ValueError("control_decimation must be positive.")
  if args.cmd_hold_seconds <= 0.0:
    raise ValueError("cmd_hold_seconds must be positive.")

  xml_path = Path(args.xml)
  if not xml_path.exists():
    raise FileNotFoundError(f"XML not found: {xml_path}")

  model_path = Path(args.model)
  if not model_path.exists():
    raise FileNotFoundError(f"ONNX model not found: {model_path}")
  if model_path.suffix.lower() != ".onnx":
    raise ValueError(f"Only ONNX model is supported, got: {model_path}")

  cmd_sequence = [
    (0.4, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (-0.4, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.3, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, -0.3, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.8),
    (0.0, 0.0, -0.8),
    (0.0, 0.0, 0.0),
  ]

  policy = OnnxPolicy(str(model_path))
  out_dir = _resolve_out_dir(args.out_dir, xml_path, model_path)

  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)
  model.opt.timestep = args.sim_dt

  base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
  if base_body_id < 0:
    raise ValueError("base_link not found in XML.")

  gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "base_gyro")
  if gyro_id < 0:
    raise ValueError("Sensor base_gyro not found in XML.")
  gyro_adr = model.sensor_adr[gyro_id]

  joint_qpos_adr = []
  joint_qvel_adr = []
  actuator_ids = []
  joint_ranges = []
  action_scales = []
  vel_limits = []
  torque_limits = []
  stiffness_gains = []
  damping_gains = []

  for jn in MOS9_JOINT_NAMES:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
    if jid < 0:
      raise ValueError(f"Joint not found in XML: {jn}")
    if aid < 0:
      raise ValueError(f"Actuator not found in XML: {jn}")

    joint_qpos_adr.append(model.jnt_qposadr[jid])
    joint_qvel_adr.append(model.jnt_dofadr[jid])
    actuator_ids.append(aid)
    joint_ranges.append(model.jnt_range[jid].copy())
    action_scales.append(_action_scale_by_name(jn))

    tau_lim, vel_lim = _joint_limits_by_name(jn)
    kp, kd = _pd_gains_by_name(jn)
    torque_limits.append(tau_lim)
    vel_limits.append(vel_lim)
    stiffness_gains.append(kp)
    damping_gains.append(kd)

  for i, jn in enumerate(MOS9_JOINT_NAMES):
    if "ankle_roll" in jn or "shoulder_pitch" in jn or "shoulder_roll" in jn:
      armature_val = ARMATURE_4310
    else:
      armature_val = ARMATURE_6408

    dof_adr = joint_qvel_adr[i]
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
    model.dof_armature[dof_adr] = armature_val
    model.jnt_stiffness[jid] = 0.0
    model.dof_damping[dof_adr] = 0.0

  mujoco.mj_setConst(model, data)

  joint_qpos_adr = np.asarray(joint_qpos_adr, dtype=np.int32)
  joint_qvel_adr = np.asarray(joint_qvel_adr, dtype=np.int32)
  actuator_ids = np.asarray(actuator_ids, dtype=np.int32)
  joint_ranges = np.asarray(joint_ranges, dtype=np.float64)
  action_scales = np.asarray(action_scales, dtype=np.float64)
  vel_limits = np.asarray(vel_limits, dtype=np.float64)
  torque_limits = np.asarray(torque_limits, dtype=np.float64)
  stiffness_gains = np.asarray(stiffness_gains, dtype=np.float64)
  damping_gains = np.asarray(damping_gains, dtype=np.float64)

  for i, aid in enumerate(actuator_ids):
    if hasattr(model, "actuator_forcerange"):
      model.actuator_forcerange[aid, 0] = -torque_limits[i]
      model.actuator_forcerange[aid, 1] = torque_limits[i]
    if hasattr(model, "actuator_forcelimited"):
      model.actuator_forcelimited[aid] = 1

  if hasattr(model, "actuator_gainprm"):
    model.actuator_gainprm[:, :] = 0.0
    model.actuator_gainprm[actuator_ids, 0] = 1.0
  if hasattr(model, "actuator_biasprm"):
    model.actuator_biasprm[:, :] = 0.0

  mujoco.mj_resetData(model, data)

  init_joint_pos = {
    "left_shoulder_roll": 1.4,
    "right_shoulder_roll": -1.4,
  }
  data.qpos[joint_qpos_adr] = 0.0
  for i, jn in enumerate(MOS9_JOINT_NAMES):
    if jn in init_joint_pos:
      data.qpos[joint_qpos_adr[i]] = init_joint_pos[jn]

  data.qpos[2] = args.base_height
  data.qvel[0:3] = 0.0

  mujoco.mj_forward(model, data)

  default_qpos = np.zeros(len(MOS9_JOINT_NAMES), dtype=np.float64)
  for i, jn in enumerate(MOS9_JOINT_NAMES):
    if jn in init_joint_pos:
      default_qpos[i] = init_joint_pos[jn]

  base_anchor_xyz = data.qpos[0:3].copy()

  def _lock_base_translation():
    data.qpos[0:3] = base_anchor_xyz
    data.qvel[0:3] = 0.0

  ctrl_dt = args.sim_dt * args.control_decimation
  prev_action = np.zeros(len(MOS9_JOINT_NAMES), dtype=np.float64)
  target_qpos = default_qpos.copy()
  command_delay_s = 0.005
  delay_steps = max(1, int(round(command_delay_s / args.sim_dt)))
  target_delay = deque(
    [default_qpos.copy() for _ in range(delay_steps)], maxlen=delay_steps
  )
  print(f"[INFO] command_delay: {command_delay_s:.3f}s ({delay_steps} steps)")

  time_log = []
  desired_qpos_log = []
  actual_qpos_log = []
  qvel_log = []
  torque_log = []
  imu_rpy_log = []
  imu_ang_vel_log = []
  cmd_log = []
  actual_cmd_log = []
  action_log = []

  def _build_obs() -> np.ndarray:
    phase_idx = int(data.time / args.cmd_hold_seconds) % len(cmd_sequence)
    cmd_vx, cmd_vy, cmd_wz = cmd_sequence[phase_idx]

    base_quat = data.qpos[3:7].copy()
    rot_w_b = _quat_to_rotmat_wxyz(base_quat).T
    projected_gravity = rot_w_b @ np.array([0.0, 0.0, -1.0], dtype=np.float64)

    projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    base_ang_vel = data.sensordata[gyro_adr : gyro_adr + 3].copy()
    base_ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    joint_pos_rel = data.qpos[joint_qpos_adr] - default_qpos
    joint_vel_rel = data.qvel[joint_qvel_adr]

    obs = np.concatenate(
      [
        base_ang_vel * 0.2,
        projected_gravity,
        np.array([cmd_vx, cmd_vy, cmd_wz], dtype=np.float64),
        joint_pos_rel,
        joint_vel_rel * 0.05,
        prev_action,
      ],
      axis=0,
    )
    return obs.astype(np.float32)

  max_steps = int(args.duration / args.sim_dt)
  decim = int(args.control_decimation)

  print(f"[INFO] xml: {xml_path}")
  print(f"[INFO] model: {model_path}")
  print(f"[INFO] sim_dt: {args.sim_dt}, control_dt: {ctrl_dt} ({1.0 / ctrl_dt:.1f} Hz)")
  print(f"[INFO] duration: {args.duration:.3f}s")
  print(f"[INFO] out_dir: {out_dir}")

  with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
  ) as viewer:
    t0 = time.time()
    for i in range(max_steps):
      if not viewer.is_running():
        break

      _lock_base_translation()

      if i % decim == 0:
        obs = _build_obs()
        action = policy(obs).astype(np.float64)
        target_qpos = default_qpos + action * action_scales
        target_qpos = np.clip(target_qpos, joint_ranges[:, 0], joint_ranges[:, 1])
        prev_action[:] = action

      target_delay.append(target_qpos.copy())
      delayed_target_qpos = target_delay[0]

      qpos_now = data.qpos[joint_qpos_adr]
      qvel_now = data.qvel[joint_qvel_adr]
      torque = (
        stiffness_gains * (delayed_target_qpos - qpos_now) - damping_gains * qvel_now
      )
      torque = np.clip(torque, -torque_limits, torque_limits)
      data.ctrl[actuator_ids] = torque

      base_quat_now = data.qpos[3:7].copy()
      imu_rpy = _quat_to_euler_xyz_wxyz(base_quat_now)
      imu_ang_vel = data.sensordata[gyro_adr : gyro_adr + 3].copy()
      rot_w_b_now = _quat_to_rotmat_wxyz(base_quat_now).T
      base_lin_vel_world = data.qvel[0:3].copy()
      base_lin_vel_body = rot_w_b_now @ base_lin_vel_world
      actual_vx = base_lin_vel_body[0]
      actual_vy = base_lin_vel_body[1]
      actual_wz = imu_ang_vel[2]

      phase_idx = int(data.time / args.cmd_hold_seconds) % len(cmd_sequence)
      cmd_vx, cmd_vy, cmd_wz = cmd_sequence[phase_idx]

      time_log.append(data.time)
      desired_qpos_log.append(delayed_target_qpos.copy())
      actual_qpos_log.append(qpos_now.copy())
      qvel_log.append(qvel_now.copy())
      torque_log.append(torque.copy())
      imu_rpy_log.append(imu_rpy)
      imu_ang_vel_log.append(imu_ang_vel)
      cmd_log.append(np.array([cmd_vx, cmd_vy, cmd_wz], dtype=np.float64))
      actual_cmd_log.append(
        np.array([actual_vx, actual_vy, actual_wz], dtype=np.float64)
      )
      action_log.append(prev_action.copy())

      mujoco.mj_step(model, data)
      _lock_base_translation()

      if i % 20 == 0:
        viewer.sync()
        if args.realtime:
          target_t = (i + 1) * args.sim_dt
          sleep_t = target_t - (time.time() - t0)
          if sleep_t > 0:
            time.sleep(sleep_t)

  if len(time_log) > 0:
    time_s = np.asarray(time_log, dtype=np.float64)
    desired_qpos = np.asarray(desired_qpos_log, dtype=np.float64)
    actual_qpos = np.asarray(actual_qpos_log, dtype=np.float64)
    qvel = np.asarray(qvel_log, dtype=np.float64)
    torque = np.asarray(torque_log, dtype=np.float64)
    imu_rpy = np.asarray(imu_rpy_log, dtype=np.float64)
    imu_ang_vel = np.asarray(imu_ang_vel_log, dtype=np.float64)
    cmd = np.asarray(cmd_log, dtype=np.float64)
    actual_cmd = np.asarray(actual_cmd_log, dtype=np.float64)
    action = np.asarray(action_log, dtype=np.float64)

    _save_plots(
      out_dir=out_dir,
      time_s=time_s,
      desired_qpos=desired_qpos,
      actual_qpos=actual_qpos,
      qvel=qvel,
      torque=torque,
      imu_rpy=imu_rpy,
      imu_ang_vel=imu_ang_vel,
      cmd=cmd,
      actual_cmd=actual_cmd,
      joint_names=MOS9_JOINT_NAMES,
      vel_limits=vel_limits,
      torque_limits=torque_limits,
    )
    npz_path = _save_plot_data_npz(
      out_dir=out_dir,
      time_s=time_s,
      desired_qpos=desired_qpos,
      actual_qpos=actual_qpos,
      qvel=qvel,
      torque=torque,
      imu_rpy=imu_rpy,
      imu_ang_vel=imu_ang_vel,
      cmd=cmd,
      actual_cmd=actual_cmd,
      action=action,
      joint_names=MOS9_JOINT_NAMES,
      vel_limits=vel_limits,
      torque_limits=torque_limits,
      action_scales=action_scales,
    )
    print(f"[INFO] plots saved to: {out_dir}")
    print(f"[INFO] plot data saved to: {npz_path}")


if __name__ == "__main__":
  main()
