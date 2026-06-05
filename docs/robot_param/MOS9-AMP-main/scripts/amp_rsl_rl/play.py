"""Script to play a checkpoint with RSL-RL (factoryIsaac)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import pickle
from pathlib import Path

import tqdm
from isaaclab import __version__ as omni_isaac_lab_version
from isaaclab.app import AppLauncher

# local imports
import argtool as rsl_arg_cli  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
if omni_isaac_lab_version < "0.21.0":
  parser.add_argument(
    "--cpu", action="store_true", default=False, help="Use CPU pipeline."
  )
parser.add_argument(
  "--target", type=str, default=None, help="Direct path to target ckpt"
)

parser.add_argument(
  "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
  "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
  "--video", action="store_true", default=False, help="Record videos during playing."
)
parser.add_argument(
  "--length", type=int, default=200, help="Length of the recorded video (in steps)."
)
parser.add_argument("--rldevice", type=str, default="cuda:0", help="Device for rl")

parser.add_argument(
  "--collect", action="store_true", default=False, help="Record data during playing."
)
parser.add_argument(
  "--web", action="store_true", default=False, help="Web videos during playing."
)
parser.add_argument(
  "--local", action="store_true", default=False, help="Using asset in local buffer"
)
parser.add_argument(
  "--determine", action="store_true", default=False, help="Clear reset terms"
)
parser.add_argument(
  "--cmd_hold_seconds",
  type=float,
  default=3.0,
  help="Seconds to hold each play command",
)

# append RSL-RL cli arguments
rsl_arg_cli.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown_args = parser.parse_known_args()

commands_overrides = {}
reward_overrides = {}
event_overrides = {}
env_overrides = {}
# backward compatibility for legacy --env_ prefix


def _parse_cli_value(val_str: str):
  if "," in val_str:
    items = [v.strip() for v in val_str.split(",")]
    try:
      return tuple(float(x) for x in items)
    except ValueError:
      return tuple(items)

  lower = val_str.lower()
  if lower == "true":
    return True
  if lower == "false":
    return False

  try:
    return int(val_str)
  except ValueError:
    try:
      return float(val_str)
    except ValueError:
      return val_str


def _set_by_path(root, path: str, value):
  parts = path.split(".")
  current = root
  for part in parts[:-1]:
    if isinstance(current, dict):
      if part not in current:
        raise KeyError(f"Key '{part}' not found while setting '{path}'")
      current = current[part]
    else:
      if not hasattr(current, part):
        raise AttributeError(f"Attribute '{part}' not found while setting '{path}'")
      current = getattr(current, part)

  leaf = parts[-1]
  if isinstance(current, dict):
    if leaf not in current:
      raise KeyError(f"Key '{leaf}' not found while setting '{path}'")
    current[leaf] = value
  else:
    if not hasattr(current, leaf):
      raise AttributeError(f"Attribute '{leaf}' not found while setting '{path}'")
    setattr(current, leaf, value)


def _get_by_path(root, path: str):
  parts = path.split(".")
  current = root
  for part in parts:
    if isinstance(current, dict):
      current = current[part]
    else:
      current = getattr(current, part)
  return current


def _apply_commands_overrides(env_cfg, overrides: dict):
  if not overrides:
    return

  if not hasattr(env_cfg, "commands"):
    print("[WARNING] env_cfg has no commands section. Ignore commands overrides.")
    return

  print("[INFO] Applying commands overrides...")
  for command_path, val in overrides.items():
    try:
      old_val = _get_by_path(env_cfg.commands, command_path)
      _set_by_path(env_cfg.commands, command_path, val)
      print(f"    -> commands.{command_path}: {old_val} -> {val}")
    except (KeyError, AttributeError) as e:
      print(f"[WARNING] {e}")


def _apply_reward_overrides(env_cfg, overrides: dict):
  if not overrides:
    return
  if not hasattr(env_cfg, "rewards"):
    print("[WARNING] env_cfg has no rewards section. Ignore reward overrides.")
    return

  print("[INFO] Applying reward overrides...")
  for reward_path, reward_val in overrides.items():
    # Handle parameter overrides (e.g. track_lin_vel_xy.params.std)
    if ".params." in reward_path:
      reward_name, param_path = reward_path.split(".params.", 1)
      if not hasattr(env_cfg.rewards, reward_name):
        print(f"[WARNING] Reward term '{reward_name}' not found.")
        continue

      reward_term = getattr(env_cfg.rewards, reward_name)
      if not hasattr(reward_term, "params"):
        print(f"[WARNING] Reward term '{reward_name}' has no params field.")
        continue

      try:
        old_val = _get_by_path(reward_term.params, param_path)
        _set_by_path(reward_term.params, param_path, reward_val)
        print(
          f"    -> rewards.{reward_name}.params.{param_path}: {old_val} -> {reward_val}"
        )
      except (KeyError, AttributeError) as e:
        print(f"[WARNING] Failed to set param for {reward_name}: {e}")
      continue

    # Backward compatibility for weight overrides (e.g. track_lin_vel_xy)
    reward_name = reward_path
    if not hasattr(env_cfg.rewards, reward_name):
      print(f"[WARNING] Reward term '{reward_name}' not found.")
      continue
    reward_term = getattr(env_cfg.rewards, reward_name)
    if not hasattr(reward_term, "weight"):
      print(f"[WARNING] Reward term '{reward_name}' has no weight field.")
      continue
    old_weight = reward_term.weight
    reward_term.weight = reward_val
    print(f"    -> rewards.{reward_name}.weight: {old_weight} -> {reward_val}")


def _apply_event_overrides(env_cfg, overrides: dict):
  if not overrides:
    return
  if not hasattr(env_cfg, "events"):
    print("[WARNING] env_cfg has no events section. Ignore event overrides.")
    return

  print("[INFO] Applying event overrides...")
  for event_path, val in overrides.items():
    try:
      old_val = _get_by_path(env_cfg.events, event_path)
      _set_by_path(env_cfg.events, event_path, val)
      print(f"    -> events.{event_path}: {old_val} -> {val}")
    except (KeyError, AttributeError) as e:
      print(f"[WARNING] {e}")


def _apply_env_overrides(env_cfg, overrides: dict):
  if not overrides:
    return

  print("[INFO] Applying env overrides (legacy --env_)...")
  for env_path, val in overrides.items():
    try:
      old_val = _get_by_path(env_cfg, env_path)
      _set_by_path(env_cfg, env_path, val)
      print(f"    -> env.{env_path}: {old_val} -> {val}")
    except (KeyError, AttributeError) as e:
      print(f"[WARNING] {e}")


for arg in unknown_args:
  if "=" not in arg:
    print(f"[WARNING] Ignore unknown arg without '=': {arg}")
    continue

  key, value = arg.split("=", 1)
  parsed_value = _parse_cli_value(value)

  if key.startswith("--commands."):
    command_path = key.replace("--commands.", "", 1)
    commands_overrides[command_path] = parsed_value
  elif key.startswith("--reward_"):
    reward_name = key.replace("--reward_", "", 1)
    reward_overrides[reward_name] = parsed_value
  elif key.startswith("--event_"):
    event_path = key.replace("--event_", "", 1)
    event_overrides[event_path] = parsed_value
  elif key.startswith("--env_"):
    env_path = key.replace("--env_", "", 1)
    env_overrides[env_path] = parsed_value
  else:
    print(f"[WARNING] Unknown override key: {key}")

if commands_overrides:
  print(f"[INFO] CLI commands overrides: {commands_overrides}")
if reward_overrides:
  print(f"[INFO] CLI reward overrides: {reward_overrides}")
if event_overrides:
  print(f"[INFO] CLI event overrides: {event_overrides}")
if env_overrides:
  print(f"[INFO] CLI env overrides (legacy --env_): {env_overrides}")

# always enable cameras to record video
if args_cli.video or args_cli.web:
  args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from amp_tasks.amp_rsl_rl.isaaclab.exporter import (
  export_policy_as_jit,
  export_policy_as_onnx,
)
from amp_tasks.amp_rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

matplotlib.use("Agg")


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


def _resolve_plot_dir(target_ckpt: str) -> Path:
  target_path = Path(target_ckpt).resolve()
  return target_path.parent / "play_plot" / target_path.stem


def _set_base_velocity_command(command_term, cmd_xyz: tuple[float, float, float]):
  command_term.vel_command_b[:, 0] = cmd_xyz[0]
  command_term.vel_command_b[:, 1] = cmd_xyz[1]
  command_term.vel_command_b[:, 2] = cmd_xyz[2]
  if hasattr(command_term, "is_standing_env"):
    command_term.is_standing_env[:] = False
  if hasattr(command_term, "is_heading_env"):
    command_term.is_heading_env[:] = False


def _save_play_plots(
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


def main():
  task_name = args_cli.task
  if args_cli.target is None:
    raise ValueError("Please use --target to specify a checkpoint.")

  resume_path = os.path.abspath(args_cli.target)
  print(f"[INFO]: Loading model checkpoint from: {resume_path}")
  log_dir = os.path.dirname(resume_path)
  run_path = os.path.dirname(resume_path)

  if args_cli.collect:
    sample_dir = os.path.join(run_path, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    output_file = os.path.join(sample_dir, "total_data.pkl")

  if task_name is None:
    raise ValueError("Please specify --task when playing with local configs.")

  agent_cfg = rsl_arg_cli.parse_rsl_rl_cfg(task_name, args_cli)
  env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")

  if args_cli.seed is not None:
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed

  if args_cli.rldevice is not None:
    agent_cfg.device = args_cli.rldevice

  if args_cli.device is not None:
    env_cfg.sim.device = args_cli.device

  if args_cli.num_envs is not None:
    env_cfg.scene.num_envs = args_cli.num_envs

  if args_cli.local:
    env_cfg.scene.robot.spawn.use_local_asset = True

  if args_cli.cmd_hold_seconds <= 0.0:
    raise ValueError("cmd_hold_seconds must be positive.")

  env_cfg.curriculum = None

  if args_cli.determine:
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.events.reset_base.params["velocity_range"] = {
      "x": (0.0, 0.0),
      "y": (0.0, 0.0),
      "yaw": (0.0, 0.0),
    }
    env_cfg.events.push_robot = None
    env_cfg.events.add_base_mass = None
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.terrain.max_init_terrain_level = None
    env_cfg.terrain.terrain_generator.curriculum = False
    env_cfg.scene.terrain.terrain_generator.curriculum = False
    if getattr(env_cfg, "curriculum", None) is not None:
      env_cfg.curriculum = None
    if hasattr(env_cfg.rewards, "action_rate_l2"):
      env_cfg.rewards.action_rate_l2.weight = 0.0
    if hasattr(env_cfg.rewards, "dof_acc_l2"):
      env_cfg.rewards.dof_acc_l2.weight = 0.0

  # default play tuning
  env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.5)
  if hasattr(env_cfg.commands.base_velocity, "resampling_time_range"):
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
  if hasattr(env_cfg.commands.base_velocity, "rel_standing_envs"):
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
  if hasattr(env_cfg.commands.base_velocity, "heading_command"):
    env_cfg.commands.base_velocity.heading_command = False

  # allow CLI overrides like train.py style:
  # --commands.base_velocity.debug_vis=True
  # --reward_track_lin_vel_xy=1.0
  # --event_reset_base.params.velocity_range.x=0.0,0.0
  _apply_commands_overrides(env_cfg, commands_overrides)
  _apply_reward_overrides(env_cfg, reward_overrides)
  _apply_event_overrides(env_cfg, event_overrides)
  _apply_env_overrides(env_cfg, env_overrides)

  from isaaclab.envs.common import ViewerCfg

  env_cfg.viewer = ViewerCfg(
    eye=(4.0, 4.0, 4.0),
    lookat=(0.0, 0.0, 0.0),
    env_index=20,
    origin_type="env",
    asset_name="robot",
  )

  render_mode = "rgb_array" if args_cli.video or args_cli.web else None
  env = gym.make(task_name, cfg=env_cfg, render_mode=render_mode)

  if args_cli.video:
    checkpoint_stem = os.path.splitext(os.path.basename(resume_path))[0]
    video_prefix = (
      checkpoint_stem.replace("model_", "isaaclab_")
      if "model_" in checkpoint_stem
      else f"video_{checkpoint_stem}"
    )
    video_kwargs = {
      "video_folder": os.path.join(log_dir, "videos", "play"),
      "step_trigger": lambda step: step == 0,
      "video_length": args_cli.length if args_cli.length > 0 else 0,
      "disable_logger": True,
      "name_prefix": video_prefix,
    }
    print(f"[INFO] Recording videos during play with prefix: {video_prefix}")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

  env, func_runner, _learn_cfg = rsl_arg_cli.prepare_wrapper(env, args_cli, agent_cfg)
  runner: OnPolicyRunner = func_runner(
    env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
  )

  runner.load(resume_path)
  print(f"[INFO]: Loading model checkpoint from: {resume_path}")

  policy = runner.get_inference_policy(device=env.unwrapped.device)
  obs = env.get_observations()

  cmd_sequence = [
    (0.6, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (-0.3, 0.0, 0.0),
    (0.0, 0.4, 0.0),
    (0.0, 0.3, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, -0.4, 0.0),
    (0.0, -0.3, 0.0),
    (0.0, 0.0, 0.0),
  ]

  command_term = env.unwrapped.command_manager.get_term("base_velocity")
  robot = env.unwrapped.scene["robot"]
  action_term_name = env.unwrapped.action_manager.active_terms[0]
  action_term = env.unwrapped.action_manager.get_term(action_term_name)

  if hasattr(action_term, "_joint_ids") and action_term._joint_ids is not None:
    joint_ids = action_term._joint_ids
  else:
    joint_ids = list(range(action_term.processed_actions.shape[-1]))

  joint_ids_tensor = torch.as_tensor(
    joint_ids, dtype=torch.long, device=robot.data.joint_pos.device
  )
  joint_names = [robot.data.joint_names[idx] for idx in joint_ids]

  if (
    hasattr(robot.data, "soft_joint_vel_limits")
    and robot.data.soft_joint_vel_limits is not None
  ):
    vel_limits_t = robot.data.soft_joint_vel_limits[0, joint_ids_tensor]
  else:
    vel_limits_t = robot.data.joint_vel_limits[0, joint_ids_tensor]

  if (
    hasattr(robot.data, "soft_joint_effort_limits")
    and robot.data.soft_joint_effort_limits is not None
  ):
    torque_limits_t = robot.data.soft_joint_effort_limits[0, joint_ids_tensor]
  else:
    torque_limits_t = robot.data.joint_effort_limits[0, joint_ids_tensor]

  sim_dt = float(getattr(env.unwrapped, "step_dt", env_cfg.sim.dt))
  plot_dir = _resolve_plot_dir(resume_path)
  print(f"[INFO] play command cycle: {cmd_sequence}, hold={args_cli.cmd_hold_seconds}s")
  print(f"[INFO] play plot_dir: {plot_dir}")

  time_log = []
  desired_qpos_log = []
  actual_qpos_log = []
  qvel_log = []
  torque_log = []
  imu_rpy_log = []
  imu_ang_vel_log = []
  cmd_log = []
  actual_cmd_log = []

  export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
  os.makedirs(export_model_dir, exist_ok=True)

  checkpoint_stem = os.path.splitext(os.path.basename(resume_path))[0]
  export_suffix = ""
  if "_" in checkpoint_stem:
    export_suffix = f"_{checkpoint_stem.rsplit('_', 1)[-1]}"

  export_pth_name = f"policy{export_suffix}.pth"
  export_onnx_name = f"policy{export_suffix}.onnx"
  export_jit_name = f"policy{export_suffix}.pt"

  torch.save(runner.alg.actor_critic, os.path.join(export_model_dir, export_pth_name))
  export_policy_as_onnx(
    runner.alg.actor_critic, export_model_dir, filename=export_onnx_name
  )
  export_policy_as_jit(
    runner.alg.actor_critic, None, export_model_dir, filename=export_jit_name
  )
  print(
    f"[INFO]: Saving policy to: {export_model_dir} "
    f"({export_pth_name}, {export_onnx_name}, {export_jit_name})"
  )

  pbar = tqdm.tqdm(range(args_cli.length)) if args_cli.length > 0 else tqdm.tqdm()

  step = 0
  try:
    while simulation_app.is_running():
      phase_idx = int((step * sim_dt) / args_cli.cmd_hold_seconds) % len(cmd_sequence)
      cmd_now = cmd_sequence[phase_idx]
      _set_base_velocity_command(command_term, cmd_now)

      with torch.inference_mode():
        actions = policy(obs)
        obs, rewards, dones, infos = env.step(actions, not_amp=True)

      desired_qpos = action_term.processed_actions[0].detach().cpu().numpy().copy()
      actual_qpos = (
        robot.data.joint_pos[0, joint_ids_tensor].detach().cpu().numpy().copy()
      )
      joint_vel = (
        robot.data.joint_vel[0, joint_ids_tensor].detach().cpu().numpy().copy()
      )
      joint_torque = (
        robot.data.applied_torque[0, joint_ids_tensor].detach().cpu().numpy().copy()
      )

      root_quat = robot.data.root_quat_w[0].detach().cpu().numpy().copy()
      imu_rpy = _quat_to_euler_xyz_wxyz(root_quat)
      imu_ang_vel = robot.data.root_ang_vel_b[0].detach().cpu().numpy().copy()
      actual_cmd = np.array(
        [
          float(robot.data.root_lin_vel_b[0, 0].item()),
          float(robot.data.root_lin_vel_b[0, 1].item()),
          float(robot.data.root_ang_vel_b[0, 2].item()),
        ],
        dtype=np.float64,
      )

      time_log.append(step * sim_dt)
      desired_qpos_log.append(desired_qpos)
      actual_qpos_log.append(actual_qpos)
      qvel_log.append(joint_vel)
      torque_log.append(joint_torque)
      imu_rpy_log.append(imu_rpy)
      imu_ang_vel_log.append(imu_ang_vel)
      cmd_log.append(np.asarray(cmd_now, dtype=np.float64))
      actual_cmd_log.append(actual_cmd)

      step += 1
      pbar.update()
      if args_cli.collect:
        trans = [obs.data, actions.data, rewards.data, dones.data]
        with open(output_file, "ab") as f:
          pickle.dump(trans, f)
      if args_cli.length > 0 and args_cli.length < step:
        break
  except KeyboardInterrupt:
    pass

  if len(time_log) > 0:
    _save_play_plots(
      out_dir=plot_dir,
      time_s=np.asarray(time_log, dtype=np.float64),
      desired_qpos=np.asarray(desired_qpos_log, dtype=np.float64),
      actual_qpos=np.asarray(actual_qpos_log, dtype=np.float64),
      qvel=np.asarray(qvel_log, dtype=np.float64),
      torque=np.asarray(torque_log, dtype=np.float64),
      imu_rpy=np.asarray(imu_rpy_log, dtype=np.float64),
      imu_ang_vel=np.asarray(imu_ang_vel_log, dtype=np.float64),
      cmd=np.asarray(cmd_log, dtype=np.float64),
      actual_cmd=np.asarray(actual_cmd_log, dtype=np.float64),
      joint_names=joint_names,
      vel_limits=vel_limits_t.detach().cpu().numpy().copy(),
      torque_limits=torque_limits_t.detach().cpu().numpy().copy(),
    )
    print(f"[INFO] play plots saved to: {plot_dir}")

  env.close()


if __name__ == "__main__":
  main()
  simulation_app.close()
