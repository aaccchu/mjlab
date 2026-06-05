"""Replay MOS9 FK motion npz files in Isaac Sim (no export).

Example:
    python scripts/replay_mos9_fk_motion.py --input_dir data/motions/mos9_fk_motion --video
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
  description="Replay MOS9 FK motion npz files in Isaac Sim."
)
parser.add_argument(
  "--input_file", type=str, default=None, help="Single input motion .npz file."
)
parser.add_argument(
  "--input_dir",
  type=str,
  default="data/motions/mos9_fk_motion",
  help="Input motion directory.",
)
parser.add_argument(
  "--input_fps",
  type=float,
  default=None,
  help="Input motion fps override. If not provided, use fps from npz file.",
)
parser.add_argument(
  "--frame_range",
  nargs=2,
  type=int,
  metavar=("START", "END"),
  help="Frame range: START END (both inclusive, starts from 1).",
)
parser.add_argument(
  "--output_fps", type=int, default=50, help="Replay fps in simulator."
)
parser.add_argument(
  "--loop", action="store_true", default=False, help="Loop each motion file."
)
parser.add_argument(
  "--video",
  action="store_true",
  default=False,
  help="Record video of the motion replay.",
)
parser.add_argument(
  "--plot_speed_curve",
  action="store_true",
  default=False,
  help="Plot and save speed curves for each motion.",
)
parser.add_argument(
  "--plot_speed_output_dir",
  type=str,
  default=None,
  help="Output directory for speed curve plots. If not set, save to <motion_dir>/speed_curves.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
from amp_tasks.robots.MOS9 import MOS9_CYLINDER_CFG
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)

robot_type = "mos9"
robot_cfg = MOS9_CYLINDER_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
  """Configuration for replay scene."""

  ground = AssetBaseCfg(
    prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
  )

  sky_light = AssetBaseCfg(
    prim_path="/World/skyLight",
    spawn=sim_utils.DomeLightCfg(
      intensity=750.0,
      texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    ),
  )

  robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

  # 添加相机用于录制视频
  camera: CameraCfg = CameraCfg(
    prim_path="/World/Camera",
    update_period=0.0,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
      focal_length=24.0,
      focus_distance=400.0,
      horizontal_aperture=20.955,
      clipping_range=(0.1, 1.0e5),
    ),
  )


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    input_fps_override: float | None,
    output_fps: int,
    device: torch.device,
    frame_range: tuple[int, int] | None,
  ):
    self.motion_file = motion_file
    self.input_fps_override = input_fps_override
    self.output_fps = output_fps
    self.current_idx = 0
    self.device = device
    self.frame_range = frame_range
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()
    self._compute_base_link_speed_stats()

  def _load_motion(self):
    motion = np.load(self.motion_file, allow_pickle=True)

    has_raw = all(k in motion for k in ("root_pos", "root_rot", "dof_pos"))
    has_fk = all(
      k in motion
      for k in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      )
    )

    self.motion_joint_names = None
    self.motion_base_lin_vels_input = None
    self.motion_base_ang_vels_input = None
    self.motion_dof_vels_input = None

    if has_raw:
      root_pos = motion["root_pos"]
      root_rot = motion["root_rot"]
      dof_pos = motion["dof_pos"]
      quat_is_xyzw = True
    elif has_fk:
      joint_pos = motion["joint_pos"]
      joint_vel = motion["joint_vel"]
      body_pos_w = motion["body_pos_w"]
      body_quat_w = motion["body_quat_w"]
      body_lin_vel_w = motion["body_lin_vel_w"]
      body_ang_vel_w = motion["body_ang_vel_w"]

      body_names = motion["body_names"].tolist() if "body_names" in motion else []
      body_names = [str(x) for x in body_names]
      base_idx = body_names.index("base_link") if "base_link" in body_names else 0

      root_pos = body_pos_w[:, base_idx, :]
      root_rot = body_quat_w[:, base_idx, :]
      dof_pos = joint_pos
      self.motion_base_lin_vels_input = (
        torch.from_numpy(body_lin_vel_w[:, base_idx, :])
        .to(torch.float32)
        .to(self.device)
      )
      self.motion_base_ang_vels_input = (
        torch.from_numpy(body_ang_vel_w[:, base_idx, :])
        .to(torch.float32)
        .to(self.device)
      )
      self.motion_dof_vels_input = (
        torch.from_numpy(joint_vel).to(torch.float32).to(self.device)
      )
      if "joint_names" in motion:
        self.motion_joint_names = [str(x) for x in motion["joint_names"].tolist()]
      quat_is_xyzw = False
    else:
      raise KeyError(
        f"{self.motion_file} has unsupported keys: {motion.files}. "
        "Expected raw format (root_pos/root_rot/dof_pos) or fk format "
        "(joint_pos/joint_vel/body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w)."
      )

    if self.frame_range is not None:
      start, end = self.frame_range
      start_idx = max(start - 1, 0)
      end_idx = min(end, root_pos.shape[0])
      root_pos = root_pos[start_idx:end_idx]
      root_rot = root_rot[start_idx:end_idx]
      dof_pos = dof_pos[start_idx:end_idx]
      if self.motion_base_lin_vels_input is not None:
        self.motion_base_lin_vels_input = self.motion_base_lin_vels_input[
          start_idx:end_idx
        ]
      if self.motion_base_ang_vels_input is not None:
        self.motion_base_ang_vels_input = self.motion_base_ang_vels_input[
          start_idx:end_idx
        ]
      if self.motion_dof_vels_input is not None:
        self.motion_dof_vels_input = self.motion_dof_vels_input[start_idx:end_idx]

    if self.input_fps_override is not None:
      self.input_fps = float(self.input_fps_override)
    elif "fps" in motion:
      self.input_fps = (
        float(motion["fps"]) if np.ndim(motion["fps"]) == 0 else float(motion["fps"][0])
      )
    else:
      raise KeyError(f"{self.motion_file} has no fps and --input_fps is not set.")

    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps

    self.motion_base_poss_input = (
      torch.from_numpy(root_pos).to(torch.float32).to(self.device)
    )
    self.motion_base_rots_input = (
      torch.from_numpy(root_rot).to(torch.float32).to(self.device)
    )
    if quat_is_xyzw:
      self.motion_base_rots_input = self.motion_base_rots_input[
        :, [3, 0, 1, 2]
      ]  # xyzw -> wxyz
    self.motion_dof_poss_input = (
      torch.from_numpy(dof_pos).to(torch.float32).to(self.device)
    )

    self.input_frames = self.motion_base_poss_input.shape[0]
    if self.input_frames < 2:
      raise ValueError(
        f"Need at least 2 frames, got {self.input_frames} in {self.motion_file}"
      )
    self.duration = (self.input_frames - 1) * self.input_dt

    print(
      f"[INFO] Loaded motion: {self.motion_file}, duration: {self.duration:.3f}s, "
      f"frames: {self.input_frames}, input_fps: {self.input_fps:.3f}"
    )

    if self.motion_joint_names is not None:
      print(f"[INFO] Use joint_names from npz ({len(self.motion_joint_names)} joints).")

  def _interpolate_motion(self):
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    self._index_0, self._index_1, self._blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[self._index_0],
      self.motion_base_poss_input[self._index_1],
      self._blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[self._index_0],
      self.motion_base_rots_input[self._index_1],
      self._blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[self._index_0],
      self.motion_dof_poss_input[self._index_1],
      self._blend.unsqueeze(1),
    )

  def _compute_velocities(self):
    if self.motion_base_lin_vels_input is not None:
      self.motion_base_lin_vels = self._lerp(
        self.motion_base_lin_vels_input[self._index_0],
        self.motion_base_lin_vels_input[self._index_1],
        self._blend.unsqueeze(1),
      )
    else:
      self.motion_base_lin_vels = torch.gradient(
        self.motion_base_poss, spacing=self.output_dt, dim=0
      )[0]

    if self.motion_dof_vels_input is not None:
      self.motion_dof_vels = self._lerp(
        self.motion_dof_vels_input[self._index_0],
        self.motion_dof_vels_input[self._index_1],
        self._blend.unsqueeze(1),
      )
    else:
      self.motion_dof_vels = torch.gradient(
        self.motion_dof_poss, spacing=self.output_dt, dim=0
      )[0]

    if self.motion_base_ang_vels_input is not None:
      self.motion_base_ang_vels = self._lerp(
        self.motion_base_ang_vels_input[self._index_0],
        self.motion_base_ang_vels_input[self._index_1],
        self._blend.unsqueeze(1),
      )
    else:
      self.motion_base_ang_vels = self._so3_derivative(
        self.motion_base_rots, self.output_dt
      )

  def _compute_base_link_speed_stats(self):
    base_speed_xyz = torch.linalg.norm(self.motion_base_lin_vels, dim=1)
    base_speed_xy = torch.linalg.norm(self.motion_base_lin_vels[:, :2], dim=1)
    base_vel_mean_xyz = self.motion_base_lin_vels.mean(dim=0)
    base_vel_mean_xy = base_vel_mean_xyz[:2]

    self.base_link_avg_speed_xyz = float(base_speed_xyz.mean().item())
    self.base_link_avg_speed_xy = float(base_speed_xy.mean().item())
    self.base_link_max_speed_xyz = float(base_speed_xyz.max().item())
    self.base_link_max_speed_xy = float(base_speed_xy.max().item())
    self.base_link_mean_vel_vec_xy = (
      base_vel_mean_xy.detach().cpu().numpy().astype(np.float32)
    )
    self.base_link_mean_vel_vec_xyz = (
      base_vel_mean_xyz.detach().cpu().numpy().astype(np.float32)
    )
    self.base_link_mean_vel_mag_xy = float(torch.linalg.norm(base_vel_mean_xy).item())
    self.base_link_mean_vel_mag_xyz = float(torch.linalg.norm(base_vel_mean_xyz).item())

    displacement = self.motion_base_poss[-1] - self.motion_base_poss[0]
    self.base_link_net_speed_xy = float(
      (torch.linalg.norm(displacement[:2]) / self.duration).item()
    )
    displacement_xy_norm = torch.linalg.norm(displacement[:2])
    if float(displacement_xy_norm.item()) > 1e-8:
      direction_xy = displacement[:2] / displacement_xy_norm
    else:
      direction_xy = torch.zeros(
        2, dtype=displacement.dtype, device=displacement.device
      )
    self.base_link_direction_xy = direction_xy.detach().cpu().numpy().astype(np.float32)

    base_ang_vel_z = self.motion_base_ang_vels[:, 2]
    self.base_link_avg_ang_vel_z = float(base_ang_vel_z.mean().item())
    self.base_link_avg_abs_ang_vel_z = float(base_ang_vel_z.abs().mean().item())

    print(f"[INFO] motion name: {Path(self.motion_file).name}")
    print("[INFO] base_link average movement speed:")
    print(f"[INFO] avg_speed_xy: {self.base_link_avg_speed_xy:.4f} m/s")
    print(f"[INFO] avg_speed_xyz: {self.base_link_avg_speed_xyz:.4f} m/s")
    print(f"[INFO] max_speed_xy: {self.base_link_max_speed_xy:.4f} m/s")
    print(f"[INFO] max_speed_xyz: {self.base_link_max_speed_xyz:.4f} m/s")
    print(f"[INFO] net_speed_xy: {self.base_link_net_speed_xy:.4f} m/s")
    print(
      f"[INFO] dir_xy: [{self.base_link_direction_xy[0]:.4f}, {self.base_link_direction_xy[1]:.4f}]"
    )
    print(
      "[INFO] mean_vel_xy: "
      f"[{self.base_link_mean_vel_vec_xy[0]:.4f}, {self.base_link_mean_vel_vec_xy[1]:.4f}] m/s"
    )
    print(f"[INFO] mean_vel_mag_xy: {self.base_link_mean_vel_mag_xy:.4f} m/s")
    print(f"[INFO] avg_wz: {self.base_link_avg_ang_vel_z:.4f} rad/s")
    print(f"[INFO] avg_abs_wz: {self.base_link_avg_abs_ang_vel_z:.4f} rad/s")

  def plot_speed_curves(self, output_dir: str | Path):
    try:
      import matplotlib.pyplot as plt
    except ImportError:
      print("[WARNING] matplotlib is not installed, skip speed curve plotting.")
      return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    time_axis = (
      (
        torch.arange(self.output_frames, device=self.device, dtype=torch.float32)
        * self.output_dt
      )
      .cpu()
      .numpy()
    )
    speed_xy = torch.linalg.norm(self.motion_base_lin_vels[:, :2], dim=1).cpu().numpy()
    speed_xyz = torch.linalg.norm(self.motion_base_lin_vels, dim=1).cpu().numpy()
    ang_vel_z = self.motion_base_ang_vels[:, 2].cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(time_axis, speed_xy, color="tab:blue", linewidth=1.2)
    axes[0].axhline(
      self.base_link_avg_speed_xy,
      color="tab:blue",
      linestyle="--",
      alpha=0.8,
      label="avg",
    )
    axes[0].axhline(
      self.base_link_max_speed_xy,
      color="tab:red",
      linestyle=":",
      alpha=0.8,
      label="max",
    )
    axes[0].set_ylabel("speed_xy (m/s)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(time_axis, speed_xyz, color="tab:green", linewidth=1.2)
    axes[1].axhline(
      self.base_link_avg_speed_xyz,
      color="tab:green",
      linestyle="--",
      alpha=0.8,
      label="avg",
    )
    axes[1].axhline(
      self.base_link_max_speed_xyz,
      color="tab:red",
      linestyle=":",
      alpha=0.8,
      label="max",
    )
    axes[1].set_ylabel("speed_xyz (m/s)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(time_axis, ang_vel_z, color="tab:orange", linewidth=1.2)
    axes[2].axhline(
      self.base_link_avg_ang_vel_z,
      color="tab:orange",
      linestyle="--",
      alpha=0.8,
      label="avg_wz",
    )
    axes[2].set_ylabel("wz (rad/s)")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.suptitle(f"Motion speed curves: {Path(self.motion_file).name}")
    fig.tight_layout()

    output_path = output_dir / f"{Path(self.motion_file).stem}_speed_curve.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] speed curve saved: {output_path}")

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    out = torch.zeros_like(a)
    for i in range(a.shape[0]):
      out[i] = quat_slerp(a[i], b[i], blend[i])
    return out

  def _compute_frame_blend(self, times: torch.Tensor):
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(
      index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device)
    )
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
    return omega

  def get_next_state(self):
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


class MotionEnv(gym.Env):
  """A minimal Gym environment wrapper to support gym.wrappers.RecordVideo"""

  def __init__(self, sim, scene, motion, robot_joint_indexes, args_cli):
    self.sim = sim
    self.scene = scene
    self.motion = motion
    self.robot = scene["robot"]
    self.camera = scene["camera"]
    self.robot_joint_indexes = robot_joint_indexes
    self.args_cli = args_cli

    self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
    self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
    self.render_mode = "rgb_array"
    self.metadata = {"render_modes": ["rgb_array"], "render_fps": args_cli.output_fps}
    self._camera_offset = np.array([2.8, -2.8, 1.8], dtype=np.float32)
    self._camera_target_offset = np.array([0.0, 0.0, 0.3], dtype=np.float32)

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.motion.current_idx = 0
    return np.zeros(1, dtype=np.float32), {}

  def step(self, action):
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = self.motion.get_next_state()

    root_states = self.robot.data.default_root_state.clone()
    root_states[:, :3] = motion_base_pos
    root_states[:, :2] += self.scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = motion_base_ang_vel
    self.robot.write_root_state_to_sim(root_states)

    joint_pos = self.robot.data.default_joint_pos.clone()
    joint_vel = self.robot.data.default_joint_vel.clone()
    joint_pos[:, self.robot_joint_indexes] = motion_dof_pos
    joint_vel[:, self.robot_joint_indexes] = motion_dof_vel
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

    pos_lookat = root_states[0, :3].detach().cpu().numpy().astype(np.float32)
    cam_pos = pos_lookat + self._camera_offset
    cam_target = pos_lookat + self._camera_target_offset
    cam_pos[2] = max(cam_pos[2], 1.2)

    camera_positions = torch.tensor(
      [cam_pos], device=self.sim.device, dtype=torch.float32
    )
    camera_targets = torch.tensor(
      [cam_target], device=self.sim.device, dtype=torch.float32
    )

    if hasattr(self.camera, "set_world_poses_from_view"):
      self.camera.set_world_poses_from_view(camera_positions, camera_targets)
    else:
      self.camera.set_world_poses(
        camera_positions,
        torch.tensor(
          [[0.8535534, 0.1464466, 0.3535534, -0.3535534]],
          device=self.sim.device,
          dtype=torch.float32,
        ),
      )

    self.sim.render()
    self.scene.update(self.sim.get_physics_dt())

    if not self.args_cli.headless:
      self.sim.set_camera_view(cam_pos, cam_target)

    return np.zeros(1, dtype=np.float32), 0.0, reset_flag, False, {}

  def render(self):
    self.camera.update(self.sim.get_physics_dt())
    rgb = self.camera.data.output["rgb"][0].cpu().numpy()
    if rgb.shape[-1] == 4:
      rgb = rgb[..., :3]  # RGBA to RGB
    return rgb


def run_simulator(
  _args_cli,
  sim: sim_utils.SimulationContext,
  scene: InteractiveScene,
  joint_names: list[str],
):
  import tqdm

  motion = MotionLoader(
    motion_file=_args_cli.input_file,
    input_fps_override=_args_cli.input_fps,
    output_fps=_args_cli.output_fps,
    device=sim.device,
    frame_range=_args_cli.frame_range,
  )

  if _args_cli.plot_speed_curve:
    if _args_cli.plot_speed_output_dir is not None:
      output_dir = Path(_args_cli.plot_speed_output_dir)
    else:
      output_dir = Path(_args_cli.input_file).parent / "speed_curves"
    motion.plot_speed_curves(output_dir)

  robot = scene["robot"]
  source_joint_names = (
    motion.motion_joint_names if motion.motion_joint_names is not None else joint_names
  )
  robot_joint_indexes = robot.find_joints(source_joint_names, preserve_order=True)[0]

  if _args_cli.video:
    env = MotionEnv(sim, scene, motion, robot_joint_indexes, _args_cli)

    video_folder = "data/output_video/mos9_fk_motion"
    os.makedirs(video_folder, exist_ok=True)

    video_kwargs = {
      "video_folder": video_folder,
      "step_trigger": lambda step: step == 0,
      "video_length": motion.output_frames,
      "disable_logger": True,
      "name_prefix": Path(_args_cli.input_file).stem,
    }
    print(f"[INFO] Recording videos to {video_folder}")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env.reset()
    pbar = tqdm.tqdm(total=motion.output_frames, desc=_args_cli.input_file)

    try:
      while simulation_app.is_running():
        _, _, terminated, truncated, _ = env.step(None)
        pbar.update(1)

        if terminated or truncated:
          if _args_cli.loop:
            env.reset()
            pbar.reset()
          else:
            break
    finally:
      pbar.close()
      env.close()

  else:
    pbar = tqdm.tqdm(total=motion.output_frames, desc=_args_cli.input_file)
    try:
      while simulation_app.is_running():
        (
          (
            motion_base_pos,
            motion_base_rot,
            motion_base_lin_vel,
            motion_base_ang_vel,
            motion_dof_pos,
            motion_dof_vel,
          ),
          reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        if not _args_cli.headless:
          pos_lookat = root_states[0, :3].cpu().numpy()
          sim.set_camera_view(
            pos_lookat + np.array([2.8, -2.8, 1.8]),
            pos_lookat + np.array([0.0, 0.0, 0.7]),
          )

        pbar.update(1)
        if reset_flag:
          if _args_cli.loop:
            pbar.reset()
          else:
            break
    finally:
      pbar.close()


def main():
  sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
  sim_cfg.dt = 1.0 / args_cli.output_fps
  sim = SimulationContext(sim_cfg)

  scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
  scene = InteractiveScene(scene_cfg)
  sim.reset()
  print("[INFO]: Setup complete...")

  joint_names = [
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

  if args_cli.input_file is not None:
    input_files = [Path(args_cli.input_file)]
  else:
    input_dir = Path(args_cli.input_dir)
    input_files = sorted(input_dir.glob("*.npz"))

  if len(input_files) == 0:
    raise FileNotFoundError(
      f"No .npz files found. input_file={args_cli.input_file}, input_dir={args_cli.input_dir}"
    )

  print(f"[INFO] Found {len(input_files)} files")
  for input_file in input_files:
    args_cli.input_file = str(input_file)
    run_simulator(args_cli, sim, scene, joint_names=joint_names)
    if not simulation_app.is_running():
      break


if __name__ == "__main__":
  main()
  simulation_app.close()
