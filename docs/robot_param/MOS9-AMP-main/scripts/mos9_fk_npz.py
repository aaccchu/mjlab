"""Replay MOS9 motion npz files in Isaac Sim and export simulated states to npz.

Example:
    python scripts/mos9_fk_npz.py --input_dir data/motions/mos9 --output_dir datasets/temp/mos9_fk
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
  description="Replay MOS9 motion npz files and output replayed npz files."
)
parser.add_argument(
  "--input_file", type=str, default=None, help="Single input motion .npz file."
)
parser.add_argument(
  "--input_dir", type=str, default="data/motions/mos9", help="Input motion directory."
)
parser.add_argument(
  "--output_dir",
  type=str,
  default="datasets/temp/mos9_fk",
  help="Output directory for replayed npz files.",
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
  help=(
    "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
    " loaded."
  ),
)
parser.add_argument(
  "--output_fps", type=int, default=50, help="The fps of the output motion."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
import torch
from amp_tasks.robots.MOS9 import MOS9_CYLINDER_CFG
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
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
  """Configuration for a replay motions scene."""

  # ground plane
  ground = AssetBaseCfg(
    prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
  )

  # lights
  sky_light = AssetBaseCfg(
    prim_path="/World/skyLight",
    spawn=sim_utils.DomeLightCfg(
      intensity=750.0,
      texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    ),
  )

  # articulation
  robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")


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

  def _load_motion(self):
    """Loads motion from npz file."""
    motion = np.load(self.motion_file, allow_pickle=True)

    if "root_pos" not in motion or "root_rot" not in motion or "dof_pos" not in motion:
      raise KeyError(
        f"{self.motion_file} missing required keys. Required: root_pos, root_rot, dof_pos; got: {motion.files}"
      )

    root_pos = motion["root_pos"]
    root_rot = motion["root_rot"]
    dof_pos = motion["dof_pos"]
    self.motion_link_names = (
      motion["link_body_list"].tolist() if "link_body_list" in motion else None
    )
    if self.motion_link_names is not None and len(self.motion_link_names) == 0:
      self.motion_link_names = None

    if self.frame_range is not None:
      start, end = self.frame_range
      start_idx = max(start - 1, 0)
      end_idx = min(end, root_pos.shape[0])
      root_pos = root_pos[start_idx:end_idx]
      root_rot = root_rot[start_idx:end_idx]
      dof_pos = dof_pos[start_idx:end_idx]

    # infer fps from file unless overridden
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

    # motion npz root_rot is xyzw; Isaac expects wxyz
    self.motion_base_rots_input = (
      torch.from_numpy(root_rot).to(torch.float32).to(self.device)
    )
    self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]

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
      f"Motion loaded ({self.motion_file}), duration: {self.duration:.4f} sec, "
      f"frames: {self.input_frames}, input_fps: {self.input_fps:.4f}"
    )

  def _interpolate_motion(self):
    """Interpolates the motion to the output fps."""
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps:.4f}, "
      f"output frames: {self.output_frames}, output fps: {self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Linear interpolation between two tensors."""
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions."""
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
    return slerped_quats

  def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
    """Computes the frame blend for the motion."""
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(
      index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device)
    )
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    """Computes the velocities of the motion."""
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Computes the derivative of a sequence of SO3 rotations.

    Args:
        rotations: shape (B, 4).
        dt: time step.
    Returns:
        shape (B, 3).
    """
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))

    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
    return omega

  def get_next_state(
    self,
  ) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
  ]:
    """Gets the next state of the motion."""
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


def run_simulator(
  _args_cli,
  sim: sim_utils.SimulationContext,
  scene: InteractiveScene,
  joint_names: list[str],
):
  """Runs the simulation loop."""
  # Load motion
  motion = MotionLoader(
    motion_file=_args_cli.input_file,
    input_fps_override=_args_cli.input_fps,
    output_fps=_args_cli.output_fps,
    device=sim.device,
    frame_range=_args_cli.frame_range,
  )

  import tqdm

  pbar = tqdm.tqdm(range(motion.output_frames), desc=_args_cli.input_file)

  # Extract scene entities
  robot = scene["robot"]
  robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

  # preserve output order to match source motion format/order
  # joint order follows input motion motor order (joint_names)
  joint_readout_indexes = robot_joint_indexes

  # body order follows input motion link_body_list when available
  if motion.motion_link_names is not None and len(motion.motion_link_names) > 0:
    body_readout_indexes = robot.find_bodies(
      motion.motion_link_names, preserve_order=True
    )[0]
    body_readout_indexes = torch.as_tensor(
      body_readout_indexes, device=sim.device, dtype=torch.long
    )
    if body_readout_indexes.numel() == 0:
      body_readout_indexes = torch.arange(
        len(robot.body_names), device=sim.device, dtype=torch.long
      )
      body_names_out = list(robot.body_names)
      print(
        f"[WARN] Could not resolve body names from motion file: {motion.motion_link_names}. Use all robot bodies."
      )
    else:
      body_names_out = motion.motion_link_names
  else:
    body_readout_indexes = torch.arange(
      len(robot.body_names), device=sim.device, dtype=torch.long
    )
    body_names_out = list(robot.body_names)

  # ------- data logger -------------------------------------------------------
  log = {
    "fps": [_args_cli.output_fps],
    "joint_names": np.array(joint_names),
    "body_names": np.asarray(body_names_out, dtype=np.str_),
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  file_saved = False
  # --------------------------------------------------------------------------

  # Simulation loop
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

      # set root state
      root_states = robot.data.default_root_state.clone()
      root_states[:, :3] = motion_base_pos
      root_states[:, :2] += scene.env_origins[:, :2]
      root_states[:, 3:7] = motion_base_rot
      root_states[:, 7:10] = motion_base_lin_vel
      root_states[:, 10:] = motion_base_ang_vel
      robot.write_root_state_to_sim(root_states)

      # set joint state
      joint_pos = robot.data.default_joint_pos.clone()
      joint_vel = robot.data.default_joint_vel.clone()
      joint_pos[:, robot_joint_indexes] = motion_dof_pos
      joint_vel[:, robot_joint_indexes] = motion_dof_vel
      robot.write_joint_state_to_sim(joint_pos, joint_vel)
      sim.render()  # We don't want physic (sim.step())
      scene.update(sim.get_physics_dt())

      if not _args_cli.headless:
        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

      if not file_saved:
        log["joint_pos"].append(
          robot.data.joint_pos[0, joint_readout_indexes].cpu().numpy().copy()
        )
        log["joint_vel"].append(
          robot.data.joint_vel[0, joint_readout_indexes].cpu().numpy().copy()
        )
        log["body_pos_w"].append(
          robot.data.body_pos_w[0, body_readout_indexes].cpu().numpy().copy()
        )
        log["body_quat_w"].append(
          robot.data.body_quat_w[0, body_readout_indexes].cpu().numpy().copy()
        )
        log["body_lin_vel_w"].append(
          robot.data.body_lin_vel_w[0, body_readout_indexes].cpu().numpy().copy()
        )
        log["body_ang_vel_w"].append(
          robot.data.body_ang_vel_w[0, body_readout_indexes].cpu().numpy().copy()
        )

        # import pdb; pdb.set_trace()  # for debugging, can be removed later

      pbar.update(1)

      if reset_flag and not file_saved:
        file_saved = True
        for k in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          log[k] = np.stack(log[k], axis=0)

        fname = f"{_args_cli.output_name}.npz"
        np.savez(fname, **log)
        print(f"save to {fname}")
        break
  finally:
    pbar.close()

  return


def main():
  """Main function."""
  # Load kit helper
  sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
  sim_cfg.dt = 1.0 / args_cli.output_fps
  sim = SimulationContext(sim_cfg)
  # Design scene
  scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
  scene = InteractiveScene(scene_cfg)
  # Play the simulator
  sim.reset()
  print("[INFO]: Setup complete...")

  # MOS9 motor order from data/motions/README.md
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

  output_dir = Path(args_cli.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  print(f"[INFO] Found {len(input_files)} files")

  for input_file in input_files:
    args_cli.input_file = str(input_file)
    args_cli.output_name = str(output_dir / input_file.stem)

    run_simulator(
      args_cli,
      sim,
      scene,
      joint_names=joint_names,
    )


if __name__ == "__main__":
  # run the main function
  main()
  # close sim app
  simulation_app.close()
  exit()
