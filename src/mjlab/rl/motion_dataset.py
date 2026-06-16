"""Human-motion reference dataset for AMP (Adversarial Motion Priors).

Loads the MOS9 reference clips (docs/robot_param/MOS9-AMP-main/data/motions/),
which were FK-generated from the same MJCF as the live mjlab MOS92 model — so the
stored joint_pos/joint_vel are in the SAME convention as the robot's runtime
joint state (verified to 0 mm by scripts/amp/check_motion_dataset.py). This lets
the AMP discriminator compare the policy's per-step joint state against the human
reference directly, with no sign/axis remapping.

This is a slimmed port of the IsaacLab MotionDataset in
docs/robot_param/MOS9-AMP-main/.../velocity/mdp/motion_dataset.py, restricted to
the joint-space AMP observation (joint_pos [+ joint_vel]) used by AMPObsBaisc. We
deliberately drop the body/root world-frame terms and their quaternion math: the
chosen AMP obs is joint-only, the most robust signal for "walk like a human"
(stride/cadence/lift live in the leg-joint trajectory), and joint-only sidesteps
any world-frame/base-frame convention drift between IsaacLab and mjlab.

The discriminator scores transition PAIRS (state_t, state_{t+1}); this dataset
samples such pairs from within single clips (never across clip boundaries).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

# Project root so motion_files paths can be given relative to the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# AMP observation terms this loader supports (joint-space only).
_SUPPORTED_TERMS = ("joint_pos", "joint_vel")


class MotionDataset:
  """Reference human-motion clips, exposed as sampleable AMP transition pairs."""

  def __init__(
    self,
    motion_files: Sequence[str],
    joint_names: Sequence[str],
    amp_obs_terms: Sequence[str] = ("joint_pos", "joint_vel"),
    device: str = "cpu",
  ) -> None:
    for term in amp_obs_terms:
      if term not in _SUPPORTED_TERMS:
        raise ValueError(
          f"MotionDataset supports only {_SUPPORTED_TERMS} (joint-space AMP); "
          f"got unsupported term '{term}'."
        )
    self.device = device
    self.joint_names = list(joint_names)
    self.observation_terms = list(amp_obs_terms)

    joint_pos_list: list[torch.Tensor] = []
    joint_vel_list: list[torch.Tensor] = []
    traj_lengths: list[int] = []
    fps_list: list[float] = []

    for motion_file in motion_files:
      path = (_REPO_ROOT / motion_file).resolve()
      if not path.is_file():
        raise FileNotFoundError(f"motion file not found: {path}")
      data = np.load(path, allow_pickle=True)
      file_joint_names = [str(n) for n in data["joint_names"].tolist()]
      name_to_idx = {n: i for i, n in enumerate(file_joint_names)}
      missing = [n for n in self.joint_names if n not in name_to_idx]
      if missing:
        raise KeyError(
          f"{path} missing required AMP joints {missing}. Available: {file_joint_names}"
        )
      order = torch.tensor([name_to_idx[n] for n in self.joint_names], dtype=torch.long)
      jp = torch.tensor(data["joint_pos"], dtype=torch.float32)[:, order]
      jv = torch.tensor(data["joint_vel"], dtype=torch.float32)[:, order]
      joint_pos_list.append(jp)
      joint_vel_list.append(jv)
      traj_lengths.append(jp.shape[0])
      fps_list.append(float(np.asarray(data["fps"]).reshape(-1)[0]))

    self.joint_pos = torch.cat(joint_pos_list, dim=0).to(device)
    self.joint_vel = torch.cat(joint_vel_list, dim=0).to(device)
    self.fps_list = fps_list
    self.num_joints = len(self.joint_names)
    self._terms = {"joint_pos": self.joint_pos, "joint_vel": self.joint_vel}

    # Per-term observation dim and total dim of a SINGLE state (not the pair).
    self.observation_dims = [self.num_joints for _ in self.observation_terms]
    self.observation_dim = sum(self.observation_dims)

    # Transition indices: (t, t+1) pairs that never cross a clip boundary.
    self.index_t, self.index_tp1 = self._build_transition_indices(traj_lengths, device)
    if len(self.index_t) == 0:
      raise ValueError("dataset has no valid transition pairs (clips too short).")

  @classmethod
  def from_cfg(cls, cfg: dict, device: str) -> "MotionDataset":
    return cls(
      motion_files=cfg["motion_files"],
      joint_names=cfg["joint_names"],
      amp_obs_terms=cfg.get("amp_obs_terms", ("joint_pos", "joint_vel")),
      device=device,
    )

  @staticmethod
  def _build_transition_indices(
    traj_lengths: list[int], device: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    idx_t, idx_tp1 = [], []
    offset = 0
    for traj_len in traj_lengths:
      if traj_len >= 2:
        t = torch.arange(offset, offset + traj_len - 1)
        idx_t.append(t)
        idx_tp1.append(t + 1)
      offset += traj_len
    if not idx_t:
      empty = torch.empty(0, dtype=torch.long, device=device)
      return empty, empty
    return torch.cat(idx_t).to(device), torch.cat(idx_tp1).to(device)

  def _assemble(self, idx: torch.Tensor) -> torch.Tensor:
    """Concatenate the configured AMP terms at the given frame indices."""
    return torch.cat(
      [self._terms[term][idx] for term in self.observation_terms], dim=-1
    )

  def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (state, next_state) AMP-observation pairs from the reference clips."""
    sel = torch.randint(0, len(self.index_t), (batch_size,), device=self.device)
    t = self.index_t[sel]
    tp1 = self.index_tp1[sel]
    return self._assemble(t), self._assemble(tp1)

  def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
    for _ in range(num_mini_batch):
      yield self.sample(mini_batch_size)
