from pathlib import Path
from typing import List, Sequence

import isaaclab.utils.math as math_utils
import numpy as np
import torch


class MotionDataset:
  def __init__(
    self,
    motion_files: Sequence[str],
    body_names: Sequence[str],
    joint_names: Sequence[str] | None,
    base_body_name: str,
    amp_obs_terms: List[str],
    motion_buckets: dict[str, list[str]] | None = None,
    use_command_conditioned_sampling: bool = False,
    bucket_names: Sequence[str] | None = None,
    strict_bucket_check: bool = False,
    device: str = "cpu",
  ):
    self.device = device
    self.body_names = list(body_names)
    self.joint_names = list(joint_names) if joint_names is not None else None
    self.base_body_name = base_body_name
    self.observation_terms = amp_obs_terms
    self.use_command_conditioned_sampling = use_command_conditioned_sampling
    self.bucket_names = (
      list(bucket_names)
      if bucket_names is not None
      else ["forward", "backward", "left", "right"]
    )
    self.motion_buckets = motion_buckets or {}
    self.strict_bucket_check = strict_bucket_check

    joint_pos_list = []
    joint_vel_list = []
    body_pos_w_list = []
    body_quat_w_list = []
    body_lin_vel_w_list = []
    body_ang_vel_w_list = []
    root_pos_w_list = []
    root_quat_w_list = []
    root_lin_vel_w_list = []
    root_ang_vel_w_list = []
    fps_list = []
    traj_lengths = []
    traj_transition_ranges: list[tuple[int, int]] = []
    loaded_motion_files: list[str] = []

    project_root_dir_path = Path(__file__).resolve().parents[5]
    requires_body_terms = any(
      term in self.observation_terms
      for term in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    )
    requires_body_local_terms = any(
      term in self.observation_terms
      for term in ("body_quat_b", "body_lin_vel_b", "body_ang_vel_b")
    )
    requires_root_local_terms = any(
      term in self.observation_terms for term in ("root_lin_vel_b", "root_ang_vel_b")
    )
    requires_joint_terms = any(
      term in self.observation_terms for term in ("joint_pos", "joint_vel")
    )

    for motion_file in motion_files:
      motion_path = project_root_dir_path / motion_file
      assert motion_path.is_file(), f"Invalid motion file: {motion_path}"
      data = np.load(motion_path)
      loaded_motion_files.append(motion_file)

      fps_list.append(float(data["fps"]))
      traj_len = data["joint_pos"].shape[0]
      traj_lengths.append(traj_len)

      joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32)
      joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32)

      if requires_joint_terms:
        if self.joint_names is None:
          raise KeyError(
            f"{motion_path} requires joint alignment for AMP terms {self.observation_terms}, but target joint_names is None."
          )
        if "joint_names" not in data:
          raise KeyError(
            f"{motion_path} missing 'joint_names', cannot align AMP joint observation terms: {self.observation_terms}"
          )

        file_joint_names = [str(name) for name in data["joint_names"].tolist()]
        file_joint_name_to_idx = {
          name: idx for idx, name in enumerate(file_joint_names)
        }
        missing_joint_names = [
          name for name in self.joint_names if name not in file_joint_name_to_idx
        ]
        if missing_joint_names:
          raise KeyError(
            f"{motion_path} missing required joint names for AMP: {missing_joint_names}. "
            f"Available: {file_joint_names}"
          )

        ordered_joint_indices = torch.tensor(
          [file_joint_name_to_idx[name] for name in self.joint_names], dtype=torch.long
        )
        joint_pos = joint_pos[:, ordered_joint_indices]
        joint_vel = joint_vel[:, ordered_joint_indices]

      joint_pos_list.append(joint_pos)
      joint_vel_list.append(joint_vel)

      body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32)
      body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32)
      body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32)
      body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32)

      if "body_names" not in data:
        raise KeyError(
          f"{motion_path} missing 'body_names', cannot resolve base_body_name='{self.base_body_name}'."
        )
      file_body_names = [str(name) for name in data["body_names"].tolist()]
      file_body_name_to_idx = {name: idx for idx, name in enumerate(file_body_names)}
      if self.base_body_name not in file_body_name_to_idx:
        raise KeyError(
          f"{motion_path} missing base_body_name='{self.base_body_name}'. Available: {file_body_names}"
        )
      base_body_idx = file_body_name_to_idx[self.base_body_name]
      root_pos_w_list.append(body_pos_w[:, base_body_idx, :])
      root_quat_w_list.append(body_quat_w[:, base_body_idx, :])
      root_lin_vel_w_list.append(body_lin_vel_w[:, base_body_idx, :])
      root_ang_vel_w_list.append(body_ang_vel_w[:, base_body_idx, :])

      if requires_body_terms or requires_body_local_terms:
        missing_names = [
          name for name in self.body_names if name not in file_body_name_to_idx
        ]
        if missing_names:
          raise KeyError(
            f"{motion_path} missing required body names for AMP: {missing_names}. "
            f"Available: {file_body_names}"
          )

        ordered_body_indices = torch.tensor(
          [file_body_name_to_idx[name] for name in self.body_names], dtype=torch.long
        )
        body_pos_w = body_pos_w[:, ordered_body_indices]
        body_quat_w = body_quat_w[:, ordered_body_indices]
        body_lin_vel_w = body_lin_vel_w[:, ordered_body_indices]
        body_ang_vel_w = body_ang_vel_w[:, ordered_body_indices]

      body_pos_w_list.append(body_pos_w)
      body_quat_w_list.append(body_quat_w)
      body_lin_vel_w_list.append(body_lin_vel_w)
      body_ang_vel_w_list.append(body_ang_vel_w)

    transition_offset = 0
    for traj_len in traj_lengths:
      transition_len = max(traj_len - 1, 0)
      traj_transition_ranges.append(
        (transition_offset, transition_offset + transition_len)
      )
      transition_offset += transition_len

    self.joint_pos = torch.cat(joint_pos_list, dim=0).to(device)
    self.joint_vel = torch.cat(joint_vel_list, dim=0).to(device)
    body_pos_w_all = torch.cat(body_pos_w_list, dim=0).to(device)
    body_quat_w_all = torch.cat(body_quat_w_list, dim=0).to(device)
    body_lin_vel_w_all = torch.cat(body_lin_vel_w_list, dim=0).to(device)
    body_ang_vel_w_all = torch.cat(body_ang_vel_w_list, dim=0).to(device)

    self.total_dataset_size = sum(traj_lengths)

    self.root_pos_w = torch.cat(root_pos_w_list, dim=0).to(device)
    self.root_quat_w = torch.cat(root_quat_w_list, dim=0).to(device)
    self.root_lin_vel_w = torch.cat(root_lin_vel_w_list, dim=0).to(device)
    self.root_ang_vel_w = torch.cat(root_ang_vel_w_list, dim=0).to(device)
    if requires_root_local_terms:
      self.root_lin_vel_b = math_utils.quat_apply_inverse(
        self.root_quat_w, self.root_lin_vel_w
      )
      self.root_ang_vel_b = math_utils.quat_apply_inverse(
        self.root_quat_w, self.root_ang_vel_w
      )

    if requires_body_local_terms:
      root_quat_expand = self.root_quat_w.unsqueeze(1).expand(
        -1, body_quat_w_all.shape[1], -1
      )
      root_quat_flat = root_quat_expand.reshape(-1, 4)

      body_quat_b_all = math_utils.quat_mul(
        math_utils.quat_conjugate(root_quat_flat),
        body_quat_w_all.reshape(-1, 4),
      ).reshape_as(body_quat_w_all)

      body_lin_rel_w = body_lin_vel_w_all - self.root_lin_vel_w.unsqueeze(1)
      body_lin_vel_b_all = math_utils.quat_apply_inverse(
        root_quat_flat,
        body_lin_rel_w.reshape(-1, 3),
      ).reshape_as(body_lin_vel_w_all)

      body_ang_rel_w = body_ang_vel_w_all - self.root_ang_vel_w.unsqueeze(1)
      body_ang_vel_b_all = math_utils.quat_apply_inverse(
        root_quat_flat,
        body_ang_rel_w.reshape(-1, 3),
      ).reshape_as(body_ang_vel_w_all)

    def subtract_flaten(target: torch.Tensor):
      return target.reshape(self.total_dataset_size, -1)

    self.body_pos_w = subtract_flaten(body_pos_w_all)
    self.body_quat_w = subtract_flaten(body_quat_w_all)
    self.body_lin_vel_w = subtract_flaten(body_lin_vel_w_all)
    self.body_ang_vel_w = subtract_flaten(body_ang_vel_w_all)
    if requires_body_local_terms:
      self.body_quat_b = subtract_flaten(body_quat_b_all)
      self.body_lin_vel_b = subtract_flaten(body_lin_vel_b_all)
      self.body_ang_vel_b = subtract_flaten(body_ang_vel_b_all)

    self.fps_list = fps_list

    self.index_t, self.index_tp1 = self._build_transition_indices(traj_lengths, device)

    self.bucket_index_t: dict[str, torch.Tensor] = {}
    self.bucket_index_tp1: dict[str, torch.Tensor] = {}
    if self.use_command_conditioned_sampling:
      if not self.motion_buckets:
        raise ValueError(
          "use_command_conditioned_sampling=True but motion_buckets is empty."
        )

      if self.strict_bucket_check:
        bucket_files_all = []
        for bucket_name in self.bucket_names:
          bucket_files_all.extend(self.motion_buckets.get(bucket_name, []))

        duplicate_files = sorted(
          {f for f in bucket_files_all if bucket_files_all.count(f) > 1}
        )
        if duplicate_files:
          raise ValueError(
            f"motion_buckets has duplicated files across buckets (strict mode): {duplicate_files}"
          )

        missing_bucket_files = sorted(set(loaded_motion_files) - set(bucket_files_all))
        if missing_bucket_files:
          raise ValueError(
            "motion_files not fully covered by motion_buckets (strict mode). Missing: "
            f"{missing_bucket_files}"
          )

      file_to_transition_range = {
        motion_file: traj_transition_ranges[idx]
        for idx, motion_file in enumerate(loaded_motion_files)
      }
      full_bucket_t = self.index_t
      full_bucket_tp1 = self.index_tp1
      for bucket_name in self.bucket_names:
        bucket_files = self.motion_buckets.get(bucket_name, [])
        bucket_t_segments = []
        bucket_tp1_segments = []
        for bucket_file in bucket_files:
          if bucket_file not in file_to_transition_range:
            continue
          start, end = file_to_transition_range[bucket_file]
          if end <= start:
            continue
          bucket_t_segments.append(full_bucket_t[start:end])
          bucket_tp1_segments.append(full_bucket_tp1[start:end])

        if not bucket_t_segments:
          raise ValueError(
            f"Bucket '{bucket_name}' has no valid transitions. Check motion_buckets and motion_files overlap."
          )

        self.bucket_index_t[bucket_name] = torch.cat(bucket_t_segments, dim=0)
        self.bucket_index_tp1[bucket_name] = torch.cat(bucket_tp1_segments, dim=0)

    self.init_observation_dims()

  @classmethod
  def from_cfg(cls, cfg: dict, env, device):
    body_names = cfg["body_names"]
    robot = env.scene[cfg["asset_name"]]
    joint_names = cfg.get("joint_names", None)
    if joint_names is None:
      joint_names = [str(name) for name in robot.joint_names]
    base_body_name = cfg.get("base_body_name", "base_link")
    obj = cls(
      motion_files=cfg["motion_files"],
      body_names=body_names,
      joint_names=joint_names,
      base_body_name=base_body_name,
      amp_obs_terms=cfg["amp_obs_terms"],
      motion_buckets=cfg.get("motion_buckets", None),
      use_command_conditioned_sampling=cfg.get(
        "use_command_conditioned_sampling", False
      ),
      bucket_names=cfg.get("bucket_names", ["forward", "backward", "left", "right"]),
      strict_bucket_check=cfg.get("strict_bucket_check", False),
      device=device,
    )
    return obj

  def observation_dim_cast(self, name) -> int:
    if hasattr(self, name):
      obs_term: torch.Tensor = getattr(self, name)
      assert isinstance(obs_term, torch.Tensor), (
        f"invalid observation name: {name} for get dim"
      )
      return obs_term.shape[-1]
    raise NotImplementedError(f"Failed for term: {name}")

  def init_observation_dims(self):
    observation_dims = []
    for obs_term in self.observation_terms:
      observation_dims.append(self.observation_dim_cast(obs_term))
    self.observation_dim = sum(observation_dims)
    self.observation_dims = observation_dims

  def _build_transition_indices(self, traj_lengths: List[int], device: str):
    idx_t = []
    idx_tp1 = []

    offset = 0
    for traj_len in traj_lengths:
      if traj_len < 2:
        offset += traj_len
        continue
      t = torch.arange(offset, offset + traj_len - 1)
      idx_t.append(t)
      idx_tp1.append(t + 1)
      offset += traj_len

    idx_t = torch.cat(idx_t).to(device)
    idx_tp1 = torch.cat(idx_tp1).to(device)
    return idx_t, idx_tp1

  def sample_batch(self, batch_size: int):
    idx = torch.randint(0, len(self.index_t), (batch_size,), device=self.device)
    t = self.index_t[idx]
    tp1 = self.index_tp1[idx]

    return {
      "joint_pos": (self.joint_pos[t], self.joint_pos[tp1]),
      "joint_vel": (self.joint_vel[t], self.joint_vel[tp1]),
      "root_lin_vel_w": (self.root_lin_vel_w[t], self.root_lin_vel_w[tp1]),
      "root_ang_vel_w": (self.root_ang_vel_w[t], self.root_ang_vel_w[tp1]),
      "root_lin_vel_b": (self.root_lin_vel_b[t], self.root_lin_vel_b[tp1])
      if hasattr(self, "root_lin_vel_b")
      else None,
      "root_ang_vel_b": (self.root_ang_vel_b[t], self.root_ang_vel_b[tp1])
      if hasattr(self, "root_ang_vel_b")
      else None,
      "body_pos_w": (self.body_pos_w[t], self.body_pos_w[tp1]),
      "body_quat_w": (self.body_quat_w[t], self.body_quat_w[tp1]),
      "body_lin_vel_w": (self.body_lin_vel_w[t], self.body_lin_vel_w[tp1]),
      "body_ang_vel_w": (self.body_ang_vel_w[t], self.body_ang_vel_w[tp1]),
      "body_quat_b": (self.body_quat_b[t], self.body_quat_b[tp1])
      if hasattr(self, "body_quat_b")
      else None,
      "body_lin_vel_b": (self.body_lin_vel_b[t], self.body_lin_vel_b[tp1])
      if hasattr(self, "body_lin_vel_b")
      else None,
      "body_ang_vel_b": (self.body_ang_vel_b[t], self.body_ang_vel_b[tp1])
      if hasattr(self, "body_ang_vel_b")
      else None,
    }

  def _gather_terms(self, t: torch.Tensor, tp1: torch.Tensor):
    return {
      "joint_pos": (self.joint_pos[t], self.joint_pos[tp1]),
      "joint_vel": (self.joint_vel[t], self.joint_vel[tp1]),
      "root_lin_vel_w": (self.root_lin_vel_w[t], self.root_lin_vel_w[tp1]),
      "root_ang_vel_w": (self.root_ang_vel_w[t], self.root_ang_vel_w[tp1]),
      "root_lin_vel_b": (self.root_lin_vel_b[t], self.root_lin_vel_b[tp1])
      if hasattr(self, "root_lin_vel_b")
      else None,
      "root_ang_vel_b": (self.root_ang_vel_b[t], self.root_ang_vel_b[tp1])
      if hasattr(self, "root_ang_vel_b")
      else None,
      "body_pos_w": (self.body_pos_w[t], self.body_pos_w[tp1]),
      "body_quat_w": (self.body_quat_w[t], self.body_quat_w[tp1]),
      "body_lin_vel_w": (self.body_lin_vel_w[t], self.body_lin_vel_w[tp1]),
      "body_ang_vel_w": (self.body_ang_vel_w[t], self.body_ang_vel_w[tp1]),
      "body_quat_b": (self.body_quat_b[t], self.body_quat_b[tp1])
      if hasattr(self, "body_quat_b")
      else None,
      "body_lin_vel_b": (self.body_lin_vel_b[t], self.body_lin_vel_b[tp1])
      if hasattr(self, "body_lin_vel_b")
      else None,
      "body_ang_vel_b": (self.body_ang_vel_b[t], self.body_ang_vel_b[tp1])
      if hasattr(self, "body_ang_vel_b")
      else None,
    }

  def sample_batch_by_bucket_ids(self, bucket_ids: torch.Tensor):
    if not self.use_command_conditioned_sampling:
      raise RuntimeError("Bucket-conditioned sampling is disabled for this dataset.")

    bucket_ids = bucket_ids.to(self.device, dtype=torch.long)
    valid_bucket_ids = torch.arange(
      len(self.bucket_names), device=self.device, dtype=torch.long
    )
    if not torch.isin(bucket_ids, valid_bucket_ids).all():
      invalid_values = torch.unique(
        bucket_ids[~torch.isin(bucket_ids, valid_bucket_ids)]
      ).tolist()
      raise RuntimeError(
        f"Invalid bucket ids requested: {invalid_values}. Expected range [0, {len(self.bucket_names) - 1}]."
      )

    batch_size = bucket_ids.shape[0]
    t = torch.empty(batch_size, dtype=torch.long, device=self.device)
    tp1 = torch.empty(batch_size, dtype=torch.long, device=self.device)

    for bucket_idx, bucket_name in enumerate(self.bucket_names):
      mask = bucket_ids == bucket_idx
      if not torch.any(mask):
        continue
      count = int(mask.sum().item())
      bucket_t = self.bucket_index_t[bucket_name]
      bucket_tp1 = self.bucket_index_tp1[bucket_name]
      sampled_idx = torch.randint(0, bucket_t.shape[0], (count,), device=self.device)
      t[mask] = bucket_t[sampled_idx]
      tp1[mask] = bucket_tp1[sampled_idx]

    sample = self._gather_terms(t, tp1)
    t1, tp1_concat = [], []
    for term in self.observation_terms:
      if sample[term] is None:
        raise KeyError(
          f"AMP term '{term}' requested but corresponding dataset tensor is unavailable."
        )
      _t1, _tp1 = sample[term]
      t1.append(_t1)
      tp1_concat.append(_tp1)
    return torch.cat(t1, dim=-1), torch.cat(tp1_concat, dim=-1)

  def feed_forward_generator(self, num_mini_batch, mini_batch_size):
    for idx in range(0, num_mini_batch):
      sample = self.sample_batch(mini_batch_size)
      t1, tp1 = [], []
      for term in self.observation_terms:
        if sample[term] is None:
          raise KeyError(
            f"AMP term '{term}' requested but corresponding dataset tensor is unavailable."
          )
        _t1, _tp1 = sample[term]
        t1.append(_t1)
        tp1.append(_tp1)
      t1, tp1 = torch.cat(t1, dim=-1), torch.cat(tp1, dim=-1)
      yield t1, tp1

  def get_random_states(self, batch_size: int):
    idx = torch.randint(0, len(self.index_t), (batch_size,), device=self.device)
    t = self.index_t[idx]

    return {
      "root_pos_w": self.root_pos_w[t],
      "root_quat_w": self.root_quat_w[t],
      "root_lin_vel_w": self.root_lin_vel_w[t],
      "root_ang_vel_w": self.root_ang_vel_w[t],
      "joint_pos": self.joint_pos[t],
      "joint_vel": self.joint_vel[t],
    }
