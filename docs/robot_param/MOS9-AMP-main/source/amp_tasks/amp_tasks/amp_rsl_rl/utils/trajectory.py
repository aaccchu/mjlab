import torch


def split_and_pad_trajectories(tensor, dones):
  dones = dones.clone()
  dones[-1] = 1
  flat_dones = dones.transpose(1, 0).reshape(-1, 1)

  done_indices = torch.cat(
    (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0])
  )
  trajectory_lengths = done_indices[1:] - done_indices[:-1]
  trajectory_lengths_list = trajectory_lengths.tolist()
  trajectories = torch.split(
    tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list
  )
  padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)

  trajectory_masks = trajectory_lengths > torch.arange(
    0, tensor.shape[0], device=tensor.device
  ).unsqueeze(1)
  return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
  return (
    trajectories.transpose(1, 0)[masks.transpose(1, 0)]
    .view(-1, trajectories.shape[0], trajectories.shape[-1])
    .transpose(1, 0)
  )
