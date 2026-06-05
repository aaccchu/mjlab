import numpy as np
import torch


class ReplayBuffer:
  def __init__(self, obs_dim, buffer_size, device):
    self.states = torch.zeros(buffer_size, obs_dim).to(device)
    self.next_states = torch.zeros(buffer_size, obs_dim).to(device)
    self.bucket_ids = torch.zeros(buffer_size, dtype=torch.long).to(device)
    self.buffer_size = buffer_size
    self.device = device

    self.step = 0
    self.num_samples = 0

  def insert(self, states, next_states, bucket_ids=None):
    num_states = states.shape[0]
    start_idx = self.step
    end_idx = self.step + num_states
    if bucket_ids is None:
      bucket_ids = torch.zeros(num_states, dtype=torch.long, device=self.device)
    else:
      bucket_ids = bucket_ids.to(device=self.device, dtype=torch.long)

    if end_idx > self.buffer_size:
      self.states[self.step : self.buffer_size] = states[: self.buffer_size - self.step]
      self.next_states[self.step : self.buffer_size] = next_states[
        : self.buffer_size - self.step
      ]
      self.bucket_ids[self.step : self.buffer_size] = bucket_ids[
        : self.buffer_size - self.step
      ]
      self.states[: end_idx - self.buffer_size] = states[self.buffer_size - self.step :]
      self.next_states[: end_idx - self.buffer_size] = next_states[
        self.buffer_size - self.step :
      ]
      self.bucket_ids[: end_idx - self.buffer_size] = bucket_ids[
        self.buffer_size - self.step :
      ]
    else:
      self.states[start_idx:end_idx] = states
      self.next_states[start_idx:end_idx] = next_states
      self.bucket_ids[start_idx:end_idx] = bucket_ids

    self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
    self.step = (self.step + num_states) % self.buffer_size

  def feed_forward_generator(self, num_mini_batch, mini_batch_size):
    for _ in range(num_mini_batch):
      sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
      yield (
        self.states[sample_idxs].to(self.device),
        self.next_states[sample_idxs].to(self.device),
        self.bucket_ids[sample_idxs].to(self.device),
      )
