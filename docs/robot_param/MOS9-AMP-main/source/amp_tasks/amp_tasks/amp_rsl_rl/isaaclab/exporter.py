import copy
import os

import torch


def export_policy_as_jit(
  policy: object, normalizer: object | None, path: str, filename="policy.pt"
):
  policy_exporter = _TorchPolicyExporter(policy, normalizer)
  policy_exporter.export(path, filename)


def export_policy_as_onnx(
  policy: object,
  path: str,
  normalizer: object | None = None,
  filename="policy.onnx",
  verbose=False,
):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
  policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
  policy_exporter.export(path, filename)


class _TorchPolicyExporter(torch.nn.Module):
  def __init__(self, policy, normalizer=None):
    super().__init__()
    self.is_recurrent = policy.is_recurrent
    if hasattr(policy, "actor"):
      self.actor = copy.deepcopy(policy.actor)
      if self.is_recurrent:
        self.rnn = copy.deepcopy(policy.memory_a.rnn)
    elif hasattr(policy, "student"):
      self.actor = copy.deepcopy(policy.student)
      if self.is_recurrent:
        self.rnn = copy.deepcopy(policy.memory_s.rnn)
    else:
      raise ValueError("Policy does not have an actor/student module.")

    if self.is_recurrent:
      self.rnn.cpu()
      self.rnn_type = type(self.rnn).__name__.lower()
      self.register_buffer(
        "hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
      )
      if self.rnn_type == "lstm":
        self.register_buffer(
          "cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
        )
        self.forward = self.forward_lstm
        self.reset = self.reset_memory
      elif self.rnn_type == "gru":
        self.forward = self.forward_gru
        self.reset = self.reset_memory
      else:
        raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")

    if normalizer:
      self.normalizer = copy.deepcopy(normalizer)
    else:
      self.normalizer = torch.nn.Identity()

  def forward_lstm(self, x):
    x = self.normalizer(x)
    x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
    self.hidden_state[:] = h
    self.cell_state[:] = c
    x = x.squeeze(0)
    return self.actor(x)

  def forward_gru(self, x):
    x = self.normalizer(x)
    x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
    self.hidden_state[:] = h
    x = x.squeeze(0)
    return self.actor(x)

  def forward(self, x):
    return self.actor(self.normalizer(x))

  @torch.jit.export
  def reset(self):
    pass

  def reset_memory(self):
    self.hidden_state[:] = 0.0
    if hasattr(self, "cell_state"):
      self.cell_state[:] = 0.0

  def export(self, path, filename):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, filename)
    self.to("cpu")
    traced_script_module = torch.jit.script(self)
    traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
  def __init__(self, policy, normalizer=None, verbose=False):
    super().__init__()
    self.verbose = verbose
    self.is_recurrent = policy.is_recurrent
    if hasattr(policy, "actor"):
      self.actor = copy.deepcopy(policy.actor)
      if self.is_recurrent:
        self.rnn = copy.deepcopy(policy.memory_a.rnn)
    elif hasattr(policy, "student"):
      self.actor = copy.deepcopy(policy.student)
      if self.is_recurrent:
        self.rnn = copy.deepcopy(policy.memory_s.rnn)
    else:
      raise ValueError("Policy does not have an actor/student module.")

    if self.is_recurrent:
      self.rnn.cpu()
      self.rnn_type = type(self.rnn).__name__.lower()
      if self.rnn_type == "lstm":
        self.forward = self.forward_lstm
      elif self.rnn_type == "gru":
        self.forward = self.forward_gru
      else:
        raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")

    if normalizer:
      self.normalizer = copy.deepcopy(normalizer)
    else:
      self.normalizer = torch.nn.Identity()

  def forward_lstm(self, x_in, h_in, c_in):
    x_in = self.normalizer(x_in)
    x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
    x = x.squeeze(0)
    return self.actor(x), h, c

  def forward_gru(self, x_in, h_in):
    x_in = self.normalizer(x_in)
    x, h = self.rnn(x_in.unsqueeze(0), h_in)
    x = x.squeeze(0)
    return self.actor(x), h

  def forward(self, x):
    return self.actor(self.normalizer(x))

  def export(self, path, filename):
    self.to("cpu")
    self.eval()
    if self.is_recurrent:
      obs = torch.zeros(1, self.rnn.input_size)
      h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)

      if self.rnn_type == "lstm":
        c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
        torch.onnx.export(
          self,
          (obs, h_in, c_in),
          os.path.join(path, filename),
          export_params=True,
          opset_version=11,
          verbose=self.verbose,
          input_names=["obs", "h_in", "c_in"],
          output_names=["actions", "h_out", "c_out"],
          dynamic_axes={},
        )
      elif self.rnn_type == "gru":
        torch.onnx.export(
          self,
          (obs, h_in),
          os.path.join(path, filename),
          export_params=True,
          opset_version=11,
          verbose=self.verbose,
          input_names=["obs", "h_in"],
          output_names=["actions", "h_out"],
          dynamic_axes={},
        )
      else:
        raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
    else:
      obs = torch.zeros(1, self.actor[0].in_features)
      torch.onnx.export(
        self,
        obs,
        os.path.join(path, filename),
        export_params=True,
        opset_version=11,
        verbose=self.verbose,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
      )
