import torch
import torch.nn as nn

from amp_tasks.amp_rsl_rl.utils.trajectory import unpad_trajectories

from .actor_critic import ActorCritic, get_activation


class ActorCriticRecurrent(ActorCritic):
  is_recurrent = True

  def __init__(
    self,
    num_actor_obs,
    num_critic_obs,
    num_actions,
    actor_hidden_dims=[256, 256, 256],
    critic_hidden_dims=[256, 256, 256],
    activation="elu",
    rnn_type="lstm",
    rnn_hidden_size=256,
    rnn_num_layers=1,
    init_noise_std=1.0,
    **kwargs,
  ):
    if kwargs:
      print(
        "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: "
        + str(kwargs.keys())
      )

    super().__init__(
      num_actor_obs=rnn_hidden_size,
      num_critic_obs=rnn_hidden_size,
      num_actions=num_actions,
      actor_hidden_dims=actor_hidden_dims,
      critic_hidden_dims=critic_hidden_dims,
      activation=activation,
      init_noise_std=init_noise_std,
    )

    _ = get_activation(activation)

    self.memory_a = Memory(
      num_actor_obs,
      type=rnn_type,
      num_layers=rnn_num_layers,
      hidden_size=rnn_hidden_size,
    )
    self.memory_c = Memory(
      num_critic_obs,
      type=rnn_type,
      num_layers=rnn_num_layers,
      hidden_size=rnn_hidden_size,
    )

    print(f"Actor RNN: {self.memory_a}")
    print(f"Critic RNN: {self.memory_c}")

  def reset(self, dones=None):
    self.memory_a.reset(dones)
    self.memory_c.reset(dones)

  def act(self, observations, masks=None, hidden_states=None):
    input_actor = self.memory_a(observations, masks, hidden_states)
    return super().act(input_actor.squeeze(0))

  def act_inference(self, observations):
    input_actor = self.memory_a(observations)
    return super().act_inference(input_actor.squeeze(0))

  def evaluate(self, critic_observations, masks=None, hidden_states=None):
    input_critic = self.memory_c(critic_observations, masks, hidden_states)
    return super().evaluate(input_critic.squeeze(0))

  def get_hidden_states(self):
    return self.memory_a.hidden_states, self.memory_c.hidden_states


class Memory(torch.nn.Module):
  def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
    super().__init__()
    rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
    self.rnn = rnn_cls(
      input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
    )
    self.hidden_states = None

  def forward(self, input, masks=None, hidden_states=None):
    batch_mode = masks is not None
    if batch_mode:
      if hidden_states is None:
        raise ValueError(
          "Hidden states not passed to memory module during policy update"
        )
      output, _ = self.rnn(input, hidden_states)
      output = unpad_trajectories(output, masks)
    else:
      output, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
    return output

  def reset(self, dones=None):
    if self.hidden_states is None:
      return
    for hidden_state in self.hidden_states:
      hidden_state[..., dones, :] = 0.0
