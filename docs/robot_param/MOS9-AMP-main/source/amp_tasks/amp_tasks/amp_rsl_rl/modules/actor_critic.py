import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
  is_recurrent = False

  def __init__(
    self,
    num_actor_obs,
    num_critic_obs,
    num_actions,
    actor_hidden_dims=[256, 256, 256],
    critic_hidden_dims=[256, 256, 256],
    activation="elu",
    init_noise_std=1.0,
    fixed_std=False,
    **kwargs,
  ):
    if kwargs:
      print(
        "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
        + str(list(kwargs.keys()))
      )
    super().__init__()

    activation = get_activation(activation)

    actor_layers = [nn.Linear(num_actor_obs, actor_hidden_dims[0]), activation]
    for idx in range(len(actor_hidden_dims)):
      if idx == len(actor_hidden_dims) - 1:
        actor_layers.append(nn.Linear(actor_hidden_dims[idx], num_actions))
      else:
        actor_layers.append(
          nn.Linear(actor_hidden_dims[idx], actor_hidden_dims[idx + 1])
        )
        actor_layers.append(activation)
    self.actor = nn.Sequential(*actor_layers)

    critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), activation]
    for idx in range(len(critic_hidden_dims)):
      if idx == len(critic_hidden_dims) - 1:
        critic_layers.append(nn.Linear(critic_hidden_dims[idx], 1))
      else:
        critic_layers.append(
          nn.Linear(critic_hidden_dims[idx], critic_hidden_dims[idx + 1])
        )
        critic_layers.append(activation)
    self.critic = nn.Sequential(*critic_layers)

    print(f"Actor MLP: {self.actor}")
    print(f"Critic MLP: {self.critic}")

    self.fixed_std = fixed_std
    std = init_noise_std * torch.ones(num_actions)
    self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
    self.distribution = None
    Normal.set_default_validate_args = False

  @staticmethod
  def init_weights(sequential, scales):
    [
      torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
      for idx, module in enumerate(
        mod for mod in sequential if isinstance(mod, nn.Linear)
      )
    ]

  def reset(self, dones=None):
    pass

  def forward(self):
    raise NotImplementedError

  @property
  def action_mean(self):
    return self.distribution.mean

  @property
  def action_std(self):
    return self.distribution.stddev

  @property
  def entropy(self):
    return self.distribution.entropy().sum(dim=-1)

  def update_distribution(self, observations):
    mean = self.actor(observations)
    std = self.std.to(mean.device)
    self.distribution = Normal(mean, mean * 0.0 + std)

  def act(self, observations, **kwargs):
    self.update_distribution(observations)
    return self.distribution.sample()

  def get_actions_log_prob(self, actions):
    return self.distribution.log_prob(actions).sum(dim=-1)

  def act_inference(self, observations):
    return self.actor(observations)

  def evaluate(self, critic_observations, **kwargs):
    return self.critic(critic_observations)


def get_activation(act_name):
  if act_name == "elu":
    return nn.ELU()
  if act_name == "selu":
    return nn.SELU()
  if act_name == "relu":
    return nn.ReLU()
  if act_name == "crelu":
    return nn.ReLU()
  if act_name == "lrelu":
    return nn.LeakyReLU()
  if act_name == "tanh":
    return nn.Tanh()
  if act_name == "sigmoid":
    return nn.Sigmoid()
  print("invalid activation function!")
  return None
