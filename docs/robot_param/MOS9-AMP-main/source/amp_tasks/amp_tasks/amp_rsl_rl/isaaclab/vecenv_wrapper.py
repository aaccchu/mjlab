import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

from amp_tasks.amp_rsl_rl.rl import VecEnv


class RslRlVecEnvWrapper(VecEnv):
  def __init__(
    self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None
  ):
    if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(
      env.unwrapped, DirectRLEnv
    ):
      raise ValueError(
        "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
        f" {type(env)}"
      )

    self.env = env
    self.clip_actions = clip_actions
    self.num_envs = self.unwrapped.num_envs
    self.device = self.unwrapped.device
    self.max_episode_length = self.unwrapped.max_episode_length

    if hasattr(self.unwrapped, "action_manager"):
      self.num_actions = self.unwrapped.action_manager.total_action_dim
    else:
      self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

    if hasattr(self.unwrapped, "observation_manager"):
      self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
    else:
      self.num_obs = gym.spaces.flatdim(
        self.unwrapped.single_observation_space["policy"]
      )

    if (
      hasattr(self.unwrapped, "observation_manager")
      and "critic" in self.unwrapped.observation_manager.group_obs_dim
    ):
      self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim[
        "critic"
      ][0]
    elif (
      hasattr(self.unwrapped, "num_states")
      and "critic" in self.unwrapped.single_observation_space
    ):
      self.num_privileged_obs = gym.spaces.flatdim(
        self.unwrapped.single_observation_space["critic"]
      )
    else:
      self.num_privileged_obs = 0

    self._modify_action_space()
    self.env.reset()

  def __str__(self):
    return f"<{type(self).__name__}{self.env}>"

  def __repr__(self):
    return str(self)

  @property
  def cfg(self) -> object:
    return self.unwrapped.cfg

  @property
  def render_mode(self) -> str | None:
    return self.env.render_mode

  @property
  def observation_space(self) -> gym.Space:
    return self.env.observation_space

  @property
  def action_space(self) -> gym.Space:
    return self.env.action_space

  @classmethod
  def class_name(cls) -> str:
    return cls.__name__

  @property
  def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
    return self.env.unwrapped

  def get_observations(self) -> tuple[torch.Tensor, dict]:
    if hasattr(self.unwrapped, "observation_manager"):
      obs_dict = self.unwrapped.observation_manager.compute()
    else:
      obs_dict = self.unwrapped._get_observations()
    return obs_dict["policy"], {"observations": obs_dict}

  def get_privileged_observations(self) -> tuple[torch.Tensor, None]:
    if hasattr(self.unwrapped, "observation_manager"):
      obs_dict = self.unwrapped.observation_manager.compute()
    else:
      obs_dict = self.unwrapped._get_observations()
    return obs_dict["critic"]

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return self.unwrapped.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor):
    self.unwrapped.episode_length_buf = value

  def seed(self, seed: int = -1) -> int:
    return self.unwrapped.seed(seed)

  def reset(self) -> tuple[torch.Tensor, dict]:
    obs_dict, _ = self.env.reset()
    return obs_dict["policy"], {"observations": obs_dict}

  def step(
    self, actions: torch.Tensor, *, use_old=False, **kwargs
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

    obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
    dones = (terminated | truncated).to(dtype=torch.long)
    obs = obs_dict["policy"]
    extras["observations"] = obs_dict

    if not self.unwrapped.cfg.is_finite_horizon:
      extras["time_outs"] = truncated

    return obs, rew, dones, extras

  def close(self):
    return self.env.close()

  def _modify_action_space(self):
    if self.clip_actions is None:
      return

    self.env.unwrapped.single_action_space = gym.spaces.Box(
      low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
    )
    self.env.unwrapped.action_space = gym.vector.utils.batch_space(
      self.env.unwrapped.single_action_space, self.num_envs
    )
