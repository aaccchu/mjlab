import torch

from .vecenv_wrapper import RslRlVecEnvWrapper


class AMPEnvWrapper(RslRlVecEnvWrapper):
  def __init__(self, env, clip_actions=None):
    super().__init__(env, clip_actions)
    self.rewards_shape = self.unwrapped.reward_manager._step_reward.shape[-1]

  def get_observations(self) -> tuple[torch.Tensor, dict]:
    if hasattr(self.unwrapped, "observation_manager"):
      obs_dict = self.unwrapped.observation_manager.compute()
    else:
      obs_dict = self.unwrapped._get_observations()
    return obs_dict["policy"]

  def get_amp_observations(self) -> tuple[torch.Tensor, dict]:
    if hasattr(self.unwrapped, "observation_manager"):
      obs_dict = self.unwrapped.observation_manager.compute()
    else:
      obs_dict = self.unwrapped._get_observations()
    return obs_dict["amp"]

  def step(self, actions, *, not_amp=True, **kwargs):
    if not_amp:
      return super().step(actions, **kwargs)

    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

    obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
    dones = (terminated | truncated).to(dtype=torch.long)
    obs = obs_dict["policy"]
    privileged_obs = obs_dict.get("critic", obs)
    terminal_amp_states = obs_dict.get("amp", obs)
    extras["observations"] = obs_dict
    reset_env_ids = torch.where(dones)[0]

    enable_rsi = getattr(self.unwrapped.cfg, "enable_rsi", False)
    if enable_rsi and len(reset_env_ids) > 0 and hasattr(self, "amp_data"):
      rsi_states = self.amp_data.get_random_states(len(reset_env_ids))
      robot = self.unwrapped.scene["robot"]

      root_states = robot.data.default_root_state[reset_env_ids].clone()
      root_states[:, :3] = rsi_states["root_pos_w"]
      root_states[:, :2] += self.unwrapped.scene.env_origins[reset_env_ids, :2]
      root_states[:, 3:7] = rsi_states["root_quat_w"]
      root_states[:, 7:10] = rsi_states["root_lin_vel_w"]
      root_states[:, 10:13] = rsi_states["root_ang_vel_w"]
      robot.write_root_state_to_sim(root_states, env_ids=reset_env_ids)

      joint_pos = robot.data.default_joint_pos[reset_env_ids].clone()
      joint_vel = robot.data.default_joint_vel[reset_env_ids].clone()

      if rsi_states["joint_pos"].shape[-1] == joint_pos.shape[-1]:
        joint_pos = rsi_states["joint_pos"]
        joint_vel = rsi_states["joint_vel"]
      else:
        min_dim = min(rsi_states["joint_pos"].shape[-1], joint_pos.shape[-1])
        joint_pos[:, :min_dim] = rsi_states["joint_pos"][:, :min_dim]
        joint_vel[:, :min_dim] = rsi_states["joint_vel"][:, :min_dim]

      robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=reset_env_ids)

      base_env = self.unwrapped
      if hasattr(base_env, "observation_manager"):
        obs_dict = base_env.observation_manager.compute()
      else:
        obs_dict = base_env._get_observations()

      obs = obs_dict["policy"]
      privileged_obs = obs_dict.get("critic", obs)
      terminal_amp_states = obs_dict.get("amp", obs)
      extras["observations"] = obs_dict

    if not self.unwrapped.cfg.is_finite_horizon:
      extras["time_outs"] = truncated

    return (
      obs,
      privileged_obs,
      rew,
      dones,
      extras,
      reset_env_ids,
      terminal_amp_states[reset_env_ids],
    )

  @property
  def dof_pos_limits(self) -> torch.Tensor:
    return self.unwrapped.scene["robot"].data.joint_pos_limits
