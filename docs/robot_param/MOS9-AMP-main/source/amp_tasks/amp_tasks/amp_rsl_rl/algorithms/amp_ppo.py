import torch
import torch.nn as nn
import torch.optim as optim

from amp_tasks.amp_rsl_rl.modules import ActorCritic
from amp_tasks.amp_rsl_rl.storage import ReplayBuffer, RolloutStorage
from amp_tasks.velocity.mdp.motion_dataset import MotionDataset

from .amp_discriminator import AMPDiscriminator


class AMPPPO:
  actor_critic: ActorCritic

  def __init__(
    self,
    actor_critic: ActorCritic,
    discriminator: AMPDiscriminator,
    amp_data: MotionDataset,
    amp_normalizer,
    num_learning_epochs=1,
    num_mini_batches=1,
    clip_param=0.2,
    gamma=0.998,
    lam=0.95,
    value_loss_coef=1.0,
    entropy_coef=0.0,
    learning_rate=1e-3,
    max_grad_norm=1.0,
    use_clipped_value_loss=True,
    schedule="fixed",
    desired_kl=0.01,
    device="cpu",
    amp_replay_buffer_size=100000,
    min_std=None,
    amp_grad_pen_coef=10.0,
    **kwargs,
  ):
    self.device = device

    self.desired_kl = desired_kl
    self.schedule = schedule
    self.learning_rate = learning_rate
    self.min_std = min_std
    self.amp_grad_pen_coef = amp_grad_pen_coef

    self.discriminator = discriminator
    self.discriminator.to(self.device)
    self.amp_transition = RolloutStorage.Transition()
    self.amp_storage = ReplayBuffer(
      discriminator.input_dim // 2, amp_replay_buffer_size, device
    )
    self.amp_data: MotionDataset = amp_data
    self.amp_normalizer = amp_normalizer

    self.actor_critic = actor_critic
    self.actor_critic.to(self.device)
    self.storage = None

    self.amp_lr_coef = getattr(self.discriminator, "amp_lr_coef", 1.0)
    self.amp_lr_coef = min(max(float(self.amp_lr_coef), 0.0), 1.0)
    self.discriminator_lr = learning_rate * self.amp_lr_coef

    params = [
      {
        "params": self.actor_critic.parameters(),
        "name": "actor_critic",
        "lr": learning_rate,
      },
      {
        "params": self.discriminator.trunk.parameters(),
        "weight_decay": 10e-4,
        "name": "amp_trunk",
        "lr": self.discriminator_lr,
      },
      {
        "params": self.discriminator.amp_linear.parameters(),
        "weight_decay": 10e-2,
        "name": "amp_head",
        "lr": self.discriminator_lr,
      },
    ]
    self.optimizer = optim.Adam(params, lr=learning_rate)
    self.transition = RolloutStorage.Transition()

    self.clip_param = clip_param
    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef
    self.gamma = gamma
    self.lam = lam
    self.max_grad_norm = max_grad_norm
    self.use_clipped_value_loss = use_clipped_value_loss

  def init_storage(
    self,
    num_envs,
    num_transitions_per_env,
    actor_obs_shape,
    critic_obs_shape,
    action_shape,
  ):
    self.storage = RolloutStorage(
      num_envs,
      num_transitions_per_env,
      actor_obs_shape,
      critic_obs_shape,
      action_shape,
      self.device,
    )

  def test_mode(self):
    self.actor_critic.test()

  def train_mode(self):
    self.actor_critic.train()

  def act(self, obs, critic_obs, amp_obs):
    if self.actor_critic.is_recurrent:
      self.transition.hidden_states = self.actor_critic.get_hidden_states()
    aug_obs, aug_critic_obs = obs.detach(), critic_obs.detach()
    self.transition.actions = self.actor_critic.act(aug_obs).detach()
    self.transition.values = self.actor_critic.evaluate(aug_critic_obs).detach()
    self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
      self.transition.actions
    ).detach()
    self.transition.action_mean = self.actor_critic.action_mean.detach()
    self.transition.action_sigma = self.actor_critic.action_std.detach()
    self.transition.observations = obs
    self.transition.critic_observations = critic_obs
    self.amp_transition.observations = amp_obs
    return self.transition.actions

  def process_env_step(self, rewards, dones, infos, amp_obs, bucket_ids=None):
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones
    if "time_outs" in infos:
      self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
      )

    self.amp_storage.insert(
      self.amp_transition.observations, amp_obs, bucket_ids=bucket_ids
    )
    self.storage.add_transitions(self.transition)
    self.transition.clear()
    self.amp_transition.clear()
    self.actor_critic.reset(dones)

  def compute_returns(self, last_critic_obs):
    aug_last_critic_obs = last_critic_obs.detach()
    last_values = self.actor_critic.evaluate(aug_last_critic_obs).detach()
    self.storage.compute_returns(last_values, self.gamma, self.lam)

  def update(self):
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_amp_loss = 0
    mean_grad_pen_loss = 0
    mean_policy_pred = 0
    mean_expert_pred = 0
    if self.actor_critic.is_recurrent:
      generator = self.storage.reccurent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    amp_policy_generator = self.amp_storage.feed_forward_generator(
      self.num_learning_epochs * self.num_mini_batches,
      self.storage.num_envs
      * self.storage.num_transitions_per_env
      // self.num_mini_batches,
    )
    expert_mini_batch_size = (
      self.storage.num_envs
      * self.storage.num_transitions_per_env
      // self.num_mini_batches
    )

    if not self.amp_data.use_command_conditioned_sampling:
      amp_expert_generator = self.amp_data.feed_forward_generator(
        self.num_learning_epochs * self.num_mini_batches,
        expert_mini_batch_size,
      )
      iter_zip = zip(generator, amp_policy_generator, amp_expert_generator)
    else:
      iter_zip = zip(generator, amp_policy_generator)

    for pack in iter_zip:
      if not self.amp_data.use_command_conditioned_sampling:
        sample, sample_amp_policy, sample_amp_expert = pack
      else:
        sample, sample_amp_policy = pack
      (
        obs_batch,
        critic_obs_batch,
        actions_batch,
        target_values_batch,
        advantages_batch,
        returns_batch,
        old_actions_log_prob_batch,
        old_mu_batch,
        old_sigma_batch,
        hid_states_batch,
        masks_batch,
      ) = sample
      aug_obs_batch = obs_batch.detach()

      self.actor_critic.act(
        aug_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
      )
      actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
      aug_critic_obs_batch = critic_obs_batch.detach()
      value_batch = self.actor_critic.evaluate(
        aug_critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
      )
      mu_batch = self.actor_critic.action_mean
      sigma_batch = self.actor_critic.action_std
      entropy_batch = self.actor_critic.entropy

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = torch.sum(
            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
            / (2.0 * torch.square(sigma_batch))
            - 0.5,
            axis=-1,
          )
          kl_mean = torch.mean(kl)

          if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
          elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

          for param_group in self.optimizer.param_groups:
            if param_group.get("name") == "actor_critic":
              param_group["lr"] = self.learning_rate
            else:
              param_group["lr"] = self.learning_rate * self.amp_lr_coef

      ratio = torch.exp(
        actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
      )
      surrogate = -torch.squeeze(advantages_batch) * ratio
      surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (returns_batch - value_batch).pow(2).mean()

      policy_state, policy_next_state, policy_bucket_ids = sample_amp_policy
      if self.amp_data.use_command_conditioned_sampling:
        sample_amp_expert = self.amp_data.sample_batch_by_bucket_ids(policy_bucket_ids)
      expert_state, expert_next_state = sample_amp_expert
      if self.amp_normalizer is not None:
        with torch.no_grad():
          policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
          policy_next_state = self.amp_normalizer.normalize_torch(
            policy_next_state, self.device
          )
          expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
          expert_next_state = self.amp_normalizer.normalize_torch(
            expert_next_state, self.device
          )
      policy_d = self.discriminator(
        torch.cat([policy_state, policy_next_state], dim=-1)
      )
      expert_d = self.discriminator(
        torch.cat([expert_state, expert_next_state], dim=-1)
      )
      expert_loss = torch.nn.MSELoss()(
        expert_d, torch.ones(expert_d.size(), device=self.device)
      )
      policy_loss = torch.nn.MSELoss()(
        policy_d, -1 * torch.ones(policy_d.size(), device=self.device)
      )
      amp_loss = 0.5 * (expert_loss + policy_loss)
      grad_pen_loss = self.discriminator.compute_grad_pen(
        *sample_amp_expert, lambda_=self.amp_grad_pen_coef
      )

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy_batch.mean()
        + amp_loss
        + grad_pen_loss
      )

      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
      self.optimizer.step()

      if not self.actor_critic.fixed_std and self.min_std is not None:
        self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

      if self.amp_normalizer is not None:
        self.amp_normalizer.update(policy_state.cpu().numpy())
        self.amp_normalizer.update(expert_state.cpu().numpy())

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_amp_loss += amp_loss.item()
      mean_grad_pen_loss += grad_pen_loss.item()
      mean_policy_pred += policy_d.mean().item()
      mean_expert_pred += expert_d.mean().item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_amp_loss /= num_updates
    mean_grad_pen_loss /= num_updates
    mean_policy_pred /= num_updates
    mean_expert_pred /= num_updates
    self.storage.clear()

    return (
      mean_value_loss,
      mean_surrogate_loss,
      mean_amp_loss,
      mean_grad_pen_loss,
      mean_policy_pred,
      mean_expert_pred,
    )
