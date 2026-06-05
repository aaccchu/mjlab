import os
import statistics
import time
from collections import deque
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

try:
  import wandb
except ImportError:
  wandb = None
except Exception:
  wandb = None


from amp_tasks.amp_rsl_rl.algorithms import AMPPPO, AMPDiscriminator
from amp_tasks.amp_rsl_rl.isaaclab.amp_wrapper import AMPEnvWrapper
from amp_tasks.amp_rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from amp_tasks.amp_rsl_rl.rl import VecEnv
from amp_tasks.amp_rsl_rl.utils import Normalizer
from amp_tasks.velocity.mdp.motion_dataset import MotionDataset


class AMPOnPolicyRunner:
  def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
    self.cfg = train_cfg
    self.alg_cfg = train_cfg["algorithm"]
    self.policy_cfg = train_cfg["policy"]
    self.amp_data_cfg = train_cfg["amp_data"]
    self.device = device
    self.env: AMPEnvWrapper = env
    if self.env.num_privileged_obs is not None:
      num_critic_obs = self.env.num_privileged_obs
    else:
      num_critic_obs = self.env.num_obs
    actor_critic_registry = {
      "ActorCritic": ActorCritic,
      "ActorCriticRecurrent": ActorCriticRecurrent,
    }
    actor_critic_name = self.policy_cfg["class_name"]
    actor_critic_class = actor_critic_registry.get(actor_critic_name)
    if actor_critic_class is None:
      raise ValueError(f"Unsupported policy class: {actor_critic_name}")
    num_actor_obs = self.env.num_obs
    actor_critic: Any = actor_critic_class(
      num_actor_obs=num_actor_obs,
      num_critic_obs=num_critic_obs,
      num_actions=self.env.num_actions,
      **self.policy_cfg,
    ).to(self.device)

    amp_data = MotionDataset.from_cfg(
      cfg=self.amp_data_cfg, env=env.unwrapped, device=device
    )
    self.env.amp_data = amp_data
    amp_normalizer = Normalizer(amp_data.observation_dim)
    amp_lr_coef = train_cfg.get("amp_lr_coef", 1.0)
    amp_grad_pen_coef = train_cfg.get("amp_grad_pen_coef", 10.0)
    discriminator = AMPDiscriminator(
      amp_data.observation_dim * 2,
      train_cfg["amp_reward_coef"],
      train_cfg["amp_discr_hidden_dims"],
      device,
      train_cfg["amp_task_reward_lerp"],
      amp_lr_coef=amp_lr_coef,
    ).to(self.device)

    alg_registry = {"AMPPPO": AMPPPO}
    alg_name = self.alg_cfg["class_name"]
    alg_class = alg_registry.get(alg_name)
    if alg_class is None:
      raise ValueError(f"Unsupported algorithm class: {alg_name}")
    min_std = torch.tensor(self.cfg["amp_min_normalized_std"], device=self.device) * (
      torch.abs(self.env.dof_pos_limits[0, :, 1] - self.env.dof_pos_limits[0, :, 0])
    )
    self.alg: AMPPPO = alg_class(
      actor_critic,
      discriminator,
      amp_data,
      amp_normalizer,
      device=self.device,
      min_std=min_std,
      amp_grad_pen_coef=amp_grad_pen_coef,
      **self.alg_cfg,
    )
    self.num_steps_per_env = self.cfg["num_steps_per_env"]
    self.save_interval = self.cfg["save_interval"]

    self.alg.init_storage(
      self.env.num_envs,
      self.num_steps_per_env,
      [num_actor_obs],
      [self.env.num_privileged_obs],
      [self.env.num_actions],
    )

    self.log_dir = log_dir
    self.writer = None
    self.wandb_run = None
    self.tot_timesteps = 0
    self.tot_time = 0
    self.current_learning_iteration = 0

    _, _ = self.env.reset()

  def _compute_command_bucket_ids(
    self, command: torch.Tensor, bucket_names: list[str]
  ) -> torch.Tensor:
    axis_threshold = float(self.amp_data_cfg.get("command_axis_threshold", 0.02))
    strict_bucket_check = bool(self.amp_data_cfg.get("strict_bucket_check", False))
    bucket_name_to_id = {name: idx for idx, name in enumerate(bucket_names)}
    default_id = 0

    abs_x = torch.abs(command[:, 0])
    abs_y = torch.abs(command[:, 1])
    abs_z = torch.abs(command[:, 2])
    x_active = abs_x > axis_threshold
    y_active = abs_y > axis_threshold
    z_active = abs_z > axis_threshold

    use_x = x_active & (~y_active) & (~z_active)
    use_y = y_active & (~x_active) & (~z_active)
    use_z = z_active & (~x_active) & (~y_active)
    ambiguous = (x_active.int() + y_active.int() + z_active.int()) != 1

    if strict_bucket_check and torch.any(ambiguous):
      ambiguous_count = int(ambiguous.sum().item())
      example_ids = torch.nonzero(ambiguous, as_tuple=False).squeeze(-1)[:8].tolist()
      raise RuntimeError(
        "Ambiguous command-to-bucket mapping detected under strict_bucket_check=True. "
        f"count={ambiguous_count}, sample_env_ids={example_ids}, threshold={axis_threshold}."
      )

    bucket_ids = torch.full(
      (command.shape[0],), default_id, device=self.device, dtype=torch.long
    )

    if "forward" in bucket_name_to_id:
      bucket_ids = torch.where(
        use_x & (command[:, 0] >= 0.0),
        torch.full_like(bucket_ids, bucket_name_to_id["forward"]),
        bucket_ids,
      )
    if "backward" in bucket_name_to_id:
      bucket_ids = torch.where(
        use_x & (command[:, 0] < 0.0),
        torch.full_like(bucket_ids, bucket_name_to_id["backward"]),
        bucket_ids,
      )
    if "left" in bucket_name_to_id:
      bucket_ids = torch.where(
        use_y & (command[:, 1] >= 0.0),
        torch.full_like(bucket_ids, bucket_name_to_id["left"]),
        bucket_ids,
      )
    if "right" in bucket_name_to_id:
      bucket_ids = torch.where(
        use_y & (command[:, 1] < 0.0),
        torch.full_like(bucket_ids, bucket_name_to_id["right"]),
        bucket_ids,
      )
    if "left_turn" in bucket_name_to_id:
      bucket_ids = torch.where(
        use_z & (command[:, 2] > 0.0),
        torch.full_like(bucket_ids, bucket_name_to_id["left_turn"]),
        bucket_ids,
      )
    if "right_turn" in bucket_name_to_id:
      bucket_ids = torch.where(
        use_z & (command[:, 2] < 0.0),
        torch.full_like(bucket_ids, bucket_name_to_id["right_turn"]),
        bucket_ids,
      )

    return bucket_ids

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    if self.log_dir is not None and self.writer is None:
      self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

    logger_type = self.cfg.get("logger", "tensorboard")
    if (
      self.log_dir is not None
      and self.wandb_run is None
      and logger_type == "wandb"
      and wandb is not None
    ):
      self.wandb_run = wandb.init(
        project=self.cfg.get("wandb_project", "isaaclab"),
        name=os.path.basename(self.log_dir),
        dir=self.log_dir,
        config=self.cfg,
      )
    elif logger_type == "wandb" and wandb is None:
      print(
        "[WARN] logger=wandb but wandb package is not installed. Falling back to tensorboard only."
      )

    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations()
    privileged_obs = self.env.get_privileged_observations()
    amp_obs = self.env.get_amp_observations()
    critic_obs = privileged_obs if privileged_obs is not None else obs
    obs, critic_obs, amp_obs = (
      obs.to(self.device),
      critic_obs.to(self.device),
      amp_obs.to(self.device),
    )

    self.alg.actor_critic.train()
    self.alg.discriminator.train()

    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    amp_rew_buffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(
      self.env.num_envs, dtype=torch.float, device=self.device
    )
    cur_episode_length = torch.zeros(
      self.env.num_envs, dtype=torch.float, device=self.device
    )

    last_amp_reward_mean = 0.0
    last_task_reward_mean = 0.0

    tot_iter = self.current_learning_iteration + num_learning_iterations
    for it in range(self.current_learning_iteration, tot_iter):
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.num_steps_per_env):
          actions = self.alg.act(obs, critic_obs, amp_obs)
          (
            obs,
            privileged_obs,
            rewards,
            dones,
            infos,
            reset_env_ids,
            terminal_amp_states,
          ) = self.env.step(actions, not_amp=False)
          next_amp_obs = self.env.get_amp_observations()

          critic_obs = privileged_obs if privileged_obs is not None else obs
          obs, critic_obs, next_amp_obs, rewards, dones = (
            obs.to(self.device),
            critic_obs.to(self.device),
            next_amp_obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )

          next_amp_obs_with_term = torch.clone(next_amp_obs)
          next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

          task_rewards = rewards.clone()
          amp_rewards, _ = self.alg.discriminator.predict_amp_reward(
            amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer
          )
          rewards = amp_rewards

          last_task_reward_mean = task_rewards.mean().item()
          last_amp_reward_mean = amp_rewards.mean().item()
          amp_rew_buffer.append(last_amp_reward_mean)
          amp_obs = torch.clone(next_amp_obs)

          bucket_ids = None
          if self.env.amp_data.use_command_conditioned_sampling:
            command_name = self.amp_data_cfg.get("command_name", "base_velocity")
            command = self.env.unwrapped.command_manager.get_command(command_name).to(
              self.device
            )
            bucket_names = self.amp_data_cfg.get(
              "bucket_names", ["forward", "backward", "left", "right"]
            )
            bucket_ids = self._compute_command_bucket_ids(command, bucket_names)

          self.alg.process_env_step(
            rewards, dones, infos, next_amp_obs_with_term, bucket_ids=bucket_ids
          )

          if self.log_dir is not None:
            if "episode" in infos:
              ep_infos.append(infos["episode"])
            if "log" in infos:
              ep_infos.append(infos["log"])
            cur_reward_sum += rewards
            cur_episode_length += 1
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

        stop = time.time()
        collection_time = stop - start
        start = stop
        self.alg.compute_returns(critic_obs)

      (
        mean_value_loss,
        mean_surrogate_loss,
        mean_amp_loss,
        mean_grad_pen_loss,
        mean_policy_pred,
        mean_expert_pred,
      ) = self.alg.update()
      stop = time.time()
      learn_time = stop - start
      if self.log_dir is not None:
        self.log(locals())
      if it % self.save_interval == 0:
        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
      ep_infos.clear()

    self.current_learning_iteration += num_learning_iterations
    self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
    if self.wandb_run is not None:
      self.wandb_run.finish()
      self.wandb_run = None

  def log(self, locs, width=80, pad=35):
    self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
    self.tot_time += locs["collection_time"] + locs["learn_time"]
    iteration_time = locs["collection_time"] + locs["learn_time"]

    ep_string = ""
    episode_metrics = {}
    if locs["ep_infos"]:
      for key in locs["ep_infos"][0]:
        infotensor = torch.tensor([], device=self.device)
        for ep_info in locs["ep_infos"]:
          if not isinstance(ep_info[key], torch.Tensor):
            ep_info[key] = torch.Tensor([ep_info[key]])
          if len(ep_info[key].shape) == 0:
            ep_info[key] = ep_info[key].unsqueeze(0)
          infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
        value = torch.mean(infotensor)

        if key.startswith("Episode_Reward/"):
          metric_tag = "Reward/" + key[len("Episode_Reward/") :]
        elif key.startswith("Metrics/"):
          metric_tag = "Metrics/" + key[len("Metrics/") :]
        elif key.startswith("Episode_Termination/"):
          metric_tag = "Termination/" + key[len("Episode_Termination/") :]
        else:
          metric_tag = "Episode/" + key

        self.writer.add_scalar(metric_tag, value, locs["it"])
        episode_metrics[metric_tag] = value.item()
        ep_string += f"{f'Mean episode {key}:':>{pad}} {value:.4f}\n"
    mean_std = self.alg.actor_critic.std.mean()
    fps = int(
      self.num_steps_per_env
      * self.env.num_envs
      / (locs["collection_time"] + locs["learn_time"])
    )

    self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
    self.writer.add_scalar(
      "Reward/amp_reward_mean_step", locs["last_amp_reward_mean"], locs["it"]
    )
    self.writer.add_scalar(
      "Reward/task_reward_mean_step", locs["last_task_reward_mean"], locs["it"]
    )
    if len(locs["amp_rew_buffer"]) > 0:
      self.writer.add_scalar(
        "Reward/amp_reward_mean_window",
        statistics.mean(locs["amp_rew_buffer"]),
        locs["it"],
      )
    self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
    self.writer.add_scalar("Loss/AMP", locs["mean_amp_loss"], locs["it"])
    self.writer.add_scalar("Loss/AMP_grad", locs["mean_grad_pen_loss"], locs["it"])
    self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
    self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
    self.writer.add_scalar(
      "Policy/AMP_mean_policy_pred", locs["mean_policy_pred"], locs["it"]
    )
    self.writer.add_scalar(
      "Policy/AMP_mean_expert_pred", locs["mean_expert_pred"], locs["it"]
    )
    self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
    self.writer.add_scalar("Perf/timesteps", self.tot_timesteps, locs["it"])
    self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
    self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
    if len(locs["rewbuffer"]) > 0:
      self.writer.add_scalar(
        "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
      )
      self.writer.add_scalar(
        "Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"]
      )
      self.writer.add_scalar(
        "Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time
      )
      self.writer.add_scalar(
        "Train/mean_episode_length/time",
        statistics.mean(locs["lenbuffer"]),
        self.tot_time,
      )

    if self.wandb_run is not None:
      wandb_metrics = {
        "Loss/value_function": locs["mean_value_loss"],
        "Loss/surrogate": locs["mean_surrogate_loss"],
        "Loss/AMP": locs["mean_amp_loss"],
        "Loss/AMP_grad": locs["mean_grad_pen_loss"],
        "Loss/learning_rate": self.alg.learning_rate,
        "Policy/mean_noise_std": mean_std.item(),
        "Policy/AMP_mean_policy_pred": locs["mean_policy_pred"],
        "Policy/AMP_mean_expert_pred": locs["mean_expert_pred"],
        "Reward/amp_reward_mean_step": locs["last_amp_reward_mean"],
        "Reward/task_reward_mean_step": locs["last_task_reward_mean"],
        "Perf/total_fps": fps,
        "Perf/collection time": locs["collection_time"],
        "Perf/learning_time": locs["learn_time"],
        "global_step": locs["it"],
        "timesteps": self.tot_timesteps,
      }
      if len(locs["rewbuffer"]) > 0:
        wandb_metrics["Train/mean_reward"] = statistics.mean(locs["rewbuffer"])
        wandb_metrics["Train/mean_episode_length"] = statistics.mean(locs["lenbuffer"])
      if len(locs["amp_rew_buffer"]) > 0:
        wandb_metrics["Reward/amp_reward_mean_window"] = statistics.mean(
          locs["amp_rew_buffer"]
        )
      if len(episode_metrics) > 0:
        wandb_metrics.update(episode_metrics)
      self.wandb_run.log(wandb_metrics, step=locs["it"])

    header = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

    if len(locs["rewbuffer"]) > 0:
      log_string = (
        f"{'#' * width}\n"
        f"{header.center(width, ' ')}\n\n"
        f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
        f"{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"
        f"{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"
        f"{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"
        f"{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"
        f"{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"
        f"{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"
        f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
        f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
        f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
      )
    else:
      log_string = (
        f"{'#' * width}\n"
        f"{header.center(width, ' ')}\n\n"
        f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
        f"{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"
        f"{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"
        f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
      )

    log_string += ep_string
    log_string += (
      f"{'-' * width}\n"
      f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
      f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
      f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
      f"{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"
    )
    print(log_string)

  def save(self, path, infos=None):
    torch.save(
      {
        "model_state_dict": self.alg.actor_critic.state_dict(),
        "optimizer_state_dict": self.alg.optimizer.state_dict(),
        "discriminator_state_dict": self.alg.discriminator.state_dict(),
        "amp_normalizer": self.alg.amp_normalizer,
        "iter": self.current_learning_iteration,
        "infos": infos,
      },
      path,
    )

  def load(self, path, load_optimizer=True):
    loaded_dict = torch.load(path, weights_only=False)
    self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
    self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
    self.alg.amp_normalizer = loaded_dict["amp_normalizer"]
    if load_optimizer:
      self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
    return loaded_dict["infos"]

  def get_inference_policy(self, device=None):
    self.alg.actor_critic.eval()
    if device is not None:
      self.alg.actor_critic.to(device)
    return self.alg.actor_critic.act_inference
