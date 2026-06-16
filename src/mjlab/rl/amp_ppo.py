"""AMP (Adversarial Motion Priors) PPO — make the policy move like human motion.

Local PPO subclass that adds an adversarial style reward on top of the full PPO
task objective, so the robot learns a human-like gait (foot lift, longer stride,
slower cadence) WITHOUT abandoning the task. A discriminator is trained to tell
the policy's per-step motion (an "amp" obs group: joint_pos + joint_vel) apart
from a human-motion reference dataset; the policy is rewarded for fooling it.

Why a subclass (mirrors src/mjlab/rl/keypoint_aux_ppo.py): the vendored rsl_rl is
never modified. Selected via ``algorithm.class_name = "mjlab.rl.amp_ppo:AMPPPO"``.

Architecture & risk control:
- The discriminator is a SEPARATE network with its OWN optimizer. It only emits a
  per-step scalar reward added to the task reward; it never shares parameters with
  the actor/critic, so it cannot pollute the control trunk (cf. the v4 perception-
  trunk failure where a shared MLP let perception gradients wreck the gait).
- Style/task competition lives only in the reward. ``task_reward_lerp`` blends them
  (r = (1-lerp)*style + lerp*task); raise it to protect the shot rate.
- AMP scores transition PAIRS (state_t, state_{t+1}). We capture the amp obs at
  ``act`` time as state_t and pair it with the post-step amp obs in
  ``process_env_step`` as state_{t+1}, mirroring the reference AMPPPO. On episode
  resets the pair straddles a reset; with thousands of envs this is negligible
  noise (same approximation the reference implementation makes).

Reference (read-only): docs/robot_param/MOS9-AMP-main/.../amp_rsl_rl/algorithms/
{amp_discriminator.py, amp_ppo.py}.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO

from mjlab.rl.motion_dataset import MotionDataset


class AMPDiscriminator(nn.Module):
  """MLP discriminator over concatenated (state, next_state) AMP observations.

  Outputs a scalar logit d; expert (dataset) is trained toward +1, policy toward
  -1 (least-squares GAN). The style reward is a clamped quadratic around d=1, so
  motion the discriminator believes is "expert-like" earns reward.
  """

  def __init__(
    self,
    input_dim: int,
    hidden_dims: list[int],
    amp_reward_coef: float,
    task_reward_lerp: float,
    device: str,
  ) -> None:
    super().__init__()
    self.device = device
    self.input_dim = input_dim
    self.amp_reward_coef = amp_reward_coef
    self.task_reward_lerp = task_reward_lerp

    layers: list[nn.Module] = []
    curr = input_dim
    for h in hidden_dims:
      layers.append(nn.Linear(curr, h))
      layers.append(nn.ReLU())
      curr = h
    self.trunk = nn.Sequential(*layers).to(device)
    self.head = nn.Linear(hidden_dims[-1], 1).to(device)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.head(self.trunk(x))

  def compute_grad_pen(
    self, expert_state: torch.Tensor, expert_next_state: torch.Tensor, lambda_: float
  ) -> torch.Tensor:
    """Zero-centered gradient penalty on the expert manifold (stabilizes GAN)."""
    expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
    expert_data.requires_grad = True
    disc = self.head(self.trunk(expert_data))
    grad = torch.autograd.grad(
      outputs=disc,
      inputs=expert_data,
      grad_outputs=torch.ones_like(disc),
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    return lambda_ * grad.norm(2, dim=1).pow(2).mean()

  def predict_reward(
    self, state: torch.Tensor, next_state: torch.Tensor, task_reward: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-step style reward (no grad) blended with the task reward via lerp."""
    with torch.no_grad():
      self.eval()
      d = self.head(self.trunk(torch.cat([state, next_state], dim=-1)))
      style = self.amp_reward_coef * torch.clamp(
        1.0 - 0.25 * torch.square(d - 1.0), min=0.0
      )
      style = style.squeeze(-1)
      if self.task_reward_lerp > 0.0:
        reward = (
          1.0 - self.task_reward_lerp
        ) * style + self.task_reward_lerp * task_reward
      else:
        reward = style
      self.train()
    return reward, d.squeeze(-1)


class AMPPPO(PPO):
  """PPO + adversarial-motion-prior style reward (human-like gait)."""

  def __init__(self, *args, amp_cfg: dict | None = None, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    cfg = amp_cfg or {}
    self._amp_group = cfg.get("obs_group", "amp")
    self._amp_obs_terms = tuple(cfg.get("amp_obs_terms", ("joint_pos", "joint_vel")))
    self._amp_grad_pen_coef = float(cfg.get("amp_grad_pen_coef", 10.0))
    self._amp_replay_size = int(cfg.get("replay_buffer_size", 1_000_000))

    # Reference human-motion dataset (single-state dim; the discriminator sees a
    # transition PAIR, hence 2x).
    self.amp_data = MotionDataset.from_cfg(cfg, device=self.device)
    single_dim = self.amp_data.observation_dim

    self.discriminator = AMPDiscriminator(
      input_dim=2 * single_dim,
      hidden_dims=list(cfg.get("discr_hidden_dims", [256, 256])),
      amp_reward_coef=float(cfg.get("amp_reward_coef", 0.2)),
      task_reward_lerp=float(cfg.get("amp_task_reward_lerp", 0.7)),
      device=self.device,
    )
    # Separate optimizer for the discriminator (does NOT touch actor/critic).
    lr = self.learning_rate * float(cfg.get("amp_lr_coef", 0.1))
    self.amp_optimizer = torch.optim.Adam(
      [
        {"params": self.discriminator.trunk.parameters(), "weight_decay": 1e-3},
        {"params": self.discriminator.head.parameters(), "weight_decay": 1e-2},
      ],
      lr=lr,
    )

    # Rolling buffer of policy transition pairs (state_t, state_{t+1}).
    self._amp_buf_state = torch.zeros(
      self._amp_replay_size, single_dim, device=self.device
    )
    self._amp_buf_next = torch.zeros(
      self._amp_replay_size, single_dim, device=self.device
    )
    self._amp_buf_step = 0
    self._amp_buf_filled = 0
    self._amp_single_dim = single_dim
    self._prev_amp_obs: torch.Tensor | None = None
    self._amp_metrics: dict[str, float] = {}

  # --- helpers -------------------------------------------------------------
  def _extract_amp_obs(self, obs) -> torch.Tensor:
    return torch.cat(
      [obs[self._amp_group]]
      if isinstance(self._amp_group, str)
      else [obs[g] for g in self._amp_group],
      dim=-1,
    )

  def _buffer_insert(self, state: torch.Tensor, next_state: torch.Tensor) -> None:
    n = state.shape[0]
    idx = (torch.arange(n, device=self.device) + self._amp_buf_step) % (
      self._amp_replay_size
    )
    self._amp_buf_state[idx] = state.detach()
    self._amp_buf_next[idx] = next_state.detach()
    self._amp_buf_step = int((self._amp_buf_step + n) % self._amp_replay_size)
    self._amp_buf_filled = min(self._amp_replay_size, self._amp_buf_filled + n)

  def _buffer_sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    sel = torch.randint(0, self._amp_buf_filled, (batch_size,), device=self.device)
    return self._amp_buf_state[sel], self._amp_buf_next[sel]

  # --- PPO hooks -----------------------------------------------------------
  def act(self, obs) -> torch.Tensor:
    # Capture the AMP obs BEFORE the env steps — this is state_t of the pair.
    self._cur_amp_obs = self._extract_amp_obs(obs)
    return super().act(obs)

  def process_env_step(self, obs, rewards, dones, extras) -> None:
    # obs here is the POST-step observation -> state_{t+1}.
    next_amp_obs = self._extract_amp_obs(obs)
    state = self._cur_amp_obs
    # Style reward blended with task reward, then hand to PPO's bootstrap/storage.
    style_blended, _ = self.discriminator.predict_reward(state, next_amp_obs, rewards)
    self._buffer_insert(state, next_amp_obs)
    super().process_env_step(obs, style_blended, dones, extras)

  def _train_discriminator(self) -> None:
    """Adversarial pass: push expert->+1, policy->-1, plus gradient penalty.

    Runs over fresh expert samples and the policy replay buffer, BEFORE
    super().update() clears the PPO storage. Uses its OWN optimizer.
    """
    if self._amp_buf_filled < 2:
      return
    num_batches = self.num_learning_epochs * self.num_mini_batches
    batch_size = max(
      1,
      (self.storage.num_envs * self.storage.num_transitions_per_env)
      // self.num_mini_batches,
    )
    mse = nn.MSELoss()
    acc_amp, acc_gp, acc_pol_d, acc_exp_d, n = 0.0, 0.0, 0.0, 0.0, 0
    for _ in range(num_batches):
      pol_s, pol_ns = self._buffer_sample(batch_size)
      exp_s, exp_ns = self.amp_data.sample(batch_size)
      pol_s, pol_ns = pol_s.to(self.device), pol_ns.to(self.device)
      exp_s, exp_ns = exp_s.to(self.device), exp_ns.to(self.device)

      policy_d = self.discriminator(torch.cat([pol_s, pol_ns], dim=-1))
      expert_d = self.discriminator(torch.cat([exp_s, exp_ns], dim=-1))
      expert_loss = mse(expert_d, torch.ones_like(expert_d))
      policy_loss = mse(policy_d, -torch.ones_like(policy_d))
      amp_loss = 0.5 * (expert_loss + policy_loss)
      grad_pen = self.discriminator.compute_grad_pen(
        exp_s, exp_ns, lambda_=self._amp_grad_pen_coef
      )
      self.amp_optimizer.zero_grad()
      (amp_loss + grad_pen).backward()
      nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
      self.amp_optimizer.step()

      acc_amp += amp_loss.item()
      acc_gp += grad_pen.item()
      acc_pol_d += policy_d.mean().item()
      acc_exp_d += expert_d.mean().item()
      n += 1
    if n:
      self._amp_metrics = {
        "amp_disc": acc_amp / n,
        "amp_grad_pen": acc_gp / n,
        "amp_policy_d": acc_pol_d / n,
        "amp_expert_d": acc_exp_d / n,
      }

  def update(self) -> dict[str, float]:
    self._train_discriminator()
    loss_dict = super().update()
    loss_dict.update(self._amp_metrics)
    return loss_dict
