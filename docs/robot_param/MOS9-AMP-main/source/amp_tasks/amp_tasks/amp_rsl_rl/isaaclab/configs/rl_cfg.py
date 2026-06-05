from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class RslRlPpoActorCriticCfg:
  class_name: str = "ActorCritic"
  init_noise_std: float = MISSING
  noise_std_type: Literal["scalar", "log"] = "scalar"
  actor_hidden_dims: list[int] = MISSING
  critic_hidden_dims: list[int] = MISSING
  activation: str = MISSING


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
  class_name: str = "ActorCriticRecurrent"
  rnn_type: str = MISSING
  rnn_hidden_dim: int = MISSING
  rnn_num_layers: int = MISSING


@configclass
class RslRlPpoAlgorithmCfg:
  class_name: str = "PPO"
  num_learning_epochs: int = MISSING
  num_mini_batches: int = MISSING
  learning_rate: float = MISSING
  schedule: str = MISSING
  gamma: float = MISSING
  lam: float = MISSING
  entropy_coef: float = MISSING
  desired_kl: float = MISSING
  max_grad_norm: float = MISSING
  value_loss_coef: float = MISSING
  use_clipped_value_loss: bool = MISSING
  clip_param: float = MISSING
  normalize_advantage_per_mini_batch: bool = False


@configclass
class RslRlOnPolicyRunnerCfg:
  seed: int = 42
  device: str = "cuda:0"
  num_steps_per_env: int = MISSING
  max_iterations: int = MISSING
  empirical_normalization: bool = MISSING
  policy: RslRlPpoActorCriticCfg = MISSING
  algorithm: RslRlPpoAlgorithmCfg = MISSING
  clip_actions: float | None = None
  save_interval: int = MISSING
  experiment_name: str = MISSING
  run_name: str = ""
  logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
  neptune_project: str = "isaaclab"
  wandb_project: str = "isaaclab"
  resume: bool = False
  load_run: str = ".*"
  load_checkpoint: str = "model_.*.pt"
