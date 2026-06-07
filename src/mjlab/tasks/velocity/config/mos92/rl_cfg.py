"""RL configuration for MOS92 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

# Spatial-softmax CNN for the head depth camera. Mirrors the manipulation
# vision tasks; output_channels [16,32] + spatial_softmax => 64-d latent.
_VISION_CNN_CFG = {
  "output_channels": [16, 32],
  "kernel_size": [5, 3],
  "stride": [2, 2],
  "padding": "zeros",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "none",
  "spatial_softmax": True,
  "spatial_softmax_temperature": 1.0,
}
_VISION_MODEL_CLS = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"

# RGB self-localization CNN (v3g Phase B/C). Same spatial-softmax shape as the
# depth branch (=> 64-d latent), but a SEPARATE branch so the validated
# depth-ball CNN is untouched: depth stays dedicated to the ball, RGB carries
# self-localization from the painted field lines/goal. The model builds one CNN
# per 2-D image obs group; passing cnn_cfg as a dict-of-dicts keyed by group
# name gives each branch its own config.
_RGB_CNN_CFG = dict(_VISION_CNN_CFG)


def mos92_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="mos92_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def mos92_vision_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Gaze-warmup vision runner: actor sees GT 1D obs + head depth via CNN.

  Actor keeps the same MLP trunk (512,256,128) as A2 so the bootstrap loads
  cleanly; the CNN latent is appended to the 1D obs (handled by the CNN model).
  The critic stays a pure MLP on GT-only obs (asymmetric, clean bootstrap).
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=_VISION_CNN_CFG,
      class_name=_VISION_MODEL_CLS,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="mos92_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=3_000,
    obs_groups={
      "actor": ("actor", "camera"),
      "critic": ("critic",),
    },
  )


def mos92_selfloc_vision_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """v3g Phase B/C: depth-ball CNN + a SECOND RGB CNN for self-localization.

  Adds a "camera_rgb" image obs group to the actor. The spatial-softmax model
  builds one CNN per 2-D group, so cnn_cfg is a dict-of-dicts keyed by group
  name: "camera" (depth, 1ch) keeps the validated ball CNN, "camera_rgb"
  (rgb, 3ch) is a fresh branch whose 64-d latent appends to the actor obs. Both
  latents concatenate onto the 1-D obs before the MLP trunk. Critic stays a pure
  MLP on GT-only obs (asymmetric) — unchanged from the rebalanced bootstrap.
  """
  cfg = mos92_vision_ppo_runner_cfg()
  assert cfg.actor.cnn_cfg is not None
  cfg.actor.cnn_cfg = {"camera": _VISION_CNN_CFG, "camera_rgb": _RGB_CNN_CFG}
  cfg.obs_groups = {
    "actor": ("actor", "camera", "camera_rgb"),
    "critic": ("critic",),
  }
  return cfg
