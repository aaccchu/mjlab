"""Task v3g temporal: distill self-localization into an RGB CNN with FRAME-STACKING.

The single-frame Phase B+C run failed: one front-facing RGB frame is ambiguous
for self-loc (center-circle / side lines are left/right symmetric, the goal is a
few distant pixels), so the RGB CNN could not localize and the estimate error
grew as the GT obs faded. Fix: stack the last N RGB frames -> (B, N*3, H, W) so
the CNN sees inter-frame parallax and can disambiguate as the robot moves/turns.

Frame-stacking touches only the observation layer: the wider image tensor flows
through the existing feed-forward RGB CNN (input_channels auto-derived from the
tensor), with ZERO changes to PPO / storage / export. The GT fade is also slower
than the failed run (iters [800, 2500] vs [400, 1200], set in the env cfg) so the
estimate converges with GT present before the crutch is removed.

Bootstrap: from the rebalanced model_1999, same as the single-frame run. The RGB
CNN is fresh either way (now with N*3 input channels); only the actor mlp.0 grows
(+64 RGB latent). Depth CNN, MLP trunk, selfloc head, std_param, the whole critic
carry over.

Usage:
  MUJOCO_GL=egl uv run python scripts/spike_v3g_selfloc_temporal.py          # full
  MUJOCO_GL=egl uv run python scripts/spike_v3g_selfloc_temporal.py --smoke  # 8-iter
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.mos92.env_cfgs import (
  mos92_soccer_e2e_dualcam_geomcurric_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import (
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

# Rebalanced explicit-selfloc run (0.3/-0.3 weights, dribbling recovered).
BASE_CKPT = Path("checkpoints/v3_soccer_solo/01_selfloc_purevision/model_2800.pt")
NUM_ENVS = 384
MAX_ITER = 3800  # > GT-fade end (3500, exp8) so the policy trains on fully-masked
# GT. exp8 bug: MAX_ITER was 2800 < fade-end 3500, so fade never completed (mask
# ~0.2 at the saved ckpt) — the policy never trained pure-vision, and the probe
# (play=True, fade-curriculum gated `if not play`) fed it FULL-scale GT (OOD for a
# fade-trained policy), collapsing the position estimate to field-center (~9m).
# The training-time metric (2.26m, falling THROUGH the fade) is the honest signal.


def _partial_load_shape_match(module, ckpt_sd) -> tuple[int, int, list[str]]:
  """Load every name+shape-matching tensor from the rebalanced ckpt; reinit the
  rest. Only the actor mlp.0 changes (input +64 for the RGB latent); the RGB CNN
  (cnns.camera_rgb.*, now N*3 input channels) is absent from the ckpt so it
  stays fresh. mlp.2/4/6, std_param, the depth CNN and the whole (pure-MLP)
  critic carry over unchanged.

  Returns (loaded, reinit-count, notes)."""
  cur = module.state_dict()
  to_load, notes = {}, []
  reinit = 0
  for k, v in ckpt_sd.items():
    if k not in cur:
      notes.append(f"{k}:absent")
      continue
    if cur[k].shape == v.shape:
      to_load[k] = v
    else:
      reinit += 1
      notes.append(f"{k}:reinit{tuple(v.shape)}->{tuple(cur[k].shape)}")
  module.load_state_dict(to_load, strict=False)
  fresh = [k for k in cur if k not in ckpt_sd]
  if fresh:
    notes.append(f"fresh(not in ckpt): {len(fresh)} tensors e.g. {fresh[:2]}")
  return len(to_load), reinit, notes


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  smoke = "--smoke" in sys.argv
  configure_torch_backends()
  device = "cuda:0"

  env_cfg = mos92_soccer_e2e_dualcam_geomcurric_env_cfg(play=False)
  env_cfg.scene.num_envs = 32 if smoke else NUM_ENVS

  # Pin the ball gt_mask to 0 (pure-vision ball, as in the bootstrap run).
  gt = env_cfg.curriculum["gt_mask"]
  gt.params["start_step"] = -1
  gt.params["end_step"] = 0
  # Anti-cheat penalties already learned upstream; keep them at full.
  pr = env_cfg.curriculum["penalty_ramp"]
  pr.params["start_step"] = -1
  pr.params["end_step"] = 0

  # GT-pose fade. EXP1B failed because the e2e env's fade is [0,1] (GT≈0 from
  # step 0) — fine when bootstrapping model_2800 (already pure-vision) but FATAL
  # here: our RGB branch is FRESH (new 96x72 cam), and a fresh CNN with no GT
  # warmup never learns pure-vision self-loc (v3 exp8 lesson). EXP1B-v2 fix: give
  # the fresh RGB branch a GT warmup [800,2500] iters before pulling the crutch.
  # In smoke, ramp instantly to exercise the masked path.
  selfloc_mask = env_cfg.curriculum["selfloc_gt_mask"]
  if smoke:
    selfloc_mask.params["start_step"] = -1
    selfloc_mask.params["end_step"] = 0
  else:
    selfloc_mask.params["start_step"] = 800 * 24
    selfloc_mask.params["end_step"] = 2500 * 24

  max_iter = 8 if smoke else MAX_ITER
  agent_cfg = mos92_selfloc_vision_ppo_runner_cfg()
  agent_cfg.max_iterations = max_iter
  agent_cfg.run_name = "spike_v4_e2e_geomcurric"
  agent_cfg.save_interval = 100

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_v4_e2e_geomcurric")
  log_dir = log_root / stamp
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)
  print(f"[INFO] Partial-load from rebalanced selfloc: {BASE_CKPT}")
  ckpt = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
  a_load, a_re, a_notes = _partial_load_shape_match(
    runner.alg._raw_actor, ckpt["actor_state_dict"]
  )
  c_load, c_re, c_notes = _partial_load_shape_match(
    runner.alg._raw_critic, ckpt["critic_state_dict"]
  )
  print(f"[INFO] actor : loaded {a_load}, reinit {a_re} -> {a_notes}")
  print(f"[INFO] critic: loaded {c_load}, reinit {c_re} -> {c_notes}")
  print(
    "[INFO] depth-ball CNN + MLP trunk + selfloc head carried over; mlp.0 reinit "
    "(+64 RGB latent) + the N-frame-stacked RGB CNN is fresh."
  )

  mode = "SMOKE" if smoke else "FULL"
  print(
    f"[INFO] v3g temporal selfloc-vision [{mode}]: {max_iter} iters, "
    f"envs={env_cfg.scene.num_envs}"
  )
  print(
    "[INFO] watch Metrics/selfloc_pos_err_m: should stay LOW even as the GT pose "
    "obs fades (iters 800->2500) — the stacked RGB CNN should localize via parallax."
  )
  runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)
  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
