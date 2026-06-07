"""Task v3g Phase B+C: distill self-localization into an RGB CNN.

Builds on the rebalanced explicit-selfloc policy (4-d cognitive estimate head +
accuracy reward, dribbling recovered). Two things happen here:

  PHASE B (vision wiring): the head camera now emits RGB too, and a second
  "camera_rgb" image obs group feeds a SEPARATE CNN branch. The validated
  depth-ball CNN is untouched (depth stays for the ball); the fresh RGB CNN is
  there to read the painted field lines / goal for self-localization.

  PHASE C (distillation): a curriculum (registered in the env cfg) ramps the GT
  robot_field_pose obs from scale 1 -> 0 over iters [400, 1200]. The
  selfloc_accuracy reward scores the estimate against robot_field_pose computed
  FRESH from true state each step, so the teacher survives the obs mask. As the
  GT obs fades, the only remaining source of pose information is the RGB CNN --
  so self-localization is forced into vision.

Bootstrap: from the rebalanced model_1999. Action dims are UNCHANGED (the
selfloc head already exists), so the partial-load is clean: only the actor
mlp.0 grows (input +64 for the RGB latent) and the RGB CNN is fresh. Everything
else -- mlp.2/4/6, std_param, the depth CNN, the whole critic -- carries over.

Usage:
  MUJOCO_GL=egl uv run python scripts/spike_v3g_selfloc_vision.py          # full
  MUJOCO_GL=egl uv run python scripts/spike_v3g_selfloc_vision.py --smoke  # 8-iter
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
  mos92_soccer_selfloc_vision_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import (
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

# Rebalanced explicit-selfloc run (0.3/-0.3 weights, dribbling recovered).
BASE_CKPT = Path(
  "logs/rsl_rl/mos92_velocity/2026-06-06_15-39-23_spike_v3g_selfloc_explicit/model_1999.pt"
)
NUM_ENVS = 1024
MAX_ITER = 2000


def _partial_load_shape_match(module, ckpt_sd) -> tuple[int, int, list[str]]:
  """Load every name+shape-matching tensor from the rebalanced ckpt; reinit the
  rest. Here only the actor mlp.0 changes (input grew +64 for the RGB CNN
  latent); the RGB CNN (cnns.camera_rgb.*) is absent from the ckpt so it stays
  at fresh init. mlp.2/4/6, std_param, the depth CNN (cnns.camera.*) and the
  whole (pure-MLP) critic carry over unchanged.

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

  env_cfg = mos92_soccer_selfloc_vision_env_cfg(play=False)
  env_cfg.scene.num_envs = 32 if smoke else NUM_ENVS

  # Pin the ball gt_mask to 0 (pure-vision ball, as in the bootstrap run).
  gt = env_cfg.curriculum["gt_mask"]
  gt.params["start_step"] = -1
  gt.params["end_step"] = 0
  # Anti-cheat penalties already learned upstream; keep them at full.
  pr = env_cfg.curriculum["penalty_ramp"]
  pr.params["start_step"] = -1
  pr.params["end_step"] = 0

  # Phase C GT-pose mask: in smoke, force it to ramp instantly so we exercise the
  # masked path; in full, keep the env's [400,1200]-iter ramp.
  selfloc_mask = env_cfg.curriculum["selfloc_gt_mask"]
  if smoke:
    selfloc_mask.params["start_step"] = -1
    selfloc_mask.params["end_step"] = 0

  max_iter = 8 if smoke else MAX_ITER
  agent_cfg = mos92_selfloc_vision_ppo_runner_cfg()
  agent_cfg.max_iterations = max_iter
  agent_cfg.run_name = "spike_v3g_selfloc_vision"
  agent_cfg.save_interval = 100

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_v3g_selfloc_vision")
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
    "[INFO] depth-ball CNN + MLP trunk + selfloc head carried over; only mlp.0 "
    "(input +64 RGB latent) reinit + the RGB CNN is fresh."
  )

  mode = "SMOKE" if smoke else "FULL"
  print(
    f"[INFO] v3g Phase B+C selfloc-vision [{mode}]: {max_iter} iters, "
    f"envs={env_cfg.scene.num_envs}"
  )
  print(
    "[INFO] watch Metrics/selfloc_pos_err_m: should stay LOW even as the GT pose "
    "obs fades (iters 400->1200) — that means the RGB CNN took over self-loc."
  )
  runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)
  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
