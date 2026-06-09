"""v4 EXP6: GT-landmark ORACLE pose belief — perception OUT of the actor trunk.

EXP5 paradigm finding: the actor is a single shared MLP trunk emitting BOTH
joint_pos and the selfloc detection slice, so any perception learning signal
(reward OR supervised loss) backprops through the trunk and corrupts the gait
(fell_over climbed 1 -> 37 -> 52 with gradient strength). The fix is structural:
perception must not be an action of the control policy.

Here the geometry chain runs in an OBSERVATION (oracle_pose_belief): project the
known field keypoints with TRUE visibility (oracle = perfect DETECTION, not
perfect pose), depth-lift, Kabsch vs the map -> pose belief [x,y,sin,cos] + quality
signals [visible_frac, residual]. The soccer policy CONSUMES this belief; it has
no selfloc action and no keypoint reward, so NO perception gradient touches the
gait trunk. codex diagnosis step #1: prove the back-end (belief -> kick) is viable
under perfect detection AND remove the fell_over pollution source.

Criteria: pos_err < 1m + goal_rate >= 0.2 + fell_over ~= 0. If met -> the back-end
is sound and the problem is purely the learned detector (EXP7+). If it still
collapses -> the problem is fusion/action/coordination, not the CNN.

Usage:
  MUJOCO_GL=egl uv run python scripts/spike_v4_e2e_oracle.py          # full
  MUJOCO_GL=egl uv run python scripts/spike_v4_e2e_oracle.py --smoke  # 8-iter
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
  mos92_soccer_e2e_dualcam_oracle_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import (
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

BASE_CKPT = Path("checkpoints/v3_soccer_solo/01_selfloc_purevision/model_2800.pt")
NUM_ENVS = 384
MAX_ITER = 3000


def _partial_load_shape_match(module, ckpt_sd) -> tuple[int, int, list[str]]:
  """Load every name+shape-matching tensor from the bootstrap ckpt; reinit the
  rest. The actor's mlp.0 grows (+RGB latent + the 6-d belief slot replaces the
  4-d GT pose) and mlp.6 reinit-s (action 24/66 -> 20, selfloc head dropped);
  the MLP trunk (mlp.2/4), depth-ball CNN and the critic carry over.

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

  env_cfg = mos92_soccer_e2e_dualcam_oracle_env_cfg(play=False)
  env_cfg.scene.num_envs = 32 if smoke else NUM_ENVS

  # Pure-vision ball (as in the bootstrap run): pin ball gt_mask to 0.
  gt = env_cfg.curriculum["gt_mask"]
  gt.params["start_step"] = -1
  gt.params["end_step"] = 0
  pr = env_cfg.curriculum["penalty_ramp"]
  pr.params["start_step"] = -1
  pr.params["end_step"] = 0
  # NOTE: no selfloc_gt_mask curriculum here — the oracle env removed it. The pose
  # slot is the geometry belief (not GT), so there is nothing to fade.

  max_iter = 8 if smoke else MAX_ITER
  agent_cfg = mos92_selfloc_vision_ppo_runner_cfg()
  agent_cfg.max_iterations = max_iter
  agent_cfg.run_name = "spike_v4_e2e_oracle"
  agent_cfg.save_interval = 100
  # Plain PPO: perception is an OBSERVATION now, no aux loss / no keypoint head.

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_v4_e2e_oracle")
  log_dir = log_root / stamp
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)
  print(f"[INFO] Partial-load from: {BASE_CKPT}")
  ckpt = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
  a_load, a_re, a_notes = _partial_load_shape_match(
    runner.alg._raw_actor, ckpt["actor_state_dict"]
  )
  c_load, c_re, c_notes = _partial_load_shape_match(
    runner.alg._raw_critic, ckpt["critic_state_dict"]
  )
  print(f"[INFO] actor : loaded {a_load}, reinit {a_re} -> {a_notes}")
  print(f"[INFO] critic: loaded {c_load}, reinit {c_re} -> {c_notes}")

  mode = "SMOKE" if smoke else "FULL"
  print(f"[INFO] v4 EXP6 oracle belief [{mode}]: {max_iter} iters, envs={env_cfg.scene.num_envs}")
  print("[INFO] watch Metrics/selfloc_pos_err_m + goal_rate + fell_over: oracle belief")
  print("[INFO] should give pos_err<1m, goal_rate>=0.2, fell_over~=0 (clean back-end test).")
  runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)
  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
