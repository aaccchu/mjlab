"""Goal-scoring spike (目标③): dribble the ball INTO the fixed goal, near-range.

Self-localization (目标①) reached sub-meter in exp8b. This attacks the
independent 目标③: actually scoring. The selfloc env's geometry (ball spawns by
the robot but goal fixed at x=+11, robot spawns full-field facing anywhere) made
dribble_success≈0 — too hard. The goal env instead spawns the robot in the
attacking half FACING the goal (x∈[6,8.5], yaw∈[-0.6,0.6]) with goal_progress
(w2) + goal_scored (w10) rewards, so near-range shooting is learnable.

Bootstrap from v3f (best dribbler: success 2.6 to random targets, anti-cheat +
arms fixed). v3f is a vision policy (1D obs 84 + 64 CNN latent); the goal env is
pure-GT (1D obs 81, no camera). So partial-load: reinit mlp.0 (input 148->81,
no CNN) + obs_normalizer; KEEP mlp.2/4/6 (deep trunk + the 20-motor action head,
unchanged) so v3f's walk/balance/kick motor skill carries over. Only the input
remap + the goal-aiming behavior relearn.

Usage:
  MUJOCO_GL=egl uv run python scripts/spike_v3g_goal.py          # full
  MUJOCO_GL=egl uv run python scripts/spike_v3g_goal.py --smoke  # 8-iter
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
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.mos92.env_cfgs import mos92_soccer_goal_env_cfg
from mjlab.tasks.velocity.config.mos92.rl_cfg import mos92_ppo_runner_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

BASE_CKPT = Path(
  "logs/rsl_rl/mos92_velocity/2026-06-07_03-55-43_spike_v3g_goal_stable/model_1600.pt"
)
NUM_ENVS = 1024
MAX_ITER = 2000


def _partial_load_shape_match(module, ckpt_sd) -> tuple[int, int, list[str]]:
  """Load every name+shape-matching tensor; reinit the rest. v3f -> goal env:
  mlp.0 changes (148 vision input -> 81 pure-GT, no CNN) and obs_normalizer
  changes; mlp.2/4/6 (trunk + 20-motor head) and distribution.std_param are
  unchanged and carry v3f's locomotion+kick skill. CNN tensors are absent in the
  goal model so they're simply not loaded.

  Returns (loaded, reinit-count, notes)."""
  cur = module.state_dict()
  to_load, notes = {}, []
  reinit = 0
  for k, v in ckpt_sd.items():
    if k not in cur:
      notes.append(f"{k}:dropped(not in goal model)")
      continue
    if cur[k].shape == v.shape:
      to_load[k] = v
    else:
      reinit += 1
      notes.append(f"{k}:reinit{tuple(v.shape)}->{tuple(cur[k].shape)}")
  module.load_state_dict(to_load, strict=False)
  return len(to_load), reinit, notes


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  smoke = "--smoke" in sys.argv
  configure_torch_backends()
  device = "cuda:0"

  env_cfg = mos92_soccer_goal_env_cfg(play=False)
  env_cfg.scene.num_envs = 32 if smoke else NUM_ENVS
  # exp11: rebalance for stability. The goal run learned to score (goal_rate
  # 0.29) but upright collapsed 0.99->0.62 because scoring rewards (~20 total)
  # dwarfed upright (1.0). Raise upright 1.0->2.5 and cut goal_scored 10->5 so
  # the policy keeps scoring while relearning to stay on its feet.
  env_cfg.rewards["upright"].weight = 2.5
  env_cfg.rewards["goal_scored"].weight = 5.0
  # exp12 (用户要求"最少步骤"): per-step cost until the goal is scored, so the
  # policy is pushed to score in fewer steps. Small weight so it shapes speed
  # without overpowering scoring/stability (which exp11 just balanced).
  from mjlab.managers.reward_manager import RewardTermCfg as _RTC
  env_cfg.rewards["time_to_goal_penalty"] = _RTC(
    func=mdp.time_to_goal_penalty, weight=-0.02,
    params={"command_name": "dribble"},
  )

  max_iter = 8 if smoke else MAX_ITER
  agent_cfg = mos92_ppo_runner_cfg()  # pure-MLP runner (no CNN) for the GT env.
  agent_cfg.max_iterations = max_iter
  agent_cfg.run_name = "spike_v3g_goal_minsteps2"
  agent_cfg.save_interval = 100

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_v3g_goal_minsteps2")
  log_dir = log_root / stamp
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)
  print(f"[INFO] Partial-load from v3f: {BASE_CKPT}")
  ckpt = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
  a_load, a_re, a_notes = _partial_load_shape_match(
    runner.alg._raw_actor, ckpt["actor_state_dict"]
  )
  c_load, c_re, c_notes = _partial_load_shape_match(
    runner.alg._raw_critic, ckpt["critic_state_dict"]
  )
  print(f"[INFO] actor : loaded {a_load}, reinit {a_re} -> {a_notes}")
  print(f"[INFO] critic: loaded {c_load}, reinit {c_re} -> {c_notes}")
  print("[INFO] v3f trunk + 20-motor head carried; mlp.0 reinit (vision->GT obs).")

  mode = "SMOKE" if smoke else "FULL"
  print(f"[INFO] v3g goal-scoring [{mode}]: {max_iter} iters, envs={env_cfg.scene.num_envs}")
  print("[INFO] watch Metrics/dribble/goal_rate + dribble_success (should rise).")
  runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)
  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
