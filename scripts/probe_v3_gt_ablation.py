"""Probe v3-MOS92: is the depth CNN dead weight? (cheap inference-time ablation)

Loads the gaze-warmup checkpoint (model_2999) and runs the SAME vision policy
twice over the same env:

  * baseline  - GT ball obs intact (what the policy was trained with),
  * ablated   - the GT ball-vector slices (robot_to_ball, ball_velocity,
                ball_gaze_uv) zeroed in the actor's 1D obs, so the ONLY remaining
                path to the ball is the head depth camera -> CNN latent.

If gaze_center / ball_visible / dribble_success survive the ablation, the CNN
learned to see. If they collapse, the GT vector was a crutch and the CNN is dead
weight (the memory's prediction). Minutes, no training.

NOTE: ablation zeroes the RAW obs (pre-normalizer), matching how the A2->v3
bootstrap zero-filled these columns. That pins the GT inputs to a fixed value
rather than signalling "absent"; the baseline/ablated RATIO is the robust signal.

Usage:
  MUJOCO_GL=egl uv run python scripts/probe_v3_gt_ablation.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.mos92.env_cfgs import mos92_soccer_vision_env_cfg
from mjlab.tasks.velocity.config.mos92.rl_cfg import mos92_vision_ppo_runner_cfg
from mjlab.utils.torch import configure_torch_backends

# Default = warmup run. Override with argv[1] to probe a different run (e.g. v3b).
RUN_DIR = Path("logs/rsl_rl/mos92_velocity/2026-06-02_20-19-43_spike_v3_vision")
if len(sys.argv) > 1:
  RUN_DIR = Path(sys.argv[1])
NUM_ENVS = 256
N_STEPS = 600
SETTLE = 150  # discard transient before averaging
# GT terms in the actor's 1D obs that leak ball POSITION (target dir is legal).
GT_BALL_TERMS = ("robot_to_ball", "ball_velocity", "ball_gaze_uv")


def _latest_ckpt() -> Path:
  ckpts = sorted(RUN_DIR.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
  if not ckpts:
    raise FileNotFoundError(f"No checkpoint in {RUN_DIR}")
  return ckpts[-1]


def _gt_slices(obs_mgr) -> tuple[list[tuple[int, int]], int]:
  """Return (GT_BALL slices, visible-flag column) inside the 'actor' group.

  Reads term names + dims from the ObservationManager so offsets can't drift if
  the obs layout changes. The 'actor' group is concatenated in declaration order.
  ball_gaze_uv is (u, v, visible); the visible flag is the term's 3rd column.
  """
  names = obs_mgr.active_terms["actor"]
  dims = obs_mgr.group_obs_term_dim["actor"]  # list of per-term shape tuples
  slices: list[tuple[int, int]] = []
  visible_col = -1
  offset = 0
  for name, shape in zip(names, dims):
    width = int(torch.tensor(shape).prod().item()) if len(shape) else 1
    if name in GT_BALL_TERMS:
      slices.append((offset, offset + width))
    if name == "ball_gaze_uv":
      visible_col = offset + 2  # (u, v, visible)
    offset += width
  found = sum(1 for n in names if n in GT_BALL_TERMS)
  assert found == len(GT_BALL_TERMS), (
    f"expected {GT_BALL_TERMS} in actor obs, found {found}: {names}"
  )
  assert visible_col >= 0, "ball_gaze_uv not found in actor obs"
  print(f"[INFO] actor 1D obs width={offset}, GT-ball slices={slices}, "
        f"visible_col={visible_col}")
  return slices, visible_col


def _run_pass(env, runner, device, ablate: bool, slices, visible_col) -> dict:
  """Roll out N_STEPS; return time-averaged metrics over post-SETTLE steps.

  ball_visible is read from the GT visible flag in the RAW (un-ablated) obs each
  step so it reflects true ball-in-frame even in the ablated pass; everything the
  POLICY sees is still zeroed when ablate=True.
  """
  policy = runner.get_inference_policy(device=device)
  rew_mgr = env.unwrapped.reward_manager
  sums: dict[str, float] = defaultdict(float)
  count = 0

  with torch.inference_mode():
    env.reset()
    obs = env.get_observations()
    for step in range(N_STEPS):
      true_visible = obs["actor"][:, visible_col].clone()
      if ablate:
        a = obs["actor"]
        for s, e in slices:
          a[:, s:e] = 0.0
      action = policy(obs)
      obs, _, _, _ = env.step(action)

      if step >= SETTLE:
        count += 1
        for name in ("gaze_center", "gaze_search", "dribble_success", "upright"):
          if name in rew_mgr.active_terms:
            idx = rew_mgr.active_terms.index(name)
            sums[name] += rew_mgr._step_reward[:, idx].mean().item()
        sums["ball_visible"] += true_visible.mean().item()

  return {k: v / max(count, 1) for k, v in sums.items()}


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  env_cfg = mos92_soccer_vision_env_cfg(play=True)
  env_cfg.scene.num_envs = NUM_ENVS
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)

  agent_cfg = mos92_vision_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  ckpt = _latest_ckpt()
  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  print(f"[INFO] Loaded {ckpt}")

  slices, visible_col = _gt_slices(env.unwrapped.observation_manager)

  # Baseline first (GT intact), then ablated. Each pass resets inside its own
  # inference_mode block, so they start from comparable distributions.
  print(f"\n[INFO] === BASELINE (GT intact), {N_STEPS} steps, {NUM_ENVS} envs ===")
  base = _run_pass(env, runner, device, ablate=False, slices=slices,
                   visible_col=visible_col)
  print(f"[INFO] === ABLATED (GT ball-vector zeroed -> camera only) ===")
  abl = _run_pass(env, runner, device, ablate=True, slices=slices,
                  visible_col=visible_col)

  env.close()

  keys = ("gaze_center", "ball_visible", "dribble_success", "upright", "gaze_search")
  print("\n" + "=" * 60)
  print(f"{'metric':<18}{'baseline':>12}{'ablated':>12}{'ratio':>10}")
  print("-" * 60)
  for k in keys:
    b, a = base.get(k, float("nan")), abl.get(k, float("nan"))
    ratio = (a / b) if b not in (0.0, float("nan")) else float("nan")
    print(f"{k:<18}{b:>12.4f}{a:>12.4f}{ratio:>10.2f}")
  print("=" * 60)
  print("READ: ratio ~1.0 => CNN carries the ball signal (vision works).")
  print("      ratio ~0.0 => GT was a crutch, CNN is dead weight (memory's bet).")


if __name__ == "__main__":
  main()
