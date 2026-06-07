"""Probe v3g Phase A: does the policy actually USE robot_field_pose to aim?

Phase A trained WITH robot_field_pose GT intact (no mask), having SWAPPED out the
goal-direction spoon-feed (robot_to_target). So the open question is whether the
policy genuinely derives "which way is the goal" from knowing where it stands —
or whether it ignored field_pose and found some other crutch.

Two inference-time passes on the SAME policy:
  * baseline - robot_field_pose intact (trained condition)
  * ablated  - robot_field_pose zeroed in the actor's 1D obs

INTERPRETATION (opposite of the ball-CNN probe): if dribble_success / goal
progress COLLAPSES under ablation, the policy DEPENDS on field_pose -> it really
is self-localizing to aim (Phase A goal met). If it survives unchanged,
field_pose was ignored and "self-localization" didn't actually take hold.

Usage:
  MUJOCO_GL=egl uv run python scripts/probe_v3g_selfloc.py <run_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.mos92.env_cfgs import mos92_soccer_selfloc_env_cfg
from mjlab.tasks.velocity.config.mos92.rl_cfg import mos92_vision_ppo_runner_cfg
from mjlab.utils.torch import configure_torch_backends

RUN_DIR = Path("logs/rsl_rl/mos92_velocity/2026-06-06_10-59-26_spike_v3g_selfloc")
if len(sys.argv) > 1:
  RUN_DIR = Path(sys.argv[1])
NUM_ENVS = 256
N_STEPS = 600
SETTLE = 150
# The self-localization GT term. We swapped robot_field_pose in under this key.
SELFLOC_TERM = "ball_to_target"
# GT ball terms the policy was TRAINED to see as zero (gt_mask pinned to 0 in the
# spike). In eval the curriculum sits at factor=1.0, so we must zero these in BOTH
# passes to reproduce the trained pure-vision condition — otherwise the policy
# gets OOD non-zero ball inputs and behaves like garbage (baseline collapses).
TRAINED_ZERO_TERMS = ("robot_to_ball", "ball_velocity", "ball_gaze_uv")


def _latest_ckpt() -> Path:
  ckpts = sorted(RUN_DIR.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
  if not ckpts:
    raise FileNotFoundError(f"No checkpoint in {RUN_DIR}")
  return ckpts[-1]


def _term_slices(obs_mgr, term_names) -> dict:
  """Map each requested actor-obs term name -> (start, end) column span."""
  names = obs_mgr.active_terms["actor"]
  dims = obs_mgr.group_obs_term_dim["actor"]
  spans, offset = {}, 0
  for name, shape in zip(names, dims, strict=False):
    width = int(torch.tensor(shape).prod().item()) if len(shape) else 1
    if name in term_names:
      spans[name] = (offset, offset + width)
    offset += width
  return spans


def _run_pass(env, runner, device, ablate: bool, selfloc_span, zero_spans) -> dict:
  """Roll out N_STEPS; time-average metrics over post-SETTLE steps.

  zero_spans (GT ball terms) are zeroed in BOTH passes to reproduce the trained
  pure-vision condition. selfloc_span (robot_field_pose) is additionally zeroed
  only when ablate=True. dribble_success collapsing under ablation == the policy
  DEPENDS on field_pose to aim.
  """

  def _mask(obs):
    for s, e in zero_spans:
      obs[:, s:e] = 0.0
    if ablate:
      obs[:, selfloc_span[0] : selfloc_span[1]] = 0.0

  policy = runner.get_inference_policy(device=device)
  cmd = env.unwrapped.command_manager.get_term("dribble")
  sums: dict = {}
  count = 0
  with torch.no_grad():
    env.reset()
    obs = env.get_observations()
    for step in range(N_STEPS):
      _mask(obs)
      action = policy(obs)
      obs, _, _, _ = env.step(action)
      _mask(obs)
      if step >= SETTLE:
        m = cmd.metrics
        sums["dribble_success"] = (
          sums.get("dribble_success", 0.0) + m["at_goal"].mean().item()
        )
        bt = m.get("ball_to_target_error")
        if bt is not None:
          sums["ball_to_target_error"] = (
            sums.get("ball_to_target_error", 0.0) + bt.mean().item()
          )
        rob = env.unwrapped.scene["robot"]
        up = rob.data.projected_gravity_b[:, 2]
        sums["upright"] = sums.get("upright", 0.0) + (-up).clamp(0, 1).mean().item()
        count += 1
  return {k: v / max(count, 1) for k, v in sums.items()}


def main() -> None:
  configure_torch_backends()
  device = "cuda:0"
  env_cfg = mos92_soccer_selfloc_env_cfg(play=True)
  env_cfg.scene.num_envs = NUM_ENVS
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=mos92_vision_ppo_runner_cfg().clip_actions)

  spans = _term_slices(
    env.unwrapped.observation_manager, (SELFLOC_TERM,) + TRAINED_ZERO_TERMS
  )
  selfloc_span = spans[SELFLOC_TERM]
  zero_spans = [spans[t] for t in TRAINED_ZERO_TERMS if t in spans]
  print(f"[INFO] field_pose slice={selfloc_span}, trained-zero ball spans={zero_spans}")

  agent_cfg = mos92_vision_ppo_runner_cfg()
  runner = MjlabOnPolicyRunner(env, _as_dict(agent_cfg), None, device)
  ckpt = _latest_ckpt()
  runner.load(str(ckpt))
  print(f"[INFO] Loaded {ckpt}")

  print(f"\n[INFO] === BASELINE (field_pose intact, ball zeroed), {N_STEPS} steps ===")
  base = _run_pass(env, runner, device, False, selfloc_span, zero_spans)
  print("\n[INFO] === ABLATED (field_pose ALSO zeroed) ===")
  abl = _run_pass(env, runner, device, True, selfloc_span, zero_spans)
  env.close()

  print("\n" + "=" * 60)
  print(f"{'metric':<26}{'baseline':>11}{'ablated':>11}{'ratio':>9}")
  print("-" * 60)
  for k in sorted(set(base) | set(abl)):
    b, a = base.get(k, float("nan")), abl.get(k, float("nan"))
    r = (a / b) if b else float("nan")
    print(f"{k:<26}{b:>11.4f}{a:>11.4f}{r:>9.2f}")
  print("=" * 60)
  print("DEPENDS on field_pose if dribble_success COLLAPSES when ablated")
  print("(ratio << 1). Survives unchanged (ratio ~1) => field_pose was ignored.")


def _as_dict(cfg):
  from dataclasses import asdict

  return asdict(cfg)


if __name__ == "__main__":
  main()
