"""Probe v3g Phase A (explicit): is the self-loc estimate (a) accurate, and
(b) actually USED to aim at the goal?

The explicit self-loc spike adds a 4-d cognitive action ("selfloc") the policy
uses to REPORT its estimate of [x_n, y_n, sin(yaw), cos(yaw)], scored against GT
robot_field_pose. The estimate feeds back to the next obs via the "actions"
term. Two questions, two checks:

  (1) CALIBRATION - is the estimate close to GT? We read the policy's
      selfloc raw action and compare to mdp.robot_field_pose each step,
      reporting mean position error in meters + heading error in degrees.

  (2) IS-IT-USED - does goal-dribbling DEPEND on the estimate? We zero the
      selfloc slice INSIDE the "actions" obs term (the feedback path) and see
      if dribble_success collapses. Collapse => the policy genuinely aims using
      its own estimate. Survives => it found another crutch (the same failure
      mode the original Phase A probe exposed).

We also keep the original GT-field_pose ablation as a cross-check: with GT still
in obs (Phase A sanity), the estimate can cheat by echoing GT, so a real test of
vision-based self-loc only comes after GT is masked (Phase C). This probe
therefore reports BOTH ablations and is explicit about which one is meaningful
at which phase.

Usage:
  MUJOCO_GL=egl uv run python scripts/probe_v3g_selfloc_explicit.py <run_dir>
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.mos92.env_cfgs import mos92_soccer_selfloc_env_cfg
from mjlab.tasks.velocity.config.mos92.rl_cfg import mos92_vision_ppo_runner_cfg
from mjlab.utils.torch import configure_torch_backends

RUN_DIR = Path("logs/rsl_rl/mos92_velocity/2026-06-06_10-59-26_spike_v3g_selfloc")
if len(sys.argv) > 1:
  RUN_DIR = Path(sys.argv[1])
NUM_ENVS = 256
N_STEPS = 600
SETTLE = 150
SELFLOC_GT_TERM = "ball_to_target"  # robot_field_pose GT swapped in under this key.
SELFLOC_ACTION = "selfloc"  # the cognitive estimate action term.
TRAINED_ZERO_TERMS = ("robot_to_ball", "ball_velocity", "ball_gaze_uv")
_ROBOT = SceneEntityCfg("robot")


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


def _action_feedback_selfloc_span(obs_mgr, action_mgr) -> tuple[int, int]:
  """Span of the selfloc slice WITHIN the 'actions' obs term (feedback path).

  The 'actions' obs term is the full flat action [joint_pos(20), selfloc(4)].
  Return the absolute (start, end) columns of the selfloc sub-slice in the
  actor obs vector, so we can zero ONLY the estimate feedback.
  """
  names = obs_mgr.active_terms["actor"]
  dims = obs_mgr.group_obs_term_dim["actor"]
  offset = 0
  for name, shape in zip(names, dims, strict=False):
    width = int(torch.tensor(shape).prod().item()) if len(shape) else 1
    if name == "actions":
      # Within the flat action, selfloc follows joint_pos (insertion order).
      pre = 0
      for tname, term in action_mgr._terms.items():
        if tname == SELFLOC_ACTION:
          return (offset + pre, offset + pre + term.action_dim)
        pre += term.action_dim
    offset += width
  raise RuntimeError("selfloc slice not found in 'actions' obs term")


def _run_pass(env, runner, device, mode, gt_span, fb_span, zero_spans) -> dict:
  """Roll out N_STEPS; time-average metrics over post-SETTLE steps.

  mode: "baseline" | "ablate_gt" (zero GT field_pose obs) |
        "ablate_estimate" (zero the selfloc estimate feedback in 'actions').
  Always zeroes the trained-zero ball terms to reproduce pure-vision training.
  Also accumulates calibration error (estimate vs GT robot_field_pose).
  """

  def _mask(obs):
    for s, e in zero_spans:
      obs[:, s:e] = 0.0
    if mode == "ablate_gt":
      obs[:, gt_span[0] : gt_span[1]] = 0.0
    elif mode == "ablate_estimate":
      obs[:, fb_span[0] : fb_span[1]] = 0.0

  policy = runner.get_inference_policy(device=device)
  uenv = env.unwrapped
  cmd = uenv.command_manager.get_term("dribble")
  sl_term = uenv.action_manager.get_term(SELFLOC_ACTION)
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
        rob = uenv.scene["robot"]
        up = rob.data.projected_gravity_b[:, 2]
        sums["upright"] = sums.get("upright", 0.0) + (-up).clamp(0, 1).mean().item()
        # Calibration: estimate vs GT field pose.
        est = sl_term.raw_action  # (B,4)
        gt = mdp.robot_field_pose(uenv, "dribble", _ROBOT)  # (B,4)
        dx = (est[:, 0] - gt[:, 0]) * cmd.cfg.half_length
        dy = (est[:, 1] - gt[:, 1]) * cmd.cfg.half_width
        pos_err_m = torch.sqrt(dx**2 + dy**2).mean().item()
        # Heading error from the sin/cos estimate vs GT.
        est_yaw = torch.atan2(est[:, 2], est[:, 3])
        gt_yaw = torch.atan2(gt[:, 2], gt[:, 3])
        head_err = (
          torch.atan2(torch.sin(est_yaw - gt_yaw), torch.cos(est_yaw - gt_yaw))
          .abs()
          .mean()
          .item()
        )
        sums["selfloc_pos_err_m"] = sums.get("selfloc_pos_err_m", 0.0) + pos_err_m
        sums["selfloc_head_err_deg"] = sums.get("selfloc_head_err_deg", 0.0) + (
          head_err * 180.0 / math.pi
        )
        count += 1
  return {k: v / max(count, 1) for k, v in sums.items()}


def main() -> None:
  configure_torch_backends()
  device = "cuda:0"
  env_cfg = mos92_soccer_selfloc_env_cfg(play=True)
  env_cfg.scene.num_envs = NUM_ENVS
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=mos92_vision_ppo_runner_cfg().clip_actions)

  obs_mgr = env.unwrapped.observation_manager
  action_mgr = env.unwrapped.action_manager
  spans = _term_slices(obs_mgr, (SELFLOC_GT_TERM,) + TRAINED_ZERO_TERMS)
  gt_span = spans[SELFLOC_GT_TERM]
  zero_spans = [spans[t] for t in TRAINED_ZERO_TERMS if t in spans]
  fb_span = _action_feedback_selfloc_span(obs_mgr, action_mgr)
  print(f"[INFO] GT field_pose slice={gt_span}, estimate-feedback slice={fb_span}")
  print(f"[INFO] trained-zero ball spans={zero_spans}")

  agent_cfg = mos92_vision_ppo_runner_cfg()
  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), None, device)
  ckpt = _latest_ckpt()
  runner.load(str(ckpt))
  print(f"[INFO] Loaded {ckpt}")

  print(f"\n[INFO] === BASELINE (all intact, ball zeroed), {N_STEPS} steps ===")
  base = _run_pass(env, runner, device, "baseline", gt_span, fb_span, zero_spans)
  print("\n[INFO] === ABLATE ESTIMATE (zero selfloc feedback in 'actions') ===")
  abl_est = _run_pass(
    env, runner, device, "ablate_estimate", gt_span, fb_span, zero_spans
  )
  print("\n[INFO] === ABLATE GT field_pose (cross-check, meaningful post-mask) ===")
  abl_gt = _run_pass(env, runner, device, "ablate_gt", gt_span, fb_span, zero_spans)
  env.close()

  print("\n" + "=" * 72)
  print(f"{'metric':<24}{'baseline':>11}{'-estimate':>12}{'-GT_pose':>12}")
  print("-" * 72)
  for k in sorted(set(base) | set(abl_est) | set(abl_gt)):
    b = base.get(k, float("nan"))
    ae = abl_est.get(k, float("nan"))
    ag = abl_gt.get(k, float("nan"))
    print(f"{k:<24}{b:>11.4f}{ae:>12.4f}{ag:>12.4f}")
  print("=" * 72)
  print("CALIBRATION: selfloc_pos_err_m / selfloc_head_err_deg should be SMALL")
  print("  in baseline (estimate tracks GT).")
  print("IS-IT-USED: dribble_success should COLLAPSE under -estimate if the")
  print("  policy aims using its own estimate. Survives => estimate ignored.")
  print("Note: with GT field_pose still in obs (Phase A), the estimate can echo")
  print("  GT; the real vision test is after GT is masked (Phase C).")


if __name__ == "__main__":
  main()
