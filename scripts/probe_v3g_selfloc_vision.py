"""Probe v3g Phase B+C: did self-localization move INTO the RGB CNN (pure vision)?

The Phase B+C policy has TWO CNN branches (depth-ball + RGB) and was trained
with a curriculum that faded the GT robot_field_pose obs from scale 1 -> 0 over
iters [400, 1200], while the selfloc_accuracy reward kept scoring the estimate
against fresh GT each step. If the RGB CNN took over, the estimate should stay
accurate even with the (already-faded) GT obs zeroed.

THE decisive test is the −GT_pose ablation:
  * Phase A (no vision): zeroing GT obs blew the estimate up to 5.7m / 85°
    — the estimate was just echoing the GT obs.
  * Phase B+C (this): if −GT_pose error stays LOW (close to baseline), the
    estimate is computed from the RGB CNN, not copied from GT => PURE VISION
    self-localization succeeded.

Three passes (same policy): baseline / −estimate (zero selfloc feedback) /
−GT_pose (zero the faded GT pose obs). Calibration (estimate vs fresh GT) is
measured every post-settle step in meters + degrees.

Usage:
  MUJOCO_GL=egl uv run python scripts/probe_v3g_selfloc_vision.py <run_dir>
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
from mjlab.tasks.velocity.config.mos92.env_cfgs import (
  mos92_soccer_selfloc_vision_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import (
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.utils.torch import configure_torch_backends

RUN_DIR = Path(
  "logs/rsl_rl/mos92_velocity/2026-06-06_17-06-56_spike_v3g_selfloc_vision"
)
if len(sys.argv) > 1:
  RUN_DIR = Path(sys.argv[1])
NUM_ENVS = 256
N_STEPS = 600
SETTLE = 150
SELFLOC_GT_TERM = "ball_to_target"  # holds GT robot_field_pose (4-d).
SELFLOC_ACTION = "selfloc"
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
  """Absolute (start,end) of the selfloc slice within the 'actions' obs term."""
  names = obs_mgr.active_terms["actor"]
  dims = obs_mgr.group_obs_term_dim["actor"]
  offset = 0
  for name, shape in zip(names, dims, strict=False):
    width = int(torch.tensor(shape).prod().item()) if len(shape) else 1
    if name == "actions":
      pre = 0
      for tname, term in action_mgr._terms.items():
        if tname == SELFLOC_ACTION:
          return (offset + pre, offset + pre + term.action_dim)
        pre += term.action_dim
    offset += width
  raise RuntimeError("selfloc slice not found in 'actions' obs term")


def _run_pass(env, runner, device, mode, gt_span, fb_span, zero_spans) -> dict:
  """Roll out N_STEPS; time-average metrics over post-SETTLE steps.

  obs is a TensorDict (dual-CNN env => keys actor/camera/camera_rgb). We only
  ever mask columns of the 1-D 'actor' group; the image groups pass through
  untouched so the RGB CNN keeps seeing the field.

  mode: "baseline" | "ablate_gt" (zero GT pose obs in actor) |
        "ablate_estimate" (zero the selfloc feedback slice in actor).
  """

  def _mask(obs):
    a = obs["actor"]
    for s, e in zero_spans:
      a[:, s:e] = 0.0
    if mode == "ablate_gt":
      a[:, gt_span[0] : gt_span[1]] = 0.0
    elif mode == "ablate_estimate":
      a[:, fb_span[0] : fb_span[1]] = 0.0

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
        est = sl_term.raw_action  # (B,4)
        gt = mdp.robot_field_pose(uenv, "dribble", _ROBOT)  # (B,4)
        dx = (est[:, 0] - gt[:, 0]) * cmd.cfg.half_length
        dy = (est[:, 1] - gt[:, 1]) * cmd.cfg.half_width
        pos_err_m = torch.sqrt(dx**2 + dy**2).mean().item()
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
  # play=False keeps the GT-fade curriculum active so the obs distribution matches
  # training. The standard play=True probe FEEDS FULL-SCALE GT (fade is gated
  # `if not play`), which is OOD for a fade-trained policy and collapses the
  # position estimate (~9m artifact). Diagnostic diag_selfloc_env_gap.py confirmed
  # model_2800 is 0.82m in play=False vs 9.47m in play=True. We advance
  # common_step_counter so the fade reaches ~0 (pure-vision condition).
  env_cfg = mos92_soccer_selfloc_vision_env_cfg(play=False)
  env_cfg.scene.num_envs = NUM_ENVS
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  agent_cfg = mos92_selfloc_vision_ppo_runner_cfg()
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  # Drive the GT mask to ~0 (pure vision) like the late-fade training condition.
  env.unwrapped.common_step_counter = 3600 * 24

  obs_mgr = env.unwrapped.observation_manager
  action_mgr = env.unwrapped.action_manager
  spans = _term_slices(obs_mgr, (SELFLOC_GT_TERM,) + TRAINED_ZERO_TERMS)
  gt_span = spans[SELFLOC_GT_TERM]
  zero_spans = [spans[t] for t in TRAINED_ZERO_TERMS if t in spans]
  fb_span = _action_feedback_selfloc_span(obs_mgr, action_mgr)
  print(f"[INFO] GT pose slice={gt_span}, estimate-feedback slice={fb_span}")
  print(f"[INFO] trained-zero ball spans={zero_spans}")

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), None, device)
  ckpt = _latest_ckpt()
  runner.load(str(ckpt))
  print(f"[INFO] Loaded {ckpt}")

  print(f"\n[INFO] === BASELINE (all intact, ball zeroed), {N_STEPS} steps ===")
  base = _run_pass(env, runner, device, "baseline", gt_span, fb_span, zero_spans)
  print("\n[INFO] === ABLATE ESTIMATE (zero selfloc feedback) ===")
  abl_est = _run_pass(
    env, runner, device, "ablate_estimate", gt_span, fb_span, zero_spans
  )
  print("\n[INFO] === ABLATE GT pose (THE pure-vision test) ===")
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
  print("PURE-VISION SUCCESS if -GT_pose selfloc_pos_err_m stays close to baseline")
  print("  (Phase A blew up to 5.7m/85° when GT was zeroed; if it stays low here,")
  print("  the estimate comes from the RGB CNN, not from echoing the GT obs).")


if __name__ == "__main__":
  main()
