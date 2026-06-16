"""Quantify gait human-likeness: foot-lift height + step cadence, EXP23 vs EXP24.

The standard training metrics don't log foot peak-height on flat terrain (the
clearance/swing rewards were deleted), so this script measures it directly. It
rolls each checkpoint in its env, tracks each foot site's world-z, and reports:

  - foot_lift_peak : mean peak height (m) a foot reaches between consecutive
                     ground contacts — the "do they pick their feet up" number.
  - step_period    : mean time (s) between a foot's successive lift-offs — higher
                     = slower, longer strides (less 小碎步).
  - airtime_frac   : fraction of steps a foot is off the ground.
  - foot_slip      : mean horizontal foot speed while planted (lower = cleaner).

Compares EXP23 (strike, flat-shuffle baseline) vs EXP24 (strike + AMP). A
human-like gait should show HIGHER foot_lift_peak and LONGER step_period.

Usage: MUJOCO_GL=egl uv run python scripts/eval_v4_gait_compare.py
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.mos92 import env_cfgs as _cfgs
from mjlab.tasks.velocity.config.mos92.rl_cfg import (
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.utils.torch import configure_torch_backends

FOOT_SITES = ("left_foot", "right_foot")
GROUND_Z = 0.04  # foot-site z below this counts as "on the ground" (contact).
STEPS = 1500
DT = 0.02

RUNS = [
  (
    "EXP23 (strike, baseline)",
    "checkpoints/v4_soccer/kick_strike_exp23/model_1999.pt",
    "mos92_soccer_e2e_dualcam_ekf_kick_strike_env_cfg",
  ),
  (
    "EXP24 (strike + AMP)",
    "checkpoints/v4_soccer/kick_strike_amp_exp24/model_1999.pt",
    "mos92_soccer_e2e_dualcam_ekf_kick_strike_amp_env_cfg",
  ),
]


def _measure(label: str, ckpt: str, env_name: str, device: str) -> dict | None:
  if not Path(ckpt).exists():
    print(f"[skip] {label}: ckpt missing ({ckpt})")
    return None
  env_cfg = getattr(_cfgs, env_name)(play=False)
  env_cfg.scene.num_envs = 64
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)
  if "gt_mask" in env_cfg.curriculum:
    env_cfg.curriculum["gt_mask"].params["start_step"] = -1
    env_cfg.curriculum["gt_mask"].params["end_step"] = 0
  if "penalty_ramp" in env_cfg.curriculum:
    env_cfg.curriculum["penalty_ramp"].params["start_step"] = -1
    env_cfg.curriculum["penalty_ramp"].params["end_step"] = 0

  agent_cfg = mos92_selfloc_vision_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env_w = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(env_w, asdict(agent_cfg), None, device)
  runner.load(ckpt, load_cfg={"actor": True}, strict=False, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = env.scene["robot"]
  sc = SceneEntityCfg("robot", site_names=FOOT_SITES, preserve_order=True)
  sc.resolve(env.scene)

  z_hist: list[np.ndarray] = []
  slip_hist: list[np.ndarray] = []
  obs = env_w.get_observations()
  for _ in range(STEPS):
    with torch.no_grad():
      actions = policy(obs)
    obs, _, _, _ = env_w.step(actions)
    z = robot.data.site_pos_w[:, sc.site_ids, 2]  # [B, F]
    vxy = robot.data.site_lin_vel_w[:, sc.site_ids, :2].norm(dim=-1)  # [B, F]
    z_hist.append(z.cpu().numpy())
    slip_hist.append(vxy.cpu().numpy())
  env_w.close()

  z = np.stack(z_hist, axis=0)  # [T, B, F]
  slip = np.stack(slip_hist, axis=0)
  on_ground = z < GROUND_Z

  # Per (env, foot) swing analysis: a swing is a run of off-ground steps; its peak
  # height and the time between successive lift-offs define lift and cadence.
  lifts, periods, airtimes = [], [], []
  T, B, F = z.shape
  for b in range(B):
    for f in range(F):
      grounded = on_ground[:, b, f]
      airtimes.append(1.0 - grounded.mean())
      liftoffs = np.where((~grounded[1:]) & (grounded[:-1]))[0]
      for i in range(len(liftoffs) - 1):
        s, e = liftoffs[i], liftoffs[i + 1]
        seg = z[s:e, b, f]
        if seg.size:
          lifts.append(float(seg.max()))
        periods.append((e - s) * DT)
  # Slip while planted.
  slip_planted = slip[on_ground]
  return {
    "label": label,
    "foot_lift_peak_m": float(np.mean(lifts)) if lifts else float("nan"),
    "step_period_s": float(np.mean(periods)) if periods else float("nan"),
    "airtime_frac": float(np.mean(airtimes)),
    "foot_slip_mps": float(np.mean(slip_planted))
    if slip_planted.size
    else float("nan"),
    "n_swings": len(lifts),
  }


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  results = []
  for label, ckpt, env_name in RUNS:
    r = _measure(label, ckpt, env_name, device)
    if r:
      results.append(r)
      print(
        f"\n[{label}]\n"
        f"  foot_lift_peak : {r['foot_lift_peak_m'] * 1000:6.1f} mm\n"
        f"  step_period    : {r['step_period_s']:6.3f} s\n"
        f"  airtime_frac   : {r['airtime_frac']:6.3f}\n"
        f"  foot_slip      : {r['foot_slip_mps']:6.3f} m/s  (n_swings={r['n_swings']})"
      )
  if len(results) == 2:
    a, b = results
    print("\n=== EXP24 vs EXP23 (human-likeness deltas) ===")
    print(
      f"  foot lift : {a['foot_lift_peak_m'] * 1000:.1f} -> "
      f"{b['foot_lift_peak_m'] * 1000:.1f} mm  "
      f"({'HIGHER' if b['foot_lift_peak_m'] > a['foot_lift_peak_m'] else 'lower'})"
    )
    print(
      f"  step period: {a['step_period_s']:.3f} -> {b['step_period_s']:.3f} s  "
      f"({'LONGER' if b['step_period_s'] > a['step_period_s'] else 'shorter'})"
    )
    print(
      f"  foot slip  : {a['foot_slip_mps']:.3f} -> {b['foot_slip_mps']:.3f} m/s  "
      f"({'less' if b['foot_slip_mps'] < a['foot_slip_mps'] else 'more'})"
    )


if __name__ == "__main__":
  main()
