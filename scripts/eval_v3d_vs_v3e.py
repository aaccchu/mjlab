"""A/B eval: v3d (pre-anti-cheat) vs v3e (anti-cheat) in the SAME instrumented env.

v3d never logged the anti-cheat metrics during training (the sensors/penalties
didn't exist yet), so comparing training logs is apples-to-oranges. This loads
BOTH checkpoints into the identical anti-cheat-instrumented camera-only env and
measures, over the same rollout:
  - trapped / holding / illegal / sticking fractions  (did anti-cheat help?)
  - dribble_success                                    (did skill survive?)
  - ball_under_robot_frac (ball within 0.25m xy of base) (straddle proxy)
  - mean |shoulder_roll| in rad                        (arm-raise: did arms drop?)

Both passes zero the GT ball slices each step (camera-only, the trained
condition). Honest A/B: same env, same steps, only the checkpoint differs.

Usage: MUJOCO_GL=egl uv run python scripts/eval_v3d_vs_v3e.py
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.mos92.env_cfgs import (
  mos92_soccer_vision_ablation_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import mos92_vision_ppo_runner_cfg
from mjlab.utils.torch import configure_torch_backends

CKPTS = {
  "v3d": "logs/rsl_rl/mos92_velocity/2026-06-05_20-46-37_spike_v3d_jumpfix/model_1499.pt",
  "v3e": "logs/rsl_rl/mos92_velocity/2026-06-05_22-43-22_spike_v3e_anticheat/model_2999.pt",
  "v3f": "logs/rsl_rl/mos92_velocity/2026-06-06_00-09-22_spike_v3f_holding/model_1499.pt",
}
NUM_ENVS = 256
N_STEPS = 600
SETTLE = 150
GT_BALL_TERMS = ("robot_to_ball", "ball_velocity", "ball_gaze_uv")


def _gt_slices(obs_mgr):
  names = obs_mgr.active_terms["actor"]
  dims = obs_mgr.group_obs_term_dim["actor"]
  slices, offset = [], 0
  for name, shape in zip(names, dims, strict=False):
    width = int(torch.tensor(shape).prod().item()) if len(shape) else 1
    if name in GT_BALL_TERMS:
      slices.append((offset, offset + width))
    offset += width
  return slices


def _run_pass(env, runner, device, ckpt_path, slices) -> dict:
  """Camera-only rollout; return time-averaged anti-cheat + skill metrics."""
  ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  runner.alg._raw_actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
  policy = runner.get_inference_policy(device=device)
  robot = env.unwrapped.scene["robot"]
  ball = env.unwrapped.scene["ball"]
  cmd = env.unwrapped.command_manager.get_term("dribble")
  rew_mgr = env.unwrapped.reward_manager
  roll_idx = [i for i, n in enumerate(robot.joint_names) if "shoulder_roll" in n]
  pitch_idx = [i for i, n in enumerate(robot.joint_names) if "shoulder_pitch" in n]
  sums: dict[str, float] = defaultdict(float)
  count = 0

  with torch.inference_mode():
    env.reset()
    obs = env.get_observations()
    for step in range(N_STEPS):
      a = obs["actor"]
      for s, e in slices:
        a[:, s:e] = 0.0
      obs, _, _, _ = env.step(policy(obs))
      if step >= SETTLE:
        count += 1
        # straddle proxy: ball within 0.25m xy of robot base
        rxy = robot.data.root_link_pos_w[:, :2]
        bxy = ball.data.root_link_pos_w[:, :2]
        under = (torch.norm(bxy - rxy, dim=-1) < 0.25).float().mean().item()
        sums["ball_under_robot_frac"] += under
        # arm raise: mean |shoulder_roll| (abduction, T-pose) AND |shoulder_pitch|
        # (forward/overhead swing). The keyframe fix only touched ROLL, so pitch
        # is the un-constrained DOF the policy can still use to fling arms up.
        roll = robot.data.joint_pos[:, roll_idx].abs().mean().item()
        sums["mean_abs_shoulder_roll"] += roll
        pitch = robot.data.joint_pos[:, pitch_idx].abs().mean().item()
        sums["mean_abs_shoulder_pitch"] += pitch
        sums["holding_time_mean"] += cmd.metrics["holding_time"].mean().item()
        sums["ball_speed_mean"] += cmd.metrics["ball_speed"].mean().item()
        # Exact penalty fractions (the precise rule definitions, written to the
        # log dict by each penalty fn this step). These are the honest anti-cheat
        # numbers — not the loose ball_under proxy.
        log = env.unwrapped.extras.get("log", {})
        for mk in (
          "ball_trapped_frac",
          "holding_ball_frac",
          "illegal_contact_frac",
          "ball_sticking_frac",
        ):
          v = log.get(f"Metrics/{mk}")
          if v is not None:
            sums[mk] += float(v)
        for name in ("dribble_success", "gaze_center", "upright"):
          if name in rew_mgr.active_terms:
            idx = rew_mgr.active_terms.index(name)
            sums[name] += rew_mgr._step_reward[:, idx].mean().item()
  return {k: v / max(count, 1) for k, v in sums.items()}


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()
  device = "cuda:0"
  env_cfg = mos92_soccer_vision_ablation_env_cfg(play=True)
  env_cfg.scene.num_envs = NUM_ENVS
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)
  agent_cfg = mos92_vision_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
  slices = _gt_slices(env.unwrapped.observation_manager)

  results = {}
  for tag, path in CKPTS.items():
    print(f"\n[INFO] === {tag}: {Path(path).parent.name} ===")
    results[tag] = _run_pass(env, runner, device, path, slices)
  env.close()

  keys = (
    "dribble_success",
    "ball_trapped_frac",
    "holding_ball_frac",
    "illegal_contact_frac",
    "ball_sticking_frac",
    "ball_under_robot_frac",
    "holding_time_mean",
    "ball_speed_mean",
    "mean_abs_shoulder_roll",
    "mean_abs_shoulder_pitch",
    "gaze_center",
    "upright",
  )
  tags = list(results.keys())
  print("\n" + "=" * 76)
  hdr = f"{'metric':<26}" + "".join(f"{t:>12}" for t in tags)
  print(hdr)
  print("-" * 76)
  for k in keys:
    row = f"{k:<26}" + "".join(
      f"{results[t].get(k, float('nan')):>12.4f}" for t in tags
    )
    print(row)
  print("=" * 76)
  print("WANT: holding_ball_frac & ball_under DOWN (anti-cheat), shoulder_roll")
  print("      DOWN (arms drop), dribble_success roughly held (skill survived).")


if __name__ == "__main__":
  main()
