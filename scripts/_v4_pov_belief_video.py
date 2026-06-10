"""Render a robot-POV + self-localization belief video for a v4 soccer policy.

Produces ONE longer clip that tells the localization story end-to-end over a
single episode (reset -> orient -> dribble -> shoot), as a 3-panel composite:

  [ robot camera POV ] [ top-down field: GT vs belief ] [ pos_err + coverage ]

  - POV: the head_cam_rgb image the policy's CNN actually consumes (96x72,
    upscaled), so you see the field come into view as the robot turns.
  - Map: a top-down pitch with the ground-truth robot pose (green), the policy's
    FUSED belief estimate (red, the SAME _last_xy_n the policy consumes), the
    ball (white) and the goal target (yellow), each with a fading trail.
  - Curves: localization error in metres and the fused field-coverage fraction
    (uniq_frac) vs time — early on coverage is low and error high; after the
    robot scans/turns, coverage rises and error drops, then it strikes.

The episode is chosen by rolling out env0 (play=False, goal-targeted) and
keeping the FIRST episode that ends in a goal if one occurs within a budget of
attempts; otherwise the best goalward attempt is kept (flagged in the result).

Used by scripts/eval_v4_pov_belief_exp16.py and ..._exp19.py.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mediapy as media  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.velocity.config.mos92.rl_cfg import (  # noqa: E402
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.tasks.velocity.mdp.observations import (  # noqa: E402
  FusedPoseBelief,
  robot_field_pose,
)
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

HALF_LENGTH = 11.0
HALF_WIDTH = 7.0
GOAL_HALF_WIDTH = 1.0
FPS = 30
MAX_EPISODE_STEPS = 1000  # 20 s cap (episode resets earlier on goal/fall).
MAX_ATTEMPTS = 12  # episodes to try before keeping the best goalward attempt.
TRAIL = 40  # trail length in steps for the top-down map.


def _find_belief_term(env: ManagerBasedRlEnv):
  om = env.observation_manager
  for grp in ("actor", "critic"):
    names = om.active_terms.get(grp, [])
    if "ball_to_target" in names:
      cand = om._group_obs_term_cfgs[grp][names.index("ball_to_target")].func
      if isinstance(cand, FusedPoseBelief):
        return cand
  return None


def _build_env(env_cfg_fn: Callable[..., object], device: str):
  # play=False -> 20 s episodes that reset on goal/fall; force the GT-pose fade
  # curriculum to its deployed end (pure belief) and bias targets to the goal.
  env_cfg = env_cfg_fn(play=False)
  env_cfg.scene.num_envs = 1
  if "actor" in env_cfg.observations:
    env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)
  if "gt_mask" in env_cfg.curriculum:
    env_cfg.curriculum["gt_mask"].params["start_step"] = -1
    env_cfg.curriculum["gt_mask"].params["end_step"] = 0
  if "penalty_ramp" in env_cfg.curriculum:
    env_cfg.curriculum["penalty_ramp"].params["start_step"] = -1
    env_cfg.curriculum["penalty_ramp"].params["end_step"] = 0
  cmd_cfg = env_cfg.commands.get("dribble")
  if cmd_cfg is not None and hasattr(cmd_cfg, "goal_target_fraction"):
    cmd_cfg.goal_target_fraction = 1.0
  return env_cfg


def _capture_episode(env, env_w, policy, belief, rgb_sensor, cmd, origin_xy):
  """Roll one episode (until reset or cap). Return per-step trace + frames."""
  pov: list[np.ndarray] = []
  gt_xy: list[tuple[float, float]] = []
  est_xy: list[tuple[float, float]] = []
  ball_xy: list[tuple[float, float]] = []
  tgt_xy: list[tuple[float, float]] = []
  err_m: list[float] = []
  cover: list[float] = []
  scored = False

  obs = env_w.get_observations()
  for _ in range(MAX_EPISODE_STEPS):
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env_w.step(actions)

    rgb_sensor.update(0.0)
    pov.append(np.asarray(rgb_sensor.data.rgb[0].cpu().numpy()))

    gt = robot_field_pose(env, "dribble")[0]
    gx = float(gt[0]) * HALF_LENGTH
    gy = float(gt[1]) * HALF_WIDTH
    gt_xy.append((gx, gy))
    if belief is not None and belief._last_xy_n is not None:
      ex = float(belief._last_xy_n[0, 0]) * HALF_LENGTH
      ey = float(belief._last_xy_n[0, 1]) * HALF_WIDTH
      cv = float(belief._last_uniq[0]) if belief._last_uniq is not None else 0.0
    else:
      ex, ey, cv = gx, gy, 0.0
    est_xy.append((ex, ey))
    cover.append(cv)
    err_m.append(float(np.hypot(ex - gx, ey - gy)))

    b = cmd.ball_pos_w[0, :2] - origin_xy
    ball_xy.append((float(b[0]), float(b[1])))
    # target_pos is world coords (goal target adds center_xy); subtract origin.
    tp = cmd.target_pos[0, :2] - origin_xy
    tgt_xy.append((float(tp[0]), float(tp[1])))

    if float(cmd.goal_scored[0]) > 0.0:
      scored = True
    done = bool(dones[0]) if hasattr(dones, "__getitem__") else bool(dones)
    if done:
      break

  return {
    "pov": pov,
    "gt": gt_xy,
    "est": est_xy,
    "ball": ball_xy,
    "tgt": tgt_xy,
    "err": err_m,
    "cover": cover,
    "scored": scored,
    "len": len(pov),
  }


def _draw_pitch(ax):
  ax.add_patch(plt.Rectangle((-HALF_LENGTH, -HALF_WIDTH), 2 * HALF_LENGTH,
                             2 * HALF_WIDTH, fill=False, color="white", lw=1.5))
  ax.plot([0, 0], [-HALF_WIDTH, HALF_WIDTH], color="white", lw=1, alpha=0.6)
  circ = plt.Circle((0, 0), 2.0, fill=False, color="white", lw=1, alpha=0.6)
  ax.add_patch(circ)
  # goal mouth at +x
  ax.plot([HALF_LENGTH, HALF_LENGTH], [-GOAL_HALF_WIDTH, GOAL_HALF_WIDTH],
          color="yellow", lw=4)
  ax.set_xlim(-HALF_LENGTH - 1, HALF_LENGTH + 1)
  ax.set_ylim(-HALF_WIDTH - 1, HALF_WIDTH + 1)
  ax.set_aspect("equal")
  ax.set_facecolor("#2e7d32")
  ax.set_xticks([])
  ax.set_yticks([])


def _compose(ep: dict, out_path: Path, title: str) -> None:
  n = ep["len"]
  gt = np.array(ep["gt"])
  est = np.array(ep["est"])
  ball = np.array(ep["ball"])
  tgt = np.array(ep["tgt"])
  err = np.array(ep["err"])
  cover = np.array(ep["cover"])
  frames = []
  for t in range(n):
    fig = plt.figure(figsize=(12, 4), dpi=100)
    fig.suptitle(f"{title}   step {t}/{n}", fontsize=11)

    # Panel 1: robot POV
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(ep["pov"][t])
    ax1.set_title("robot camera POV (head_cam_rgb)", fontsize=9)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Panel 2: top-down GT vs belief
    ax2 = fig.add_subplot(1, 3, 2)
    _draw_pitch(ax2)
    lo = max(0, t - TRAIL)
    ax2.plot(gt[lo : t + 1, 0], gt[lo : t + 1, 1], color="lime", lw=1, alpha=0.5)
    ax2.plot(est[lo : t + 1, 0], est[lo : t + 1, 1], color="red", lw=1, alpha=0.5)
    ax2.scatter(*gt[t], color="lime", s=60, label="GT pose", zorder=5)
    ax2.scatter(*est[t], color="red", s=60, marker="x", label="belief", zorder=5)
    ax2.scatter(*ball[t], color="white", s=40, label="ball", zorder=5,
                edgecolors="black")
    ax2.scatter(*tgt[t], color="yellow", s=50, marker="*", label="target", zorder=4)
    ax2.set_title("field: GT (green) vs belief (red x)", fontsize=9)
    ax2.legend(loc="upper left", fontsize=6, framealpha=0.5)

    # Panel 3: error + coverage curves
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(np.arange(t + 1), err[: t + 1], color="red", label="loc err (m)")
    ax3.set_xlim(0, n)
    ax3.set_ylim(0, max(1.0, float(err.max()) * 1.1))
    ax3.set_ylabel("loc err (m)", color="red", fontsize=8)
    ax3.tick_params(labelsize=7)
    ax3b = ax3.twinx()
    ax3b.plot(np.arange(t + 1), cover[: t + 1], color="cyan", label="field coverage")
    ax3b.set_ylim(0, 1.0)
    ax3b.set_ylabel("coverage frac", color="cyan", fontsize=8)
    ax3b.tick_params(labelsize=7)
    ax3.set_title("localization error vs field coverage", fontsize=9)
    ax3.set_xlabel("step", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    frames.append(buf.reshape(h, w, 4)[..., :3].copy())
    plt.close(fig)

  media.write_video(str(out_path), frames, fps=FPS)


def run(env_cfg_fn, ckpt: Path, out_dir: Path, prefix: str, title: str) -> dict:
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()
  out_dir.mkdir(parents=True, exist_ok=True)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  if not ckpt.exists():
    raise FileNotFoundError(f"checkpoint missing: {ckpt}")

  env_cfg = _build_env(env_cfg_fn, device)
  agent_cfg = mos92_selfloc_vision_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env_w = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(env_w, asdict(agent_cfg), str(out_dir), device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=False, map_location=device)
  policy = runner.get_inference_policy(device=device)
  belief = _find_belief_term(env)
  rgb_sensor = env.scene["head_cam_rgb"]
  cmd = env.command_manager.get_term("dribble")
  origin_xy = env.scene.env_origins[0, :2]

  best = None
  env_w.reset()
  for attempt in range(MAX_ATTEMPTS):
    ep = _capture_episode(env, env_w, policy, belief, rgb_sensor, cmd, origin_xy)
    print(
      f"[INFO] attempt {attempt}: len={ep['len']} scored={ep['scored']} "
      f"final_err={ep['err'][-1]:.2f}m"
    )
    # Prefer a scored episode; else keep the one whose ball got deepest in +x.
    deepest = max(b[0] for b in ep["ball"])
    ep["_deepest"] = deepest
    if ep["scored"]:
      best = ep
      break
    if best is None or deepest > best["_deepest"]:
      best = ep
  env_w.close()

  out_path = out_dir / f"{prefix}_pov_belief.mp4"
  _compose(best, out_path, title)
  result = {
    "ckpt": str(ckpt),
    "video": str(out_path),
    "episode_len": best["len"],
    "scored": best["scored"],
    "fps": FPS,
    "final_loc_err_m": round(float(best["err"][-1]), 3),
    "mean_loc_err_m": round(float(np.mean(best["err"])), 3),
    "start_coverage": round(float(best["cover"][0]), 3),
    "max_coverage": round(float(np.max(best["cover"])), 3),
  }
  print(f"[INFO] wrote POV+belief video -> {out_path} (scored={best['scored']})")
  return result
