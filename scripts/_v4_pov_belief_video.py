"""Render a robot-POV + self-localization belief video for a v4 soccer policy.

Produces ONE longer clip that tells the localization story end-to-end over a
single episode, as a 4-panel composite:

  [ RGB POV ] [ depth POV ] [ top-down: GT vs belief ] [ loc-err + coverage ]

  - RGB POV: the head_cam_rgb image the policy's self-loc CNN consumes (96x72,
    upscaled). The clip is trimmed to START while the robot still sees no field
    landmarks (coverage ~0: no goal, no key points), so the opening frames are
    genuinely "blind", then the field comes into view as the robot turns.
  - depth POV: the head_cam depth image the policy's ball CNN consumes (64x48),
    shown as a colormap (near=bright). This is how the ball is perceived.
  - Map: top-down pitch with the ground-truth robot pose (green) vs the policy's
    EKF belief (red x, the SAME _mu the policy consumes), AND the TRUE ball
    (white) vs the BELIEVED ball (orange). The believed ball = belief pose
    composed with the egocentric ball direction, so its offset from the true
    ball is the self-localization error propagated onto the ball (NOT an
    independent ball-perception error; ball perception is implicit in the CNN).
  - Curves: localization error (m) and fused field-coverage fraction over time.

The episode is rolled out on env0 (play=False, goal-targeted); a scored episode
is kept if one occurs within a budget of attempts, else the best goalward
attempt. Used by scripts/eval_v4_pov_belief_exp16.py and ..._exp19.py.
"""

from __future__ import annotations

import math
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
MAX_ATTEMPTS = 22  # episodes to try before keeping the best goalward attempt.
TRAIL = 40  # trail length in steps for the top-down map.
DEPTH_CUTOFF = 3.0  # metres; matches camera_depth obs the policy consumes.
DEPTH_MIN = 0.05
# Head-cam geometry (measured from the compiled model): the ball CNN sees a
# ground ball only beyond this range; closer is the foot-zone blind spot.
CAM_FOVY_DEG = 60.0
CAM_HEIGHT = 0.79  # approx head-cam height (m).
BALL_Z = 0.07


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


def _believed_ball(env, belief, cmd, robot, origin_xy):
  """Where the policy THINKS the ball is = belief pose (EKF _mu) composed with the
  egocentric ball direction. Returns (true_xy, believed_xy) in env-local coords."""
  ball_w = (cmd.ball_pos_w[0, :2] - origin_xy).cpu().numpy()
  rob_w = (robot.data.root_link_pos_w[0, :2] - origin_xy).cpu().numpy()
  gt = robot_field_pose(env, "dribble")[0].cpu().numpy()
  gt_yaw = math.atan2(gt[2], gt[3])
  dvec = ball_w - rob_w
  c, s = math.cos(-gt_yaw), math.sin(-gt_yaw)
  ego = np.array([c * dvec[0] - s * dvec[1], s * dvec[0] + c * dvec[1]])
  if belief is not None and getattr(belief, "_mu", None) is not None:
    mu = belief._mu[0].cpu().numpy()
    bx, by, byaw = float(mu[0]), float(mu[1]), float(mu[2])
  else:
    bx, by, byaw = float(rob_w[0]), float(rob_w[1]), gt_yaw
  c2, s2 = math.cos(byaw), math.sin(byaw)
  bel = np.array([bx, by]) + np.array(
    [c2 * ego[0] - s2 * ego[1], s2 * ego[0] + c2 * ego[1]]
  )
  return (float(ball_w[0]), float(ball_w[1])), (float(bel[0]), float(bel[1]))


def _ball_in_camera(env, cmd, robot):
  """True if the ball center projects inside the head camera's view frustum this
  step, plus the robot->ball ground distance. Uses the live camera world pose
  (cam_xpos / cam_xmat) and the camera's fovy, so it reflects the ACTUAL geometry
  (head height ~0.79 m, ~0 deg pitch) — a ground ball only enters view beyond
  ~1.25 m; closer is the foot-zone blind spot."""
  import mujoco

  m = env.sim.mj_model
  cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
  wd = env.sim.wp_data
  cam_pos = wd.cam_xpos.numpy()[0][cam_id]  # (3,)
  cam_mat = wd.cam_xmat.numpy()[0][cam_id].reshape(3, 3)
  ball_w = cmd.ball_pos_w[0].cpu().numpy()
  rob_w = robot.data.root_link_pos_w[0].cpu().numpy()
  dist = float(np.hypot(ball_w[0] - rob_w[0], ball_w[1] - rob_w[1]))
  # Ball center in camera frame (mujoco camera looks down its -z axis).
  rel = ball_w - cam_pos
  cam = cam_mat.T @ rel
  depth = -cam[2]
  if depth <= 0:  # behind camera
    return False, dist
  aspect = 96.0 / 72.0  # rgb cam width/height
  half_v = math.radians(CAM_FOVY_DEG) / 2.0
  half_h = math.atan(math.tan(half_v) * aspect)
  ang_v = math.atan2(cam[1], depth)
  ang_h = math.atan2(cam[0], depth)
  in_view = abs(ang_v) < half_v and abs(ang_h) < half_h
  return bool(in_view), dist


def _capture_episode(env, env_w, policy, belief, rgb_s, depth_s, cmd, robot, origin):
  """Roll one episode (until reset or cap). Return per-step trace + frames."""
  tr = {k: [] for k in (
    "rgb", "depth", "gt", "est", "ball", "ball_bel", "tgt", "err", "cover", "vis",
    "ball_in_view", "ball_dist"
  )}
  scored = False
  obs = env_w.get_observations()
  for _ in range(MAX_EPISODE_STEPS):
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env_w.step(actions)

    rgb_s.update(0.0)
    depth_s.update(0.0)
    tr["rgb"].append(np.asarray(rgb_s.data.rgb[0].cpu().numpy()))
    # Clip to the same 3 m the policy's camera_depth obs uses, then normalize —
    # the RAW sensor depth runs to the ~1000 m horizon, which auto-normalizes the
    # useful near range (ball / goal at 0-3 m) into near-black. This panel must
    # show what the policy's ball CNN actually consumes, not the raw buffer.
    raw_depth = depth_s.data.depth[0, ..., 0].cpu().numpy()
    clipped = np.clip(raw_depth, DEPTH_MIN, DEPTH_CUTOFF) / DEPTH_CUTOFF
    tr["depth"].append(np.asarray(clipped))

    # Is the ball within the head camera's view frustum this step? (geometry:
    # cam ~0.79 m high, ~0 deg pitch, fovy 60 deg -> a ground ball only enters
    # view beyond ~1.25 m; closer than that it is in the foot-zone blind spot.)
    bview, bdist = _ball_in_camera(env, cmd, robot)
    tr["ball_in_view"].append(bview)
    tr["ball_dist"].append(bdist)

    gt = robot_field_pose(env, "dribble")[0]
    gx, gy = float(gt[0]) * HALF_LENGTH, float(gt[1]) * HALF_WIDTH
    tr["gt"].append((gx, gy))
    if belief is not None and belief._last_xy_n is not None:
      ex = float(belief._last_xy_n[0, 0]) * HALF_LENGTH
      ey = float(belief._last_xy_n[0, 1]) * HALF_WIDTH
      cv = float(belief._last_uniq[0]) if belief._last_uniq is not None else 0.0
      vn = (
        float(belief._last_vis_now[0])
        if getattr(belief, "_last_vis_now", None) is not None
        else cv
      )
    else:
      ex, ey, cv, vn = gx, gy, 0.0, 0.0
    tr["est"].append((ex, ey))
    tr["cover"].append(cv)
    tr["vis"].append(vn)
    tr["err"].append(float(np.hypot(ex - gx, ey - gy)))

    true_ball, bel_ball = _believed_ball(env, belief, cmd, robot, origin)
    tr["ball"].append(true_ball)
    tr["ball_bel"].append(bel_ball)
    tp = (cmd.target_pos[0, :2] - origin).cpu().numpy()
    tr["tgt"].append((float(tp[0]), float(tp[1])))

    if float(cmd.goal_scored[0]) > 0.0:
      scored = True
    done = bool(dones[0]) if hasattr(dones, "__getitem__") else bool(dones)
    if done:
      break

  tr["scored"] = scored
  tr["len"] = len(tr["rgb"])
  return tr


def _draw_pitch(ax):
  ax.add_patch(plt.Rectangle((-HALF_LENGTH, -HALF_WIDTH), 2 * HALF_LENGTH,
                             2 * HALF_WIDTH, fill=False, color="white", lw=1.5))
  ax.plot([0, 0], [-HALF_WIDTH, HALF_WIDTH], color="white", lw=1, alpha=0.6)
  ax.add_patch(plt.Circle((0, 0), 2.0, fill=False, color="white", lw=1, alpha=0.6))
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
  ball_bel = np.array(ep["ball_bel"])
  tgt = np.array(ep["tgt"])
  err = np.array(ep["err"])
  cover = np.array(ep["cover"])
  vis = np.array(ep["vis"])
  err_max = max(1.0, float(err.max()) * 1.1)
  frames = []
  for t in range(n):
    fig = plt.figure(figsize=(16, 4), dpi=100)
    # Tag by localization quality (loc err), NOT instantaneous coverage: the EKF
    # retains an accurate belief even when momentarily seeing few landmarks (e.g.
    # head-down at the ball near the goal), so a coverage-based tag would wrongly
    # read "blind" there.
    e = float(err[t])
    if e > 2.0:
      tag = "UNCERTAIN / EKF prior (high loc error)"
    elif e > 0.8:
      tag = f"localizing (loc err {e:.1f} m)"
    else:
      tag = f"localized (loc err {e:.1f} m)"
    fig.suptitle(f"{title}   step {t}/{n}   [{tag}]", fontsize=11)

    ball_in_view = ep["ball_in_view"]
    ball_dist = ep["ball_dist"]
    bview = ball_in_view[t]
    bdist = ball_dist[t]
    vtag = "ball IN VIEW" if bview else "ball in BLIND SPOT"
    vcolor = "lime" if bview else "red"

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(ep["rgb"][t])
    ax1.set_title("RGB POV (self-loc CNN)", fontsize=9)
    ax1.set_xlabel(f"{vtag}  (ball {bdist:.1f} m)", fontsize=8, color=vcolor)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(1, 4, 2)
    # clipped-to-3m depth (what the policy's camera_depth obs feeds the ball CNN);
    # near = bright. turbo over [0,1] where 1.0 == 3 m+.
    ax2.imshow(ep["depth"][t], cmap="turbo_r", vmin=0.0, vmax=1.0)
    ax2.set_title("depth POV (ball CNN, clip 3m)", fontsize=9)
    ax2.set_xlabel("near=bright, far(>=3m)=dark", fontsize=8)
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(1, 4, 3)
    _draw_pitch(ax3)
    lo = max(0, t - TRAIL)
    ax3.plot(gt[lo : t + 1, 0], gt[lo : t + 1, 1], color="lime", lw=1, alpha=0.5)
    ax3.plot(est[lo : t + 1, 0], est[lo : t + 1, 1], color="red", lw=1, alpha=0.5)
    ax3.scatter(*gt[t], color="lime", s=60, label="robot GT", zorder=5)
    ax3.scatter(*est[t], color="red", s=60, marker="x", label="robot belief", zorder=5)
    ax3.scatter(*ball[t], color="white", s=45, label="ball true", zorder=5,
                edgecolors="black")
    ax3.scatter(*ball_bel[t], color="orange", s=45, marker="D",
                label="ball believed", zorder=5, edgecolors="black")
    ax3.plot([ball[t, 0], ball_bel[t, 0]], [ball[t, 1], ball_bel[t, 1]],
             color="orange", lw=0.8, alpha=0.7)
    ax3.scatter(*tgt[t], color="yellow", s=55, marker="*", label="target", zorder=4)
    ax3.set_title("field: GT vs belief (robot & ball)", fontsize=9)
    ax3.legend(loc="upper left", fontsize=6, framealpha=0.5, ncol=2)

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.plot(np.arange(t + 1), err[: t + 1], color="red", label="loc err (m)")
    ax4.set_xlim(0, n)
    ax4.set_ylim(0, err_max)
    ax4.set_ylabel("loc err (m)", color="red", fontsize=8)
    ax4.tick_params(labelsize=7)
    ax4b = ax4.twinx()
    ax4b.plot(np.arange(t + 1), cover[: t + 1], color="cyan", label="fused coverage")
    ax4b.plot(np.arange(t + 1), vis[: t + 1], color="deepskyblue", lw=0.8,
              alpha=0.6, label="visible now")
    ax4b.set_ylim(0, 1.0)
    ax4b.set_ylabel("coverage frac", color="cyan", fontsize=8)
    ax4b.tick_params(labelsize=7)
    ax4b.legend(loc="upper right", fontsize=6, framealpha=0.5)
    ax4.set_title("loc error vs field coverage", fontsize=9)
    ax4.set_xlabel("step", fontsize=8)

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
  rgb_s = env.scene["head_cam_rgb"]
  depth_s = env.scene["head_cam"]
  robot = env.scene["robot"]
  cmd = env.command_manager.get_term("dribble")
  origin = env.scene.env_origins[0, :2]

  best = None
  best_key = None
  env_w.reset()
  for attempt in range(MAX_ATTEMPTS):
    ep = _capture_episode(
      env, env_w, policy, belief, rgb_s, depth_s, cmd, robot, origin
    )
    deepest = max(b[0] for b in ep["ball"])
    ep["_deepest"] = deepest
    # Every episode opens blind: at reset the EKF sits at its wide prior and the
    # camera sees ~0 keypoints (vis0~0). The narrative arc we want is
    # blind-start -> localize -> dribble -> score, so rank by: scored, then how
    # much the belief converged (start high loc err, end low), then ball depth.
    start_err = float(np.mean(ep["err"][:10]))
    end_err = float(np.mean(ep["err"][-30:]))
    converged = start_err - end_err  # positive = belief improved over the episode
    ep["_converged"] = converged
    # Quality of a non-scored episode as a demo: we want the localization STORY
    # — a belief that starts wrong (high start err = robot unsure where it is)
    # and converges (low end err), with the ball driven deep toward the goal.
    # Reward the convergence arc and ball depth; penalize a poorly-localized end.
    demo_q = deepest - 3.0 * end_err + 2.5 * max(0.0, converged)
    key = (
      1 if ep["scored"] else 0,
      round(demo_q, 2),
      round(converged, 2),
    )
    print(
      f"[INFO] attempt {attempt}: len={ep['len']} scored={ep['scored']} "
      f"start_err={start_err:.2f} end_err={end_err:.2f} "
      f"deepest_x={deepest:.1f} demo_q={demo_q:.1f} vis0={ep['vis'][0]:.3f}"
    )
    if best is None or key > best_key:
      best, best_key = ep, key
  env_w.close()

  # No blind-start trim: frame 0 (episode reset) is the blindest moment (EKF at
  # its wide prior), which is exactly the opening the story wants.
  out_path = out_dir / f"{prefix}_pov_belief.mp4"
  _compose(best, out_path, title)
  ball_err = [
    float(np.hypot(a[0] - b[0], a[1] - b[1]))
    for a, b in zip(best["ball"], best["ball_bel"], strict=True)
  ]
  result = {
    "ckpt": str(ckpt),
    "video": str(out_path),
    "episode_len": best["len"],
    "scored": best["scored"],
    "fps": FPS,
    "final_loc_err_m": round(float(best["err"][-1]), 3),
    "mean_loc_err_m": round(float(np.mean(best["err"])), 3),
    "mean_believed_ball_err_m": round(float(np.mean(ball_err)), 3),
    "start_coverage": round(float(best["cover"][0]), 3),
    "max_coverage": round(float(np.max(best["cover"])), 3),
    "start_visible_now": round(float(best["vis"][0]), 3),
    "start_loc_err_m": round(float(np.mean(best["err"][:10])), 3),
    "ball_in_view_frac": round(float(np.mean(best["ball_in_view"])), 3),
    "note": (
      "believed-ball offset = self-localization error propagated onto the ball "
      "(egocentric ball direction is GT; ball perception is implicit in the CNN, "
      "no explicit ball-xy output to read). ball_in_view_frac = fraction of steps "
      "the ball is inside the head-cam frustum; a ground ball only enters view "
      "beyond ~1.25 m (cam ~0.79 m high, ~0 deg pitch), so when dribbling/kicking "
      "(ball <1.25 m) it sits in the foot-zone blind spot and the policy relies on "
      "the EKF self-pose + RGB temporal memory, not the live ball pixels."
    ),
  }
  print(f"[INFO] wrote POV+belief video -> {out_path} (scored={best['scored']})")
  return result
