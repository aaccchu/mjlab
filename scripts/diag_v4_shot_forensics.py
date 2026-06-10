"""EXP20b shot-failure forensics: WHERE and HOW do goal-targeted episodes fail?

Motivation (cold analysis, 2026-06-10): EXP20b ended at real shot rate 0.368 with
episode_success 0.581 — i.e. a large fraction of goal episodes get the ball NEAR
the target but do not score. The aggregate metrics cannot say WHY. This script
rolls out the deployment checkpoint over many goal-targeted episodes and logs a
per-episode forensic record:

  - scored / not
  - spawn-to-goal distance (does far spawn even have time to reach?)
  - deepest ball x reached + |y| at the moment of deepest x  -> miss geometry:
      * x >= 11 but |y| > 1   -> WIDE MISS (aim error, crossed end line outside mouth)
      * x in [9, 11)          -> SHORT (reached the box but never crossed)
      * x < 9                 -> NEVER ARRIVED (locomotion/time budget, not aim)
  - per-kick records: ball position at each kick contact (zone: near-goal x>7 /
    mid x in [0,7] / back x<0) and the goalward-aim error of the kick direction
    (angle between ball velocity right after contact and ball->goal-mouth ray)
  - time of first kick, total kicks

Aggregates printed at the end give the zone-conditional kick accuracy the user
asked for (near-goal vs midfield) and the miss-type distribution, which decide
what EXP21 should attack (aim width? shot power? approach time?).

Usage: MUJOCO_GL=egl uv run python scripts/diag_v4_shot_forensics.py [--envs N] [--steps N]
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.mos92.env_cfgs import (
  mos92_soccer_e2e_dualcam_ekf_kick_aim_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import (
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.utils.torch import configure_torch_backends

CKPT = Path("checkpoints/v4_soccer/kick_aim_exp20b/model_1999.pt")
OUT = Path("soccer_eval/2026-06-10_v4/kick_aim_exp20b/shot_forensics.json")
GOAL_X = 11.0
GOAL_HALF_W = 1.0
NEAR_GOAL_X = 7.0  # near-goal zone: x > 7 (attacking third-ish)


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()
  num_envs = 64
  steps = 3000  # ~3 episodes per env -> ~190 episode records
  for i, a in enumerate(sys.argv):
    if a == "--envs":
      num_envs = int(sys.argv[i + 1])
    if a == "--steps":
      steps = int(sys.argv[i + 1])
  device = "cuda:0"

  env_cfg = mos92_soccer_e2e_dualcam_ekf_kick_aim_env_cfg(play=False)
  env_cfg.scene.num_envs = num_envs
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)
  for cur in ("gt_mask", "penalty_ramp"):
    if cur in env_cfg.curriculum:
      env_cfg.curriculum[cur].params["start_step"] = -1
      env_cfg.curriculum[cur].params["end_step"] = 0
  env_cfg.commands["dribble"].goal_target_fraction = 1.0  # every episode = a shot trial

  agent_cfg = mos92_selfloc_vision_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env_w = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(env_w, asdict(agent_cfg), "/tmp", device)
  runner.load(str(CKPT), load_cfg={"actor": True}, strict=False, map_location=device)
  policy = runner.get_inference_policy(device=device)

  cmd = env.command_manager.get_term("dribble")
  robot = env.scene["robot"]
  contact = env.scene["foot_ball_contact"]
  origin = env.scene.env_origins[:, :2]

  B = num_envs
  # per-env episode accumulators
  deep_x = torch.full((B,), -1e9, device=device)
  y_at_deep = torch.zeros(B, device=device)
  spawn_goal_dist = torch.zeros(B, device=device)
  kicks = torch.zeros(B, device=device)
  first_kick_t = torch.full((B,), -1.0, device=device)
  ep_t = torch.zeros(B, device=device)
  prev_contact = torch.zeros(B, dtype=torch.bool, device=device)
  # per-kick logs (flat lists)
  kick_records: list[dict] = []
  episodes: list[dict] = []

  def goal_dir(ball_xy_local: torch.Tensor) -> torch.Tensor:
    """Unit ray ball -> goal mouth center-clamped (same as goal_progress)."""
    gx = torch.full_like(ball_xy_local[:, 0], GOAL_X)
    gy = ball_xy_local[:, 1].clamp(-GOAL_HALF_W, GOAL_HALF_W)
    to_goal = torch.stack([gx, gy], dim=-1) - ball_xy_local
    return to_goal / (to_goal.norm(dim=-1, keepdim=True) + 1e-6)

  obs = env_w.get_observations()
  scored_latch = torch.zeros(B, dtype=torch.bool, device=device)
  # initialize spawn distances
  rl = robot.data.root_link_pos_w[:, :2] - origin
  spawn_goal_dist[:] = ((rl[:, 0] - GOAL_X) ** 2 + rl[:, 1] ** 2).sqrt()

  print(f"[INFO] forensics rollout: {B} envs x {steps} steps from {CKPT.name}")
  for t in range(steps):
    with torch.no_grad():
      a = policy(obs)
    obs, _, dones, _ = env_w.step(a)

    # IMPORTANT step-ordering: _reset_idx runs INSIDE env.step() before it
    # returns, so for done envs every post-step read (ball pos, goal_scored,
    # contacts) already belongs to the NEW episode. Therefore: (1) latch scoring
    # via newly_scored each step (it fires once at the crossing step, before the
    # goal-triggered termination on the next step); (2) close out done episodes
    # using the accumulators BEFORE folding in this step's readings.
    scored_latch |= cmd.newly_scored > 0

    done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
    if done_ids.numel() > 0:
      for e in done_ids.tolist():
        dx = float(deep_x[e])
        dy = float(y_at_deep[e])
        if bool(scored_latch[e]):
          miss = "SCORED"
        elif dx >= GOAL_X - 0.2 and abs(dy) >= GOAL_HALF_W:
          miss = "WIDE"  # crossed end line but outside the mouth
        elif dx >= 9.0:
          miss = "SHORT"  # reached the box, never crossed
        else:
          miss = "NEVER_ARRIVED"
        episodes.append({
          "scored": bool(scored_latch[e]),
          "miss_type": miss,
          "deep_x": round(dx, 2),
          "y_at_deep": round(dy, 2),
          "spawn_goal_dist": round(float(spawn_goal_dist[e]), 2),
          "kicks": int(kicks[e]),
          "first_kick_t": int(first_kick_t[e]),
          "ep_len": int(ep_t[e]),
        })
        # reset accumulators for the new episode (already underway post-reset)
        deep_x[e] = -1e9
        kicks[e] = 0
        first_kick_t[e] = -1
        ep_t[e] = 0
        scored_latch[e] = False
        prev_contact[e] = False
      rl = robot.data.root_link_pos_w[done_ids, :2] - origin[done_ids]
      spawn_goal_dist[done_ids] = (
        (rl[:, 0] - GOAL_X) ** 2 + rl[:, 1] ** 2
      ).sqrt()

    ep_t += 1

    ball_l = cmd.ball_pos_w[:, :2] - origin
    # deepest-x tracking
    deeper = ball_l[:, 0] > deep_x
    y_at_deep = torch.where(deeper, ball_l[:, 1], y_at_deep)
    deep_x = torch.where(deeper, ball_l[:, 0], deep_x)

    # kick contact rising edge
    assert contact.data.found is not None
    in_c = contact.data.found.sum(dim=-1) > 0
    new_kick = in_c & ~prev_contact
    prev_contact = in_c
    if new_kick.any():
      ids = new_kick.nonzero(as_tuple=False).squeeze(-1)
      bv = cmd.ball_lin_vel_w[ids, :2]
      bl = ball_l[ids]
      gdir = goal_dir(bl)
      speed = bv.norm(dim=-1)
      # aim error: angle between post-contact ball velocity and goalward ray.
      cosang = (bv * gdir).sum(-1) / (speed + 1e-6)
      ang = torch.rad2deg(torch.arccos(cosang.clamp(-1, 1)))
      for j, e in enumerate(ids.tolist()):
        kicks[e] += 1
        if first_kick_t[e] < 0:
          first_kick_t[e] = ep_t[e]
        kick_records.append({
          "x": round(float(bl[j, 0]), 2),
          "y": round(float(bl[j, 1]), 2),
          "aim_err_deg": round(float(ang[j]), 1),
          "speed": round(float(speed[j]), 2),
          "dist_to_goal": round(float(
            ((bl[j, 0] - GOAL_X) ** 2 + bl[j, 1] ** 2).sqrt()
          ), 2),
        })

  env_w.close()

  # ---- aggregate report ----
  n = len(episodes)
  sc = [e for e in episodes if e["scored"]]
  print(f"\n=== {n} goal-targeted episodes, scored {len(sc)} ({len(sc) / max(n, 1):.3f}) ===")
  from collections import Counter
  mt = Counter(e["miss_type"] for e in episodes)
  print("miss types:", dict(mt))

  def stats(vals):
    if not vals:
      return "n=0"
    v = np.array(vals)
    return f"mean={v.mean():.2f} med={np.median(v):.2f} n={len(v)}"

  print("\n-- spawn dist: scored vs not --")
  print("  scored :", stats([e["spawn_goal_dist"] for e in sc]))
  print("  missed :", stats([e["spawn_goal_dist"] for e in episodes if not e["scored"]]))
  print("  NEVER_ARRIVED spawn:", stats([
    e["spawn_goal_dist"] for e in episodes if e["miss_type"] == "NEVER_ARRIVED"
  ]))

  print("\n-- kick aim error by zone (deg, lower=better) --")
  near = [k for k in kick_records if k["x"] > NEAR_GOAL_X]
  mid = [k for k in kick_records if 0 <= k["x"] <= NEAR_GOAL_X]
  back = [k for k in kick_records if k["x"] < 0]
  for name, ks in (("near-goal x>7", near), ("midfield 0..7", mid), ("own-half x<0", back)):
    if ks:
      errs = [k["aim_err_deg"] for k in ks]
      spd = [k["speed"] for k in ks]
      print(f"  {name:14s}: aim_err {stats(errs)} | kick_speed {stats(spd)}")

  print("\n-- WIDE misses: |y| at end line --")
  print("  ", stats([abs(e["y_at_deep"]) for e in episodes if e["miss_type"] == "WIDE"]))

  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps({
    "ckpt": str(CKPT), "episodes": episodes,
    "kicks": kick_records[:2000],  # cap file size
    "summary": {
      "n_episodes": n, "scored": len(sc), "rate": round(len(sc) / max(n, 1), 3),
      "miss_types": dict(mt),
    },
  }, indent=1))
  print(f"\n[INFO] full records -> {OUT}")


if __name__ == "__main__":
  main()
