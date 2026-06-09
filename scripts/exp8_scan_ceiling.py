"""EXP8 feasibility ceiling (pure geometry, NO sim/GPU): can turning the body
raise unique-keypoint coverage enough for Kabsch to break 1m?

EXP7 proved passive multi-frame fusion is futile: walking forward, ~4 of 23
keypoints stay visible, unique coverage barely moves (3.9->4.4), pos_err only
-13%. The hypothesis for EXP8 (active vision) is that ROTATING the body sweeps the
60-deg camera across DIFFERENT keypoints, raising unique coverage 4->15+ so Kabsch
becomes well-conditioned. This script tests that ceiling analytically: sample
field positions, at each scan yaw over 360deg, compute keypoint visibility from
the pinhole geometry (analytic depth = distance to the known 3D point, so NO
rendered depth needed), and compare single-orientation vs full-scan Kabsch error.

If a full scan reaches ~15+ unique keypoints and sub-1m Kabsch, EXP8 is worth the
cost. If even a full turn can't, the camera FOV / keypoint set is the wall.
"""

from __future__ import annotations

import math

import torch

_HL, _HW = 11.0, 7.0
FOVY = 60.0
W, H = 64, 48
CAM_HEIGHT = 0.5  # approx head-cam world height (robot standing); pitch ~level.


def main() -> None:
  from mjlab.tasks.velocity.mdp.field_keypoints import field_keypoints_3d, kabsch_se2

  dev = "cpu"
  kp = field_keypoints_3d(dev)  # (K,3) field-local, z=0
  K = kp.shape[0]
  f = (H / 2.0) / math.tan(FOVY * math.pi / 360.0)

  # Sample positions across the attacking half (where EXP6 spawns/plays).
  xs = torch.linspace(-2.0, 9.0, 12)
  ys = torch.linspace(-5.0, 5.0, 11)
  gx, gy = torch.meshgrid(xs, ys, indexing="ij")
  pos = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (P,2)
  P = pos.shape[0]
  yaws = torch.linspace(0, 2 * math.pi, 24)[:-1]  # 23 scan orientations

  # Visibility (P, n_yaw, K): keypoint in front + inside 60deg-fovy frame.
  cam_xy = pos  # camera ~ at base xy
  vis = torch.zeros(P, yaws.shape[0], K)
  for yi, yaw in enumerate(yaws):
    # Camera looks along +x of base rotated by yaw; build look dir.
    fwd = torch.tensor([math.cos(yaw), math.sin(yaw)])
    left = torch.tensor([-math.sin(yaw), math.cos(yaw)])
    for ki in range(K):
      d = kp[ki, :2] - cam_xy  # (P,2)
      fdepth = d @ fwd  # forward distance
      lateral = d @ left
      vert_ok = True  # ground points; vertical fov rarely the limiter at z=0
      # horizontal half-angle limit = atan((W/2)/f) but fov set by fovy*aspect;
      # use horizontal fov from aspect ratio.
      hfov = 2 * math.atan((W / H) * math.tan(FOVY * math.pi / 360.0))
      ang = torch.atan2(lateral.abs(), fdepth.clamp(min=1e-3))
      vis[:, yi, ki] = ((fdepth > 0.2) & (ang < hfov / 2) & torch.tensor(vert_ok)).float()

  # Single best orientation vs full scan: unique coverage.
  per_yaw_cov = vis.sum(-1)  # (P, n_yaw) count visible per orientation
  best_single = per_yaw_cov.max(dim=1).values  # (P,)
  full_scan_uniq = (vis.sum(1) > 0).float().sum(-1)  # (P,) union over all yaws

  print(f"RESULT positions={P}, keypoints={K}")
  print(f"RESULT single-orientation visible kp: mean={best_single.mean():.1f} median={best_single.median():.1f}")
  print(f"RESULT FULL-SCAN unique kp: mean={full_scan_uniq.mean():.1f} median={full_scan_uniq.median():.1f} max={full_scan_uniq.max():.0f}")

  # Kabsch error: single-orientation (best) vs full-scan union, analytic 3D points.
  def kabsch_err(use_full):
    errs = []
    for p in range(P):
      if use_full:
        seen = (vis[p].sum(0) > 0)
      else:
        bo = per_yaw_cov[p].argmax()
        seen = vis[p, bo] > 0
      if seen.sum() < 2:
        continue
      src = (kp[seen, :2] - pos[p]).unsqueeze(0)  # base-frame (no rotation needed for err)
      dst = kp[seen, :2].unsqueeze(0)
      w = torch.ones(1, int(seen.sum()))
      _, t = kabsch_se2(src + pos[p], dst, w)  # recover translation
      errs.append(torch.linalg.norm(t[0] - pos[p]))
    return torch.tensor(errs)

  es = kabsch_err(False)
  ef = kabsch_err(True)
  print(f"RESULT single Kabsch pos_err: mean={es.mean():.3f} median={es.median():.3f} (n={len(es)})")
  print(f"RESULT FULL-SCAN Kabsch pos_err: mean={ef.mean():.3f} median={ef.median():.3f} (n={len(ef)})")


if __name__ == "__main__":
  main()
