"""v4 EXP5c: field keypoint 3D map + Kabsch self-localization geometry.

Paradigm shift from regression: instead of a CNN regressing continuous pose
(x,y,yaw) — which collapsed once the GT-pose crutch faded across all v4 EXP1-5b
runs — the CNN only DETECTS field keypoints (2D pixel coords). Pose is then
solved geometrically: lift detected pixels to camera-frame 3D via depth, then
Kabsch rigid-fit against this KNOWN field map. The detection supervision is the
per-frame projection of these 3D points (always available, never masked), so the
net learns a stable image->keypoint map with no fade-out crutch to lose.

All coordinates are FIELD-LOCAL meters (env origin subtracted), z = ground plane
(0.0) for painted marks. Mirrors src/mjlab/terrains/soccer_field.py SoccerFieldCfg
defaults: half_length=11, half_width=7, center_circle_radius=2, goal_area
0.75x1.95, penalty_area 2.25x3.45, penalty_mark |x|=7.4, goal opening +-1.
"""

from __future__ import annotations

import torch

# Field dims (meters) — keep in sync with SoccerFieldCfg defaults.
_HL = 11.0  # half_length, x in [-11, 11]
_HW = 7.0  # half_width,  y in [-7, 7]
_CC = 2.0  # center_circle_radius
_GA_D = 0.75  # goal_area_depth
_GA_W = 1.95  # goal_area_half_width
_PA_D = 2.25  # penalty_area_depth
_PA_W = 3.45  # penalty_area_half_width
_PM = 7.4  # penalty_mark |x|
_GOAL_W = 1.0  # goal_inner_half_width (opening y in [-1, 1])


# Geometrically salient, map-known points (field-local meters, z=0 ground).
# Symmetric across both halves; Kabsch fits the subset currently visible.
_KEYPOINTS_XY = [
    # Field corners (4)
    (_HL, _HW), (_HL, -_HW), (-_HL, _HW), (-_HL, -_HW),
    # Midline x=0 endpoints at the touchlines (2)
    (0.0, _HW), (0.0, -_HW),
    # Center mark + center-circle x-axis intersections (3)
    (0.0, 0.0), (_CC, 0.0), (-_CC, 0.0),
    # Penalty marks (2)
    (_PM, 0.0), (-_PM, 0.0),
    # +x penalty-area corners (outer edge x=HL-PA_D) (2)
    (_HL - _PA_D, _PA_W), (_HL - _PA_D, -_PA_W),
    # -x penalty-area corners (2)
    (-(_HL - _PA_D), _PA_W), (-(_HL - _PA_D), -_PA_W),
    # +x goal-area corners (outer edge x=HL-GA_D) (2)
    (_HL - _GA_D, _GA_W), (_HL - _GA_D, -_GA_W),
    # -x goal-area corners (2)
    (-(_HL - _GA_D), _GA_W), (-(_HL - _GA_D), -_GA_W),
    # Goal-post bases on each goal line (4)
    (_HL, _GOAL_W), (_HL, -_GOAL_W), (-_HL, _GOAL_W), (-_HL, -_GOAL_W),
]
NUM_KEYPOINTS = len(_KEYPOINTS_XY)  # 23


def field_keypoints_3d(device: torch.device | str) -> torch.Tensor:
    """(K, 3) field-local 3D keypoint map in meters, z=0 ground plane."""
    xy = torch.tensor(_KEYPOINTS_XY, dtype=torch.float32, device=device)
    z = torch.zeros(xy.shape[0], 1, dtype=torch.float32, device=device)
    return torch.cat([xy, z], dim=-1)


def project_keypoints(
    kp_world: torch.Tensor,
    cam_pos: torch.Tensor,
    cam_mat: torch.Tensor,
    fovy_deg: float,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project (K,3) field keypoints into each env's camera image.

    Generates the per-frame 2D supervision labels for the detection head. MuJoCo
    camera looks down its local -Z, with +X right and +Y up. Returns:
      uv:  (B, K, 2) pixel coords, normalized to [-1, 1] (matches SpatialSoftmax).
      vis: (B, K) bool — point is in front of the cam AND inside the frame.

    Args:
      kp_world: (B, K, 3) keypoints in WORLD frame (env origin already added).
      cam_pos:  (B, 3) camera world position.
      cam_mat:  (B, 3, 3) camera world rotation (columns = cam x,y,z axes in world).
      fovy_deg: vertical field of view in degrees.
    """
    # World point -> camera frame: p_cam = R^T (p_world - cam_pos).
    rel = kp_world - cam_pos.unsqueeze(1)  # (B, K, 3)
    p_cam = torch.einsum("bij,bkj->bki", cam_mat.transpose(1, 2), rel)  # (B, K, 3)
    x_c, y_c, z_c = p_cam[..., 0], p_cam[..., 1], p_cam[..., 2]
    # Camera looks down -Z; depth in front is -z_c > 0.
    depth = -z_c
    eps = 1e-6
    in_front = depth > eps
    # Pinhole with vertical fovy. f in pixels.
    f = (height / 2.0) / torch.tan(
        torch.tensor(fovy_deg, device=kp_world.device) * 3.14159265 / 360.0
    )
    u = f * (x_c / depth.clamp(min=eps)) + width / 2.0
    v = -f * (y_c / depth.clamp(min=eps)) + height / 2.0
    in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    vis = in_front & in_frame
    # Normalize to [-1, 1].
    u_n = (u / (width - 1)) * 2.0 - 1.0
    v_n = (v / (height - 1)) * 2.0 - 1.0
    uv = torch.stack([u_n, v_n], dim=-1)
    return uv, vis


def kabsch_se2(
    src_xy: torch.Tensor,
    dst_xy: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted Kabsch fit of a planar rigid transform (SE2): src -> dst.

    Solves for rotation yaw + translation t minimizing sum w_k |R*src_k + t - dst_k|^2
    over the 2D ground plane (keypoints are all z=0, so the field-plane fit is 2D).
    Used at inference: src = detected keypoints lifted to WORLD-frame xy via depth,
    dst = known map xy. The recovered (t, yaw) IS the robot/camera field pose — no
    learned regression, a closed-form geometric solve.

    Args:
      src_xy:  (B, K, 2) source points (depth-lifted detections, world xy).
      dst_xy:  (B, K, 2) target map points.
      weights: (B, K) non-negative per-point weights (detection confidence * vis).
    Returns:
      yaw: (B,) recovered rotation angle.
      t:   (B, 2) recovered translation.
    """
    w = weights.clamp(min=0.0)
    wsum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (B,1)
    mu_s = (w.unsqueeze(-1) * src_xy).sum(dim=1) / wsum  # (B,2)
    mu_d = (w.unsqueeze(-1) * dst_xy).sum(dim=1) / wsum
    s_c = src_xy - mu_s.unsqueeze(1)
    d_c = dst_xy - mu_d.unsqueeze(1)
    # Cross-covariance H = sum w * s_c^T d_c (2x2).
    H = torch.einsum("bk,bki,bkj->bij", w, s_c, d_c)  # (B,2,2)
    # Closed-form 2D rotation from H: yaw = atan2(H01 - H10, H00 + H11).
    yaw = torch.atan2(H[:, 0, 1] - H[:, 1, 0], H[:, 0, 0] + H[:, 1, 1])
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    # R = [[cos,-sin],[sin,cos]]; t = mu_d - R mu_s.
    Rx = cos * mu_s[:, 0] - sin * mu_s[:, 1]
    Ry = sin * mu_s[:, 0] + cos * mu_s[:, 1]
    t = mu_d - torch.stack([Rx, Ry], dim=-1)
    return yaw, t


def lift_pixels_to_world(
    uv_n: torch.Tensor,
    depth_at_uv: torch.Tensor,
    cam_pos: torch.Tensor,
    cam_mat: torch.Tensor,
    fovy_deg: float,
    width: int,
    height: int,
) -> torch.Tensor:
    """Back-project normalized pixels + metric depth to WORLD-frame 3D points.

    Inverse of project_keypoints' pinhole. uv_n in [-1,1], depth_at_uv in meters
    (sampled from the RAW metric depth buffer at each detected keypoint pixel).
    Returns (B, K, 3) world coords for the Kabsch fit.
    """
    u = (uv_n[..., 0] + 1.0) * 0.5 * (width - 1)
    v = (uv_n[..., 1] + 1.0) * 0.5 * (height - 1)
    f = (height / 2.0) / torch.tan(
        torch.tensor(fovy_deg, device=uv_n.device) * 3.14159265 / 360.0
    )
    # Camera frame: looks down -Z, +X right, +Y up.
    x_c = (u - width / 2.0) / f * depth_at_uv
    y_c = -(v - height / 2.0) / f * depth_at_uv
    z_c = -depth_at_uv
    p_cam = torch.stack([x_c, y_c, z_c], dim=-1)  # (B, K, 3)
    # Camera -> world: p_world = R p_cam + cam_pos.
    p_world = torch.einsum("bij,bkj->bki", cam_mat, p_cam) + cam_pos.unsqueeze(1)
    return p_world




