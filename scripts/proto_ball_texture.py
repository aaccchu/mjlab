"""Prototype + render a truncated-icosahedron soccer-ball texture (viewer/RGB only).

Pattern: 12 pentagon centers (icosahedron vertices) = RED; 20 hexagon centers
(icosahedron face centroids) = BLUE/WHITE alternating; one hexagon marked darker
for rotation observability. Colored by spherical Voronoi (nearest face center).

Built as a MuJoCo CUBE texture (6 stacked faces). This script renders the ball
to a PNG so we can VISUALLY verify before integrating into soccer_field.py.

Usage: MUJOCO_GL=egl uv run python scripts/proto_ball_texture.py
"""

from __future__ import annotations

import mujoco
import numpy as np


def _face_centers():
  """Return (pentagon_centers[12,3], hexagon_centers[20,3]) on the unit sphere."""
  from scipy.spatial import ConvexHull

  phi = (1.0 + 5.0**0.5) / 2.0
  verts = np.array(
    [
      (0, 1, phi), (0, 1, -phi), (0, -1, phi), (0, -1, -phi),
      (1, phi, 0), (1, -phi, 0), (-1, phi, 0), (-1, -phi, 0),
      (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1),
    ],
    dtype=np.float64,
  )
  verts /= np.linalg.norm(verts, axis=1, keepdims=True)
  hull = ConvexHull(verts)
  hexc = np.array([verts[s].mean(axis=0) for s in hull.simplices])
  hexc /= np.linalg.norm(hexc, axis=1, keepdims=True)
  return verts, hexc


# Colors (RGB 0-255).
_RED = (220, 30, 30)
_BLUE = (30, 60, 200)
_WHITE = (245, 245, 245)
_MARK = (15, 15, 15)  # Single darker hexagon for rotation observability.


def _color_for_direction(dirs, pent, hexc):
  """dirs: (N,3) unit vectors -> (N,3) uint8 RGB by nearest face center."""
  # Cosine similarity to every center; nearest = max dot.
  pent_dot = dirs @ pent.T  # (N,12)
  hex_dot = dirs @ hexc.T  # (N,20)
  best_pent = pent_dot.max(axis=1)
  best_hex_idx = hex_dot.argmax(axis=1)
  best_hex = hex_dot.max(axis=1)

  out = np.zeros((dirs.shape[0], 3), dtype=np.uint8)
  is_pent = best_pent >= best_hex
  out[is_pent] = _RED
  hex_mask = ~is_pent
  hi = best_hex_idx[hex_mask]
  hex_rgb = np.where((hi % 2)[:, None] == 0, np.array(_BLUE), np.array(_WHITE))
  hex_rgb[hi == 0] = _MARK  # Mark hexagon index 0.
  out[hex_mask] = hex_rgb
  return out


def _cube_face_dirs(face, n):
  """Unit direction for each pixel of one cube face. MuJoCo cube layout: the
  texture is 6 square faces stacked vertically (width=n, height=6n)."""
  u = (np.arange(n) + 0.5) / n * 2.0 - 1.0  # [-1,1)
  v = (np.arange(n) + 0.5) / n * 2.0 - 1.0
  uu, vv = np.meshgrid(u, v)
  ones = np.ones_like(uu)
  # MuJoCo cube face order: right, left, up, down, front, back (+x,-x,+y,-y,+z,-z).
  faces = {
    0: (ones, -vv, -uu),   # +x
    1: (-ones, -vv, uu),   # -x
    2: (uu, ones, vv),     # +y
    3: (uu, -ones, -vv),   # -y
    4: (uu, -vv, ones),    # +z
    5: (-uu, -vv, -ones),  # -z
  }
  x, y, z = faces[face]
  d = np.stack([x, y, z], axis=-1).reshape(-1, 3)
  d /= np.linalg.norm(d, axis=1, keepdims=True)
  return d


def build_cube_texture_bytes(n=128):
  """Return (6n, n, 3) uint8 cube texture for the soccer ball."""
  pent, hexc = _face_centers()
  faces = []
  for f in range(6):
    dirs = _cube_face_dirs(f, n)
    rgb = _color_for_direction(dirs, pent, hexc).reshape(n, n, 3)
    faces.append(rgb)
  return np.concatenate(faces, axis=0)  # (6n, n, 3)


def main():
  import imageio.v3 as iio

  n = 128
  tex = build_cube_texture_bytes(n)
  print(f"[tex] shape={tex.shape} dtype={tex.dtype}")

  spec = mujoco.MjSpec()
  spec.add_texture(
    name="ball_tex",
    type=mujoco.mjtTexture.mjTEXTURE_CUBE,
    width=n,
    height=6 * n,
  )
  spec.textures[0].data = tex.tobytes()
  mat = spec.add_material(name="ball_mat")
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "ball_tex"

  spec.worldbody.add_light(pos=(0.3, 0.3, 0.5), dir=(-0.3, -0.3, -0.5))
  b = spec.worldbody.add_body(name="ball", pos=(0, 0, 0))
  b.add_geom(
    type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.11, 0, 0), material="ball_mat"
  )
  # Camera at +x looking back toward the origin: z_cam=+x so -z_cam=-x points at
  # the ball. xyaxes = (x_axis=(0,1,0), y_axis=(0,0,1)) -> z = x_axis X y_axis = +x.
  cam = spec.worldbody.add_camera(
    name="cam", pos=(0.45, 0, 0), xyaxes=(0, 1, 0, 0, 0, 1)
  )

  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  with mujoco.Renderer(model, height=400, width=400) as r:
    r.update_scene(data, camera="cam")
    img = r.render()
  out = "scripts/_ball_texture_preview.png"
  iio.imwrite(out, img)
  print(f"[OK] rendered ball preview -> {out}")


if __name__ == "__main__":
  main()
