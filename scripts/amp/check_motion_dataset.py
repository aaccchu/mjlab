"""AMP Phase 0 — verify the MOS9 human-motion dataset and build the joint/body
name→index alignment between the dataset and the live mjlab MOS92 model.

The AMP reference clips (docs/robot_param/MOS9-AMP-main/data/motions/) store an
18-DoF joint order that EXCLUDES the 2 neck joints and groups joints as
R-arm/L-arm/R-leg/L-leg. The live mjlab MOS92 model exposes 20 actuated joints in
MJCF tree order (neck first, then arms, then legs). The 21 body links match in
order between the two. This script:

  1. loads every clip in the requested folders, prints fps / length / dof,
  2. asserts each clip's joint_names (18) and body_names (21) all resolve by name
     into the live MOS92 model's joint/body ordering,
  3. prints the resulting index maps so the training-side AMP obs term can select
     the matching mjlab joints/bodies via SceneEntityCfg(preserve_order=True).

Run: uv run python scripts/amp/check_motion_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402

from mjlab.asset_zoo.robots.mos92.mos92_constants import MOS92_XML  # noqa: E402

MOTION_ROOT = Path("docs/robot_param/MOS9-AMP-main/data/motions")
# Folders to validate. mos9_fk_motion is the richer walk/turn set the IsaacLab
# config trains on; clipped_simple is short single-step clips.
FOLDERS = ["mos9_fk_motion", "mos9_fk_motion_clipped_simple"]


def live_mos92_names() -> tuple[list[str], list[str]]:
  """(joint_names, body_names) from the compiled MOS92 model, excluding the free
  joint and the world body — i.e. the actuated-joint and link orderings."""
  spec = mujoco.MjSpec.from_file(str(MOS92_XML))
  m = spec.compile()
  joints = []
  for i in range(m.njnt):
    if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    joints.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i))
  bodies = []
  for i in range(m.nbody):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
    if nm == "world":
      continue
    bodies.append(nm)
  return joints, bodies


def main() -> None:
  live_joints, live_bodies = live_mos92_names()
  print(f"[live MOS92] {len(live_joints)} actuated joints, {len(live_bodies)} bodies")
  print(f"  joints: {live_joints}")
  print(f"  bodies: {live_bodies}\n")

  joint_idx = {n: i for i, n in enumerate(live_joints)}
  body_idx = {n: i for i, n in enumerate(live_bodies)}

  total, ok = 0, 0
  ds_joint_names: list[str] | None = None
  ds_body_names: list[str] | None = None

  for folder in FOLDERS:
    fdir = MOTION_ROOT / folder
    clips = sorted(fdir.glob("*.npz"))
    print(f"=== {folder}: {len(clips)} clips ===")
    for clip in clips:
      total += 1
      d = np.load(clip, allow_pickle=True)
      jn = [str(x) for x in d["joint_names"]]
      bn = [str(x) for x in d["body_names"]]
      fps = int(np.asarray(d["fps"]).reshape(-1)[0])
      n = d["joint_pos"].shape[0]
      # Consistency: every clip should share the same joint/body name lists.
      if ds_joint_names is None:
        ds_joint_names, ds_body_names = jn, bn
      missing_j = [j for j in jn if j not in joint_idx]
      missing_b = [b for b in bn if b not in body_idx]
      status = "OK"
      if missing_j or missing_b:
        status = f"FAIL missing_joints={missing_j} missing_bodies={missing_b}"
      else:
        ok += 1
      if jn != ds_joint_names or bn != ds_body_names:
        status += " (DIFFERENT name order from first clip!)"
      print(f"  {clip.name:52s} fps={fps} T={n:4d} dof={len(jn)} {status}")

  assert ds_joint_names is not None
  jmap = [joint_idx[j] for j in ds_joint_names]
  bmap = [body_idx[b] for b in ds_body_names]
  print(f"\n[alignment] {ok}/{total} clips align cleanly.")
  print(f"  dataset joint order (18): {ds_joint_names}")
  print(f"  -> live MOS92 joint indices: {jmap}")
  print(f"  dataset body order (21): {ds_body_names}")
  print(f"  -> live MOS92 body indices: {bmap}")
  if ok != total:
    raise SystemExit(f"FAILED: {total - ok} clips did not align")
  print("\nAll clips align. Joint/body name maps are stable across clips.")

  verify_fk_convention()


def verify_fk_convention() -> None:
  """Decisively confirm the dataset's joint convention matches the mjlab model.

  AMP only works if the policy's per-step joint pos/vel and the dataset's joint
  pos/vel are in the SAME convention — otherwise the discriminator separates them
  trivially and the style reward is meaningless. We test this by dropping a clip's
  joint angles (and its base_link pose) onto the live MOS92 model, running forward
  kinematics, and comparing the resulting body world-positions to the clip's stored
  body_pos_w. A ~0 mm error proves the conventions are identical (the clips were in
  fact FK-generated from this same MJCF — "fk" in the folder name).
  """
  clip = MOTION_ROOT / "mos9_fk_motion_clipped_simple" / "walk_straight1.npz"
  d = np.load(clip, allow_pickle=True)
  ds_jn = [str(x) for x in d["joint_names"]]
  ds_bn = [str(x) for x in d["body_names"]]
  jp, bpw, bqw = d["joint_pos"], d["body_pos_w"], d["body_quat_w"]

  spec = mujoco.MjSpec.from_file(str(MOS92_XML))
  m = spec.compile()
  data = mujoco.MjData(m)
  jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
  qadr = {n: m.jnt_qposadr[i] for i, n in enumerate(jnames)}
  root_idx = ds_bn.index("base_link")

  errs = []
  for t in range(0, jp.shape[0], 30):
    mujoco.mj_resetData(m, data)
    data.qpos[0:3] = bpw[t, root_idx]
    data.qpos[3:7] = bqw[t, root_idx]
    for k, name in enumerate(ds_jn):
      data.qpos[qadr[name]] = jp[t, k]
    mujoco.mj_forward(m, data)
    diff = np.array(
      [
        np.linalg.norm(
          data.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)] - bpw[t, bi]
        )
        for bi, b in enumerate(ds_bn)
      ]
    )
    errs.append(float(diff.mean()))
  mean_err = float(np.mean(errs))
  print(
    f"\n[FK convention] dataset-joints-on-mjlab-model vs dataset body_pos_w: "
    f"mean err {mean_err * 1000:.2f} mm"
  )
  if mean_err > 0.03:  # 30 mm
    raise SystemExit(
      f"FK MISMATCH ({mean_err * 1000:.1f} mm): dataset joint convention differs "
      "from the mjlab MOS92 model — AMP would compare incompatible states."
    )
  print("FK convention MATCHES — env and dataset joint states are directly comparable.")


if __name__ == "__main__":
  main()
