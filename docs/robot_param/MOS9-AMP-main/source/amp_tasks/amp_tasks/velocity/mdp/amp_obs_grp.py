from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import (
  body_ang_vel_b,
  body_ang_vel_w,
  body_lin_vel_b,
  body_lin_vel_w,
  body_pose_w,
  body_quat_b,
  body_quat_w,
  joint_pos,
  joint_vel,
  root_ang_vel_b,
  root_lin_vel_b,
)


@configclass
class AMPObsGrpCfg(ObsGroup):
  def adjust_key_body_indexes(self, terms: list, key_bodys: list):
    for term_name in terms:
      term: ObsTerm = getattr(self, term_name)
      if "asset_cfg" in term.params:
        term.params["asset_cfg"].body_names = key_bodys
        term.params["asset_cfg"].preserve_order = True
      else:
        term.params["asset_cfg"] = SceneEntityCfg(
          name="robot", body_names=key_bodys, preserve_order=True
        )
    return self

  def adjust_key_joint_and_body_indexes(
    self,
    joint_terms: list,
    body_terms: list,
    key_joints: list,
    key_bodys: list,
  ):
    for term_name in joint_terms:
      term: ObsTerm = getattr(self, term_name)
      if "asset_cfg" in term.params:
        term.params["asset_cfg"].joint_names = key_joints
        term.params["asset_cfg"].preserve_order = True
      else:
        term.params["asset_cfg"] = SceneEntityCfg(
          name="robot", joint_names=key_joints, preserve_order=True
        )

    for term_name in body_terms:
      term: ObsTerm = getattr(self, term_name)
      if "asset_cfg" in term.params:
        term.params["asset_cfg"].body_names = key_bodys
        term.params["asset_cfg"].preserve_order = True
      else:
        term.params["asset_cfg"] = SceneEntityCfg(
          name="robot", body_names=key_bodys, preserve_order=True
        )

    return self


@configclass
class AMPObsJointPosCfg(AMPObsGrpCfg):
  joint_pos = ObsTerm(func=joint_pos)


AMPObsJointPosTerms = ["joint_pos"]


@configclass
class AMPObsBaiscCfg(AMPObsGrpCfg):
  joint_pos = ObsTerm(func=joint_pos)
  joint_vel = ObsTerm(func=joint_vel)


AMPObsBaiscTerms = ["joint_pos", "joint_vel"]


@configclass
class AMPObsSoft1(AMPObsGrpCfg):
  joint_pos = ObsTerm(func=joint_pos)
  joint_vel = ObsTerm(func=joint_vel)
  body_lin_vel_b = ObsTerm(func=body_lin_vel_b)


AMPObsSoft1Terms = ["joint_pos", "joint_vel", "body_lin_vel_b"]


@configclass
class AMPObsSoft1BaseVelBCfg(AMPObsGrpCfg):
  joint_pos = ObsTerm(func=joint_pos)
  joint_vel = ObsTerm(func=joint_vel)
  body_lin_vel_b = ObsTerm(func=body_lin_vel_b)
  root_lin_vel_b = ObsTerm(func=root_lin_vel_b)
  root_ang_vel_b = ObsTerm(func=root_ang_vel_b)


AMPObsSoft1BaseVelBTerms = [
  "joint_pos",
  "joint_vel",
  "body_lin_vel_b",
  "root_lin_vel_b",
  "root_ang_vel_b",
]


@configclass
class AMPObsSoftTrackCfg(AMPObsGrpCfg):
  joint_pos = ObsTerm(func=joint_pos)
  joint_vel = ObsTerm(func=joint_vel)
  body_quat_w = ObsTerm(func=body_quat_w)
  body_lin_vel_w = ObsTerm(func=body_lin_vel_w)
  body_ang_vel_w = ObsTerm(func=body_ang_vel_w)


AMPObsSoftTrackTerms = [
  "joint_pos",
  "joint_vel",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
]


@configclass
class AMPObsSoftTrackLocalCfg(AMPObsGrpCfg):
  joint_pos = ObsTerm(func=joint_pos)
  joint_vel = ObsTerm(func=joint_vel)
  body_quat_b = ObsTerm(func=body_quat_b)
  body_lin_vel_b = ObsTerm(func=body_lin_vel_b)
  body_ang_vel_b = ObsTerm(func=body_ang_vel_b)


AMPObsSoftTrackLocalTerms = [
  "joint_pos",
  "joint_vel",
  "body_quat_b",
  "body_lin_vel_b",
  "body_ang_vel_b",
]


@configclass
class AMPObsHardTrackCfg(AMPObsGrpCfg):
  joint_pos = ObsTerm(func=joint_pos)
  joint_vel = ObsTerm(func=joint_vel)
  body_pos_w = ObsTerm(func=body_pose_w)
  body_quat_w = ObsTerm(func=body_quat_w)
  body_lin_vel_w = ObsTerm(func=body_lin_vel_w)
  body_ang_vel_w = ObsTerm(func=body_ang_vel_w)


AMPObsHardTrackTerms = [
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
]
