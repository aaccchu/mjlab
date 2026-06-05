"""Objective dribble-vs-trap check for the v3d jump-fix policy (camera-only).

The render frames show the robot straddling the ball (ball under the torso).
Still frames can't distinguish "trapping the ball under the body" from "dribbling
it forward". This rolls out the policy in the same camera-only ablation condition
and logs, per step:
  - ball world xy, robot base xy, target xy
  - whether the ball is "under" the robot (xy distance base<->ball < UNDER_R)
and reports, per env:
  - ball path length (how far the ball actually travelled)
  - net ball->target progress (start dist - end dist)
  - fraction of steps the ball was under the robot

Trapping  => high under-fraction, low path length, low/negative net progress.
Dribbling => ball path length large, net progress positive, under-fraction modest.

Usage: MUJOCO_GL=egl uv run python scripts/check_v3d_dribble.py <run_dir>
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.mos92.env_cfgs import mos92_soccer_vision_ablation_env_cfg
from mjlab.tasks.velocity.config.mos92.rl_cfg import mos92_vision_ppo_runner_cfg
from mjlab.tasks.velocity.mdp.dribble_command import DribbleCommand, DribbleCommandCfg
from mjlab.utils.torch import configure_torch_backends

RUN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else None
NUM_ENVS = 64
N_STEPS = 500
SETTLE = 50
UNDER_R = 0.25  # ball within this xy radius of base => "under the robot"
GT_BALL_TERMS = ("robot_to_ball", "ball_velocity", "ball_gaze_uv")
