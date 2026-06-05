from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  mos92_flat_env_cfg,
  mos92_soccer_env_cfg,
  mos92_soccer_gaze_env_cfg,
  mos92_soccer_goal_env_cfg,
  mos92_soccer_search_env_cfg,
  mos92_soccer_vision_env_cfg,
)
from .rl_cfg import mos92_ppo_runner_cfg, mos92_vision_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-MOS92",
  env_cfg=mos92_flat_env_cfg(),
  play_env_cfg=mos92_flat_env_cfg(play=True),
  rl_cfg=mos92_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Soccer-MOS92",
  env_cfg=mos92_soccer_env_cfg(),
  play_env_cfg=mos92_soccer_env_cfg(play=True),
  rl_cfg=mos92_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# A-MOS92 Group 1: weight=0.3, loose neck std (参考 G1 a2 平衡点)
register_mjlab_task(
  task_id="Mjlab-Velocity-Soccer-Gaze-MOS92",
  env_cfg=mos92_soccer_gaze_env_cfg(gaze_weight=0.3, neck_pose_std_val=10.0),
  play_env_cfg=mos92_soccer_gaze_env_cfg(
    play=True, gaze_weight=0.3, neck_pose_std_val=10.0
  ),
  rl_cfg=mos92_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# A-MOS92 Group 2: weight=1.0, loose neck std
register_mjlab_task(
  task_id="Mjlab-Velocity-Soccer-Gaze-MOS92-W10",
  env_cfg=mos92_soccer_gaze_env_cfg(gaze_weight=1.0, neck_pose_std_val=10.0),
  play_env_cfg=mos92_soccer_gaze_env_cfg(
    play=True, gaze_weight=1.0, neck_pose_std_val=10.0
  ),
  rl_cfg=mos92_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# A-MOS92 Group 3: weight=1.0, tight neck std (限制转头幅度)
register_mjlab_task(
  task_id="Mjlab-Velocity-Soccer-Gaze-MOS92-W10-Tight",
  env_cfg=mos92_soccer_gaze_env_cfg(gaze_weight=1.0, neck_pose_std_val=1.0),
  play_env_cfg=mos92_soccer_gaze_env_cfg(
    play=True, gaze_weight=1.0, neck_pose_std_val=1.0
  ),
  rl_cfg=mos92_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Spike E-MOS92: goal-scoring (dribble the ball into the +x goal mouth)
register_mjlab_task(
  task_id="Mjlab-Velocity-Soccer-Goal-MOS92",
  env_cfg=mos92_soccer_goal_env_cfg(),
  play_env_cfg=mos92_soccer_goal_env_cfg(play=True),
  rl_cfg=mos92_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Spike A2-MOS92: vision-centered gaze + search/track behavior sequence
register_mjlab_task(
  task_id="Mjlab-Velocity-Soccer-Search-MOS92",
  env_cfg=mos92_soccer_search_env_cfg(),
  play_env_cfg=mos92_soccer_search_env_cfg(play=True),
  rl_cfg=mos92_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# v3 Stage 3 (gaze warmup): A2 + head depth camera into actor CNN.
# Actor keeps GT ball vector AND sees depth; critic stays GT-only (asymmetric).
register_mjlab_task(
  task_id="Mjlab-Velocity-Soccer-Vision-MOS92",
  env_cfg=mos92_soccer_vision_env_cfg(),
  play_env_cfg=mos92_soccer_vision_env_cfg(play=True),
  rl_cfg=mos92_vision_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
