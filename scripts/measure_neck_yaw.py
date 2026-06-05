"""测量 A-MOS92 gaze 策略的 neck_yaw 转头幅度。"""

import numpy as np
import torch
from dataclasses import asdict
from pathlib import Path

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.envs import ManagerBasedRlEnv

TASK = "Mjlab-Velocity-Soccer-Gaze-MOS92-W10-Tight"
CKPT = "logs/rsl_rl/mos92_velocity/2026-06-02_13-33-44/model_3998.pt"
DEVICE = "cuda:0"
N_STEPS = 400

env_cfg = load_env_cfg(TASK, play=True)
agent_cfg = load_rl_cfg(TASK)
env_cfg.scene.num_envs = 16

env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device=DEVICE)
runner.load(CKPT, load_cfg={"actor": True}, strict=True, map_location=DEVICE)
policy = runner.get_inference_policy(device=DEVICE)

robot = env.unwrapped.scene["robot"]
neck_idx = robot.joint_names.index("neck_yaw")
print(f"neck_yaw joint index: {neck_idx}")

obs = env.get_observations()
neck_angles = []
with torch.inference_mode():
    for _ in range(N_STEPS):
        act = policy(obs)
        obs, _, _, _ = env.step(act)
        neck_q = robot.data.joint_pos[:, neck_idx]
        neck_angles.append(neck_q.cpu().numpy())

arr = np.array(neck_angles)  # (steps, envs)
deg = np.degrees(arr)
print(f"neck_yaw 角度统计 (deg) over {N_STEPS} steps x {arr.shape[1]} envs:")
print(f"  mean abs: {np.abs(deg).mean():.1f}")
print(f"  std:      {deg.std():.1f}")
print(f"  min/max:  {deg.min():.1f} / {deg.max():.1f}")
print(f"  p90 abs:  {np.percentile(np.abs(deg), 90):.1f}")
print(f"  范围>10deg的时间占比: {(np.abs(deg) > 10).mean()*100:.0f}%")
print(f"  范围>20deg的时间占比: {(np.abs(deg) > 20).mean()*100:.0f}%")
