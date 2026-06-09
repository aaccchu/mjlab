"""Validate FusedPoseBelief: does temporal fusion (in the stateful obs) beat the
single-frame oracle belief during a real rollout? EXP8 gate before training."""
import traceback

import torch


def main(ckpt):
  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.velocity.config.mos92.env_cfgs import (
    mos92_soccer_e2e_dualcam_oracle_env_cfg as f,
  )
  from mjlab.tasks.velocity.config.mos92.rl_cfg import (
    mos92_selfloc_vision_ppo_runner_cfg,
  )
  from mjlab.tasks.velocity.mdp.observations import (
    FusedPoseBelief,
    oracle_pose_belief,
    robot_field_pose,
  )

  dev = "cuda:0"
  cfg = f(play=False)
  cfg.scene.num_envs = 64
  env = ManagerBasedRlEnv(cfg=cfg, device=dev)
  agent = mos92_selfloc_vision_ppo_runner_cfg()
  env_w = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = MjlabOnPolicyRunner(env_w, asdict(agent), "/tmp/exp8_dummy", dev)
  ck = torch.load(ckpt, map_location=dev, weights_only=False)
  runner.alg._raw_actor.load_state_dict(ck["actor_state_dict"], strict=False)
  policy = runner.get_inference_policy(dev)

  fused = FusedPoseBelief(env, "dribble", "head_cam", num_frames=8, stride=4)
  cmd = env.command_manager.get_term("dribble")
  HL, HW = cmd.cfg.half_length, cmd.cfg.half_width

  obs, _ = env_w.reset()
  fused.reset(slice(None))
  s_err, f_err, s_uniq, f_uniq = [], [], [], []
  for t in range(150):
    with torch.inference_mode():
      act = policy(obs)
    obs, _, _, _ = env_w.step(act)
    single = oracle_pose_belief(env, "dribble")  # (B,6)
    fb = fused(env, "dribble", "head_cam", 8, 4)  # (B,7)
    gt = robot_field_pose(env, "dribble")
    gx, gy = gt[:, 0] * HL, gt[:, 1] * HW
    if t < 30:
      continue  # let the window fill
    se = torch.sqrt((single[:, 0] * HL - gx) ** 2 + (single[:, 1] * HW - gy) ** 2)
    fe = torch.sqrt((fb[:, 0] * HL - gx) ** 2 + (fb[:, 1] * HW - gy) ** 2)
    s_err.append(se)
    f_err.append(fe)
    s_uniq.append(single[:, 4])
    f_uniq.append(fb[:, 5])
  se = torch.cat(s_err)
  fe = torch.cat(f_err)
  print(f"RESULT single belief pos_err: mean={se.mean():.3f} median={se.median():.3f}")
  print(f"RESULT FUSED  belief pos_err: mean={fe.mean():.3f} median={fe.median():.3f}")
  print(f"RESULT improvement: {(1-fe.mean()/se.mean())*100:.1f}%")
  print(f"RESULT vis_frac single={torch.cat(s_uniq).mean():.3f} uniq_frac fused={torch.cat(f_uniq).mean():.3f}")
  env.close()


if __name__ == "__main__":
  import sys
  try:
    main(sys.argv[1])
  except Exception:
    traceback.print_exc()
