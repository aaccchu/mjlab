"""Task B (MOS92): jump-fix finetune from the v3b pure-vision checkpoint.

Continues from the v3b GT-ablation checkpoint (model_2999 = a pure-vision policy:
depth CNN carries ball bearing, GT ball obs fully masked) and finetunes with the
new `flight_phase` reward active, so the policy learns to TURN BY STEPPING instead
of hopping. The gt_mask curriculum is pinned to factor=0 from step 0 (start=-1,
end=0) so the finetune NEVER re-introduces the GT crutch that v3b removed.

Success signal: Metrics/flight_phase_frac falls toward 0 while dribble_success /
gaze_center / upright hold. Verify on rendered video that reorientation is done
by stepping, not jumping.

Usage:
  MUJOCO_GL=egl uv run python scripts/spike_v3d_jumpfix.py            # full run
  MUJOCO_GL=egl uv run python scripts/spike_v3d_jumpfix.py --smoke    # 8-iter smoke
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.mos92.env_cfgs import (
  mos92_soccer_vision_ablation_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import mos92_vision_ppo_runner_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

BASE_CKPT = Path(
  "logs/rsl_rl/mos92_velocity/2026-06-05_13-40-53_spike_v3b_ablation/model_2999.pt"
)
NUM_ENVS = 1024
MAX_ITER = 1500


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  smoke = "--smoke" in sys.argv
  configure_torch_backends()
  device = "cuda:0"

  env_cfg = mos92_soccer_vision_ablation_env_cfg(play=False)
  env_cfg.scene.num_envs = 32 if smoke else NUM_ENVS

  # CRITICAL: pin the GT-ablation mask to factor=0 from step 0. The finetune's
  # step counter restarts at 0, and v3b's mask holds factor=1 until step 500*24
  # then ramps down — reusing it verbatim would RE-INTRODUCE the GT ball crutch
  # that v3b/v3c removed. start_step=-1, end_step=0 => step(>=0) >= end_step =>
  # factor=0 always: GT ball obs stay fully masked for the entire finetune.
  gt_mask = env_cfg.curriculum["gt_mask"]
  gt_mask.params["start_step"] = -1
  gt_mask.params["end_step"] = 0

  max_iter = 8 if smoke else MAX_ITER
  agent_cfg = mos92_vision_ppo_runner_cfg()
  agent_cfg.max_iterations = max_iter
  agent_cfg.run_name = "spike_v3d_jumpfix"
  agent_cfg.save_interval = 100

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_v3d_jumpfix")
  log_dir = log_root / stamp
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)

  # Strict load: v3b checkpoint has the exact CNN-actor + GT-critic layout this
  # env produces (mask keeps obs dims unchanged). Fresh optimizer/iteration so
  # the new flight_phase reward can reshape the policy.
  print(f"[INFO] Bootstrapping (strict) from: {BASE_CKPT}")
  ckpt = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
  runner.alg._raw_actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
  runner.alg._raw_critic.load_state_dict(ckpt["critic_state_dict"], strict=True)
  print("[INFO] Actor (CNN) + critic loaded strict; GT mask pinned to 0.")

  mode = "SMOKE" if smoke else "FULL"
  print(
    f"[INFO] Task B jump-fix [{mode}]: {max_iter} iters, envs={env_cfg.scene.num_envs}"
  )
  print("[INFO] watch Metrics/flight_phase_frac -> 0 while dribble/gaze/upright hold.")
  runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)

  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
