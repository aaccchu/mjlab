"""v4 EXP20b: CONTINUE EXP20 (aim-the-kick) for 2000 more iters.

EXP20 ended with every aim-symptom improved (pos_err 0.851 / OOB 0.212 /
ball_to_tgt 2.16 all best-ever; shot rate 0.340-0.346) and the shot-rate trend
still rising at iter 2000 (last400 0.324 -> last200 0.346). Same env, same
incentives; bootstrap from EXP20 model_1999 to see how far the aim improvements
carry the shot rate (target >=0.40).

Shot rate plateaued at 0.30-0.33 across EXP16-19 while ball_speed_peak is 3.2 and
ball_to_tgt_err rebounds after kicks: kicks hard, lands scattered — "doesn't aim".
Three verified causes/levers (research log: EXP20 调研与瓶颈诊断):

  1. GAZE PITCH SIGN BUG (verified by rendering the real camera across a
     neck_pitch sweep): positive neck_pitch = camera DOWN, but _gaze_uv_visible
     computed `elev - neck_pitch`, so gaze rewards taught the policy to look UP
     at a close ground ball. Fixed globally in observations.py (codex EXP7C3 hit
     the same bug; their fix took ball-visible 0.015 -> 0.254).
  2. Pose reward clamps neck_pitch (std 0.1/0.15) — relaxed to 1.0 in this env.
  3. kick_lateral_alignment: reward |y_b| -> 0 in the frontal approach band
     (codex C21l2 measured 13x contact-window from this one lever).

Bootstrap: EXP19 model_1999 (deployment model). NO std reset — codex lesson: the
"low-speed ball-hugging attractor" can wash out a kick chain within 50-200 iters;
keep the tuned std and WATCH ball_speed_peak (>2.8 guardrail).

Criteria: real shot rate (goal_rate/target_is_goal) 0.32 -> >=0.40.
Secondary: neck_pitch usage extends positive (looks down), Metrics/kick_lat_align
rises. Guardrails: fell_over<0.1, pos_err<1.1 (looking down trades off landmark
coverage — if pos_err breaks, back off gaze weight), ball_speed_peak>2.8.

Usage: MUJOCO_GL=egl uv run python scripts/spike_v4_e2e_ekf_kick_aim2.py [--smoke]
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
  mos92_soccer_e2e_dualcam_ekf_kick_aim_env_cfg,
)
from mjlab.tasks.velocity.config.mos92.rl_cfg import (
  mos92_selfloc_vision_ppo_runner_cfg,
)
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

BASE_CKPT = Path(
  "logs/rsl_rl/mos92_velocity/2026-06-10_19-01-49_spike_v4_e2e_ekf_kick_aim/model_1999.pt"
)
NUM_ENVS = 384
MAX_ITER = 2000


def _partial_load_shape_match(module, ckpt_sd) -> tuple[int, int, list[str]]:
  """Load every name+shape-matching tensor; reinit the rest. EXP20 keeps the obs
  layout identical to EXP19 so this should load everything (reinit 0)."""
  cur = module.state_dict()
  to_load, notes = {}, []
  reinit = 0
  for k, v in ckpt_sd.items():
    if k not in cur:
      notes.append(f"{k}:absent")
      continue
    if cur[k].shape == v.shape:
      to_load[k] = v
    else:
      reinit += 1
      notes.append(f"{k}:reinit{tuple(v.shape)}->{tuple(cur[k].shape)}")
  module.load_state_dict(to_load, strict=False)
  return len(to_load), reinit, notes


def main() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  smoke = "--smoke" in sys.argv
  configure_torch_backends()
  device = "cuda:0"

  env_cfg = mos92_soccer_e2e_dualcam_ekf_kick_aim_env_cfg(play=False)
  env_cfg.scene.num_envs = 32 if smoke else NUM_ENVS

  gt = env_cfg.curriculum["gt_mask"]
  gt.params["start_step"] = -1
  gt.params["end_step"] = 0
  pr = env_cfg.curriculum["penalty_ramp"]
  pr.params["start_step"] = -1
  pr.params["end_step"] = 0

  max_iter = 8 if smoke else MAX_ITER
  agent_cfg = mos92_selfloc_vision_ppo_runner_cfg()
  agent_cfg.max_iterations = max_iter
  agent_cfg.run_name = "spike_v4_e2e_ekf_kick_aim2"
  agent_cfg.save_interval = 100

  log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_spike_v4_e2e_ekf_kick_aim2")
  log_dir = log_root / stamp
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)
  if BASE_CKPT.exists():
    print(f"[INFO] Partial-load from EXP20: {BASE_CKPT}")
    ckpt = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
    a_load, a_re, a_notes = _partial_load_shape_match(
      runner.alg._raw_actor, ckpt["actor_state_dict"]
    )
    c_load, c_re, c_notes = _partial_load_shape_match(
      runner.alg._raw_critic, ckpt["critic_state_dict"]
    )
    print(f"[INFO] actor : loaded {a_load}, reinit {a_re} -> {a_notes}")
    print(f"[INFO] critic: loaded {c_load}, reinit {c_re} -> {c_notes}")
    # NO std reset (codex "ball-hugging attractor" lesson): keep EXP19's tuned
    # action std; the new incentives reshape gaze/alignment, not the repertoire.
    if a_re != 0 or c_re != 0:
      print(f"[WARN] expected 0 reinit (same obs layout); got a={a_re} c={c_re}")
  else:
    print(f"[WARN] bootstrap ckpt missing ({BASE_CKPT}); training from scratch.")

  mode = "SMOKE" if smoke else "FULL"
  print(
    f"[INFO] v4 EXP20b aim-continue [{mode}]: {max_iter} iters, "
    f"envs={env_cfg.scene.num_envs}"
  )
  print("[INFO] target: shot rate 0.32 -> >=0.40; watch Metrics/kick_lat_align UP,")
  print("[INFO] fell_over<0.1, pos_err<1.1 (look-down vs landmarks!), peak>2.8.")
  runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)
  env.close()
  print(f"[INFO] Done. Logs at: {log_dir}")


if __name__ == "__main__":
  main()
