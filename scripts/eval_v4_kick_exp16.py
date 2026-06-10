"""EXP16 (kick-chain best baseline, pre-OOB-fix) — representative SCENARIO clips.

Loads the archived EXP16 checkpoint (e2e EKF kick, before the EXP17-19
out-of-bounds fix arc) and writes the three dribble-arc clips into
soccer_eval/2026-06-09_v4/kick_exp16/ via scripts/_v4_scenario_video.py:
  - kick_exp16_approach.mp4  : robot walking up to the ball
  - kick_exp16_strike.mp4    : the kick (peak ball speed)
  - kick_exp16_goalward.mp4  : ball driven toward the goal mouth (scored flag in
                               scenarios.json if a real goal was captured)
plus one mid-clip still per scenario and scenarios.json. Side-by-side with the
EXP19 clips this shows the same kick skill; the OOB difference is a steady-state
rate (0.345 vs 0.294), not something a single env0 demo reliably exhibits.

Usage: MUJOCO_GL=egl uv run python scripts/eval_v4_kick_exp16.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _v4_scenario_video import run  # noqa: E402

from mjlab.tasks.velocity.config.mos92.env_cfgs import (  # noqa: E402
  mos92_soccer_e2e_dualcam_ekf_kick_env_cfg,
)

CKPT = Path("checkpoints/v4_soccer/kick_exp16/model_1999.pt")
OUT_DIR = Path("soccer_eval/2026-06-09_v4/kick_exp16")
PREFIX = "kick_exp16"


def main() -> None:
  info = run(mos92_soccer_e2e_dualcam_ekf_kick_env_cfg, CKPT, OUT_DIR, PREFIX)
  (OUT_DIR / "scenarios.json").write_text(json.dumps(info, indent=2))
  print(json.dumps(info, indent=2))


if __name__ == "__main__":
  main()
