"""EXP23 (strike, current-best) — representative SCENARIO demo clips.

Loads the archived EXP23 checkpoint (near-goal kick_impulse x3, bootstrapped from
EXP20b) and writes the three dribble-arc clips into
soccer_eval/2026-06-16_v4_amp/kick_strike_amp_exp24/ via _v4_scenario_video.py:
  - kick_strike_amp_exp24_approach.mp4 : robot walking up to the ball
  - kick_strike_amp_exp24_strike.mp4   : the kick (peak ball speed)
  - kick_strike_amp_exp24_goalward.mp4 : ball driven toward the goal mouth (scored flag
                                     in scenarios.json if a real goal was captured)
plus one mid-clip still per scenario and scenarios.json.

Usage: MUJOCO_GL=egl uv run python scripts/eval_v4_kick_strike_amp_exp24.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _v4_scenario_video import run  # noqa: E402

from mjlab.tasks.velocity.config.mos92.env_cfgs import (  # noqa: E402
  mos92_soccer_e2e_dualcam_ekf_kick_strike_amp_env_cfg,
)

CKPT = Path("checkpoints/v4_soccer/kick_strike_amp_exp24/model_1999.pt")
OUT_DIR = Path("soccer_eval/2026-06-16_v4_amp/kick_strike_amp_exp24")
PREFIX = "kick_strike_amp_exp24"


def main() -> None:
  info = run(
    mos92_soccer_e2e_dualcam_ekf_kick_strike_amp_env_cfg, CKPT, OUT_DIR, PREFIX
  )
  (OUT_DIR / "scenarios.json").write_text(json.dumps(info, indent=2))
  print(json.dumps(info, indent=2))


if __name__ == "__main__":
  main()
