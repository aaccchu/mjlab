"""EXP20b (best deployment model) — representative SCENARIO demo clips.

Loads the archived EXP20b checkpoint (aim-the-kick: gaze pitch sign fix +
neck_pitch unclamp + lateral alignment) and writes the three dribble-arc clips
into soccer_eval/2026-06-10_v4/kick_aim_exp20b/ via scripts/_v4_scenario_video.py:
  - kick_aim_exp20b_approach.mp4 : robot walking up to the ball
  - kick_aim_exp20b_strike.mp4   : the kick (peak ball speed)
  - kick_aim_exp20b_goalward.mp4 : ball driven toward the goal mouth (scored flag
                                   in scenarios.json if a real goal was captured)
plus one mid-clip still per scenario and scenarios.json.

Usage: MUJOCO_GL=egl uv run python scripts/eval_v4_kick_aim_exp20b.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _v4_scenario_video import run  # noqa: E402

from mjlab.tasks.velocity.config.mos92.env_cfgs import (  # noqa: E402
  mos92_soccer_e2e_dualcam_ekf_kick_aim_env_cfg,
)

CKPT = Path("checkpoints/v4_soccer/kick_aim_exp20b/model_1999.pt")
OUT_DIR = Path("soccer_eval/2026-06-10_v4/kick_aim_exp20b")
PREFIX = "kick_aim_exp20b"


def main() -> None:
  info = run(mos92_soccer_e2e_dualcam_ekf_kick_aim_env_cfg, CKPT, OUT_DIR, PREFIX)
  (OUT_DIR / "scenarios.json").write_text(json.dumps(info, indent=2))
  print(json.dumps(info, indent=2))


if __name__ == "__main__":
  main()
