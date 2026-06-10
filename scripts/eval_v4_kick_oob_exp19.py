"""EXP19 (representative deployment model) — representative SCENARIO demo clips.

Loads the archived EXP19 checkpoint (e2e EKF kick + out-of-bounds fix + strong
soft-boundary shaping) and writes the three dribble-arc clips into
soccer_eval/2026-06-09_v4/kick_oob_exp19/ via scripts/_v4_scenario_video.py:
  - kick_oob_exp19_approach.mp4  : robot walking up to the ball
  - kick_oob_exp19_strike.mp4    : the kick (peak ball speed)
  - kick_oob_exp19_goalward.mp4  : ball driven toward the goal mouth (scored flag
                                   in scenarios.json if a real goal was captured)
plus one mid-clip still per scenario, and writes the picked windows to
scenarios.json. The quantitative metrics.json (EXP16-19 comparison) is NOT
touched.

Usage: MUJOCO_GL=egl uv run python scripts/eval_v4_kick_oob_exp19.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _v4_scenario_video import run  # noqa: E402

from mjlab.tasks.velocity.config.mos92.env_cfgs import (  # noqa: E402
  mos92_soccer_e2e_dualcam_ekf_kick_oob_soft2_env_cfg,
)

CKPT = Path("checkpoints/v4_soccer/kick_oob_exp19/model_1999.pt")
OUT_DIR = Path("soccer_eval/2026-06-09_v4/kick_oob_exp19")
PREFIX = "kick_oob_exp19"


def main() -> None:
  info = run(mos92_soccer_e2e_dualcam_ekf_kick_oob_soft2_env_cfg, CKPT, OUT_DIR, PREFIX)
  (OUT_DIR / "scenarios.json").write_text(json.dumps(info, indent=2))
  print(json.dumps(info, indent=2))


if __name__ == "__main__":
  main()
