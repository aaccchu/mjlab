"""EXP23 (strike, current-best) — robot-POV + self-localization belief video.

8-panel composite over one episode for the EXP23 near-goal-strike model, into
soccer_eval/2026-06-11_v4_showdown/kick_strike_exp23/:
  - kick_strike_exp23_pov_belief.mp4
  - pov_belief.json   (episode stats incl. first_ball_sighting_step, kicks,
                       distance, ball_in_view_frac)

Usage: MUJOCO_GL=egl uv run python scripts/eval_v4_pov_belief_exp23.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _v4_pov_belief_video import run  # noqa: E402

from mjlab.tasks.velocity.config.mos92.env_cfgs import (  # noqa: E402
  mos92_soccer_e2e_dualcam_ekf_kick_strike_env_cfg,
)

CKPT = Path("checkpoints/v4_soccer/kick_strike_exp23/model_1999.pt")
OUT_DIR = Path("soccer_eval/2026-06-11_v4_showdown/kick_strike_exp23")


def main() -> None:
  info = run(
    mos92_soccer_e2e_dualcam_ekf_kick_strike_env_cfg,
    CKPT,
    OUT_DIR,
    "kick_strike_exp23",
    "EXP23 (near-goal strike): robot POV + self-localization belief",
  )
  (OUT_DIR / "pov_belief.json").write_text(json.dumps(info, indent=2))
  print(json.dumps(info, indent=2))


if __name__ == "__main__":
  main()
