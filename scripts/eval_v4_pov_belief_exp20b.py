"""EXP20b — robot-POV + self-localization belief + ball-knowledge video.

8-panel composite over one episode for the EXP20b aim-the-kick model, into
soccer_eval/2026-06-10_v4/kick_aim_exp20b/:
  - kick_aim_exp20b_pov_belief.mp4
  - pov_belief.json   (episode stats incl. first_ball_sighting_step, kicks,
                       distance, ball_in_view_frac)

Of special interest for THIS model: neck_pitch is now incentivized correctly
(gaze pitch sign fixed), so close-range ball visibility should be better than
EXP16/19 (watch ball_in_view_frac and the look-down behavior in the POV).

Usage: MUJOCO_GL=egl uv run python scripts/eval_v4_pov_belief_exp20b.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _v4_pov_belief_video import run  # noqa: E402

from mjlab.tasks.velocity.config.mos92.env_cfgs import (  # noqa: E402
  mos92_soccer_e2e_dualcam_ekf_kick_aim_env_cfg,
)

CKPT = Path("checkpoints/v4_soccer/kick_aim_exp20b/model_1999.pt")
OUT_DIR = Path("soccer_eval/2026-06-10_v4/kick_aim_exp20b")


def main() -> None:
  info = run(
    mos92_soccer_e2e_dualcam_ekf_kick_aim_env_cfg,
    CKPT,
    OUT_DIR,
    "kick_aim_exp20b",
    "EXP20b (aim-the-kick): robot POV + self-localization belief",
  )
  (OUT_DIR / "pov_belief.json").write_text(json.dumps(info, indent=2))
  print(json.dumps(info, indent=2))


if __name__ == "__main__":
  main()
