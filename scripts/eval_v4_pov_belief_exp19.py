"""EXP19 — robot-POV + self-localization belief video (the localization story).

Loads the EXP19 deployment checkpoint and writes a 3-panel composite video
(robot camera POV | top-down GT-vs-belief field map | loc-error & coverage
curves) over one episode, into soccer_eval/2026-06-09_v4/kick_oob_exp19/:
  - kick_oob_exp19_pov_belief.mp4
  - pov_belief.json   (episode stats: scored, final/mean loc err, coverage)

Usage: MUJOCO_GL=egl uv run python scripts/eval_v4_pov_belief_exp19.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _v4_pov_belief_video import run  # noqa: E402

from mjlab.tasks.velocity.config.mos92.env_cfgs import (  # noqa: E402
  mos92_soccer_e2e_dualcam_ekf_kick_oob_soft2_env_cfg,
)

CKPT = Path("checkpoints/v4_soccer/kick_oob_exp19/model_1999.pt")
OUT_DIR = Path("soccer_eval/2026-06-09_v4/kick_oob_exp19")


def main() -> None:
  info = run(
    mos92_soccer_e2e_dualcam_ekf_kick_oob_soft2_env_cfg,
    CKPT,
    OUT_DIR,
    "kick_oob_exp19",
    "EXP19 (deployment): robot POV + self-localization belief",
  )
  (OUT_DIR / "pov_belief.json").write_text(json.dumps(info, indent=2))
  print(json.dumps(info, indent=2))


if __name__ == "__main__":
  main()
