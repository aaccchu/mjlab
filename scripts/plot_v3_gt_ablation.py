"""Plot the v3 GT-ablation result as a three-way comparison that makes the
"CNN learned to see (partially)" finding visible at a glance:

  * warmup baseline  - GT intact, the GT-crutch policy (dribble 4.69)
  * warmup ablated    - GT zeroed on the warmup ckpt -> CNN dead weight (collapse)
  * v3b native        - GT zeroed on the GT-ablation-trained ckpt -> camera only

For v3b, GT=0 is its IN-DISTRIBUTION trained condition, so its "ablated" column
is its true camera-only performance. dribble 0.01 (warmup ablated) -> 1.11 (v3b)
is the CNN now carrying usable ball bearing.

Numbers from scripts/probe_v3_gt_ablation.py (256 envs, 600 steps, settle avg).

Usage:
  uv run python scripts/plot_v3_gt_ablation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("soccer_eval/2026-06-05_spikes/v3b_gt_ablation")

METRICS = ["gaze_center", "ball_visible", "dribble_success", "upright", "gaze_search"]
WARMUP_BASE = [0.0061, 0.0097, 4.6890, 0.9893, 0.4752]  # GT crutch
WARMUP_ABL = [0.0200, 0.1528, 0.0105, 0.6132, 0.1427]  # CNN dead weight
V3B_NATIVE = [0.1720, 0.3911, 1.1105, 0.9466, 0.2802]  # camera only (trained)


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  x = np.arange(len(METRICS))
  w = 0.27

  fig, ax = plt.subplots(figsize=(11, 5.5))
  ax.bar(x - w, WARMUP_BASE, w, label="warmup baseline (GT crutch)", color="#2c7fb8")
  ax.bar(x, WARMUP_ABL, w, label="warmup ablated (CNN dead weight)", color="#bdbdbd")
  b3 = ax.bar(x + w, V3B_NATIVE, w, label="v3b native (camera only, trained)",
              color="#31a354")

  ax.set_ylabel("reward / rate (post-settle avg)")
  ax.set_title(
    "v3 GT-ablation: the depth CNN learned to see (partially)\n"
    "warmup collapses without GT (0.01); v3b retains dribbling on camera (1.11)"
  )
  ax.set_xticks(x)
  ax.set_xticklabels(METRICS, rotation=15, ha="right")
  ax.legend()
  ax.bar_label(b3, fmt="%.3f", fontsize=8, padding=2)

  ax.annotate(
    "dribble survives\non camera: 0.01 -> 1.11",
    xy=(2 + w, 1.11), xytext=(2.3, 3.0),
    fontsize=9, color="#006d2c",
    arrowprops=dict(arrowstyle="->", color="#006d2c"),
  )
  fig.tight_layout()
  out = OUT / "probe_three_way.png"
  fig.savefig(out, dpi=130)
  print(f"[INFO] saved {out}")


if __name__ == "__main__":
  main()
