"""Plot training curves from a run's tensorboard events into a multi-panel PNG.

Used as the standard quantitative-analysis artifact for every v3g self-loc
experiment. Reads the scalar tags that matter for the self-localization +
dribble task and writes a single annotated figure to soccer_eval/.

Usage:
  uv run python scripts/plot_run_curves.py <run_dir> [out_png]
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# (tag, label) grouped into panels. Each inner list is one subplot.
PANELS = [
  (
    "Self-localization accuracy",
    [
      ("Metrics/selfloc_pos_err_m", "pos err (m)"),
      ("Metrics/selfloc_err_l2", "L2 err (norm)"),
    ],
  ),
  (
    "Self-loc reward + GT mask",
    [
      ("Episode_Reward/selfloc_accuracy", "accuracy reward"),
      ("Episode_Reward/selfloc_error_penalty", "error penalty"),
      ("Curriculum/selfloc_gt_mask/mask_factor", "GT mask factor"),
    ],
  ),
  (
    "Dribble task",
    [
      ("Episode_Reward/dribble_approach", "approach"),
      ("Episode_Reward/dribble_to_target", "to_target"),
      ("Episode_Reward/dribble_success", "success"),
      ("Metrics/dribble/ball_path_length", "ball path (m)"),
    ],
  ),
  (
    "Stability",
    [
      ("Episode_Reward/upright", "upright"),
      ("Episode_Termination/fell_over", "fell_over (count)"),
    ],
  ),
]


def _load(run_dir: Path) -> dict[str, tuple[list[int], list[float]]]:
  files = sorted(glob.glob(str(run_dir / "events.*")))
  if not files:
    raise FileNotFoundError(f"No tensorboard events in {run_dir}")
  ea = EventAccumulator(files[0], size_guidance={"scalars": 0})
  ea.Reload()
  avail = set(ea.Tags()["scalars"])
  out = {}
  for tag in avail:
    evs = ea.Scalars(tag)
    out[tag] = ([e.step for e in evs], [e.value for e in evs])
  return out


def main() -> None:
  if len(sys.argv) < 2:
    raise SystemExit("usage: plot_run_curves.py <run_dir> [out_png]")
  run_dir = Path(sys.argv[1])
  data = _load(run_dir)
  out_png = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else Path("soccer_eval") / f"{run_dir.name}_curves.png"
  )
  out_png.parent.mkdir(parents=True, exist_ok=True)

  n = len(PANELS)
  fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n), squeeze=False)
  for ax, (title, series) in zip(axes[:, 0], PANELS, strict=False):
    plotted = False
    for tag, label in series:
      if tag in data and data[tag][0]:
        steps, vals = data[tag]
        ax.plot(steps, vals, label=label, linewidth=1.3)
        plotted = True
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.3)
    if plotted:
      ax.legend(fontsize=8, loc="best")
    else:
      ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
  fig.suptitle(run_dir.name, fontsize=12, y=0.995)
  fig.tight_layout()
  fig.savefig(out_png, dpi=110)
  print(f"[OK] wrote {out_png}")


if __name__ == "__main__":
  main()
