"""Supervisor monitor for v4 runs.

Reads a TensorBoard run directory, summarizes the main metrics, and emits a
status plus next-action suggestions. Intended for supervising Claude Code's
experiments with a consistent rubric instead of ad-hoc eyeballing.

Usage:
  python scripts/monitor_v4_run.py --run-dir logs/rsl_rl/mos92_velocity/<run>
  python scripts/monitor_v4_run.py --run-dir logs/... --write-md soccer_eval/.../report.md
  python scripts/monitor_v4_run.py --latest-prefix spike_v4_dualcam
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass
class SeriesSummary:
  tag: str
  step: int
  current: float
  best: float
  best_step: int
  recent_delta: float


def _load_events(run_dir: Path) -> EventAccumulator:
  files = sorted(glob.glob(str(run_dir / "events.*")))
  if not files:
    raise FileNotFoundError(f"No tensorboard events in {run_dir}")
  ea = EventAccumulator(files[0], size_guidance={"scalars": 0})
  ea.Reload()
  return ea


def _summary(ea: EventAccumulator, tag: str, lower_is_better: bool) -> SeriesSummary:
  scalars = ea.Scalars(tag)
  if not scalars:
    raise ValueError(f"Missing scalar tag: {tag}")
  vals = [s.value for s in scalars]
  steps = [s.step for s in scalars]
  if lower_is_better:
    best = min(vals)
  else:
    best = max(vals)
  idx = vals.index(best)
  tail_n = min(50, len(vals) - 1)
  recent_delta = vals[-1] - vals[-1 - tail_n] if tail_n > 0 else 0.0
  return SeriesSummary(
    tag=tag,
    step=steps[-1],
    current=vals[-1],
    best=best,
    best_step=steps[idx],
    recent_delta=recent_delta,
  )


def _discover_latest(root: Path, prefix: str) -> Path:
  runs = sorted(
    [p for p in root.iterdir() if p.is_dir() and prefix in p.name],
    key=lambda p: p.stat().st_mtime,
  )
  if not runs:
    raise FileNotFoundError(f"No runs under {root} with prefix '{prefix}'")
  return runs[-1]


def _grade(
  dribble: SeriesSummary,
  goal: SeriesSummary,
  selfloc: SeriesSummary,
  fell: SeriesSummary,
) -> tuple[str, list[str]]:
  notes: list[str] = []
  status = "GREEN"

  if fell.current > 0.05:
    status = "RED"
    notes.append("Stability regressed: fell_over is above 0.05.")

  if selfloc.current > 5.0 and selfloc.step >= 1000:
    status = "RED"
    notes.append("Self-localization is still too weak after 1000+ iters.")

  if dribble.best < 0.30 and dribble.step >= 1500:
    status = "RED"
    notes.append(
      "Best dribble_success is still below 0.30; dualcam selfloc-chain is not enough."
    )

  if goal.best < 0.20 and goal.step >= 1500:
    status = "RED"
    notes.append("Best goal_rate is still below 0.20 after enough training.")

  if dribble.current < max(0.05, 0.4 * dribble.best) and dribble.step >= 500:
    if status != "RED":
      status = "YELLOW"
    notes.append("Current dribble_success is far below its historical best; possible late collapse.")

  if goal.current < max(0.05, 0.4 * goal.best) and goal.step >= 500:
    if status != "RED":
      status = "YELLOW"
    notes.append("Current goal_rate is far below its historical best; check for late collapse.")

  if (
    selfloc.current <= 3.0
    and fell.current == 0.0
    and dribble.best >= 0.15
    and goal.best >= 0.15
    and status == "GREEN"
  ):
    notes.append("Run is stable and localization is strong, but behavior recovery should still be benchmarked against 04_e2e.")

  if not notes:
    notes.append("Metrics are within the configured guardrails.")

  return status, notes


def _recommendations(
  run_name: str,
  dribble: SeriesSummary,
  goal: SeriesSummary,
  selfloc: SeriesSummary,
  status: str,
) -> list[str]:
  recs: list[str] = []
  if "dualcam" in run_name and selfloc.current < 3.5 and dribble.best < 0.30:
    recs.append(
      "Highest-priority next step: build realspec_e2e_dualcam from mos92_soccer_e2e_env_cfg and bootstrap from 04_e2e_integrated/model_1499.pt."
    )
  if selfloc.current > 5.0:
    recs.append("Raise RGB resolution only after behavior is stabilized, or split into a pure self-loc ablation.")
  if dribble.current < 0.4 * dribble.best and dribble.step >= 500:
    recs.append("Record current/best divergence explicitly and consider early-stop or branch instead of only waiting for the final iter.")
  if goal.best >= 0.20 and dribble.best >= 0.20 and status != "RED":
    recs.append("Let the run finish, then compare current vs best checkpoint before choosing the artifact.")
  if not recs:
    recs.append("No special recommendation generated.")
  return recs


def _render(
  run_dir: Path,
  status: str,
  notes: list[str],
  recs: list[str],
  dribble: SeriesSummary,
  goal: SeriesSummary,
  selfloc: SeriesSummary,
  fell: SeriesSummary,
) -> str:
  lines = [
    f"# v4 Supervisor Report: {run_dir.name}",
    "",
    f"- Status: `{status}`",
    f"- Step: `{dribble.step}`",
    "",
    "## Metrics",
    f"- `dribble_success`: current `{dribble.current:.4f}`, best `{dribble.best:.4f}` @ `{dribble.best_step}`, recent_delta `{dribble.recent_delta:+.4f}`",
    f"- `goal_rate`: current `{goal.current:.4f}`, best `{goal.best:.4f}` @ `{goal.best_step}`, recent_delta `{goal.recent_delta:+.4f}`",
    f"- `selfloc_pos_err_m`: current `{selfloc.current:.4f}`, best `{selfloc.best:.4f}` @ `{selfloc.best_step}`, recent_delta `{selfloc.recent_delta:+.4f}`",
    f"- `fell_over`: current `{fell.current:.4f}`, best `{fell.best:.4f}` @ `{fell.best_step}`, recent_delta `{fell.recent_delta:+.4f}`",
    "",
    "## Flags",
  ]
  lines.extend([f"- {note}" for note in notes])
  lines.append("")
  lines.append("## Recommendations")
  lines.extend([f"- {rec}" for rec in recs])
  lines.append("")
  return "\n".join(lines)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-dir", type=Path, default=None)
  ap.add_argument(
    "--root",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "logs" / "rsl_rl" / "mos92_velocity",
    help="Root directory for auto-discovery.",
  )
  ap.add_argument("--latest-prefix", default=None)
  ap.add_argument("--write-md", type=Path, default=None)
  args = ap.parse_args()

  if args.run_dir is not None:
    run_dir = args.run_dir
  elif args.latest_prefix:
    run_dir = _discover_latest(args.root, args.latest_prefix)
  else:
    raise SystemExit("Provide --run-dir or --latest-prefix.")

  ea = _load_events(run_dir)
  dribble = _summary(ea, "Episode_Reward/dribble_success", lower_is_better=False)
  goal = _summary(ea, "Metrics/dribble/goal_rate", lower_is_better=False)
  selfloc = _summary(ea, "Metrics/selfloc_pos_err_m", lower_is_better=True)
  fell = _summary(ea, "Episode_Termination/fell_over", lower_is_better=True)

  status, notes = _grade(dribble, goal, selfloc, fell)
  recs = _recommendations(run_dir.name, dribble, goal, selfloc, status)
  report = _render(run_dir, status, notes, recs, dribble, goal, selfloc, fell)
  print(report)

  if args.write_md is not None:
    args.write_md.parent.mkdir(parents=True, exist_ok=True)
    args.write_md.write_text(report + "\n", encoding="utf-8")
    print(f"[OK] wrote {args.write_md}")


if __name__ == "__main__":
  main()
