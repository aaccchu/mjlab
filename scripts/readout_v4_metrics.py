#!/usr/bin/env python3
"""Multi-metric readout for v4 plain stdout training logs (EXP13+). codex's hard
lesson: never judge a run by pos_err (or reward) alone — a run can improve
localization while the kick chain stays broken. This reports all three dimensions
side by side over a chosen tail window so success/failure is unambiguous.

Usage: python scripts/readout_v4_metrics.py /tmp/v4_exp13_full.log [window]
"""
import re
import sys

# (log key, short label, "higher better"? for the arrow) grouped by dimension.
DIMENSIONS = {
  "自定位 LOCALIZATION": [
    ("Metrics/selfloc_pos_err_m", "pos_err_m", False),
    ("Metrics/scan_uniq_frac", "uniq_frac", True),
  ],
  "踢球链 KICK-CHAIN (codex 命门)": [
    ("Metrics/dribble/goal_rate", "goal_rate", True),
    ("Metrics/dribble/target_is_goal", "tgt_is_goal", True),
    ("Metrics/dribble/episode_success", "episode_success", True),
    ("Metrics/dribble/possession", "possession", True),
    ("Metrics/dribble/ball_path_length", "ball_path", True),
    ("Metrics/dribble/ball_speed", "ball_speed", True),
    ("Metrics/dribble/ball_speed_peak", "ball_speed_peak", True),
    ("Metrics/dribble/ball_to_target_error", "ball_to_tgt_err", False),
    ("Metrics/dribble/ball_stuck_time", "ball_stuck_s", False),
    ("Metrics/dribble/robot_to_ball_error", "robot_to_ball", False),
  ],
  "稳定性 STABILITY": [
    ("Episode_Termination/fell_over", "fell_over", False),
    ("Episode_Termination/out_of_field_bounds", "out_of_bounds", False),
  ],
}


def extract(log, key):
  vals = []
  pat = re.compile(re.escape(key) + r":\s*([-+0-9.eE]+)")
  with open(log, errors="ignore") as f:
    for line in f:
      m = pat.search(line)
      if m:
        try:
          vals.append(float(m.group(1)))
        except ValueError:
          pass
  return vals


def stats(vals, window):
  if not vals:
    return None
  tail = vals[-window:]
  n = len(tail)
  mean = sum(tail) / n
  srt = sorted(tail)
  median = srt[n // 2]
  return mean, median, min(tail), max(tail), n


def main(log, window=100):
  print(f"\n=== 多指标判读: {log}  (末 {window} 采样点) ===")
  for dim, keys in DIMENSIONS.items():
    print(f"\n── {dim} ──")
    print(f"  {'指标':<20}{'mean':>10}{'median':>10}{'min':>9}{'max':>9}{'n':>5}")
    for key, label, _hib in keys:
      s = stats(extract(log, key), window)
      if s is None:
        print(f"  {label:<20}{'(无数据)':>10}")
        continue
      mean, median, lo, hi, n = s
      print(f"  {label:<20}{mean:>10.4f}{median:>10.4f}{lo:>9.3f}{hi:>9.3f}{n:>5}")
  # True shooting rate: goals per goal-target episode. The global goal_rate is
  # capped by goal_target_fraction (only goal-target episodes can score), so this
  # divides goal_rate by the realized goal-target fraction for the real signal.
  gr = stats(extract(log, "Metrics/dribble/goal_rate"), window)
  tg = stats(extract(log, "Metrics/dribble/target_is_goal"), window)
  if gr and tg and tg[0] > 1e-6:
    subset = gr[0] / tg[0]
    print(
      f"\n  ★ 真实射门率 (goal_rate/target_is_goal) = "
      f"{gr[0]:.4f}/{tg[0]:.4f} = {subset:.3f}  (球门子集进球率)"
    )


if __name__ == "__main__":
  log = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v4_exp13_full.log"
  window = int(sys.argv[2]) if len(sys.argv) > 2 else 100
  main(log, window)
