#!/usr/bin/env python3
"""Extract current/best/best@iter/last-trend for v4 keypoint runs.

Usage: python extract_kp_metrics.py <logfile>
Objective judging — no subjective tail-watching. Tracks the 5 metrics the
supervisor flagged: kp_visible, kp_pixel_err, kp_selfloc_pos_err_m, goal_rate,
fell_over (+ keypoint_detection reward to catch saturation/gaming).
"""
from __future__ import annotations

import re
import sys

# (metric_key, lower_is_better)
METRICS = [
    ("kp_visible", False),
    ("kp_pixel_err", True),
    ("kp_selfloc_pos_err_m", True),
    ("goal_rate", False),
    ("fell_over", True),
    ("keypoint_detection", None),  # reward term: just report, 0 => saturated/gamed
]


def parse(path: str):
    lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
    it = 0
    series: dict[str, list[tuple[int, float]]] = {k: [] for k, _ in METRICS}
    for ln in lines:
        g = re.search(r"Learning iteration\s+(\d+)/", ln)
        if g:
            it = int(g.group(1))
        for k, _ in METRICS:
            m = re.search(rf"{re.escape(k)}:\s*([-\d.]+)\s*$", ln)
            if m:
                series[k].append((it, float(m.group(1))))
    return it, series


def trend(vals: list[float]) -> str:
    """Compare last-10% mean vs the prior-10% mean."""
    if len(vals) < 10:
        return "n/a"
    n = max(1, len(vals) // 10)
    late, prior = vals[-n:], vals[-2 * n : -n]
    if not prior:
        return "n/a"
    lm, pm = sum(late) / len(late), sum(prior) / len(prior)
    d = lm - pm
    arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
    return f"{arrow} ({pm:.3f}->{lm:.3f})"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v4_kp5d_full.log"
    last_it, series = parse(path)
    print(f"== {path}  (last iter {last_it}) ==")
    for k, lower in METRICS:
        v = series[k]
        if not v:
            print(f"{k:22s} : (no data)")
            continue
        cur = v[-1][1]
        vals = [x[1] for x in v]
        if lower is None:
            print(f"{k:22s} : cur={cur:.4f}  last_trend={trend(vals)}  (reward term)")
            continue
        best = (min if lower else max)(v, key=lambda x: x[1])
        # late-segment summary (iters > 2500 = post-fade region if present).
        post = [x[1] for x in v if x[0] > 2500]
        post_s = (
            f"  post2500: min={min(post):.3f} max={max(post):.3f} last={post[-1]:.3f}"
            if post
            else ""
        )
        print(
            f"{k:22s} : cur={cur:.3f}  best={best[1]:.3f}@{best[0]}  "
            f"trend={trend(vals)}{post_s}"
        )


if __name__ == "__main__":
    main()

