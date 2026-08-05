#!/usr/bin/env python3
"""Compare the adaptation arm against the no-adaptation control.

Reads the two CSVs written by submit_stream_cl.sh and reports what continual
learning bought, per drift event and over the stream.

    python examples/matey/compare_stream_arms.py <outdir>

Both `stream_cl.csv` and `stream_nocl.csv` must be present. The control is not
optional: the whole point is that a falling error curve on its own proves
nothing, because the later arrivals might simply be easier.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# The metric the shipped config's detector monitors (metric_index = 3).
MONITORED = "eval/nrmse_mean"
PRE, POST = "eval/val_pre_cur_nrmse_mean", "eval/val_post_cur_nrmse_mean"
PRE_HIST, POST_HIST = "eval/val_pre_hist_nrmse_mean", "eval/val_post_hist_nrmse_mean"


def _read(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        sys.exit(f"missing {path}. Run both arms into the same OUTDIR first.")
    series: dict[str, list[float]] = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            try:
                series[row["metric"]].append(float(row["value"]))
            except (KeyError, ValueError):
                continue  # non-numeric rows (e.g. drift/regime) are not metrics
    return series


def _pct(control: float, treated: float) -> str:
    """Improvement of `treated` over `control`, positive = better (lower error)."""
    if control == 0:
        return "n/a"
    return f"{100.0 * (control - treated) / control:+.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path, help="directory holding stream_*.csv")
    args = parser.parse_args()

    cl = _read(args.outdir / "stream_cl.csv")
    nocl = _read(args.outdir / "stream_nocl.csv")

    cl_curve, nocl_curve = cl[MONITORED], nocl[MONITORED]
    if not cl_curve or not nocl_curve:
        sys.exit(f"no {MONITORED} rows found; did the run finish?")
    if len(cl_curve) != len(nocl_curve):
        print(
            f"! arms cover different window counts ({len(cl_curve)} vs "
            f"{len(nocl_curve)}); comparing the shared prefix",
            file=sys.stderr,
        )
        width = min(len(cl_curve), len(nocl_curve))
        cl_curve, nocl_curve = cl_curve[:width], nocl_curve[:width]

    print(f"\nwindows: {len(cl_curve)}   metric: {MONITORED}\n")

    print("per drift event, on the arriving bundle")
    print(f"  {'event':>5}  {'pre':>9}  {'post':>9}  {'change':>8}")
    pre, post = cl[PRE], cl[POST]
    if not pre:
        print("  (none: the detector never fired)")
    for i, (before, after) in enumerate(zip(pre, post), start=1):
        print(f"  {i:>5}  {before:9.5f}  {after:9.5f}  {_pct(before, after):>8}")

    # The control must show pre == post at every event; if it does not, the
    # "no adaptation" arm adapted and the comparison below is meaningless.
    drifted = [
        i
        for i, (b, a) in enumerate(zip(nocl[PRE], nocl[POST]), start=1)
        if abs(b - a) > 1e-12
    ]
    if drifted:
        print(f"\n! control changed at events {drifted} -- it should be inert")

    print("\nagainst the control")
    print(f"  {'':22} {'control':>9}  {'CL':>9}  {'delta':>8}")
    whole = (statistics.mean(nocl_curve), statistics.mean(cl_curve))
    print(f"  {'whole stream':22} {whole[0]:9.5f}  {whole[1]:9.5f}  {_pct(*whole):>8}")
    tail = (statistics.mean(nocl_curve[-50:]), statistics.mean(cl_curve[-50:]))
    print(f"  {'last 50 windows':22} {tail[0]:9.5f}  {tail[1]:9.5f}  {_pct(*tail):>8}")

    # Forgetting: the historical domain is pinned to the stream's first case, so
    # a rise here is the cost of having adapted.
    hist = cl[PRE_HIST] + cl[POST_HIST][-1:]
    if hist:
        print(
            f"\nhistorical-domain NRMSE {hist[0]:.5f} -> {hist[-1]:.5f} "
            f"({_pct(hist[0], hist[-1])} vs its own start)"
        )
        if hist[-1] > hist[0]:
            print("  a rise is forgetting: adaptation cost accuracy on the old data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
