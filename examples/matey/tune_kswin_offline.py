"""Pick KSWIN's window sizes by replaying a recorded control run.

The shipped 60/20 was chosen on a 12-arrival stream with ~59 monitoring windows
per arrival. On a stream with more, shorter arrivals the reference window spans
several arrivals at once, and ``reset_after_learning`` then blanks the detector
for several more after every round -- so the same numbers behave completely
differently. Rather than guess, replay the control arm's recorded error series
through candidate settings and see where each one fires.

The replay goes through ``load_drift_detector``, so this exercises the same
detector code the run does rather than a re-implementation of it.

Usage::

    python examples/matey/tune_kswin_offline.py $OUTDIR/stream_nocl.csv \\
        --run-log $OUTDIR/run_nocl.log --baseline-until 7
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from apeiron.config.configuration import build_config  # noqa: E402
from apeiron.drift_detection.load_drift_detector import load_drift_detector  # noqa: E402

STEP_ARRIVAL_RE = re.compile(r"step=(\d+) \| model_stream \| ==== arrival (\d+)/")


def read_series(path: Path, metric: str) -> list[tuple[int, float]]:
    with path.open() as fh:
        return sorted(
            (int(r["step"]), float(r["value"]))
            for r in csv.DictReader(fh)
            if r["metric"] == metric
        )


def arrival_starts(csv_path: Path, run_log: Path | None) -> list[tuple[int, int]]:
    """(step, arrival_index) boundaries, 0-based arrivals.

    Preferred source is the ``stream/arrival`` row the stream harness writes
    into the metrics CSV, because it shares the step axis with the metric being
    replayed. The console banner is a fallback for older runs, and a poor one:
    it reports the wrong step counter.
    """
    from_csv = read_series(csv_path, "stream/arrival")
    if from_csv:
        return [(step, int(value)) for step, value in from_csv]
    if run_log is None or not run_log.is_file():
        return []
    return [
        (int(m.group(1)), int(m.group(2)) - 1)
        for m in (
            STEP_ARRIVAL_RE.search(line)
            for line in run_log.read_text(errors="ignore").splitlines()
        )
        if m
    ]


def arrival_of(step: int, starts: list[tuple[int, int]]) -> int:
    arrival = -1
    for start, idx in starts:
        if step >= start:
            arrival = idx
    return arrival


def fires(cfg, series, window: int, stat: int) -> list[int]:
    """Steps at which the detector reports drift, for one candidate setting."""
    dd = replace(cfg.drift_detection, kswin_window_size=window, kswin_stat_size=stat)
    detector = load_drift_detector(replace(cfg, drift_detection=dd))
    return [step for step, value in series if detector.update(value).drift_detected]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", help="control-arm metrics CSV (stream_nocl.csv)")
    p.add_argument("--config", default="examples/matey/matey_stream.toml")
    p.add_argument("--run-log", default="", help="run log, to report arrivals")
    p.add_argument("--metric", default="eval/nrmse_mean")
    p.add_argument(
        "--baseline-until",
        type=int,
        default=-1,
        help="last arrival still in the starting regime; firings at or before "
        "it are false alarms",
    )
    p.add_argument("--windows", default="20,24,30,40,60")
    p.add_argument("--stats", default="8,10,15,20")
    args, passthrough = p.parse_known_args(argv)

    cfg = build_config(["--config", args.config, *passthrough])
    series = read_series(Path(args.csv), args.metric)
    if not series:
        raise SystemExit(f"no {args.metric!r} rows in {args.csv}")
    starts = arrival_starts(
        Path(args.csv), Path(args.run_log) if args.run_log else None
    )

    print(f"{len(series)} windows, metric={args.metric}")
    print(f"{'window':>7} {'stat':>5} {'fires':>6}  arrivals (false alarms marked *)")
    for window in [int(w) for w in args.windows.split(",")]:
        for stat in [int(s) for s in args.stats.split(",")]:
            # KSWIN samples stat_size points out of the window_size - stat_size
            # it holds back as reference, so anything above half is not a
            # configuration at all -- river raises inside the first update.
            if 2 * stat > window:
                continue
            steps = fires(cfg, series, window, stat)
            marks = []
            for step in steps:
                arrival = arrival_of(step, starts) if starts else -1
                early = arrival >= 0 and arrival <= args.baseline_until
                marks.append(f"{arrival}{'*' if early else ''}")
            print(f"{window:>7} {stat:>5} {len(steps):>6}  {' '.join(marks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
