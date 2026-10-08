"""Plot a run's stream VRMSE with regime boundaries and drift events.

Reads the metrics CSV a run writes (``[logging] metrics_output_path``) and
draws the trace the detector watched: per-batch VRMSE over the whole stream,
its rolling mean at the detection interval, the regime boundaries, and the
batches where drift fired.

Usage::

    poetry run python -m examples.well.plot \
        --config examples/well/well_trl2d.toml \
        --metrics output/well_trl2d.csv \
        --out examples/well/figures/drift_detection.png
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from bisect import bisect_left
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from apeiron.config.configuration import Config, build_config
from examples.well.datasets import RegimeCache, get_dataset, read
from examples.well.model import N_STEPS_INPUT, N_STEPS_OUTPUT


def parse_metrics(path: Path) -> Tuple[List[float], List[int]]:
    """The per-batch VRMSE trace, and the batch index of each drift firing.

    The CSV holds one row per (step, metric, value). Eval rows arrive in
    stream order; a firing's logger step falls between eval steps, so its
    batch index is where that step would insert into the eval step sequence.
    """
    steps: List[int] = []
    trace: List[float] = []
    fire_steps: List[int] = []
    with open(path, newline="") as handle:
        for row in csv.reader(handle):
            if row[1] == "eval/vrmse":
                steps.append(int(row[0]))
                trace.append(float(row[2]))
            elif row[1] == "drift/detected" and float(row[2]) > 0:
                fire_steps.append(int(row[0]))
    fires = [bisect_left(steps, step) for step in fire_steps]
    return trace, fires


def window_boundaries(cfg: Config) -> Tuple[List[int], List[str]]:
    """Cumulative batch count at each window start, and the regime labels."""
    spec = get_dataset(cfg.data.name.split(":", 1)[1])
    cache = RegimeCache(spec, cfg.data.path)
    starts: List[int] = []
    labels: List[str] = []
    total = 0
    for regime in cache.regimes():
        shape = read(cache.array("train", regime)).shape
        samples = shape[0] * (shape[1] - N_STEPS_INPUT - N_STEPS_OUTPUT + 1)
        starts.append(total)
        labels.append(regime.rsplit("_", 1)[-1])
        total += math.ceil(samples / cfg.data.batch_size)
    return starts, labels


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot stream VRMSE and drift events")
    parser.add_argument("--config", required=True, help="the run's TOML")
    parser.add_argument(
        "--metrics", help="metrics CSV (default: [logging] metrics_output_path)"
    )
    parser.add_argument(
        "--out",
        default="examples/well/figures/drift_detection.png",
        help="output image path",
    )
    args = parser.parse_args(argv)

    cfg: Config = build_config(["--config", args.config])
    metrics = Path(
        args.metrics or (cfg.logging.metrics_output_path if cfg.logging else None) or ""
    )
    if not metrics.is_file():
        parser.error(f"no metrics CSV at {metrics!r}")

    trace, fires = parse_metrics(metrics)
    if not trace:
        parser.error(f"{metrics} holds no eval/vrmse rows")
    starts, labels = window_boundaries(cfg)
    interval = cfg.drift_detection.detection_interval

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(trace)
    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=150)
    ax.plot(values, color="#9ecae1", linewidth=0.6, label="per-batch VRMSE")
    if len(values) >= interval:
        mean = np.convolve(values, np.ones(interval) / interval, mode="valid")
        ax.plot(
            np.arange(interval - 1, len(values)),
            mean,
            color="#2171b5",
            linewidth=1.4,
            label=f"{interval}-batch mean",
        )
    for start, label in zip(starts, labels):
        ax.axvline(start, color="#bbbbbb", linestyle="--", linewidth=0.7)
        ax.annotate(
            label,
            (start, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -2),
            textcoords="offset points",
            fontsize=7,
            color="#888888",
            va="top",
        )
    for index, fire in enumerate(fires, start=1):
        ax.axvline(fire, color="#d62728", linewidth=1.2)
        ax.annotate(
            f"event {index}",
            (fire, 0.82),
            xycoords=("data", "axes fraction"),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color="#d62728",
        )
    ax.set_xlabel("stream batch")
    ax.set_ylabel("VRMSE")
    ax.set_xlim(0, len(values) - 1)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out} ({len(trace)} batches, {len(fires)} drift events)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
