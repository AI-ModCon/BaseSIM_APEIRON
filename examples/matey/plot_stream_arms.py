#!/usr/bin/env python3
"""Plot the adaptation arm against the no-adaptation control.

    python examples/matey/plot_stream_arms.py <outdir> [-o figure.png]

Reads the two CSVs written by submit_stream_cl.sh and draws:

  (a) per-arrival mean +/- std of the monitored error, both arms, with the
      arrivals at which drift fired marked;
  (b) the before/after error on the arriving bundle at each drift event.

Panel (a) is what shows the control is not simply an easier stream; panel (b)
is the adaptation itself.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MONITORED = "eval/nrmse_mean"
PRE, POST = "eval/val_pre_cur_nrmse_mean", "eval/val_post_cur_nrmse_mean"

# Categorical slots 1 and 2. Identity is carried by marker shape as well as hue,
# so the panels survive greyscale printing and colour-vision deficiency.
CONTROL, ADAPTED = "#2a78d6", "#eb6834"
GRID = "#d8d8d5"
INK, MUTED = "#1a1a19", "#6b6b68"


def _series(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        sys.exit(f"missing {path}. Run both arms into the same OUTDIR first.")
    out: dict[str, list[float]] = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            try:
                out[row["metric"]].append(float(row["value"]))
            except (KeyError, ValueError):
                continue
    return out


def _per_arrival(values: list[float], n_arrivals: int):
    """Mean and population std within each arrival.

    Window-to-arrival assignment is by equal division: every arrival contributes
    the same number of evaluation windows, since eval.max_val_batches caps each
    one identically. Boundaries are therefore accurate to within a window.
    """
    step = len(values) / n_arrivals
    means, sds = [], []
    for i in range(n_arrivals):
        chunk = values[round(i * step) : round((i + 1) * step)]
        means.append(statistics.mean(chunk))
        sds.append(statistics.pstdev(chunk) if len(chunk) > 1 else 0.0)
    return means, sds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--arrivals", type=int, default=24)
    ap.add_argument(
        "--events-at",
        type=int,
        nargs="*",
        default=None,
        help="arrivals after which adaptation ran (1-based)",
    )
    args = ap.parse_args()

    cl = _series(args.outdir / "stream_cl.csv")
    nocl = _series(args.outdir / "stream_nocl.csv")
    if not cl[MONITORED] or not nocl[MONITORED]:
        sys.exit(f"no {MONITORED} rows found; did the run finish?")

    width = min(len(cl[MONITORED]), len(nocl[MONITORED]))
    n = args.arrivals
    cl_m, cl_s = _per_arrival(cl[MONITORED][:width], n)
    no_m, no_s = _per_arrival(nocl[MONITORED][:width], n)
    x = range(1, n + 1)

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.75, 1]}
    )
    fig.patch.set_facecolor("#fcfcfb")

    for a in (ax, bx):
        a.set_facecolor("#fcfcfb")
        a.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=9)

    # -- (a) per-arrival error, both arms -----------------------------------
    for series, sd, colour, marker, label in (
        (no_m, no_s, CONTROL, "o", "no adaptation (control)"),
        (cl_m, cl_s, ADAPTED, "s", "continual learning"),
    ):
        ax.errorbar(
            x,
            series,
            yerr=sd,
            color=colour,
            marker=marker,
            markersize=4.5,
            linewidth=2,
            elinewidth=1,
            capsize=2.5,
            alpha=0.95,
            label=label,
        )

    if args.events_at:
        for i, arrival in enumerate(args.events_at):
            ax.axvline(arrival, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
            ax.text(
                arrival,
                ax.get_ylim()[1],
                " adapt" if i == 0 else "",
                color=MUTED,
                fontsize=8,
                va="top",
                ha="left",
            )

    ax.set_xlabel("simulation arrival", color=INK, fontsize=10)
    ax.set_ylabel("NRMSE  (mean ± sd within arrival)", color=INK, fontsize=10)
    ax.set_title(
        "(a) error across the stream", color=INK, fontsize=11, loc="left", pad=8
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    # -- (b) before / after at each drift event -----------------------------
    pre, post = cl[PRE], cl[POST]
    idx = range(1, len(pre) + 1)
    bar = 0.36
    bx.bar(
        [i - bar / 2 for i in idx],
        pre,
        bar,
        color=CONTROL,
        label="before adaptation",
        zorder=3,
    )
    bx.bar(
        [i + bar / 2 for i in idx],
        post,
        bar,
        color=ADAPTED,
        label="after adaptation",
        zorder=3,
    )

    for i, (a_val, b_val) in zip(idx, zip(pre, post)):
        if a_val:
            bx.text(
                i,
                max(a_val, b_val) * 1.04,
                f"−{100 * (a_val - b_val) / a_val:.0f}%",
                ha="center",
                color=INK,
                fontsize=9,
            )

    bx.set_xticks(list(idx))
    bx.set_xlabel("drift event", color=INK, fontsize=10)
    bx.set_ylabel("NRMSE on the arriving bundle", color=INK, fontsize=10)
    bx.set_title(
        "(b) adaptation at each detection", color=INK, fontsize=11, loc="left", pad=8
    )
    bx.set_ylim(0, max(pre + post) * 1.22)
    bx.legend(frameon=False, fontsize=9, labelcolor=INK)

    fig.tight_layout()
    out = args.output or (args.outdir / "stream_arms.png")
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
