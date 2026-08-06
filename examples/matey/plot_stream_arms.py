#!/usr/bin/env python3
"""Plot the drift, its detection, and what adaptation does about it.

    python examples/matey/plot_stream_arms.py <outdir> [--fields showcase.npz]

Reads the two CSVs written by submit_stream_cl.sh and draws the story in order:

  (a,b,c) the physics: the mean electron-density field in the baseline and the
          held-out scenario, and the difference between them -- what "drift"
          is here, before any model sees it;
  (d)     the detector's view: the monitored error per window, and where KSWIN
          fired;
  (e)     the consequence: error across the stream for the adapting arm and the
          no-adaptation control, with the drop at each adaptation.

Panels (a-c) need field snapshots from a drift-showcase run (--fields); without
that the figure is drawn from the stream CSVs alone.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MONITORED = "eval/nrmse_mean"
PRE, POST = "eval/val_pre_cur_nrmse_mean", "eval/val_post_cur_nrmse_mean"

# Categorical slots 1 and 2; identity is also carried by marker shape, so the
# panels survive greyscale printing and colour-vision deficiency.
CONTROL, ADAPTED = "#2a78d6", "#eb6834"
GRID, INK, MUTED, SURFACE = "#d8d8d5", "#1a1a19", "#6b6b68", "#fcfcfb"


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


def _shade_regimes(ax, spec, n) -> None:
    """Shade each regime and name it, so the reader sees what changed when."""
    bands = ["#eef2f8", "#fdf0e9", "#eef6f1"]
    start = 0.5
    for i, item in enumerate(spec or []):
        label, _, end = item.rpartition(":")
        end = float(end)
        ax.axvspan(start, end + 0.5, color=bands[i % len(bands)], zorder=0)
        ax.text(
            (start + end + 0.5) / 2,
            ax.get_ylim()[1],
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color=MUTED,
        )
        start = end + 0.5


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def _per_arrival(values: list[float], n: int):
    """Mean and population sd within each arrival, by equal division.

    Every arrival contributes the same number of evaluation windows, since
    eval.max_val_batches caps each one identically, so boundaries are accurate
    to within a window.
    """
    step = len(values) / n
    means, sds = [], []
    for i in range(n):
        chunk = values[round(i * step) : round((i + 1) * step)]
        means.append(statistics.mean(chunk))
        sds.append(statistics.pstdev(chunk) if len(chunk) > 1 else 0.0)
    return means, sds


def _draw_fields(fig, gs, npz_path: Path) -> None:
    z = np.load(npz_path, allow_pickle=True)
    base, ood = z["base_ne_mean"], z["ood_ne_mean"]
    floor = base.max() * 1e-3  # ignore the near-vacuum region, where % blows up
    rel = np.where(base > floor, 100.0 * (ood - base) / np.maximum(base, floor), 0.0)
    lim = float(np.percentile(np.abs(rel), 99))
    vmax = float(max(base.max(), ood.max()))

    panels = (
        (base, "(a) baseline scenario", "viridis", dict(vmin=0, vmax=vmax)),
        (ood, "(b) held-out scenario", "viridis", dict(vmin=0, vmax=vmax)),
        (rel, "(c) change, % of baseline", "RdBu_r", dict(vmin=-lim, vmax=lim)),
    )
    for col, (data, title, cmap, kw) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(data, aspect="auto", cmap=cmap, **kw)
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(colors=MUTED, labelsize=7)
        cb.outline.set_visible(False)
        if col == 0:
            ax.set_ylabel("mean $n_e$ field", color=INK, fontsize=9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--fields", type=Path, default=None)
    ap.add_argument("--events-at", type=int, nargs="*", default=None)
    ap.add_argument(
        "--regimes",
        nargs="*",
        default=None,
        help="label:last_arrival, e.g. 'baseline:8' 'held-out:16' 'KSTAR:24'",
    )
    ap.add_argument("--arrivals", type=int, default=24)
    args = ap.parse_args()

    cl = _series(args.outdir / "stream_cl.csv")
    nocl = _series(args.outdir / "stream_nocl.csv")
    if not cl[MONITORED] or not nocl[MONITORED]:
        sys.exit(f"no {MONITORED} rows found; did the run finish?")

    fields = args.fields if args.fields and args.fields.exists() else None
    rows = 3 if fields else 2
    heights = [0.85, 1.0, 1.15] if fields else [1.0, 1.15]

    fig = plt.figure(figsize=(12, 3.2 * rows))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(rows, 3, height_ratios=heights, hspace=0.5, wspace=0.25)
    if fields:
        _draw_fields(fig, gs, fields)

    base_row = 1 if fields else 0
    n = args.arrivals
    width = min(len(cl[MONITORED]), len(nocl[MONITORED]))

    # -- what the detector sees ---------------------------------------------
    ax = fig.add_subplot(gs[base_row, :])
    _style(ax)
    ax.plot(
        np.linspace(1, n, width),
        cl[MONITORED][:width],
        color=MUTED,
        linewidth=1.1,
        label="monitored error per window",
    )
    for k, arrival in enumerate(args.events_at or []):
        ax.axvline(
            arrival,
            color=ADAPTED,
            linewidth=1.4,
            linestyle=(0, (4, 3)),
            label="KSWIN fires → adapt" if k == 0 else None,
        )
    ax.set_xlim(0.5, n + 0.5)
    _shade_regimes(ax, args.regimes, n)
    ax.set_ylabel("NRMSE", color=INK, fontsize=10)
    ax.set_title(
        f"({'d' if fields else 'a'}) what the detector sees",
        color=INK,
        fontsize=11,
        loc="left",
        pad=8,
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="center right")

    # -- what adaptation does ------------------------------------------------
    bx = fig.add_subplot(gs[base_row + 1, :])
    _style(bx)
    for vals, colour, marker, label in (
        (nocl[MONITORED][:width], CONTROL, "o", "no adaptation (control)"),
        (cl[MONITORED][:width], ADAPTED, "s", "continual learning"),
    ):
        m, s = _per_arrival(vals, n)
        bx.errorbar(
            range(1, n + 1),
            m,
            yerr=s,
            color=colour,
            marker=marker,
            markersize=4.5,
            linewidth=2,
            elinewidth=1,
            capsize=2.5,
            label=label,
        )

    for arrival, before, after in zip(args.events_at or [], cl[PRE], cl[POST]):
        bx.annotate(
            f"−{100 * (before - after) / before:.0f}%",
            xy=(arrival, after),
            xytext=(arrival, before * 1.02),
            arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.1),
            color=INK,
            fontsize=9,
            ha="center",
            va="bottom",
        )

    bx.set_xlim(0.5, n + 0.5)
    _shade_regimes(bx, args.regimes, n)
    bx.set_ylabel("NRMSE  (mean ± sd)", color=INK, fontsize=10)
    bx.set_xlabel("simulation arrival", color=INK, fontsize=10)
    bx.set_title(
        f"({'e' if fields else 'b'}) what adaptation does",
        color=INK,
        fontsize=11,
        loc="left",
        pad=8,
    )
    bx.legend(frameon=False, fontsize=9, labelcolor=INK)

    base_mean = statistics.mean(nocl[MONITORED][:width])
    cl_mean = statistics.mean(cl[MONITORED][:width])
    drops = [100 * (b - a) / b for b, a in zip(cl[PRE], cl[POST])]
    fig.text(
        0.01,
        -0.01,
        f"pretrained base model   mean NRMSE: {base_mean:.5f}\n"
        f"with continual learning mean NRMSE: {cl_mean:.5f}"
        f"   ({100 * (base_mean - cl_mean) / base_mean:.1f}% lower)\n"
        f"drop on the arriving bundle at each adaptation "
        f"(validation split, not the streamed windows): "
        f"{', '.join(f'{d:.0f}%' for d in drops)}",
        family="monospace",
        fontsize=9.5,
        color=INK,
        va="top",
    )

    out = args.output or (args.outdir / "stream_arms.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
