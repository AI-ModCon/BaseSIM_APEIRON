#!/usr/bin/env python3
"""Drift detected -> CL applied -> is the adapted model better than the pretrained one?

Three panels on one shared arrival axis (arrival k occupies x in [k, k+1)).

Panel 1 plots a drift SCORE, -log10(p) of a two-sample KS test, so it rises when
the stream changes. The line comes from a continuous monitor that never resets --
the deployed KSWIN blanks its own window after every detection, so its raw
p_value is undefined for most of the stream and cannot be drawn as a curve. Where
the deployed detector actually fired is marked separately, and the two agree: the
monitor crosses alpha at the same window the detector fires.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics as st
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

# Categorical slots: pretrained, adapted, oracle, replay.
ORIGINAL, ADAPTED, ORACLE, REPLAY = "#2a78d6", "#d62828", "#1a1a19", "#9a9a96"
REGIME = "#eb6834"
GRID, INK, MUTED, SURFACE = "#d8d8d5", "#1a1a19", "#6b6b68", "#fcfcfb"
WIN, LOSS = "#3f8f6b", "#8c62c4"
DENSITY = "#6f6f6b"  # grey: a physics reference, not one of the model arms
M = "eval/nrmse_mean"

ap = argparse.ArgumentParser()
ap.add_argument("rundir", type=Path, help="OUTDIR the arms were run into")
ap.add_argument("--alpha", type=float, default=0.005)
ap.add_argument("--stat", type=int, default=20)
ap.add_argument("--ref", type=int, default=60)
ap.add_argument(
    "--stream",
    type=Path,
    default=None,
    help="stream root; enables the mean-density trace in panel 2",
)
ap.add_argument("-o", "--output", type=Path, default=None)
args = ap.parse_args()
D = args.rundir
OUT = args.output or (D / "adaptation_sequence.png")


def col(name, metric=M):
    p = D / name
    if not p.exists():
        return None
    rows = [
        (int(r["step"]), float(r["value"]))
        for r in csv.DictReader(open(p))
        if r["metric"] == metric
    ]
    rows.sort()
    return [v for _, v in rows] or None


def _window_density(stream_root, counts):
    """Mean n_e per monitoring window, for the physics trace in panel 2.

    Returns None when the stream root or netCDF4 is unavailable, so the figure
    still draws from the run CSVs alone.
    """
    if stream_root is None:
        return None
    try:
        import json

        import netCDF4
    except ImportError:
        return None
    manifest = Path(stream_root) / "stream_manifest.json"
    if not manifest.exists():
        return None
    arrivals = {a["index"]: a for a in json.load(open(manifest))["arrivals"]}
    cache, out = {}, []
    for i, count in enumerate(counts):
        info = arrivals[i]
        src = info["source"]
        if src not in cache:
            ds = netCDF4.Dataset(src)
            ne = np.asarray(ds["ne2d"][:])
            cache[src] = ne.reshape(ne.shape[0], -1).mean(axis=1)
            ds.close()
        lo, hi = info["valid_range"]
        out.append(cache[src][np.linspace(lo, hi - 1, count).round().astype(int)])
    return np.concatenate(out)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


ct, cl = col("stream_nocl.csv"), col("stream_cl.csv")
rp, orc = col("stream_nocl_replay.csv"), col("stream_oracle.csv")
have = [a for a in (ct, cl, rp, orc) if a]
n = min(len(a) for a in have)
ct, cl = ct[:n], cl[:n]
rp = rp[:n] if rp else None
orc = orc[:n] if orc else None

pat = re.compile(r"step=(\d+) \| model_stream \| ==== arrival (\d+)/\d+: (\w+) seg")
starts = [
    (int(m.group(2)), int(m.group(1)), m.group(3))
    for line in open(D / "run_cl.log", errors="ignore")
    for m in pat.finditer(line)
]
ev = sorted(
    int(r["step"])
    for r in csv.DictReader(open(D / "stream_cl.csv"))
    if r["metric"] == M
)
edges, labels, cum = [0], [], 0
for i, (a, s, case) in enumerate(starts):
    nxt = starts[i + 1][1] if i + 1 < len(starts) else 10**9
    cum += sum(1 for e in ev if s <= e < nxt)
    if cum > edges[-1] and cum <= n:
        edges.append(cum)
        labels.append((a, case))
x = np.empty(n)
for i in range(len(edges) - 1):
    lo, hi = edges[i], edges[i + 1]
    x[lo:hi] = labels[i][0] + np.linspace(0, 1, hi - lo, endpoint=False)
ARR = [a for a, _ in labels]
X0, X1 = ARR[0], ARR[-1] + 1


def arrival_of(window):
    """Which arrival a monitoring-window ordinal falls in."""
    return next(
        a for (a, _), lo, hi in zip(labels, edges[:-1], edges[1:]) if lo <= window < hi
    )


end = {}
for a, case in labels:
    end[case] = a + 1
REGIMES = [
    ("DIII-D baseline\n(in pre-training)", end["baseline_d3d"], "#eef2f8"),
    ("DIII-D held-out scenario\n(NOT in pre-training)", end["ood_d3d"], "#fdf0e9"),
    ("KSTAR\n(in pre-training)", X1, "#eef6f1"),
]
CHANGES = [e for _, e, _ in REGIMES[:-1]]  # stream/regime change points

fires_step = [
    int(r["step"])
    for r in csv.DictReader(open(D / "stream_cl.csv"))
    if r["metric"] == "drift/detected" and r["value"] in ("True", "1", "1.0")
]
fires = [sum(1 for e in ev if e <= f) - 1 for f in fires_step]
fires = [f for f in fires if 0 <= f < n]
pre, post = (
    col("stream_cl.csv", "eval/val_pre_cur_nrmse_mean"),
    col("stream_cl.csv", "eval/val_post_cur_nrmse_mean"),
)
K = min(len(pre), len(post), len(fires))
fire_arr = [arrival_of(f) for f in fires[:K]]

# --- continuous drift score: KS of the last `stat` windows vs a fixed reference
REF = np.array(ct[: args.ref])
score = np.full(n, np.nan)
for i in range(args.stat, n):
    _, p = stats.ks_2samp(REF, np.array(cl[i - args.stat : i]))
    score[i] = -np.log10(max(p, 1e-300))
THRESH = -np.log10(args.alpha)


def per_arr(values):
    """Collapse a per-window series to one mean per arrival."""
    return [st.mean(values[edges[i] : edges[i + 1]]) for i in range(len(edges) - 1)]


pa_ct, pa_cl = per_arr(ct), per_arr(cl)

fig = plt.figure(figsize=(13.5, 10.6))
fig.patch.set_facecolor(SURFACE)
gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 1.4, 0.9], hspace=0.38)


def frame(ax, label_regimes=False, mark_adapt=True):
    start = X0
    for lab, e, colour in REGIMES:
        ax.axvspan(start, e, color=colour, zorder=0)
        if label_regimes:
            ax.text(
                (start + e) / 2,
                ax.get_ylim()[1],
                lab,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=MUTED,
                linespacing=1.3,
            )
        start = e
    for a in ARR:
        ax.axvline(a, color=GRID, linewidth=0.7, zorder=1)
    for k, e in enumerate(CHANGES):
        ax.axvline(
            e,
            color=REGIME,
            linewidth=2.6,
            zorder=6,
            label="stream / regime change" if (k == 0 and label_regimes) else None,
        )
    if mark_adapt:
        for j, f in enumerate(fires[:K]):
            ax.axvline(
                x[f],
                color=ADAPTED,
                linewidth=1.3,
                linestyle=(0, (5, 3)),
                zorder=5,
                label="continual learning applied"
                if (j == 0 and label_regimes)
                else None,
            )
    ax.set_xlim(X0, X1)
    ax.set_xticks(ARR)


# 1. drift is detected
ax = fig.add_subplot(gs[0])
style(ax)
ax.set_ylim(0, float(np.nanmax(score)) * 1.25)
ax.plot(
    x,
    score,
    color=INK,
    linewidth=1.5,
    zorder=4,
    label="drift score  $-\\log_{10}p$  (KS test)",
)
ax.axhline(
    THRESH,
    color=REGIME,
    linewidth=1.4,
    linestyle="--",
    zorder=4,
    label=rf"detection threshold  $\alpha={args.alpha}$",
)
frame(ax, label_regimes=True)
d = fires[1] - edges[3] if len(fires) > 1 else 0
ax.set_ylabel("drift score", color=INK, fontsize=10)
ax.set_title(
    f"1.  drift is detected   —   the score jumps at the stream change and the "
    f"detector fires {d} windows later",
    color=INK,
    fontsize=11.5,
    loc="left",
    pad=24,
)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left", ncol=2)

# 2. CL applied and re-evaluated
bx = fig.add_subplot(gs[1])
style(bx)
series = [(ct, ORIGINAL, "pretrained model (no adaptation)", 1.4, "-")]
if orc:
    series.append((orc, ORACLE, "oracle (trained on all arrivals)", 1.3, "--"))
if rp:
    series.append((rp, REPLAY, "replay of final adapted model", 1.2, "-"))
series.append((cl, ADAPTED, "APEIRON continual learning", 1.5, "-"))
for vals, c, lab, lw, ls in series:
    bx.plot(x, vals, color=c, linewidth=lw, linestyle=ls, label=lab, alpha=0.92)
for j, f in enumerate(fires[:K]):
    xf = x[f]
    bx.plot([xf, xf], [pre[j], post[j]], color=INK, linewidth=1.8, zorder=6)
    bx.plot(
        xf,
        pre[j],
        "o",
        color=ORIGINAL,
        markersize=7,
        markeredgecolor=INK,
        zorder=7,
        label="error before CL round" if j == 0 else None,
    )
    bx.plot(
        xf,
        post[j],
        "o",
        color=ADAPTED,
        markersize=7,
        markeredgecolor=INK,
        zorder=7,
        label="error after CL round" if j == 0 else None,
    )
    dd = 100 * (1 - post[j] / pre[j])
    bx.text(
        xf,
        max(pre[j], post[j]) * 1.07,
        f"{'−' if dd > 0 else '+'}{abs(dd):.0f}%",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=INK if dd > 0 else LOSS,
    )
frame(bx, mark_adapt=True)
# What the plasma is actually doing, on its own axis: the error tracks how far
# the density has drifted from the regime the surrogate was pre-trained on.
dens = _window_density(
    args.stream, counts=[edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
)
if dens is not None:
    dens = dens[:n]
    tx = bx.twinx()
    tx.plot(
        x,
        dens,
        color=DENSITY,
        linewidth=2.2,
        alpha=0.55,
        zorder=2,
        label=r"mean density $\bar{n}_e$",
    )
    tx.set_ylabel(r"mean density  $\bar{n}_e$  (m$^{-3}$)", color=DENSITY, fontsize=10)
    tx.tick_params(axis="y", colors=DENSITY, labelsize=9)
    tx.spines["right"].set_color(DENSITY)
    for s in ("top", "left", "bottom"):
        tx.spines[s].set_visible(False)
    # Matplotlib draws whole axes in zorder order, so the twin has to sit above
    # bx or bx's regime shading paints over the density line.
    tx.set_zorder(3)
    bx.set_zorder(2)
    tx.patch.set_visible(False)
    tx.legend(frameon=False, fontsize=8.5, labelcolor=DENSITY, loc="upper right")
bx.set_ylabel("NRMSE per window", color=INK, fontsize=10)
bx.set_title(
    "2.  continual learning is applied, then re-evaluated on the arriving simulation",
    color=INK,
    fontsize=11.5,
    loc="left",
    pad=10,
)
bx.legend(frameon=False, fontsize=8.5, labelcolor=INK, ncol=3, loc="upper left")

# 3. adapted vs pretrained
cx = fig.add_subplot(gs[2])
style(cx)
diff = 100.0 * (np.array(ct) - np.array(cl)) / np.array(ct)
cx.fill_between(
    x,
    0,
    diff,
    where=diff >= 0,
    color=WIN,
    alpha=0.65,
    interpolate=True,
    label="adapted model BETTER than pretrained",
)
cx.fill_between(
    x,
    0,
    diff,
    where=diff < 0,
    color=LOSS,
    alpha=0.55,
    interpolate=True,
    label="adapted model worse",
)
cx.axhline(0, color=INK, linewidth=1.0)
cx.set_ylim(max(-120, float(np.min(diff)) * 1.05), float(np.max(diff)) * 1.35)
for k, a in enumerate(ARR):
    dd = 100 * (1 - pa_cl[k] / pa_ct[k])
    cx.text(
        a + 0.5,
        cx.get_ylim()[1] * 0.80,
        f"{dd:+.0f}%",
        ha="center",
        fontsize=8.5,
        color=WIN if dd >= 0 else LOSS,
        fontweight="bold",
    )
frame(cx)
cx.set_ylabel("error reduction vs\npretrained (%)", color=INK, fontsize=10)
cx.set_xlabel("simulation arrival", color=INK, fontsize=10)
cx.set_title(
    "3.  where the adapted model beats the pretrained one   (numbers = per-arrival mean)",
    color=INK,
    fontsize=11.5,
    loc="left",
    pad=10,
)
cx.legend(frameon=False, fontsize=8.5, labelcolor=INK, ncol=2, loc="lower left")

macro = st.mean(100 * (1 - b / a) for a, b in zip(pa_ct, pa_cl))
rows = [f"{'pretrained (no adaptation)':30s}{st.mean(ct):.5f}"]
rows.append(
    f"{'APEIRON continual learning':30s}{st.mean(cl):.5f}   "
    f"{100 * (1 - st.mean(cl) / st.mean(ct)):+.1f}%"
)
if rp:
    rows.append(
        f"{'replay of final model':30s}{st.mean(rp):.5f}   "
        f"{100 * (1 - st.mean(rp) / st.mean(ct)):+.1f}%"
    )
if orc:
    rows.append(
        f"{'oracle (all arrivals)':30s}{st.mean(orc):.5f}   "
        f"{100 * (1 - st.mean(orc) / st.mean(ct)):+.1f}%"
    )
rows.append("")
rows.append(
    f"per-arrival mean {macro:+.1f}%    adapted better in "
    f"{sum(1 for a, b in zip(cl, ct) if a < b)}/{n} windows    "
    f"{n} windows over {len(ARR)} arrivals ({edges[2] - edges[1]}/arrival)    "
    f"each CL round: 500 steps over 105 samples (~4.8 epochs)"
)
fig.text(
    0.02, 0.02, "\n".join(rows), family="monospace", fontsize=9, color=INK, va="top"
)

fig.savefig(OUT, dpi=185, facecolor=SURFACE, bbox_inches="tight")
print(
    "wrote",
    OUT,
    "| arms:",
    "control cl",
    "replay" if rp else "",
    "oracle" if orc else "",
)
print("adaptations at arrivals", fire_arr)
