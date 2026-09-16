#!/usr/bin/env python3
"""Drift detected -> CL applied -> is it better, and what did it cost the old data?

Three panels on one shared arrival axis (arrival k occupies x in [k, k+1)), plus
a fourth whenever ``eval_retrospective.py`` has been run: the same adapted models
re-evaluated, with no further training, on the opening arrivals they never
re-trained on. That panel is what separates adaptation from forgetting -- and it
is where replaying historical data shows its effect, so pass ``--mix`` to draw a
replaying arm alongside the plain one.

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
# Frozen reference in near-black: it is the baseline every arm is read
# against, not one more series competing for a hue.
ORIGINAL, ADAPTED, ORACLE, REPLAY = "#1a1a19", "#d62828", "#7a5c9e", "#9a9a96"
MIXED = "#0b8a6b"  # CL that also replays historical data
REGIME = "#eb6834"
GRID, INK, MUTED, SURFACE = "#d8d8d5", "#1a1a19", "#6b6b68", "#fcfcfb"
WIN, LOSS = "#3f8f6b", "#8c62c4"
# The four continual-learning outcomes, in the terms the literature uses.
ADAPT = WIN  # better on the arriving data: plasticity
NEG_TRANSFER = LOSS  # worse than not adapting at all
BWT_POS = "#2a4d8f"  # positive backward transfer
FORGET = "#c0392b"  # negative backward transfer: catastrophic forgetting
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
ap.add_argument("--control", default="nocl", help="frozen-model arm")
ap.add_argument("--cl", default="cl", help="continual-learning arm")
ap.add_argument(
    "--mix", default="", help="second CL arm that replays history, e.g. base_mix"
)
ap.add_argument(
    "--baseline-arrivals",
    default="",
    help="arrivals the forgetting panel scores, e.g. 0-7; "
    "default is the stream's first case",
)
ap.add_argument("-o", "--output", type=Path, default=None)
args = ap.parse_args()
D = args.rundir
OUT = args.output or (D / "adaptation_sequence.png")


def _manifest():
    """The stream manifest, from --stream or a copy left in the run directory."""
    import json

    for c in ([args.stream / "stream_manifest.json"] if args.stream else []) + [
        D / "stream_manifest.json"
    ]:
        if c.is_file():
            return json.loads(c.read_text())
    return {}


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


CTRL_CSV, CL_CSV = f"stream_{args.control}.csv", f"stream_{args.cl}.csv"
MIX_CSV = f"stream_{args.mix}.csv" if args.mix else None
ct, cl = col(CTRL_CSV), col(CL_CSV)
mx = col(MIX_CSV) if MIX_CSV else None
rp, orc = col("stream_nocl_replay.csv"), col("stream_oracle.csv")
have = [a for a in (ct, cl, mx, rp, orc) if a]
n = min(len(a) for a in have)
ct, cl = ct[:n], cl[:n]
mx = mx[:n] if mx else None
rp = rp[:n] if rp else None
orc = orc[:n] if orc else None

# Arrival boundaries. The stream harness writes `stream/arrival` into the metrics
# CSV, which shares the step axis with the metric being plotted; the console
# banner does not -- it carries a logger step counter that stays at 0 -- so the
# run log is only a fallback for runs recorded before that marker existed.
CASES = [a.get("case") for a in _manifest().get("arrivals", [])]
marks = [
    (int(r["step"]), int(float(r["value"])))
    for r in csv.DictReader(open(D / CL_CSV))
    if r["metric"] == "stream/arrival"
]
if marks:
    starts = [
        (idx, step, CASES[idx] if idx < len(CASES) else str(idx)) for step, idx in marks
    ]
else:
    pat = re.compile(r"step=(\d+) \| model_stream \| ==== arrival (\d+)/\d+: (\w+) seg")
    log = D / f"run_{args.cl}.log"
    starts = [
        (int(m.group(2)) - 1, int(m.group(1)), m.group(3))
        for line in open(log, errors="ignore")
        for m in pat.finditer(line)
    ]
ev = sorted(
    int(r["step"]) for r in csv.DictReader(open(D / CL_CSV)) if r["metric"] == M
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
# Rendered instead of the manifest's own key, so a device under a distribution
# restriction cannot reach a figure by way of its directory names.
CASE_LABEL = {
    "baseline_d3d": "DIII-D baseline\n(in pre-training)",
    "ood_d3d": "DIII-D held-out scenario\n(NOT in pre-training)",
    "kstar": "KSTAR\n(in pre-training)",
    "sparc": "third device\n(in pre-training)",
}
TINT = ["#eef2f8", "#fdf0e9", "#eef6f1", "#f3eff8"]
_seen = []
for _, case in labels:
    if case not in _seen:
        _seen.append(case)
REGIMES = []
for i, case in enumerate(_seen):
    if case not in CASE_LABEL:
        raise KeyError(
            f"No display label for case {case!r}. Add one to CASE_LABEL rather "
            f"than letting a raw manifest key reach a figure."
        )
    REGIMES.append((CASE_LABEL[case], end.get(case, X1), TINT[i % len(TINT)]))
CHANGES = [e for _, e, _ in REGIMES[:-1]]  # stream/regime change points

fires_step = [
    int(r["step"])
    for r in csv.DictReader(open(D / CL_CSV))
    if r["metric"] == "drift/detected" and r["value"] in ("True", "1", "1.0")
]
fires = [sum(1 for e in ev if e <= f) - 1 for f in fires_step]
fires = [f for f in fires if 0 <= f < n]
pre, post = (
    col(CL_CSV, "eval/val_pre_cur_nrmse_mean"),
    col(CL_CSV, "eval/val_post_cur_nrmse_mean"),
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

RETRO = sorted(D.glob("retro_*.csv"))
fig = plt.figure(figsize=(13.5, 12.3))
fig.patch.set_facecolor(SURFACE)
gs = fig.add_gridspec(4, 1, height_ratios=[0.9, 1.7, 1.0, 0.64], hspace=0.10)


def arrival_spec(spec, n):
    """Parse "0-7" or "0,4,8" into a set of arrival indices; empty means all."""
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    return out or set(range(n))


def retro_by_arrival(arm):
    """{arrival: [window values]} for the pretrained model and for the final one.

    Same rows ``replayed()`` draws, kept grouped by arrival so the summary can
    restrict backward transfer to the arrivals that are in MATEY's pre-training
    corpus and that no adaptation round trained on.
    """
    from collections import defaultdict

    path = D / f"retro_{arm}.csv"
    if not path.exists():
        return None
    rows_ = [r for r in csv.DictReader(open(path)) if r["metric"] == "nrmse_mean"]
    if not rows_:
        return None
    last = max(int(r["event_id"]) for r in rows_)
    out = []
    for want in (0, last):
        d = defaultdict(list)
        for r in rows_:
            if int(r["event_id"]) == want:
                d[int(r["eval_arrival"])].append((int(r["window"]), float(r["value"])))
        out.append({a: [v for _, v in sorted(d[a])] for a in d})
    return out


def gains(ref, new):
    """(mean of per-window relative gains, reduction of the mean error), percent.

    The two disagree whenever the error varies across the stream: the first
    weights every window equally, the second is dominated by the windows where
    the error was largest. Quoting one as though it were the other is what makes
    "the same" result read as 3.4% in one place and 8.6% in another, so the
    figure prints both.
    """
    ref_a, new_a = np.asarray(ref, float), np.asarray(new, float)
    return (
        float(np.mean(100.0 * (ref_a - new_a) / ref_a)),
        float(100.0 * (ref_a.mean() - new_a.mean()) / ref_a.mean()),
    )


def replayed(arm):
    """(x, y) for an arm's FINAL model, re-evaluated with no further training.

    Resolution follows the CSV: per monitoring window where
    ``eval_retrospective.py`` recorded one, otherwise one point per arrival.
    Windows are spread across their arrival exactly as the online curve is, so
    the two are directly comparable point for point.

    Event 0 is the un-adapted model, so the control's name gives the pretrained
    reference -- and it coincides with that arm's own online curve, which is the
    check that both paths measure the same thing.
    """
    from collections import defaultdict

    # The control writes no checkpoints and so has no retrospective of its own,
    # but event 0 of every arm IS the un-adapted model -- so borrow the first
    # file available rather than requiring a run that cannot exist.
    path = D / f"retro_{arm}.csv"
    if arm == args.control and not path.exists():
        path = next(iter(RETRO), path)
    if not path.exists():
        return None, None
    rows_ = [r for r in csv.DictReader(open(path)) if r["metric"] == "nrmse_mean"]
    if not rows_:
        return None, None
    events = {int(r["event_id"]) for r in rows_}
    want = 0 if arm == args.control else max(events)
    per_arr_ = defaultdict(list)
    has_window = "window" in rows_[0]
    for r in rows_:
        if int(r["event_id"]) != want:
            continue
        key = int(r["window"]) if has_window else 0
        per_arr_[int(r["eval_arrival"])].append((key, float(r["value"])))
    if not per_arr_:
        return None, None
    xs_, ys_ = [], []
    for a in sorted(per_arr_):
        pts = [v for _, v in sorted(per_arr_[a])]
        for i, v in enumerate(pts):
            xs_.append(a + (i + 0.5) / len(pts))
            ys_.append(v)
    return xs_, ys_


def tag(ax, text):
    """The panel's name, boxed inside it, so the panels can sit flush."""
    ax.text(
        0.006,
        0.955,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=INK,
        zorder=12,
        bbox=dict(
            boxstyle="square,pad=0.42",
            facecolor=SURFACE,
            edgecolor=INK,
            linewidth=1.1,
        ),
    )


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
# Windows between the first regime change and the first detection after it.
_first_change = edges[next(i for i, (a, _) in enumerate(labels) if a >= CHANGES[0])]
d = next((f - _first_change for f in fires if f >= _first_change), 0)
ax.set_ylabel("drift score", color=INK, fontsize=10)
tag(ax, "1   Drift detection")
ax.set_xlabel("")
ax.legend(
    frameon=False,
    fontsize=8.5,
    labelcolor=INK,
    loc="upper left",
    bbox_to_anchor=(0.155, 1.0),
    ncol=2,
)

# 2. CL applied and re-evaluated
bx = fig.add_subplot(gs[1])
style(bx)
series = [(ct, ORIGINAL, "pretrained (frozen)", 1.4, "-")]
if orc:
    series.append((orc, ORACLE, "joint oracle", 1.3, "--"))
if rp:
    series.append((rp, REPLAY, "final model, replay", 1.2, "-"))
series.append((cl, ADAPTED, f"CL ({args.cl})", 1.6, "-"))
if mx:
    # The winning arm carries the panel: heaviest line, drawn last.
    series.append((mx, MIXED, f"CL + history ({args.mix})  ← best", 1.8, "-"))
for vals, c, lab, lw, ls in series:
    bx.plot(
        x,
        vals,
        color=c,
        linewidth=lw,
        linestyle=ls,
        label=lab,
        alpha=0.95,
        zorder=8 if c == ORIGINAL else 6,
    )
if mx:
    # Direct-label the winner where it is most separated from the others rather
    # than relying on legend order.
    j = int(np.argmax(np.array(cl) - np.array(mx)))
    bx.annotate(
        "best arm",
        (x[j], mx[j]),
        xytext=(-6, -46),
        textcoords="offset points",
        fontsize=9.5,
        fontweight="bold",
        color=MIXED,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=MIXED, linewidth=1.4),
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor=SURFACE,
            alpha=0.9,
            edgecolor="none",
        ),
        zorder=8,
    )
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
        label="pre-CL" if j == 0 else None,
    )
    bx.plot(
        xf,
        post[j],
        "o",
        color=ADAPTED,
        markersize=7,
        markeredgecolor=INK,
        zorder=7,
        label="post-CL" if j == 0 else None,
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
# The same models after the whole stream, evaluated without training: dash-dot,
# in each arm's own colour. The pretrained model measured the same way is the
# reference, and the shaded gap between them is the forgetting.
rx0, ry0 = replayed(args.control)
for arm, colour in ((args.cl, ADAPTED), (args.mix, MIXED)):
    if not arm:
        continue
    rx, ry = replayed(arm)
    if not rx:
        continue
    bx.plot(
        rx,
        ry,
        color=colour,
        linewidth=1.6,
        linestyle=(0, (6, 2, 1, 2)),
        zorder=5,
        label=f"{arm}, replay",
    )
    if rx0:
        ref_i = np.interp(rx, rx0, ry0)
        worse = np.array(ry) > ref_i
        bx.fill_between(
            rx,
            ref_i,
            ry,
            where=worse,
            interpolate=True,
            color=colour,
            alpha=0.30,
            lw=0,
            zorder=3,
            label=f"negative BWT ({arm})" if arm == args.cl else None,
        )
if rx0:
    bx.plot(
        rx0,
        ry0,
        color=ORIGINAL,
        linewidth=1.4,
        linestyle=(0, (6, 2, 1, 2)),
        zorder=5,
        label="pretrained, replay",
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
    for side in ("top", "left", "bottom"):
        tx.spines[side].set_visible(False)
    # Matplotlib draws whole axes in zorder order, so the twin has to sit above
    # bx or bx's regime shading paints over the density line.
    tx.set_zorder(3)
    bx.set_zorder(2)
    tx.patch.set_visible(False)
    tx.legend(frameon=False, fontsize=8.5, labelcolor=DENSITY, loc="upper right")
# Log: the forgetting signal lives near 0.011 while the held-out excursion
# reaches 0.055, so on a linear axis a 13% gap on the baseline arrivals is a
# hairline. The regime structure survives the change; the gap becomes readable.
bx.set_yscale("log")
bx.set_ylabel("NRMSE per window", color=INK, fontsize=10)
tag(bx, "2   Adaptation")
bx.legend(
    frameon=False,
    fontsize=8.5,
    labelcolor=INK,
    ncol=3,
    loc="upper left",
    bbox_to_anchor=(0.155, 1.0),
)

# 3. the two continual-learning outcomes, named as the literature names them
#
#    Solid, on the arriving simulation: positive is adaptation (plasticity),
#    negative is negative transfer -- adapting made it worse than not adapting.
#    Hatched, the finished model re-evaluated everywhere: this is backward
#    transfer. Negative BWT is catastrophic forgetting; positive BWT is the
#    opposite, learning later arrivals having *helped* the earlier ones.
cx = fig.add_subplot(gs[2])
style(cx)

SHOW = args.mix if mx else args.cl
SHOW_VALS = mx if mx else cl
gain = 100.0 * (np.array(ct) - np.array(SHOW_VALS)) / np.array(ct)
cx.fill_between(
    x,
    0,
    gain,
    where=gain >= 0,
    color=ADAPT,
    alpha=0.60,
    interpolate=True,
    zorder=2,
    label="adaptation",
)
cx.fill_between(
    x,
    0,
    gain,
    where=gain < 0,
    color=NEG_TRANSFER,
    alpha=0.50,
    interpolate=True,
    zorder=2,
    label="negative transfer",
)

rgx, rgy = replayed(SHOW)
rp_gain = None
if rgx and rx0:
    ref_i = np.interp(rgx, rx0, ry0)
    rp_gain = 100.0 * (ref_i - np.array(rgy)) / ref_i
    cx.fill_between(
        rgx,
        0,
        rp_gain,
        where=rp_gain >= 0,
        facecolor=BWT_POS,
        alpha=0.38,
        edgecolor=BWT_POS,
        hatch="////",
        linewidth=1.0,
        interpolate=True,
        zorder=3,
        label="positive BWT",
    )
    cx.fill_between(
        rgx,
        0,
        rp_gain,
        where=rp_gain < 0,
        facecolor=FORGET,
        alpha=0.38,
        edgecolor=FORGET,
        hatch="\\\\",
        linewidth=1.0,
        interpolate=True,
        zorder=3,
        label="negative BWT",
    )
cx.axhline(0, color=INK, linewidth=1.1, zorder=5)

lo_y = min(float(np.min(gain)), float(np.min(rp_gain)) if rp_gain is not None else 0.0)
hi_y = max(float(np.max(gain)), float(np.max(rp_gain)) if rp_gain is not None else 0.0)
cx.set_ylim(max(-120, lo_y * 1.15), hi_y * 1.18)

frame(cx)
cx.set_ylabel("error reduction vs\npretrained (%)", color=INK, fontsize=10)
cx.set_xlabel("simulation arrival", color=INK, fontsize=10)
_net = f"{SHOW}:  net adaptation {np.mean(gain):+.1f}%"
if rp_gain is not None:
    _net += f"      net BWT, all arrivals {np.mean(rp_gain):+.1f}%"
_leg = cx.legend(
    frameon=False,
    fontsize=9,
    labelcolor=INK,
    ncol=2,
    loc="lower left",
    bbox_to_anchor=(0.02, 0.0),
    title=_net,
    title_fontsize=9.5,
)
_leg.get_title().set_fontweight("bold")
_leg.get_title().set_color(INK)
tag(cx, "3   Plasticity  &  backward transfer")


def _train_frames():
    """Frames in an arrival's train split, from the manifest."""
    for a in _manifest().get("arrivals", []):
        r = a.get("train_range")
        if r:
            return r[1] - r[0]
    return "?"


# Absolute errors and stream provenance. The percentages all live in panel 4
# now; repeating them here is what let one number be quoted for another.
abs_ = [f"{'pretrained (frozen)':22s}{st.mean(ct):.5f}", f"{'CL':22s}{st.mean(cl):.5f}"]
if mx:
    abs_.append(f"{'CL + history':22s}{st.mean(mx):.5f}")
if rp:
    abs_.append(f"{'final model, replay':22s}{st.mean(rp):.5f}")
if orc:
    abs_.append(f"{'joint oracle':22s}{st.mean(orc):.5f}")
abs_.append("")
abs_.append(
    f"mean NRMSE above.  CL better in "
    f"{sum(1 for a, b in zip(cl, ct) if a < b)}/{n} windows;   "
    f"{n} windows over {len(ARR)} arrivals ({edges[2] - edges[1]}/arrival);   "
    f"{_train_frames()} training frames per arrival"
)
fig.text(
    0.02,
    0.035,
    "\n".join(abs_),
    family="monospace",
    fontsize=8.6,
    color=MUTED,
    va="top",
)

fig.suptitle(
    "Drift detection → continual learning → what it cost the original data\n"
    "solid: on the arriving simulation      dash-dot: replay of the same models "
    "after the stream",
    fontsize=13,
    fontweight="bold",
    color=INK,
    x=0.055,
    ha="left",
    y=0.995,
    linespacing=1.5,
)

# 4. the numbers behind panels 2 and 3, stated rather than left to be inferred
#
#    Three questions, one row per arm: does adapting help on the simulation that
#    just arrived; does the finished model still hold the arrivals MATEY was
#    pre-trained on and that no round ever trained on; and does it hold the
#    stream as a whole.
dx = fig.add_subplot(gs[3])
dx.set_facecolor(SURFACE)
dx.set_xlim(0, 1)
dx.set_ylim(0, 1)
dx.axis("off")

BASE_SET = arrival_spec(args.baseline_arrivals, len(ARR))
BASE_TXT = args.baseline_arrivals or "all"


def per_round(arm):
    """Mean pre-CL -> post-CL drop over the rounds, which is what panel 2 marks.

    Its reference is the arm's own model as it stood when drift fired, not the
    pre-trained model -- so a round that is undoing the previous round's damage
    scores just as well as one that learned something. That is why the arm with
    the larger number here can be the worse arm two columns to the right.
    """
    pre_ = col(f"stream_{arm}.csv", "eval/val_pre_cur_nrmse_mean")
    post_ = col(f"stream_{arm}.csv", "eval/val_post_cur_nrmse_mean")
    if not pre_ or not post_:
        return None
    k = min(len(pre_), len(post_))
    a_, b_ = np.array(pre_[:k]), np.array(post_[:k])
    return float(np.mean(100.0 * (a_ - b_) / a_)), None, k


def summary(arm, online):
    """One row: per round, over the stream, then backward transfer twice."""
    cells = [per_round(arm), gains(ct, online)]
    rb = retro_by_arrival(arm)
    for sel in (BASE_SET, None):
        if rb is None:
            cells.append(None)
            continue
        pre_, post_ = rb
        keys = [a for a in sorted(pre_) if sel is None or a in sel]
        cells.append(
            gains(
                [v for a in keys for v in pre_[a]],
                [v for a in keys for v in post_[a]],
            )
        )
    return cells


COLX = [0.005, 0.30, 0.51, 0.72, 0.93]
NR = len(per_round(args.cl) or (0, 0, 0)) and (per_round(args.cl) or (0, 0, 0))[2]
for cx_, head in zip(
    COLX[1:],
    [
        f"within one round\npre-CL → post-CL, {NR} events",
        "over the whole stream\nvs the frozen model, every window",
        f"backward transfer\npre-training arrivals {BASE_TXT}",
        "backward transfer\nall arrivals",
    ],
):
    dx.text(
        cx_,
        0.74,
        head,
        fontsize=9.2,
        color=MUTED,
        ha="center",
        va="top",
        linespacing=1.45,
    )

ROWS = [(f"CL ({args.cl})", cl, args.cl, ADAPTED)]
if mx:
    ROWS.append((f"CL + history ({args.mix})", mx, args.mix, MIXED))
for i, (lab, vals, arm_, colour) in enumerate(ROWS):
    y = 0.30 - 0.26 * i
    dx.text(
        COLX[0],
        y,
        lab,
        fontsize=10,
        color=colour,
        ha="left",
        va="center",
        fontweight="bold",
    )
    for k, (cx_, cell) in enumerate(zip(COLX[1:], summary(arm_, vals))):
        if cell is None:
            dx.text(cx_, y, "--", fontsize=11, color=MUTED, ha="center", va="center")
            continue
        per_win, total = cell[0], cell[1]
        # Column 1 stays neutral: it is measured against a moving reference, so a
        # larger number there is not a better arm -- as these two rows show.
        dx.text(
            cx_,
            y,
            f"{per_win:+.1f}%",
            fontsize=13,
            ha="right",
            va="center",
            fontweight="bold",
            color=INK if k == 0 else (MIXED if per_win >= 0 else FORGET),
        )
        if total is not None:
            dx.text(
                cx_ + 0.008,
                y,
                f"  ({total:+.1f}%)",
                fontsize=10,
                ha="left",
                va="center",
                color=MUTED,
            )

dx.axhline(0.50, color=GRID, linewidth=1.0)
dx.text(
    COLX[0],
    -0.22,
    "positive = error reduced.  Column 1 is measured against the arm's own model just "
    "before that round; the other three against the frozen pre-trained model, so the "
    "four are not on one scale.\nBold is the mean over monitoring windows, which is "
    "what panel 3 shades; bracketed is the reduction of the mean error, which the "
    "largest-error windows dominate.",
    fontsize=8.6,
    color=MUTED,
    ha="left",
    va="center",
)
tag(dx, "4   In numbers")

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
