#!/usr/bin/env python3
"""Publication figures for APEIRON drift detection on MATEY / SOLPS-ITER data.

Produces four figures sized for an Elsevier ``elsarticle`` full-width
(a4paper, 12pt) layout:

  fig1_solps_drift_detection   data-based drift detection across the regime change
  fig2_detector_response       detector behaviour vs. monitoring cadence
  fig3_matey_performance       performance-based monitoring + root-cause diagnosis
  fig4_field_maps              where in the plasma the held-out regime differs
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

# --- validated categorical palette (CVD-checked, fixed order) ---------------
C_BLUE = "#0072B2"  # in-pre-training / baseline
C_VERM = "#D55E00"  # held-out / OOD
C_GREEN = "#009E73"
C_ORANGE = "#E69F00"
C_PINK = "#CC79A7"
C_SKY = "#56B4E9"
INK = "#1a1a1a"
INK2 = "#4d4d4d"
MUTED = "#8a8a8a"
GRID = "#dcdcdc"

SEQ = "viridis"
DIV = "RdBu_r"
TEXTWIDTH = 6.3  # inches

DETECTORS = [
    ("ADWIN library default", C_GREEN, "o"),
    ("ADWIN tuned to stream", C_ORANGE, "s"),
    ("KSWIN library default", C_PINK, "^"),
    ("KSWIN tuned to stream", C_SKY, "v"),
    ("Page-Hinkley library default", C_BLUE, "D"),
    ("Page-Hinkley tuned to stream", C_VERM, "P"),
]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.edgecolor": INK2,
            "axes.linewidth": 0.7,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": GRID,
            "lines.linewidth": 1.5,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def save(fig, outdir: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")
    print(f"  wrote {outdir / name}.pdf / .png")
    plt.close(fig)


def window_reduce(x: np.ndarray, w: int) -> np.ndarray:
    n = (len(x) // w) * w
    return x[:n].reshape(-1, w).mean(1)


def panel_tag(ax, txt, x=0.008, y=0.88):
    ax.text(
        x,
        y,
        txt,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="top",
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2),
    )


def corner_tag(ax, txt, dx=-0.02):
    """Panel tag placed outside the axes, clear of titles and legends."""
    ax.text(
        dx,
        1.02,
        txt,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def first_fire(det: dict, boundary: int):
    after = [f for f in det["fired"] if f >= boundary]
    return after[0] if after else None


def leaves_envelope(score: np.ndarray, thr: float, boundary: int, run: int = 3) -> int:
    for i in range(boundary, len(score) - run):
        if np.all(score[i : i + run] > thr):
            return i
    return boundary


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------
def fig1(res: dict, npz, outdir: Path) -> dict:
    meta, dense = res["meta"], res["runs"]["dense"]
    w = dense["window_frames"]
    score = np.array(dense["score"])
    bnd = dense["boundary_window"]
    thr = meta["ref_score_p99"]
    n = len(score)
    exceed = leaves_envelope(score, thr, bnd)

    tflux = window_reduce(npz["stream_desc"][:, 0], w) / 1e23
    ref_tflux = npz["base_desc"][: meta["n_ref_frames"], 0] / 1e23

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(TEXTWIDTH, 6.1),
        sharex=True,
        height_ratios=[1.0, 1.35, 0.8, 0.95],
    )
    fig.subplots_adjust(hspace=0.13)
    x = np.arange(n)

    def bg(ax, top=None):
        ax.axvspan(0, bnd, color=C_BLUE, alpha=0.05, lw=0)
        ax.axvspan(bnd, n, color=C_VERM, alpha=0.05, lw=0)
        ax.axvline(bnd, color=INK2, lw=1.0)

    # (a) actuator
    ax = axes[0]
    bg(ax)
    ax.fill_between(
        [0, n], ref_tflux.min(), ref_tflux.max(), color=C_BLUE, alpha=0.18, lw=0
    )
    ax.plot(x[: bnd + 1], tflux[: bnd + 1], color=C_BLUE, lw=1.5)
    ax.plot(x[bnd:], tflux[bnd:], color=C_VERM, lw=1.5)
    ax.set_ylabel("gas puff $\\Gamma$\n[$10^{23}\\,$s$^{-1}$]")
    lo, hi = tflux.min(), tflux.max()
    ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.30 * (hi - lo))
    ax.set_title(
        "APEIRON data-based drift detection for the MATEY TurBT surrogate\n"
        "SOLPS-ITER edge-plasma stream, DIII-D 174310",
        pad=5,
    )
    ax.annotate(
        "in pre-training\n(Sequence_sin4)",
        xy=(bnd * 0.5, 0.13),
        xycoords=("data", "axes fraction"),
        ha="center",
        fontsize=7.0,
        color=C_BLUE,
    )
    ax.annotate(
        "held out of pre-training\n(noLat_dribble)",
        xy=(bnd + (n - bnd) * 0.22, 0.06),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=C_VERM,
    )
    panel_tag(ax, "(a)")

    # (b) drift score
    ax = axes[1]
    bg(ax)
    ax.plot(x[: bnd + 1], score[: bnd + 1], color=C_BLUE, lw=1.5)
    ax.plot(x[bnd:], score[bnd:], color=C_VERM, lw=1.5)
    ax.axhline(thr, color=MUTED, lw=1.0, ls=(0, (1.5, 1.5)))
    ax.annotate(
        f"in-distribution $p_{{99}}$ = {thr:.2f}",
        xy=(n * 0.985, thr),
        ha="right",
        va="top",
        fontsize=6.6,
        color=MUTED,
    )
    ax.set_ylabel("data drift score\n(mean KS vs. pre-training)")
    ax.set_ylim(0, min(1.0, score.max() * 1.45))
    ax.annotate(
        f"peak KS = {score.max():.2f}  "
        f"($\\times${score[bnd:].max() / score[:bnd].mean():.1f} the in-distribution level)",
        xy=(int(np.argmax(score)), score.max()),
        xytext=(int(np.argmax(score)) - 8, min(0.99, score.max() * 1.30)),
        fontsize=6.8,
        color=C_VERM,
        ha="right",
        arrowprops=dict(arrowstyle="->", color=C_VERM, lw=0.7),
    )
    panel_tag(ax, "(b)")

    # (c) per-field KS, so each field is compared against its own reference
    ax = axes[2]
    bg(ax)
    pf = np.array(dense["per_field_ks"])
    order = dense["field_order"]
    pretty = {"ne": "$n_e$", "te": "$T_e$", "ti": "$T_i$"}
    for j, nm in enumerate(order):
        ax.plot(
            np.arange(len(pf)),
            pf[:, j],
            lw=1.3,
            color=[C_VERM, C_GREEN, C_ORANGE][j % 3],
            label=pretty.get(nm, nm),
        )
    ax.set_ylabel("per-field KS\nvs. pre-training")
    ax.set_ylim(0, pf.max() * 1.35)
    ax.legend(
        loc="upper left", ncol=3, fontsize=6.4, columnspacing=1.0, handlelength=1.4
    )
    panel_tag(ax, "(c)")

    # (d) detector event strip
    ax = axes[3]
    bg(ax)
    ax.set_ylim(-0.6, len(DETECTORS) - 0.4)
    ax.set_yticks(range(len(DETECTORS)))
    ax.set_yticklabels([d[0] for d in DETECTORS], fontsize=6.4)
    ax.grid(axis="y", color=GRID, lw=0.4)
    fires = {}
    for i, (name, col, mk) in enumerate(DETECTORS):
        f0 = first_fire(dense["detectors"][name], bnd)
        fires[name] = f0
        ax.hlines(i, 0, n, color=GRID, lw=0.6, zorder=1)
        if f0 is None:
            ax.plot([n * 0.5], [i], marker="x", ms=6, mew=1.6, color=MUTED, zorder=4)
            ax.text(
                n * 0.5 + 4, i, "never fires", va="center", fontsize=6.2, color=MUTED
            )
        else:
            ax.hlines(i, bnd, f0, color=col, lw=2.4, alpha=0.55, zorder=3)
            ax.plot(
                [f0], [i], marker=mk, ms=5.5, color=col, mec="white", mew=0.8, zorder=5
            )
            ax.text(
                f0 + 3,
                i,
                f"+{(f0 - bnd) * w} frames",
                va="center",
                fontsize=6.2,
                color=col,
            )
    ax.set_xlabel(f"monitoring window   (1 window = {w} SOLPS frames)")
    ax.set_xlim(0, n)
    panel_tag(ax, "(d)  detection latency", x=0.008, y=0.99)

    save(fig, outdir, "fig1_solps_drift_detection")
    return {
        "boundary_window": int(bnd),
        "window_frames": int(w),
        "envelope_p99": float(thr),
        "window_leaving_envelope": int(exceed),
        "baseline_score_mean": float(score[:bnd].mean()),
        "baseline_score_max": float(score[:bnd].max()),
        "ood_score_mean": float(score[bnd:].mean()),
        "ood_score_max": float(score[bnd:].max()),
        "peak_excursion_x": float(score[bnd:].max() / score[:bnd].mean()),
        "per_field_ks_baseline": {
            n: float(np.array(dense["per_field_ks"])[:bnd, j].mean())
            for j, n in enumerate(dense["field_order"])
        },
        "per_field_ks_ood": {
            n: float(np.array(dense["per_field_ks"])[bnd:, j].mean())
            for j, n in enumerate(dense["field_order"])
        },
        "first_fire_windows": {k: v for k, v in fires.items()},
    }


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------
def fig2(res: dict, outdir: Path) -> dict:
    dense, short = res["runs"]["dense"], res["runs"]["short"]
    thr = res["meta"]["ref_score_p99"]
    s = np.array(short["score"])
    b = short["boundary_window"]
    w = short["window_frames"]

    fig = plt.figure(figsize=(TEXTWIDTH, 3.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.24, top=0.80)

    # (a) coarse-cadence stream
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(s))
    ax.axvspan(0, b, color=C_BLUE, alpha=0.05, lw=0)
    ax.axvspan(b, len(s), color=C_VERM, alpha=0.05, lw=0)
    ax.plot(x[: b + 1], s[: b + 1], color=C_BLUE, lw=1.5, marker="o", ms=3.2)
    ax.plot(x[b:], s[b:], color=C_VERM, lw=1.5, marker="o", ms=3.2)
    ax.axvline(b, color=INK2, lw=1.0)
    ax.axhline(thr, color=MUTED, lw=1.0, ls=(0, (1.5, 1.5)))
    ax.set_xlabel(f"monitoring window ({w} frames)")
    ax.set_ylabel("data drift score (KS)")
    ax.set_ylim(0, min(1.0, s.max() * 1.5))
    ax.set_title(f"Coarse cadence ({len(s)} windows)", fontsize=8, pad=4)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=C_BLUE,
                lw=1.5,
                marker="o",
                ms=3.2,
                label="in pre-training",
            ),
            Line2D([], [], color=C_VERM, lw=1.5, marker="o", ms=3.2, label="held out"),
            Line2D(
                [],
                [],
                color=MUTED,
                ls=(0, (1.5, 1.5)),
                lw=1.0,
                label="in-distribution $p_{99}$",
            ),
        ],
        loc="upper left",
        fontsize=6.4,
    )
    corner_tag(ax, "(a)", dx=-0.16)

    # (b) delay per detector / cadence
    ax = fig.add_subplot(gs[0, 1])
    names = [d[0] for d in DETECTORS]
    cadences = [
        (
            "fine",
            dense,
            C_BLUE,
            f"fine: {dense['n_windows']} win ({dense['window_frames']} fr)",
        ),
        ("coarse", short, C_ORANGE, f"coarse: {short['n_windows']} win ({w} fr)"),
    ]
    vals = {}
    for tag, run, _, _ in cadences:
        for nm in names:
            d = run["detectors"][nm]
            f0 = first_fire(d, run["boundary_window"])
            vals[(tag, nm)] = (
                (f0 - run["boundary_window"]) * run["window_frames"]
                if f0 is not None
                else None
            )
            vals[(tag, nm, "fa")] = d["false_alarms_before_boundary"]

    finite = [v for v in vals.values() if isinstance(v, int) and v is not None]
    cap = max(finite) * 1.18
    xpos = np.arange(len(names))
    bw = 0.38
    for j, (tag, run, col, _) in enumerate(cadences):
        for i, nm in enumerate(names):
            v = vals[(tag, nm)]
            fa = vals[(tag, nm, "fa")]
            xx = xpos[i] + (j - 0.5) * (bw + 0.02)
            if v is None:
                ax.bar(
                    xx,
                    cap,
                    bw,
                    color="white",
                    edgecolor=MUTED,
                    hatch="///",
                    lw=0.9,
                    zorder=3,
                )
                ax.text(
                    xx,
                    cap * 0.5,
                    "never\nfires",
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color=INK2,
                    rotation=90,
                    linespacing=0.9,
                )
            else:
                ax.bar(xx, v, bw, color=col, edgecolor="white", lw=0.7, zorder=3)
                lbl = f"{v}" + (f"$^{{{fa}}}$" if fa else "")
                ax.text(
                    xx,
                    v + cap * 0.025,
                    lbl,
                    ha="center",
                    va="bottom",
                    fontsize=6.2,
                    color=INK2,
                )
    ax.set_xticks(xpos)
    ax.set_xticklabels(
        [
            n.replace("Page-Hinkley", "Page-\nHinkley").replace(" ", "\n", 1)
            if "Page" not in n
            else n.replace("Page-Hinkley ", "Page-\nHinkley\n")
            for n in names
        ],
        fontsize=6.2,
    )
    ax.set_ylabel("detection delay [SOLPS frames]")
    ax.set_ylim(0, cap * 1.42)
    ax.set_title("Detection delay after the change point", fontsize=8, pad=4)
    ax.legend(
        handles=[mpatches.Patch(color=c, label=lab) for _, _, c, lab in cadences]
        + [
            mpatches.Patch(
                facecolor="white", edgecolor=MUTED, hatch="///", label="no detection"
            )
        ],
        loc="upper center",
        ncol=3,
        fontsize=6.0,
        columnspacing=0.9,
        handlelength=1.3,
        borderpad=0.3,
    )
    ax.text(
        0.995,
        0.845,
        "superscript = false alarms before the change point",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=MUTED,
    )
    corner_tag(ax, "(b)", dx=-0.10)

    save(fig, outdir, "fig2_detector_response")
    return {f"{k[0]}/{k[1]}": v for k, v in vals.items() if len(k) == 2}


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------
def fig3(matey_run: Path, base_nc: str, outdir: Path) -> dict:
    rows = list(csv.DictReader(open(matey_run / "matey_inference_drift.csv")))
    series = collections.defaultdict(list)
    for r in rows:
        series[r["metric"]].append((int(r["step"]), r["value"]))

    def get(k):
        return np.array([float(a) for _, a in sorted(series[k])])

    fields = ["ne2d", "te2d", "ti2d"]
    nr = {f: get(f"eval/nrmse_{f}") for f in fields}
    nmean = get("eval/nrmse_mean")
    nb = len(nmean) // 2

    fig = plt.figure(figsize=(TEXTWIDTH, 4.7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.62, wspace=0.34)

    # (a) per-field NRMSE
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(nmean))
    ax.axvspan(-0.5, nb - 0.5, color=C_BLUE, alpha=0.05, lw=0)
    ax.axvspan(nb - 0.5, len(x) - 0.5, color=C_VERM, alpha=0.05, lw=0)
    ax.axvline(nb - 0.5, color=INK2, lw=1.0)
    for f, col, mk in zip(fields, (C_VERM, C_GREEN, C_ORANGE), ("o", "s", "^")):
        ax.plot(x, nr[f], color=col, marker=mk, ms=3.0, lw=1.3, label=f)
    ax.plot(x, nmean, color=INK, lw=1.7, ls="--", label="mean (monitored)")
    ax.axhline(0.011, color=MUTED, lw=1.0, ls=(0, (1.5, 1.5)))
    ax.text(
        len(x) - 0.8,
        0.05,
        "FusionBench reference NRMSE = 0.011",
        va="bottom",
        ha="right",
        fontsize=6.4,
        color=MUTED,
    )
    ax.set_ylim(0, 1.30)
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylabel("NRMSE")
    ax.set_xlabel("evaluation batch")
    ax.set_title(
        "Performance-based monitoring of MATEY is blind to the regime change:\n"
        "the surrogate is already at no-skill error on data it was trained on",
        fontsize=8.2,
        pad=4,
    )
    ax.text(nb * 0.5, 1.20, "in pre-training", ha="center", fontsize=7.0, color=C_BLUE)
    ax.text(nb * 1.5, 1.20, "held out", ha="center", fontsize=7.0, color=C_VERM)
    ax.legend(
        loc="lower left", ncol=4, fontsize=6.4, columnspacing=1.0, handlelength=1.6
    )
    corner_tag(ax, "(a)", dx=-0.055)

    # (b,c) predicted vs ground truth
    stats = {}
    P, T = [], []
    for p in sorted((matey_run / "artifacts" / "stream_01_baseline").glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        P.append(z["pred"])
        T.append(z["target"])
    P, T = np.stack(P), np.stack(T)
    for j, f in enumerate(["ne2d", "te2d"]):
        ax = fig.add_subplot(gs[1, j])
        p, t = P[:, j].ravel(), T[:, j].ravel()
        r = float(np.corrcoef(p, t)[0, 1])
        stats[f] = {"corr": r, "sigma_ratio": float(p.std() / t.std())}
        ax.hexbin(t, p, gridsize=40, bins="log", cmap="Greys", mincnt=1, linewidths=0)
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lim, lim, color=C_VERM, lw=1.1, ls="--", zorder=4)
        ax.set_xlabel(f"ground truth {f} (norm.)")
        if j == 0:
            ax.set_ylabel("MATEY prediction (norm.)")
        ax.set_title(f, fontsize=8)
        ax.text(
            0.96,
            0.06,
            f"$r$ = {r:+.2f}\n$\\sigma_{{\\rm pred}}/\\sigma_{{\\rm true}}$ = {p.std() / t.std():.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.8,
            bbox=dict(fc="white", ec=GRID, alpha=0.9, pad=2.0),
        )
        corner_tag(ax, f"({'bc'[j]})", dx=-0.16)

    # (d) normalisation root cause
    ax = fig.add_subplot(gs[1, 2])
    nf = sio.netcdf_file(base_nc, "r", mmap=False)
    ne = nf.variables["ne2d"][:200].astype(np.float64).copy()
    nf.close()
    NE = (3.993869294415e16, 1.539813668964e21)
    lin = ((ne - NE[0]) / (NE[1] - NE[0])).ravel()
    lg = (
        (np.log10(np.clip(ne, 1e10, None)) - np.log10(NE[0]))
        / (np.log10(NE[1]) - np.log10(NE[0]))
    ).ravel()
    bins = np.linspace(0, 1, 55)
    ax.hist(
        lin,
        bins=bins,
        color=C_VERM,
        alpha=0.8,
        density=True,
        label="linear min-max\n(current loader)",
    )
    ax.hist(
        lg,
        bins=bins,
        color=C_BLUE,
        alpha=0.65,
        density=True,
        label="log$_{10}$ min-max",
    )
    ax.set_yscale("log")
    ax.set_xlabel("normalised $n_e$ input")
    ax.set_ylabel("density")
    ax.set_title("root cause: $n_e$ scaling", fontsize=8)
    ax.legend(loc="upper center", fontsize=6.2, handlelength=1.2)
    corner_tag(ax, "(d)", dx=-0.16)
    stats["ne_linear_median"] = float(np.median(lin))
    stats["ne_linear_p99"] = float(np.percentile(lin, 99))
    stats["ne_log_median"] = float(np.median(lg))
    stats["nrmse_baseline_mean"] = float(nmean[:nb].mean())
    stats["nrmse_ood_mean"] = float(nmean[nb:].mean())

    save(fig, outdir, "fig3_matey_performance")
    return stats


# ---------------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------------
def fig4(npz, outdir: Path) -> dict:
    pairs = [
        (npz["base_ne_mean"], npz["ood_ne_mean"], "$n_e$", "m$^{-3}$", True),
        (npz["base_te_mean"], npz["ood_te_mean"], "$T_e$", "eV", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 3.6))
    out = {}
    for row, (b, o, lab, unit, uselog) in enumerate(pairs):
        vmax = max(b.max(), o.max())
        vmin = max(min(b[b > 0].min(), o[o > 0].min()), vmax * 1e-4)
        norm = LogNorm(vmin=vmin, vmax=vmax) if uselog else None
        for col, (arr, ttl) in enumerate(
            [(b, "in pre-training\n(Sequence_sin4)"), (o, "held out\n(noLat_dribble)")]
        ):
            ax = axes[row, col]
            im = ax.imshow(
                np.clip(arr, vmin, None) if uselog else arr,
                origin="lower",
                aspect="auto",
                cmap=SEQ,
                norm=norm,
                vmin=None if uselog else 0,
                vmax=None if uselog else vmax,
            )
            if row == 0:
                ax.set_title(ttl, fontsize=7.2, pad=4)
            if col == 0:
                ax.set_ylabel(f"{lab}\nradial index", fontsize=7.5)
            if row == 1:
                ax.set_xlabel("poloidal index", fontsize=7.5)
            ax.tick_params(labelsize=6.2)
            ax.grid(False)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=5.6)
            cb.set_label(unit, fontsize=6.0)
        ax = axes[row, 2]
        rel = 100.0 * (o - b) / np.maximum(b, 1e-30)
        lim = float(np.percentile(np.abs(rel), 99))
        im = ax.imshow(
            rel, origin="lower", aspect="auto", cmap=DIV, vmin=-lim, vmax=lim
        )
        if row == 0:
            ax.set_title(
                "relative change\n(held out $-$ pre-training)", fontsize=7.2, pad=4
            )
        if row == 1:
            ax.set_xlabel("poloidal index", fontsize=7.5)
        ax.tick_params(labelsize=6.2)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=5.6)
        cb.set_label("%", fontsize=6.0)
        out[lab] = {
            "p99_abs_rel_pct": lim,
            "mean_rel_pct": float(rel.mean()),
            "min_rel_pct": float(rel.min()),
            "max_rel_pct": float(rel.max()),
        }
    fig.suptitle(
        "Time-averaged plasma state: where the held-out case leaves the pre-training domain",
        fontsize=8.5,
        y=1.02,
    )
    fig.tight_layout(h_pad=0.7, w_pad=1.0)
    save(fig, outdir, "fig4_field_maps")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--showcase", required=True)
    ap.add_argument(
        "--matey-run", default="output/matey_inference_drift_20260717_191534"
    )
    ap.add_argument(
        "--baseline-nc",
        default="/lustre/orion/lrn097/scratch/asvillar/mateydata/solps_drift/baseline/valid/d3d_sequence_sin4.nc",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_style()
    sdir = Path(args.showcase)
    outdir = Path(args.out) if args.out else sdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    res = json.load(open(sdir / "drift_showcase.json"))
    npz = np.load(sdir / "drift_showcase.npz", allow_pickle=True)

    print("[fig1]")
    s1 = fig1(res, npz, outdir)
    print("[fig2]")
    s2 = fig2(res, outdir)
    print("[fig3]")
    s3 = fig3(Path(args.matey_run), args.baseline_nc, outdir)
    print("[fig4]")
    s4 = fig4(npz, outdir)

    with open(outdir / "figure_stats.json", "w") as fh:
        json.dump(
            {"fig1": s1, "fig2": s2, "fig3": s3, "fig4": s4}, fh, indent=2, default=str
        )
    print(f"[done] figures + stats in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
