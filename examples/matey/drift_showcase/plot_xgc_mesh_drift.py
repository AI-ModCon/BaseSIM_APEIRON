#!/usr/bin/env python3
"""Publication figures for mesh-resolved XGC cross-device drift.

fig5_xgc_device_maps    electron density on each device's true triangulation,
                        with separatrix and vessel wall
fig6_xgc_same_vs_cross  same-machine vs. different-machine drift: hierarchy,
                        similarity block structure, radial localisation,
                        per-field breakdown and coverage
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

_here = str(Path(__file__).resolve().parent)
sys.path.insert(0, _here)
from plot_solps_drift_showcase import (  # noqa: E402
    C_BLUE,
    C_GREEN,
    C_SKY,
    C_VERM,
    GRID,
    INK,
    INK2,
    MUTED,
    TEXTWIDTH,
    corner_tag,
    save,
    set_style,
)

# Panel order: pre-trained first, then the two held-out same-machine pairs
# side by side so the within-device similarity is visible by eye.
MAP_ORDER = [
    "ITER PFPO",
    "KSTAR RMP",
    "DIII-D PT",
    "DIII-D NT",
    "ASDEX-U",
    "ASDEX-U favB",
]
MAP_FIELD = "e_den"

FIELD_TEX = {
    "e_den": r"$n_e$",
    "e_T_perp": r"$T_{e\perp}$",
    "e_T_para": r"$T_{e\parallel}$",
    "e_u_para": r"$u_{e\parallel}$",
    "i_T_perp": r"$T_{i\perp}$",
    "i_T_para": r"$T_{i\parallel}$",
    "i_u_para": r"$u_{i\parallel}$",
    "dpot": r"$\delta\phi$",
}


def key(lab: str) -> str:
    return lab.replace(" ", "_").replace("-", "")


def windowed_matrix(res: dict) -> dict:
    """W[a][b] = per-frame KS of a's frames against b's pooled reference."""
    w: dict[str, dict[str, list[float]]] = {}
    for lab, vals in res["levels"]["L0_temporal_windowed"].items():
        w.setdefault(lab, {})[lab] = list(vals)
    for rec in res["same_machine_pairs"] + res["cross_machine_pairs"]:
        a, b = rec["a"], rec["b"]
        w.setdefault(a, {})[b] = list(rec["ks_windowed_ab"])
        w.setdefault(b, {})[a] = list(rec["ks_windowed_ba"])
    return w


def coverage_self_inclusive(res: dict) -> dict:
    """Coverage against the closest pre-trained device, self included.

    ``xgc_mesh_drift.py`` excludes the case itself so that a reference is
    never scored against its own pool. That is the right guard for the
    reference-vs-reference number, but it makes a nonsense of the coverage
    *score*: a pre-trained device is by definition covered, and excluding
    itself reports how far ITER is from KSTAR instead.
    """
    w = windowed_matrix(res)
    refs = res["reference_cases"]
    return {
        lab: float(np.min([w[lab][r] for r in refs if r in w[lab]], axis=0).mean())
        for lab in res["labels"]
    }


def boundary_segments(rz: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Line segments for the mesh boundary, i.e. the vessel outline.

    ``grid_wall_nodes`` in xgc.mesh.bp is a node *set*, not an ordered
    polygon, so joining it point-to-point draws chords across the plasma.
    The domain boundary is instead the set of triangle edges belonging to
    exactly one triangle, which traces the wall exactly.
    """
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e = np.sort(e, axis=1).astype(np.int64)
    code = e[:, 0] * (int(rz.shape[0]) + 1) + e[:, 1]
    uniq, first, counts = np.unique(code, return_index=True, return_counts=True)
    b = e[first[counts == 1]]
    return np.stack([rz[b[:, 0]], rz[b[:, 1]]], axis=1)


def draw_device(
    ax, npz, lab: str, fidx: int, vmin: float, vmax: float, show_ylabel: bool
):
    """Field on the true triangulation, with separatrix and wall."""
    k = key(lab)
    rz = npz[f"mesh_rz_{k}"]
    tri = npz[f"mesh_tri_{k}"]
    psin = npz[f"mesh_psin_{k}"]
    snap = npz[f"snapshot_{k}"][:, fidx].astype(np.float64)

    triang = mtri.Triangulation(rz[:, 0], rz[:, 1], tri)
    val = np.clip(snap, vmin, vmax)
    # gouraud shading rather than tricontourf: the ITER mesh carries 2.55M
    # triangles and contour banding at that size costs minutes and megabytes.
    tpc = ax.tripcolor(
        triang,
        val,
        shading="gouraud",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="magma",
        rasterized=True,
    )
    try:
        ax.tricontour(
            triang, psin, levels=[1.0], colors="#00d4ff", linewidths=0.8, zorder=5
        )
    except Exception:  # noqa: BLE001 -- separatrix may fall outside the domain
        pass
    ax.add_collection(
        LineCollection(
            boundary_segments(rz, tri), colors=INK, linewidths=0.55, zorder=6
        )
    )
    ax.set_aspect("equal")
    ax.grid(False)
    ax.tick_params(labelsize=5.4)
    ax.set_xlabel("R [m]", fontsize=6.6, labelpad=1.5)
    if show_ylabel:
        ax.set_ylabel("Z [m]", fontsize=6.6, labelpad=1.5)
    return tpc


def fig5(res: dict, npz, outdir: Path) -> dict:
    cases = res["cases"]
    fields = res["fields"]
    fidx = fields.index(MAP_FIELD)
    labs = [lab for lab in MAP_ORDER if lab in cases and cases[lab]["has_snapshot"]]

    # One shared log colour scale so panels are comparable across devices.
    lo, hi = [], []
    for lab in labs:
        v = npz[f"snapshot_{key(lab)}"][:, fidx]
        v = v[np.isfinite(v) & (v > 0)]
        lo.append(np.percentile(v, 1))
        hi.append(np.percentile(v, 99))
    vmin, vmax = float(min(lo)), float(max(hi))

    fig = plt.figure(figsize=(TEXTWIDTH, 4.9))
    gs = fig.add_gridspec(
        2, 3, hspace=0.46, wspace=0.40, left=0.08, right=0.86, top=0.815, bottom=0.10
    )
    tpc = None
    for n, lab in enumerate(labs):
        ax = fig.add_subplot(gs[n // 3, n % 3])
        tpc = draw_device(ax, npz, lab, fidx, vmin, vmax, show_ylabel=(n % 3 == 0))
        info = cases[lab]
        col = C_BLUE if info["in_pretraining"] else C_VERM
        # Title carries the case name only; pre-trained vs. held out is in the
        # colour, explained once in the figure legend. Keeping it short stops
        # it from running under the panel tag.
        ax.set_title(lab, fontsize=6.8, color=col, pad=3.0)
        ax.text(
            0.03,
            0.97,
            f"{info['mesh_nodes']:,} nodes",
            transform=ax.transAxes,
            fontsize=5.0,
            color=INK2,
            va="top",
            bbox=dict(fc="white", ec="none", alpha=0.72, pad=0.8),
        )
        corner_tag(ax, f"({'abcdef'[n]})", dx=-0.20)

    cax = fig.add_axes([0.885, 0.16, 0.017, 0.62])
    cb = fig.colorbar(tpc, cax=cax)
    cb.ax.tick_params(labelsize=6)
    cb.set_label(r"$n_e$  [m$^{-3}$]", fontsize=7)

    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=C_BLUE,
                lw=0,
                marker="s",
                ms=4,
                label="in XGC pre-training",
            ),
            Line2D(
                [],
                [],
                color=C_VERM,
                lw=0,
                marker="s",
                ms=4,
                label="held out from XGC branch",
            ),
            Line2D([], [], color="#00d4ff", lw=1.0, label="separatrix ($\\psi_N=1$)"),
            Line2D([], [], color=INK, lw=0.9, label="vessel wall"),
        ],
        loc="lower center",
        ncol=4,
        fontsize=6.0,
        frameon=False,
        bbox_to_anchor=(0.47, -0.005),
        columnspacing=1.6,
        handletextpad=0.5,
    )

    # "Held out" is scoped to the XGC graph branch on purpose: MATEY's SOLPS
    # branch does include DIII-D and KSTAR, so an unqualified "held out" would
    # claim the model has never seen these devices, which is false.
    fig.suptitle(
        "Electron density on the native XGC triangulation\n"
        "blue: in MATEY's XGC graph pre-training (ITER, KSTAR)   |   "
        "vermilion: held out from it\n"
        "(DIII-D and KSTAR also appear in MATEY's separate SOLPS branch)",
        fontsize=7.8,
        y=1.0,
    )
    save(fig, outdir, "fig5_xgc_device_maps")
    return {
        "field": MAP_FIELD,
        "vmin": vmin,
        "vmax": vmax,
        "panels": labs,
        "mesh_nodes": {lab: cases[lab]["mesh_nodes"] for lab in labs},
        "mesh_tri": {lab: cases[lab]["mesh_tri"] for lab in labs},
    }


def fig6(res: dict, npz, outdir: Path) -> dict:
    lv = res["levels"]
    labs = res["labels"]
    fields = res["fields"]
    dev = {lab: res["cases"][lab]["device"] for lab in labs}
    same = res["same_machine_pairs"]
    cross = res["cross_machine_pairs"]

    fig = plt.figure(figsize=(TEXTWIDTH, 5.9))
    gs = fig.add_gridspec(
        2, 3, hspace=0.78, wspace=0.70, top=0.87, bottom=0.10, left=0.085, right=0.975
    )

    # --- (a) the drift hierarchy -----------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    groups = [
        ("sampling\nfloor", list(lv["L0_sampling_floor"].values()), MUTED),
        (
            "temporal\nsame run",
            [x for v in lv["L0_temporal_windowed"].values() for x in v],
            C_SKY,
        ),
        ("same\nmachine", [r["ks_windowed"] for r in same], C_GREEN),
        ("different\nmachine\n(all pairs)", [r["ks_windowed"] for r in cross], C_VERM),
    ]
    rng = np.random.default_rng(0)
    means = []
    for i, (name, vals, col) in enumerate(groups):
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        means.append(v.mean())
        ax.scatter(
            i + rng.uniform(-0.16, 0.16, v.size),
            v,
            s=7,
            color=col,
            alpha=0.55,
            lw=0,
            zorder=3,
        )
        ax.hlines(v.mean(), i - 0.34, i + 0.34, color=col, lw=2.0, zorder=4)
    ax.set_ylim(min(means) * 0.35, max(means) * 3.4)
    for i, (name, vals, col) in enumerate(groups):
        ax.annotate(
            f"{means[i]:.3f}",
            xy=(i, means[i]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=5.8,
            color=col,
            fontweight="bold",
        )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(
        [g[0] for g in groups],
        fontsize=5.2,
        linespacing=0.95,
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_yscale("log")
    ax.set_ylabel("KS statistic", fontsize=7.5)
    ax.set_title("drift hierarchy\n(pairwise, all combinations)", fontsize=7.0, pad=4)
    corner_tag(ax, "(a)", dx=-0.30)

    # --- (b) pairwise similarity, blocked by device ----------------------
    ax = fig.add_subplot(gs[0, 1])
    mat = np.array(res["pair_matrix"])
    im = ax.imshow(mat, cmap="magma_r", vmin=0, vmax=mat.max())
    ax.set_xticks(range(len(labs)))
    ax.set_yticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=55, ha="right", fontsize=5.2)
    ax.set_yticklabels(labs, fontsize=5.2)
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=4.6,
                color="white" if mat[i, j] > mat.max() * 0.55 else INK2,
            )
    # Outline the same-machine blocks.
    start = 0
    for d in dict.fromkeys(dev[lab] for lab in labs):
        n = sum(1 for lab in labs if dev[lab] == d)
        if n > 1:
            ax.add_patch(
                mpatches.Rectangle(
                    (start - 0.5, start - 0.5),
                    n,
                    n,
                    fill=False,
                    ec=C_GREEN,
                    lw=1.4,
                    zorder=6,
                )
            )
        start += n
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.04)
    cb.ax.tick_params(labelsize=4.8)
    cb.ax.set_title("KS", fontsize=5.2, pad=2)
    ax.set_title("pairwise similarity", fontsize=7.5, pad=4)
    ax.text(
        0.5,
        -0.62,
        "green outline: same machine",
        transform=ax.transAxes,
        ha="center",
        fontsize=5.2,
        color=C_GREEN,
    )
    corner_tag(ax, "(b)", dx=-0.36)

    # --- (c) coverage vs. the pre-trained devices ------------------------
    ax = fig.add_subplot(gs[0, 2])
    cov = coverage_self_inclusive(res)
    order = [lab for lab in labs if lab in cov and np.isfinite(cov[lab])]
    vals = [cov[lab] for lab in order]
    cols = [C_BLUE if res["cases"][lab]["in_pretraining"] else C_VERM for lab in order]
    y = np.arange(len(order))
    ax.barh(y, vals, color=cols, edgecolor="white", lw=0.6)
    for i, v in enumerate(vals):
        ax.text(
            v + max(vals) * 0.03, i, f"{v:.2f}", va="center", fontsize=5.8, color=INK2
        )
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=5.4)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.42)
    ax.set_xlabel("min KS vs. XGC pre-training", fontsize=6.4)
    ax.set_title(
        "coverage score\n(min over refs, not the (a) mean)", fontsize=6.8, pad=3
    )
    # Upper right: the two pre-trained bars are short, so the legend sits in
    # empty space rather than over the held-out values.
    ax.legend(
        handles=[
            mpatches.Patch(color=C_BLUE, label="in XGC pre-training"),
            mpatches.Patch(color=C_VERM, label="held out (XGC)"),
        ],
        loc="upper right",
        fontsize=5.2,
        borderpad=0.3,
        handlelength=1.1,
        handletextpad=0.4,
    )
    corner_tag(ax, "(c)", dx=-0.42)

    # --- (d) where the drift lives radially ------------------------------
    ax = fig.add_subplot(gs[1, 0])
    centres = np.array(res["psi_centres"])
    rad = res["radial_pairs"]
    same_keys = [k for k in rad if dev[k.split(" | ")[0]] == dev[k.split(" | ")[1]]]
    cross_keys = [k for k in rad if k not in same_keys]
    for k in same_keys:
        ax.plot(
            centres,
            rad[k],
            color=C_GREEN,
            lw=1.2,
            alpha=0.85,
            label="same machine" if k == same_keys[0] else None,
        )
    for k in cross_keys:
        ax.plot(
            centres,
            rad[k],
            color=C_VERM,
            lw=0.8,
            alpha=0.45,
            label="different machine" if k == cross_keys[0] else None,
        )
    ax.axvline(1.0, color=INK2, ls="--", lw=0.8)
    ax.text(
        0.985,
        0.45,
        "separatrix",
        fontsize=5.2,
        color=INK2,
        transform=ax.get_xaxis_transform(),
        rotation=90,
        ha="right",
        va="center",
    )
    ax.set_xlabel(r"normalised poloidal flux $\psi_N$", fontsize=7)
    ax.set_ylabel("KS statistic", fontsize=7.5)
    ax.set_title("radial localisation", fontsize=7.5, pad=3)
    ax.legend(
        fontsize=5.4,
        loc="center left",
        borderpad=0.3,
        handlelength=1.2,
        handletextpad=0.4,
    )
    corner_tag(ax, "(d)", dx=-0.30)

    # --- (e) per-field breakdown -----------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    sm = np.array([[r["per_field"][f] for f in fields] for r in same]).mean(0)
    cm = np.array([[r["per_field"][f] for f in fields] for r in cross]).mean(0)
    y = np.arange(len(fields))
    ax.barh(
        y - 0.2,
        sm,
        height=0.38,
        color=C_GREEN,
        label="same machine",
        edgecolor="white",
        lw=0.4,
    )
    ax.barh(
        y + 0.2,
        cm,
        height=0.38,
        color=C_VERM,
        label="different machine",
        edgecolor="white",
        lw=0.4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([FIELD_TEX.get(f, f) for f in fields], fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlabel("KS statistic", fontsize=7)
    ax.set_title("per field", fontsize=7.5, pad=3)
    ax.legend(fontsize=5.6, loc="lower right", borderpad=0.3)
    corner_tag(ax, "(e)", dx=-0.32)

    # --- (f) mesh-density sensitivity ------------------------------------
    # KS is invariant under monotone rescaling, so normalisation cannot bias
    # it -- but mesh refinement can, because a uniform draw over nodes
    # over-weights the finely-resolved edge by a different factor on each
    # device. Volume-weighted vs. uniform sampling brackets that effect.
    ax = fig.add_subplot(gs[1, 2])
    pv = np.array([r["ks_pooled"] for r in same + cross])
    pu = np.array([r["ks_pooled_uniform"] for r in same + cross])
    iscross = np.array([False] * len(same) + [True] * len(cross))
    ax.scatter(
        pu[~iscross],
        pv[~iscross],
        s=16,
        color=C_GREEN,
        label="same machine",
        zorder=3,
        lw=0,
    )
    ax.scatter(
        pu[iscross],
        pv[iscross],
        s=16,
        color=C_VERM,
        label="different machine",
        zorder=3,
        lw=0,
    )
    lim = [0, max(pv.max(), pu.max()) * 1.08]
    ax.plot(lim, lim, color=MUTED, ls="--", lw=0.8, zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("KS, uniform node draw", fontsize=6.6)
    ax.set_ylabel("KS, volume-weighted", fontsize=6.6)
    ax.set_title("mesh-density sensitivity", fontsize=7.5, pad=3)
    ax.legend(fontsize=5.4, loc="upper left", borderpad=0.3)
    corner_tag(ax, "(f)", dx=-0.34)

    fig.suptitle(
        "Same-machine vs. different-machine drift on XGC gyrokinetic data",
        fontsize=8.5,
        y=0.975,
    )
    save(fig, outdir, "fig6_xgc_same_vs_cross")

    return {
        "levels": {
            k: v
            for k, v in lv.items()
            if isinstance(v, (int, float))
            or k.startswith("cross")
            or k.startswith("same")
        },
        "same_machine_pairs": [
            {
                "pair": f"{r['a']} | {r['b']}",
                "ks_windowed": r["ks_windowed"],
                "ks_pooled": r["ks_pooled"],
            }
            for r in same
        ],
        "cross_machine_pairs": [
            {
                "pair": f"{r['a']} | {r['b']}",
                "ks_windowed": r["ks_windowed"],
                "ks_pooled": r["ks_pooled"],
            }
            for r in cross
        ],
        "per_field_same": {f: float(sm[i]) for i, f in enumerate(fields)},
        "per_field_cross": {f: float(cm[i]) for i, f in enumerate(fields)},
        "coverage_self_inclusive": cov,
        "coverage_excluding_self": res["coverage_mean"],
    }


def fig7(res: dict, det: dict, outdir: Path) -> dict:
    """Detector response on the device stream, with a same-machine control."""
    ds = det["device_stream"]
    ctrl = det["same_machine_control"]
    stream = np.array(ds["stream"])
    boundary = ds["boundary"]

    fig = plt.figure(figsize=(TEXTWIDTH, 4.4))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.95],
        hspace=0.72,
        wspace=0.30,
        top=0.87,
        bottom=0.11,
        left=0.085,
        right=0.975,
    )

    # --- (a) the monitored stream ----------------------------------------
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(stream))
    ax.axvspan(boundary - 0.5, len(stream) - 0.5, color=C_VERM, alpha=0.07, lw=0)
    ax.plot(x, stream, color=INK2, lw=1.1, zorder=3)
    ax.scatter(
        x[:boundary],
        stream[:boundary],
        s=11,
        color=C_BLUE,
        zorder=4,
        lw=0,
        label="in XGC pre-training",
    )
    ax.scatter(
        x[boundary:],
        stream[boundary:],
        s=11,
        color=C_VERM,
        zorder=4,
        lw=0,
        label="held out from XGC branch",
    )
    prev = 0
    for b in ds["bounds"]:
        if b["end"] < len(stream):
            ax.axvline(b["end"] - 0.5, color=GRID, lw=0.7, zorder=1)
        ax.text(
            (prev + b["end"] - 1) / 2,
            ax.get_ylim()[1],
            b["label"],
            ha="center",
            va="bottom",
            fontsize=4.8,
            color=C_BLUE if b["in_pretraining"] else C_VERM,
            rotation=0,
        )
        prev = b["end"]
    ax.axvline(boundary - 0.5, color=C_VERM, ls="--", lw=1.0, zorder=5)
    ax.set_xlabel("monitoring window", fontsize=7)
    ax.set_ylabel("coverage score\n(min KS vs. XGC pre-training)", fontsize=6.8)
    ax.set_title("device stream: the monitored signal", fontsize=7.5, pad=10)
    ax.legend(fontsize=5.6, loc="center left", borderpad=0.3)
    ylim_a = ax.get_ylim()
    corner_tag(ax, "(a)", dx=-0.075)

    # --- (b) detection delay ---------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    names = list(ds["detectors"].keys())
    delays = [ds["detectors"][n]["delay"] for n in names]
    finite = [d for d in delays if d is not None]
    cap = (max(finite) if finite else 1) * 1.3 + 1
    for i, (n, d) in enumerate(zip(names, delays)):
        if d is None:
            ax.barh(i, cap, color="white", edgecolor=MUTED, hatch="///", lw=0.7)
            ax.text(
                cap * 0.5,
                i,
                "never fires",
                ha="center",
                va="center",
                fontsize=5.4,
                color=INK2,
            )
        else:
            ax.barh(i, d, color=C_GREEN, edgecolor="white", lw=0.5)
            ax.text(d + cap * 0.03, i, f"{d}", va="center", fontsize=5.8, color=INK2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(
        [
            n.replace(" library default", "\nlibrary default").replace(" tuned to stream", "\ntuned to stream")
            for n in names
        ],
        fontsize=5.0,
        linespacing=0.9,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, cap * 1.28)
    ax.set_xlabel("detection delay [windows]", fontsize=6.8)
    ax.set_title("response to the device change", fontsize=7.2, pad=3)
    corner_tag(ax, "(b)", dx=-0.55)

    # --- (c) same-machine control ----------------------------------------
    # Plotted on the same y-scale as (a): the flat trace is the reason no
    # detector fires, which six zero-length bars would not have shown.
    ax = fig.add_subplot(gs[1, 1])
    if ctrl is not None:
        cs = np.array(ctrl["stream"])
        cb = ctrl["boundary"]
        cx = np.arange(len(cs))
        ax.plot(cx, cs, color=INK2, lw=1.1, zorder=3)
        ax.scatter(cx[:cb], cs[:cb], s=11, color=C_GREEN, lw=0, zorder=4)
        ax.scatter(cx[cb:], cs[cb:], s=11, color=C_GREEN, lw=0, zorder=4, alpha=0.55)
        ax.axvline(cb - 0.5, color=C_GREEN, ls="--", lw=1.0, zorder=5)
        ax.set_ylim(*ylim_a)
        ax.set_xlabel("monitoring window", fontsize=6.8)
        ax.set_ylabel("coverage score", fontsize=6.8)
        ax.set_title(
            f"control: {ctrl['reference']} $\\rightarrow$ "
            f"{ctrl['second_case']}\n(same machine, scenario change only)",
            fontsize=6.4,
            pad=3,
        )
        nfire = sum(len(v["fired"]) for v in ctrl["detectors"].values())
        ax.text(
            0.03,
            0.93,
            f"{sum(1 for v in ctrl['detectors'].values() if v['fired'])}"
            f"/{len(ctrl['detectors'])} detectors fire "
            f"({nfire} detections)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.6,
            color=C_GREEN,
            fontweight="bold",
        )
        ax.text(
            cb - 0.4,
            ylim_a[1] * 0.52,
            " scenario change",
            fontsize=5.0,
            color=C_GREEN,
            va="center",
        )
    corner_tag(ax, "(c)", dx=-0.22)

    fig.suptitle(
        "APEIRON detector response on the corrected XGC stream", fontsize=8.5, y=0.975
    )
    save(fig, outdir, "fig7_xgc_detectors")
    return {
        "boundary": boundary,
        "n_windows": int(len(stream)),
        "device_delays": {n: ds["detectors"][n]["delay"] for n in names},
        "device_false_alarms": {n: ds["detectors"][n]["false_alarms"] for n in names},
        "control_fires": (
            {n: len(ctrl["detectors"][n]["fired"]) for n in ctrl["detectors"]}
            if ctrl
            else None
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--showcase", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_style()
    sdir = Path(args.showcase)
    outdir = Path(args.out) if args.out else sdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    res = json.load(open(sdir / "xgc_mesh_drift.json"))
    npz = np.load(sdir / "xgc_mesh_drift.npz", allow_pickle=True)

    stats = {"fig5": fig5(res, npz, outdir), "fig6": fig6(res, npz, outdir)}
    detpath = sdir / "xgc_detectors.json"
    if detpath.exists():
        stats["fig7"] = fig7(res, json.load(open(detpath)), outdir)
    else:
        print(f"  [skip] fig7: {detpath} not found (run xgc_detector_sweep.py first)")
    with open(outdir / "fig56_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"[done] {outdir}/fig56_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
