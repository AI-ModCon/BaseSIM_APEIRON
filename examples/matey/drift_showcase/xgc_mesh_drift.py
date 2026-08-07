#!/usr/bin/env python3
"""Mesh-resolved cross-device drift extraction for XGC gyrokinetic data.

Supersedes the node-subsample-only extraction in ``xgc_drift_showcase.py``.
Two things change.

**1. The feature-column mapping is corrected.**  ``Data.x`` written by the
preprocessing has **eleven** columns, not the ten that ``GraphXGCDataset.
field_names`` advertises::

    0 pos_r  1 pos_z  2 pos_phi  3 e_den  4 e_T_perp  5 e_T_para
    6 e_u_para  7 i_T_perp  8 i_T_para  9 i_u_para  10 dpot

The toroidal angle ``pos_phi`` sits at index 2 and is *not* in ``field_names``.
The old showcase sliced ``x[:, :10]`` against the ten-name list, so every
physical field was read one column early -- what it monitored as ``e_den`` was
actually ``pos_phi`` (identically zero for the single-plane XGCa cases), and
the real ``dpot`` was never read at all.  ``MIN_FEAT``/``MAX_FEAT`` (ten
entries, from ``_minmax_features``) likewise align to columns
``[0,1,3,4,5,6,7,8,9,10]``, skipping ``pos_phi``.

**2. Fields stay on the real mesh.**  The raw ``xgc.mesh.bp`` supplies node
coordinates, the triangle connectivity, the poloidal flux and the wall-node
list; ``xgc.equil.bp`` supplies the X-point flux, so ``psi_n = psi/eq_x_psi``
puts the separatrix at 1.0.  Graph node ``i`` of plane 0 is mesh node ``i``
(verified: ``pos[:nnodes] == rz``), so a plane-0 slice drops straight onto the
triangulation.

Drift is then measured at three levels, which is the comparison the figures
make:

    L0  temporal      -- frame vs. frame within one case (the noise floor)
    L1  same machine  -- different scenario, same device (DIII-D PT vs. NT,
                         ASDEX-U vs. ASDEX-U favB)
    L2  cross device  -- different machine entirely

KS is invariant under any monotone rescaling applied to both samples, so the
min-max normalisation cancels and is not applied here.  What does *not* cancel
is mesh density: XGC meshes are far finer in the edge, so a uniform draw over
nodes over-weights the edge and does so differently for each device.  Node
samples are therefore drawn with probability proportional to ``node_vol``, and
the uniform-draw result is computed alongside so the sensitivity is reported
rather than assumed.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
from pathlib import Path

import adios2
from typing import Any

import numpy as np
import torch
from scipy.stats import ks_2samp

RAW_ROOT = "/lustre/orion/fus183/proj-shared/fusiond-seed-xgc1-data"
PROC_ROOT = RAW_ROOT + "/processed"

# Column layout of Data.x as actually written (11 columns).
COL = {
    "pos_r": 0,
    "pos_z": 1,
    "pos_phi": 2,
    "e_den": 3,
    "e_T_perp": 4,
    "e_T_para": 5,
    "e_u_para": 6,
    "i_T_perp": 7,
    "i_T_para": 8,
    "i_u_para": 9,
    "dpot": 10,
}
FIELDS = [
    "e_den",
    "e_T_perp",
    "e_T_para",
    "e_u_para",
    "i_T_perp",
    "i_T_para",
    "i_u_para",
    "dpot",
]
UNITS = {
    "e_den": "m$^{-3}$",
    "e_T_perp": "eV",
    "e_T_para": "eV",
    "e_u_para": "m s$^{-1}$",
    "i_T_perp": "eV",
    "i_T_para": "eV",
    "i_u_para": "m s$^{-1}$",
    "dpot": "V",
}

# (case dir, short label, device, in pre-training?, frame budget)
# Frame budgets track file size: one ITER graphdata_*.pt is ~6.9 GB
# (40.9M rows) while an ASDEX-U one is a few MB.
CASES = [
    # ITER used to be capped at 4 frames for I/O reasons, but an unequal frame
    # count makes its temporal statistics incomparable to the others (a pool of
    # 4 frames spread over the same run is more heterogeneous than a pool of 8),
    # so it is now 8 like everything else. Costs ~55 GB of reads.
    ("n560fr_ITER_PFPO_W_Ne", "ITER PFPO", "ITER", True, 8),
    ("n613fr_KSTART_30306_q4_rmp_turbulence", "KSTAR RMP", "KSTAR", True, 10),
    ("n565pe_PT_xgc1_d3d_adjust_flow2_for_C", "DIII-D PT", "DIII-D", False, 10),
    ("n585pe_NT_XGC1_d3d_flow5_ti_d05_tanh", "DIII-D NT", "DIII-D", False, 10),
    ("n579fr_ASDEX_U_XGCa_neutral", "ASDEX-U", "ASDEX-U", False, 12),
    ("n582fr_ASDEX_U_fav_gradb_XGCa_neutral", "ASDEX-U favB", "ASDEX-U", False, 12),
]

SNAPSHOT_MAX_NODES = 1_400_000  # full-mesh snapshot cap (ITER is 1.28M)


def key(lab: str) -> str:
    return lab.replace(" ", "_").replace("-", "")


def read_mesh(case: str, root: str) -> dict:
    """Node coords, triangles, psi_n, wall polygon and node volumes."""
    with adios2.FileReader(f"{root}/{case}/xgc.mesh.bp") as f:
        rz = np.asarray(f.read("rz"), dtype=np.float64)
        tri = np.asarray(f.read("nd_connect_list"), dtype=np.int64)
        psi = np.asarray(f.read("psi"), dtype=np.float64)
        node_vol = np.asarray(f.read("node_vol"), dtype=np.float64)
        wall_idx = np.asarray(f.read("grid_wall_nodes"), dtype=np.int64)
    with adios2.FileReader(f"{root}/{case}/xgc.equil.bp") as f:
        x_psi = float(f.read("eq_x_psi"))
        axis_r = float(f.read("eq_axis_r"))
        axis_z = float(f.read("eq_axis_z"))
    # grid_wall_nodes is 1-based in some XGC writers; detect and correct.
    if wall_idx.min() >= 1 and wall_idx.max() >= rz.shape[0]:
        wall_idx = wall_idx - 1
    return {
        "rz": rz,
        "tri": tri,
        "psi_n": psi / x_psi,
        "node_vol": node_vol,
        "wall": rz[wall_idx],
        "x_psi": x_psi,
        "axis": (axis_r, axis_z),
        "nnodes": int(rz.shape[0]),
        "ntri": int(tri.shape[0]),
    }


def load_case(
    case: str,
    n_frames: int,
    n_nodes: int,
    seed: int,
    mesh: dict,
    proc_root: str,
    skip_frac: float = 0.1,
):
    """Plane-0 physical fields per frame, volume- and uniform-weighted samples.

    Returns (samples_vol, samples_uni, node_idx_vol, snapshot, times) where the
    sample arrays are [n_frames, n_nodes, n_fields] and ``snapshot`` holds the
    full-mesh plane-0 fields of the first frame for the contour figures.
    """
    files = sorted(glob.glob(os.path.join(proc_root, case, "graphdata_*.pt")))
    if not files:
        raise FileNotFoundError(f"no graphdata under {proc_root}/{case}")
    # Drop the initial transient. The first file is timestep 2-10 of the run,
    # i.e. the initial condition before turbulence saturates, and it showed up
    # as a systematic outlier in the first monitoring window of *every* case
    # (e.g. ITER 0.167 against ~0.075 for its remaining frames). That is a
    # simulation start-up artifact, not drift.
    first = int(round(skip_frac * len(files)))
    files = files[first:] or files
    pick = np.linspace(0, len(files) - 1, min(n_frames, len(files))).astype(int)
    pick = np.unique(pick)

    nn = mesh["nnodes"]
    rng = np.random.default_rng(seed)
    # Volume-weighted draw makes the sampled distribution represent plasma
    # volume rather than mesh refinement, which differs device to device.
    w = np.clip(mesh["node_vol"], 0, None)
    w = w / w.sum()
    idx_vol = rng.choice(nn, size=min(n_nodes, nn), replace=False, p=w)
    idx_vol.sort()
    idx_uni = rng.choice(nn, size=min(n_nodes, nn), replace=False)
    idx_uni.sort()
    cols = [COL[f] for f in FIELDS]

    out_v, out_u, times, snapshot = [], [], [], None
    for n, i in enumerate(pick):
        d = torch.load(files[i], weights_only=False, map_location="cpu")
        x = d.x.numpy()
        if x.shape[1] != 11:
            raise ValueError(f"{case}: expected 11 feature columns, got {x.shape[1]}")
        if x.shape[0] % nn != 0:
            raise ValueError(
                f"{case}: {x.shape[0]} rows not a multiple of {nn} mesh nodes"
            )
        plane0 = x[:nn]  # graph node i of plane 0 == mesh node i
        out_v.append(plane0[idx_vol][:, cols].astype(np.float64))
        out_u.append(plane0[idx_uni][:, cols].astype(np.float64))
        times.append(int(getattr(d, "t", i)))
        if n == 0 and nn <= SNAPSHOT_MAX_NODES:
            snapshot = plane0[:, cols].astype(np.float32)
        del d, x, plane0
        gc.collect()
    return np.stack(out_v), np.stack(out_u), idx_vol, snapshot, times


def ks_mean(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean and per-field two-sample KS between two [N, n_fields] samples."""
    per = np.array([ks_2samp(a[:, j], b[:, j]).statistic for j in range(a.shape[1])])
    return float(per.mean()), per


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default=RAW_ROOT)
    ap.add_argument("--proc-root", default=PROC_ROOT)
    ap.add_argument("--frames", type=int, default=12, help="max frames per case")
    ap.add_argument("--nodes", type=int, default=20000, help="nodes sampled per frame")
    ap.add_argument("--psi-bins", type=int, default=24)
    ap.add_argument(
        "--skip-frac",
        type=float,
        default=0.1,
        help="fraction of each run discarded as initial transient",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--only", default=None, help="comma-separated case labels, for smoke tests"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    keep = set(args.only.split(",")) if args.only else None

    meshes, sam_v, sam_u, snaps, psin_s, times = {}, {}, {}, {}, {}, {}
    meta = {}
    for case, lab, dev, in_pre, budget in CASES:
        if keep is not None and lab not in keep:
            continue
        nf = min(args.frames, budget)
        print(f"[load] {lab:14} ({dev:8}) frames={nf}", flush=True)
        try:
            mesh = read_mesh(case, args.raw_root)
            v, u, idx, snap, ts = load_case(
                case, nf, args.nodes, args.seed, mesh, args.proc_root, args.skip_frac
            )
        except Exception as exc:  # noqa: BLE001
            print(f"   SKIPPED: {exc}", flush=True)
            continue
        meshes[lab], sam_v[lab], sam_u[lab] = mesh, v, u
        snaps[lab], times[lab] = snap, ts
        psin_s[lab] = mesh["psi_n"][idx]
        meta[lab] = {
            "case": case,
            "device": dev,
            "in_pretraining": in_pre,
            "n_frames": int(v.shape[0]),
            "n_nodes_sampled": int(v.shape[1]),
            "mesh_nodes": mesh["nnodes"],
            "mesh_tri": mesh["ntri"],
            "n_planes": None,
            "times": ts,
            "has_snapshot": snap is not None,
        }
        print(
            f"   mesh {mesh['nnodes']:,} nodes / {mesh['ntri']:,} tri   "
            f"psi_n max {mesh['psi_n'].max():.2f}",
            flush=True,
        )

    labs = [lab for _, lab, _, _, _ in CASES if lab in sam_v]
    dev_of = {lab: meta[lab]["device"] for lab in labs}
    pre_of = {lab: meta[lab]["in_pretraining"] for lab in labs}

    # ---- L0: the two noise floors --------------------------------------
    # (a) sampling floor: two disjoint node halves of the SAME frame, so any
    #     nonzero KS is pure finite-sample noise.
    # (b) temporal floor: frame t vs. frame 0 of the same case.
    floor_sampling, floor_temporal = {}, {}
    rng = np.random.default_rng(args.seed + 1)
    for lab in labs:
        a = sam_v[lab][0]
        perm = rng.permutation(a.shape[0])
        h = a.shape[0] // 2
        floor_sampling[lab] = ks_mean(a[perm[:h]], a[perm[h : 2 * h]])[0]
        tv = [
            ks_mean(sam_v[lab][t], sam_v[lab][0])[0]
            for t in range(1, sam_v[lab].shape[0])
        ]
        floor_temporal[lab] = tv
        print(
            f"[L0 ] {lab:14} sampling={floor_sampling[lab]:.4f}  "
            f"temporal={np.mean(tv) if tv else float('nan'):.4f}",
            flush=True,
        )

    # ---- L1/L2: pairwise ------------------------------------------------
    # Two views of the same comparison, and they answer different questions.
    #
    #   pooled   -- all frames of A vs. all frames of B.  Symmetric, but it
    #               averages away the temporal excursions, so two runs that
    #               wander over the same distribution look identical.
    #   windowed -- one frame of A vs. the pooled reference B, averaged over
    #               frames.  This is what a detector actually consumes: a
    #               single monitoring window scored against a reference.
    #
    # The windowed form is the primary metric because the L0 temporal floor
    # (frame of A vs. pool of A) is then the same quantity with B = A, making
    # the three levels directly comparable.
    pool_v = {lab: sam_v[lab].reshape(-1, len(FIELDS)) for lab in labs}
    pool_u = {lab: sam_u[lab].reshape(-1, len(FIELDS)) for lab in labs}
    n = len(labs)
    mat_v = np.zeros((n, n))
    mat_u = np.zeros((n, n))
    mat_field = np.zeros((n, n, len(FIELDS)))
    for i, a in enumerate(labs):
        for j, b in enumerate(labs):
            if j <= i:
                continue
            m, per = ks_mean(pool_v[a], pool_v[b])
            mat_v[i, j] = mat_v[j, i] = m
            mat_field[i, j] = mat_field[j, i] = per
            mat_u[i, j] = mat_u[j, i] = ks_mean(pool_u[a], pool_u[b])[0]
    print("[pair] pairwise KS done", flush=True)

    def windowed(a: str, b: str) -> list[float]:
        """KS of each frame of A against the pooled reference B.

        When ``a is b`` the reference is built **leave-one-out**: including the
        very frame being scored pulls the pool toward it and biases the score
        low, and the size of that bias depends on the frame count (a frame is
        1/8 of an 8-frame pool). Since this self-comparison is what gives a
        pre-trained device its coverage score, the bias would otherwise inflate
        the pre-trained vs. held-out separation.
        """
        n = sam_v[a].shape[0]
        if a != b:
            return [ks_mean(sam_v[a][t], pool_v[b])[0] for t in range(n)]
        out = []
        for t in range(n):
            ref = np.concatenate([sam_v[a][u] for u in range(n) if u != t])
            out.append(ks_mean(sam_v[a][t], ref)[0])
        return out

    # L0 restated in the windowed form, so it shares units with L1/L2.
    floor_window = {lab: windowed(lab, lab) for lab in labs}
    for lab in labs:
        print(
            f"[L0w] {lab:14} frame-vs-own-pool={np.mean(floor_window[lab]):.4f}",
            flush=True,
        )

    same_machine: list[dict[str, object]] = []
    cross_machine: list[dict[str, object]] = []
    for i, a in enumerate(labs):
        for j, b in enumerate(labs):
            if j <= i:
                continue
            wab, wba = windowed(a, b), windowed(b, a)
            rec = {
                "a": a,
                "b": b,
                "ks_pooled": float(mat_v[i, j]),
                "ks_pooled_uniform": float(mat_u[i, j]),
                "ks_windowed": float(np.mean(wab + wba)),
                "ks_windowed_ab": [float(x) for x in wab],
                "ks_windowed_ba": [float(x) for x in wba],
                "per_field": {
                    f: float(mat_field[i, j, k]) for k, f in enumerate(FIELDS)
                },
            }
            (same_machine if dev_of[a] == dev_of[b] else cross_machine).append(rec)

    # ---- coverage vs. the pre-trained devices --------------------------
    refs = [lab for lab in labs if pre_of[lab]]
    coverage: dict[str, list[float]] = {}
    for lab in labs:
        # Distance to the CLOSEST pre-trained device, not to a pooled
        # reference: pooling ITER and KSTAR gives a bimodal reference against
        # which even an ITER frame scores as far.
        others = [r for r in refs if r != lab]
        if not others:
            coverage[lab] = []
            continue
        coverage[lab] = [
            min(ks_mean(sam_v[lab][t], pool_v[r])[0] for r in others)
            for t in range(sam_v[lab].shape[0])
        ]
        print(f"[cov] {lab:14} {np.mean(coverage[lab]):.4f}", flush=True)

    # ---- radial structure: psi_n-binned profiles and per-bin KS --------
    edges = np.linspace(0.0, 1.15, args.psi_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    profiles = {}
    for lab in labs:
        pn = psin_s[lab]
        frame_mean = sam_v[lab].mean(axis=0)  # [n_nodes, n_fields]
        prof = np.full((args.psi_bins, len(FIELDS)), np.nan)
        for bin_i in range(args.psi_bins):
            m = (pn >= edges[bin_i]) & (pn < edges[bin_i + 1])
            if m.sum() >= 20:
                prof[bin_i] = frame_mean[m].mean(axis=0)
        profiles[lab] = prof

    def radial_ks(a: str, b: str) -> np.ndarray:
        pa, pb = psin_s[a], psin_s[b]
        fa, fb = pool_v[a], pool_v[b]
        # pooled samples repeat the node set once per frame
        ta = np.tile(pa, sam_v[a].shape[0])
        tb = np.tile(pb, sam_v[b].shape[0])
        out = np.full(args.psi_bins, np.nan)
        for k in range(args.psi_bins):
            ma = (ta >= edges[k]) & (ta < edges[k + 1])
            mb = (tb >= edges[k]) & (tb < edges[k + 1])
            if ma.sum() >= 50 and mb.sum() >= 50:
                out[k] = ks_mean(fa[ma], fb[mb])[0]
        return out

    radial_pairs = {}
    for rec in same_machine:
        radial_pairs[f"{rec['a']} | {rec['b']}"] = radial_ks(
            str(rec["a"]), str(rec["b"])
        ).tolist()
    # one representative cross-machine pair per held-out device vs each reference
    for lab in labs:
        for r in refs:
            if r == lab or dev_of[r] == dev_of[lab]:
                continue
            radial_pairs[f"{lab} | {r}"] = radial_ks(lab, r).tolist()
    print("[radial] per-psi_n KS done", flush=True)

    # ---- save -----------------------------------------------------------
    arrays = {
        "pair_matrix": mat_v,
        "pair_matrix_uniform": mat_u,
        "pair_matrix_field": mat_field,
        "pair_labels": np.array(labs, dtype=object),
        "fields": np.array(FIELDS, dtype=object),
        "psi_edges": edges,
        "psi_centres": centres,
    }
    for lab in labs:
        k = key(lab)
        arrays[f"mesh_rz_{k}"] = meshes[lab]["rz"].astype(np.float32)
        arrays[f"mesh_tri_{k}"] = meshes[lab]["tri"].astype(np.int32)
        arrays[f"mesh_psin_{k}"] = meshes[lab]["psi_n"].astype(np.float32)
        arrays[f"mesh_wall_{k}"] = meshes[lab]["wall"].astype(np.float32)
        arrays[f"profile_{k}"] = profiles[lab]
        arrays[f"psin_sample_{k}"] = psin_s[lab].astype(np.float32)
        arrays[f"sample_{k}"] = sam_v[lab].astype(np.float32)
        if snaps[lab] is not None:
            arrays[f"snapshot_{k}"] = snaps[lab]
    for name, vals in radial_pairs.items():
        arrays[f"radial_{name.replace(' ', '_').replace('|', 'vs')}"] = np.array(vals)
    # numpy's stub types savez_compressed's second positional as bool, so the
    # documented **arrays spread does not type-check.
    npz: Any = np.savez_compressed
    npz(str(outdir / "xgc_mesh_drift.npz"), **arrays)

    summary = {
        "cases": meta,
        "fields": FIELDS,
        "units": UNITS,
        "column_map": COL,
        "reference_cases": refs,
        "labels": labs,
        "pair_matrix": mat_v.tolist(),
        "pair_matrix_uniform": mat_u.tolist(),
        "same_machine_pairs": same_machine,
        "cross_machine_pairs": cross_machine,
        "levels": {
            "L0_sampling_floor": floor_sampling,
            "L0_temporal_vs_frame0": {
                k: [float(x) for x in v] for k, v in floor_temporal.items()
            },
            "L0_temporal_windowed": {
                k: [float(x) for x in v] for k, v in floor_window.items()
            },
            "L0_sampling_floor_mean": float(np.mean(list(floor_sampling.values()))),
            "L0_temporal_windowed_mean": float(
                np.mean([x for v in floor_window.values() for x in v])
            ),
            "L1_same_machine_mean": float(
                np.mean([r["ks_windowed"] for r in same_machine])
            )
            if same_machine
            else None,
            "L2_cross_machine_mean": float(
                np.mean([r["ks_windowed"] for r in cross_machine])
            )
            if cross_machine
            else None,
            "L1_same_machine_pooled": float(
                np.mean([r["ks_pooled"] for r in same_machine])
            )
            if same_machine
            else None,
            "L2_cross_machine_pooled": float(
                np.mean([r["ks_pooled"] for r in cross_machine])
            )
            if cross_machine
            else None,
        },
        "coverage": {k: [float(x) for x in v] for k, v in coverage.items()},
        "coverage_mean": {k: float(np.mean(v)) for k, v in coverage.items()},
        "radial_pairs": radial_pairs,
        "psi_centres": centres.tolist(),
        "settings": {
            "frames": args.frames,
            "nodes": args.nodes,
            "seed": args.seed,
            "psi_bins": args.psi_bins,
            "skip_frac": args.skip_frac,
        },
    }
    lv = summary["levels"]
    if lv["L1_same_machine_mean"] and lv["L2_cross_machine_mean"]:
        lv["cross_over_same"] = lv["L2_cross_machine_mean"] / lv["L1_same_machine_mean"]
        lv["same_over_temporal"] = (
            lv["L1_same_machine_mean"] / lv["L0_temporal_windowed_mean"]
        )
        lv["cross_over_temporal"] = (
            lv["L2_cross_machine_mean"] / lv["L0_temporal_windowed_mean"]
        )
    with open(outdir / "xgc_mesh_drift.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    print(f"\n[done] {outdir}/xgc_mesh_drift.npz + .json")
    print(f"  L0 sampling floor   {fmt(lv['L0_sampling_floor_mean'])}")
    print(f"  L0 temporal         {fmt(lv['L0_temporal_windowed_mean'])}")
    print(f"  L1 same machine     {fmt(lv['L1_same_machine_mean'])}")
    print(f"  L2 cross machine    {fmt(lv['L2_cross_machine_mean'])}")
    if "cross_over_same" in lv:
        print(f"  cross / same        {lv['cross_over_same']:.1f}x")
        print(f"  same  / temporal    {lv['same_over_temporal']:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
