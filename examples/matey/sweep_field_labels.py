#!/usr/bin/env python3
"""Find the field-embedding indices the MATEY checkpoint expects for SOLPS2DwION.

Background
----------
MATEY assigns each dataset's fields a slice of a *global* field-embedding table.
The slice is computed by walking ``DSET_NAME_TO_OBJECT`` in insertion order
(``datasets.py:_build_subset_dict``), so the indices a dataset receives depend on
the entire registry, not on which datasets are loaded.

Frontier's shared MATEY has no ``SOLPS2DwION``; APEIRON registers a custom class
at runtime, which *appends* it and therefore assigns it indices ``[532, 533, 534]``.
The checkpoint's embedding table has 536 columns, so the bounds assert in
``SubsampledLinear`` passes and inference runs silently -- but with the wrong
columns, i.e. the model decodes ne2d/te2d/ti2d as three unrelated variables.

This script sweeps the candidate index triple ``[k, k+1, k+2]`` and reports NRMSE
per field, which recovers the indices used during pre-training.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
# This file lives next to examples/matey/model.py, which shadows APEIRON's
# src/model package if the script directory stays ahead on sys.path.
_here = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(_here)]
for p in (str(REPO), str(REPO / "src")):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, help="in-pre-training SOLPS root")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--batches", type=int, default=3, help="batches averaged per offset"
    )
    ap.add_argument("--kmin", type=int, default=0)
    ap.add_argument("--kmax", type=int, default=-1, help="-1 = n_states-3")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--mode", choices=["contiguous", "per-field"], default="contiguous")
    ap.add_argument("--iters", type=int, default=2, help="coordinate-descent passes")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from apeiron.config.configuration import build_config  # noqa: E402
    from examples.matey.model import MATEYHarness  # noqa: E402

    # Dataset type, rollout horizon and step-inference come from the data root's
    # matey_settings.json, not from --set: they describe the data and the
    # checkpoint rather than this sweep.
    argv = [
        "--config",
        str(REPO / "examples/matey/matey.toml"),
        "--set",
        f"data.path={args.baseline}",
        "--set",
        f"model.pretrained_path={args.checkpoint}",
        "--set",
        "logging.backend=none",
        "--set",
        "drift_detection.max_stream_updates=1",
    ]
    cfg = build_config(argv)
    harness = MATEYHarness(cfg)
    harness.update_data_stream()
    _, val_loader = harness.get_train_dataloaders()

    # Cache a few batches so each offset sees identical data.
    batches = []
    for i, b in enumerate(val_loader):
        x, y = harness._unpack(b)
        batches.append((x.to(cfg.device), y.to(cfg.device)))
        if len(batches) >= args.batches:
            break
    if not batches:
        raise RuntimeError("no batches produced by the stream loader")

    n_states = int(harness.model.matey_model.space_bag[0].weight.shape[1])
    orig = batches[0][0].field_labels
    print(f"[info] checkpoint n_states (embedding columns) = {n_states}")
    print(f"[info] field_labels currently supplied by the loader = {orig.tolist()}")
    print(f"[info] batches cached = {len(batches)}")

    kmax = (n_states - 3) if args.kmax < 0 else args.kmax
    metric_names = list(harness.eval_metrics.keys())
    fields = [m for m in metric_names if m.startswith("nrmse_") and m != "nrmse_mean"]

    def evaluate(labels: list[int]) -> dict:
        lab = torch.tensor([labels], dtype=torch.long, device=cfg.device)
        acc: dict[str, list[float]] = {m: [] for m in metric_names}
        for x, y in batches:
            xk = dataclasses.replace(x, field_labels=lab, field_labels_out=lab)
            y_hat = harness.model(xk)
            for m, fn in harness.eval_metrics.items():
                acc[m].append(harness._to_scalar(fn(y_hat, y)))
        return {m: float(np.mean(v)) for m, v in acc.items()}

    if args.mode == "per-field":
        # Coordinate descent: the fields need not be contiguous, so sweep each
        # channel over the whole embedding table with the others held fixed.
        cur = [int(c) for c in orig[0].tolist()]
        history = []
        harness.model.eval()
        with torch.no_grad():
            for it in range(args.iters):
                for ch, fname in enumerate(fields):
                    best_j, best_v, curve = cur[ch], float("inf"), []
                    for j in range(0, n_states):
                        trial = list(cur)
                        trial[ch] = j
                        r = evaluate(trial)
                        curve.append(
                            {
                                "j": j,
                                **{f: r[f] for f in fields},
                                "nrmse_mean": r["nrmse_mean"],
                            }
                        )
                        if r[fname] < best_v:
                            best_v, best_j = r[fname], j
                    cur[ch] = best_j
                    print(
                        f"[iter {it}] {fname}: best index {best_j} "
                        f"({fname}={best_v:.5f})  labels now {cur}"
                    )
                    history.append(
                        {
                            "iter": it,
                            "channel": ch,
                            "field": fname,
                            "best_index": best_j,
                            "best_value": best_v,
                            "curve": curve,
                        }
                    )
        final = evaluate(cur)
        print("\n=== per-field result ===")
        print(
            f"loader (buggy) labels {orig[0].tolist()} -> "
            f"nrmse_mean={evaluate([int(c) for c in orig[0].tolist()])['nrmse_mean']:.5f}"
        )
        print(f"recovered labels      {cur} -> nrmse_mean={final['nrmse_mean']:.5f}")
        for f in fields:
            print(f"   {f}: {final[f]:.5f}")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(
                {
                    "n_states": n_states,
                    "loader_field_labels": orig.tolist(),
                    "fields": fields,
                    "recovered_labels": cur,
                    "final_metrics": final,
                    "history": history,
                },
                fh,
                indent=2,
            )
        print(f"[done] wrote {out}")
        return 0

    print(f"[info] sweeping k = {args.kmin}..{kmax} step {args.stride}")

    rows = []
    t0 = time.time()
    harness.model.eval()
    with torch.no_grad():
        for k in range(args.kmin, kmax + 1, args.stride):
            lab = torch.tensor([[k, k + 1, k + 2]], dtype=torch.long, device=cfg.device)
            acc: dict[str, list[float]] = {m: [] for m in metric_names}
            for x, y in batches:
                xk = dataclasses.replace(x, field_labels=lab, field_labels_out=lab)
                y_hat = harness.model(xk)
                for m, fn in harness.eval_metrics.items():
                    acc[m].append(harness._to_scalar(fn(y_hat, y)))
            row = {"k": k, **{m: float(np.mean(v)) for m, v in acc.items()}}
            rows.append(row)
            if (k - args.kmin) % (25 * args.stride) == 0:
                el = time.time() - t0
                print(f"  k={k:4d}  nrmse_mean={row['nrmse_mean']:.5f}  ({el:.0f}s)")

    rows.sort(key=lambda r: r["nrmse_mean"])
    best = rows[0]
    print("\n=== best 10 offsets by nrmse_mean ===")
    hdr = "   k    " + "".join(f"{f:>13}" for f in fields) + f"{'nrmse_mean':>13}"
    print(hdr)
    for r in rows[:10]:
        print(
            f"{r['k']:5d}   "
            + "".join(f"{r[f]:13.5f}" for f in fields)
            + f"{r['nrmse_mean']:13.5f}"
        )
    baseline_row = next((r for r in rows if r["k"] == int(orig[0][0])), None)
    print(
        f"\ncurrent (buggy) k={int(orig[0][0])}: "
        f"nrmse_mean={baseline_row['nrmse_mean']:.5f}"
        if baseline_row
        else ""
    )
    print(f"best k={best['k']}: nrmse_mean={best['nrmse_mean']:.5f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(
            {
                "n_states": n_states,
                "loader_field_labels": orig.tolist(),
                "metric_names": metric_names,
                "n_batches": len(batches),
                "rows": sorted(rows, key=lambda r: r["k"]),
                "best": best,
            },
            fh,
            indent=2,
        )
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
