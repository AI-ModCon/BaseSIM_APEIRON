#!/usr/bin/env python3
"""Stage a sequential multi-simulation SOLPS stream for the paper's Figure 2.

Figure 2 shows one time axis along which simulations arrive one after another:
first more of the machine the surrogate already handles, then simulations from
*different machines*. The claim is that drift detection fires where the machine
changes and continual learning then reduces error on the newly arrived machine.

To have enough monitoring windows for that, each simulation is cut into several
consecutive time **segments**, and each segment is staged as its own little
bundle -- one "arrival". Within a segment the train and valid ranges are
disjoint and separated by a gap, so an adaptation step can never train on the
frames it is scored against (the mistake ``stage_solps_fusionbench_bundles.sh``
makes by symlinking ``train/`` to ``valid/``).

The arrival order and the metadata each arrival carries are written to
``stream_manifest.json`` at the stream root, so the harness needs no new config
keys to walk them.

Case taxonomy, stated as it must be in the paper:

  baseline_d3d  DIII-D Sequence_sin4   same machine, in pre-training
  ood_d3d       DIII-D noLat_dribble   same machine, new scenario, HELD OUT
  kstar         KSTAR linear ramp      different machine, in pre-training

Only ``ood_d3d`` is genuinely unseen -- the checkpoint's ``train_data_paths``
covers the whole ``SOLPS2DwION/`` tree. The cross-machine arrivals are therefore
"different machine, under-fit", not "never seen".

Further machines are staged without editing this file, by passing
``--cases extra.json``. That is also how a device whose name may not appear in
this repository gets added, matching the way its normalisation envelope is
supplied through the data root's ``matey_settings.json``.

Usage
-----
    MATEYDATA=/path/to/mateydata python examples/matey/stage_solps_stream.py \\
        --out /path/to/solps_stream --segments 8 --window 60
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import netCDF4 as nc

MATEYDATA = Path(os.environ.get("MATEYDATA", "."))
PRETRAIN = Path(
    os.environ.get("SOLPS_PRETRAIN_ROOT", str(MATEYDATA / "Datasets_pretraining/solps"))
)

# Where each case's source b2time.nc lives and where it sits inside a staged
# bundle. Inlined rather than imported: this is the only staging script the
# example ships, so a second module would exist purely to hold this table.
CASES: dict[str, dict[str, Any]] = {
    "baseline_d3d": {
        "src": PRETRAIN
        / "SOLPS2DwION/D3D/174310_D"
        / "puff2.5e21_ss_Sequence_sin4_308_2d_output/b2time.nc",
        "rel": "D3D/174310_D",
        "in_pretraining": True,
    },
    "ood_d3d": {
        "src": MATEYDATA
        / "Datasets_notusedinpretraining/D3D/174310_D"
        / "puff2.5e21_ss_noLat_dribble_308_2d_output/b2time.nc",
        "rel": "D3D/174310_D",
        "in_pretraining": False,
    },
    "kstar": {
        "src": PRETRAIN / "SOLPS2DwION/KSTAR/19077_D/puff5e20_td_linear_ramp/b2time.nc",
        "rel": "KSTAR/19077_D",
        "in_pretraining": True,
    },
}


def write_time_subset(src_path: Path, dst_path: Path, start: int, stop: int) -> None:
    """Copy one netCDF file, keeping only frames ``[start, stop)``."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        nc.Dataset(str(src_path)) as src,
        nc.Dataset(str(dst_path), "w", format="NETCDF4") as dst,
    ):
        for name, dim in src.dimensions.items():
            size = max(0, stop - start) if name == "time" else len(dim)
            dst.createDimension(name, None if dim.isunlimited() else size)
        for name, var in src.variables.items():
            out = dst.createVariable(name, var.dtype, var.dimensions)
            for a in var.ncattrs():
                out.setncattr(a, var.getncattr(a))
            if "time" in var.dimensions:
                sl = [slice(None)] * len(var.dimensions)
                sl[var.dimensions.index("time")] = slice(start, stop)
                out[:] = var[tuple(sl)]
            else:
                out[:] = var[:]
        for a in src.ncattrs():
            dst.setncattr(a, src.getncattr(a))


# Arrival order along the horizontal axis of Figure 2. Same machine first so the
# cross-machine change point is unambiguous and late in the stream.
DEFAULT_ORDER = ["baseline_d3d", "ood_d3d", "kstar"]

# Machine label and whether the arrival is a machine change relative to the
# stream's starting machine. Kept here rather than inferred so the figure's
# annotations come from data, not from parsing directory names.
CASE_META = {
    "baseline_d3d": {
        "machine": "DIII-D",
        "scenario": "Sequence_sin4",
        "in_pretraining": True,
        "held_out": False,
    },
    "ood_d3d": {
        "machine": "DIII-D",
        "scenario": "noLat_dribble",
        "in_pretraining": False,
        "held_out": True,
    },
    "kstar": {
        "machine": "KSTAR",
        "scenario": "linear ramp",
        "in_pretraining": True,
        "held_out": False,
    },
}


def load_extra_cases(path: str) -> None:
    """Merge additional cases from a JSON file into CASES and CASE_META.

    Schema, one entry per case::

        {"<case>": {"src": "/abs/path/b2time.nc", "rel": "MACHINE/shot",
                    "machine": "...", "scenario": "...",
                    "in_pretraining": true, "held_out": false}}
    """
    for case, spec in json.loads(Path(path).read_text()).items():
        CASES[case] = {
            "src": Path(spec["src"]),
            "rel": spec["rel"],
            "in_pretraining": bool(spec.get("in_pretraining", True)),
        }
        CASE_META[case] = {
            k: spec[k]
            for k in ("machine", "scenario", "in_pretraining", "held_out")
            if k in spec
        }


def segment_ranges(
    n_time: int, n_segments: int, window: int, margin: int
) -> list[tuple[int, int]]:
    """Evenly spaced, non-overlapping time windows covering the run.

    Starts after ``margin`` frames: the first frames of a SOLPS run are the
    initial condition rather than a converged state, and including them puts a
    start-up transient at the head of the stream that reads as drift.
    """
    usable_lo, usable_hi = margin, n_time
    span = usable_hi - usable_lo
    if span < window:
        raise ValueError(f"only {span} usable frames, need >= {window}")
    max_segments = span // window
    k = min(n_segments, max_segments)
    if k < 1:
        raise ValueError(f"cannot fit a {window}-frame segment in {span} frames")
    step = (span - window) / (k - 1) if k > 1 else 0
    return [
        (int(usable_lo + i * step), int(usable_lo + i * step) + window)
        for i in range(k)
    ]


def stage_segment(
    src: Path, rel: str, dst_root: Path, lo: int, hi: int, train_frac: float, gap: int
) -> dict:
    """Write one arrival as a bundle with disjoint train/valid ranges."""
    span = hi - lo
    train_len = int(train_frac * span)
    train_lo, train_hi = lo, lo + train_len
    valid_lo, valid_hi = train_hi + gap, hi
    if valid_hi - valid_lo < 2:
        raise ValueError(f"segment [{lo},{hi}) too short for a {gap}-frame gap")

    if dst_root.exists():
        shutil.rmtree(dst_root)
    write_time_subset(
        src,
        dst_root / "train" / rel / f"train_t{train_lo:05d}_{train_hi:05d}.nc",
        train_lo,
        train_hi,
    )
    write_time_subset(
        src,
        dst_root / "valid" / rel / f"valid_t{valid_lo:05d}_{valid_hi:05d}.nc",
        valid_lo,
        valid_hi,
    )
    return {
        "train_range": [train_lo, train_hi],
        "valid_range": [valid_lo, valid_hi],
        "disjoint": train_hi <= valid_lo,
        "gap": gap,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Required: defaulting this into somebody's scratch directory is how a
    # stream root ends up somewhere nobody expects.
    ap.add_argument("--out", required=True, help="stream root to create")
    ap.add_argument("--order", default=",".join(DEFAULT_ORDER))
    ap.add_argument(
        "--cases",
        default="",
        help="JSON file of additional cases; see load_extra_cases",
    )
    ap.add_argument("--segments", type=int, default=8, help="arrivals per simulation")
    ap.add_argument("--window", type=int, default=60, help="frames per arrival")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument(
        "--margin",
        type=int,
        default=15,
        help="frames skipped at the start of each run (start-up transient)",
    )
    ap.add_argument(
        "--gap",
        type=int,
        default=5,
        help="frames between an arrival's train and valid ranges",
    )
    args = ap.parse_args()

    if args.cases:
        load_extra_cases(args.cases)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    order = [c.strip() for c in args.order.split(",") if c.strip()]

    manifest: list[dict] = []
    index = 0
    for case in order:
        if case not in CASES:
            print(f"[skip] unknown case {case!r}")
            continue
        spec = CASES[case]
        src = Path(spec["src"])
        if not src.is_file():
            print(f"[skip] {case}: missing source {src}")
            continue
        with nc.Dataset(str(src)) as d:
            n_time = int(d.dimensions["time"].size)
        try:
            ranges = segment_ranges(n_time, args.segments, args.window, args.margin)
        except ValueError as exc:
            print(f"[skip] {case}: {exc}")
            continue

        meta = CASE_META.get(case, {})
        for seg_i, (lo, hi) in enumerate(ranges):
            name = f"seg_{index:03d}_{case}_{seg_i:02d}"
            split = stage_segment(
                src, spec["rel"], out_root / name, lo, hi, args.train_frac, args.gap
            )
            manifest.append(
                {
                    "index": index,
                    "dir": name,
                    "case": case,
                    "segment": seg_i,
                    "source": str(src),
                    "rel": spec["rel"],
                    "time_range": [lo, hi],
                    **meta,
                    **split,
                }
            )
            index += 1
        print(
            f"[ok] {case:<13} n_time={n_time:<5} arrivals={len(ranges)} "
            f"window={args.window} machine={meta.get('machine')}"
        )

    if not manifest:
        print("nothing staged")
        return 1

    # Where the machine changes -- the change points Figure 2 annotates.
    change_points = [
        m["index"]
        for i, m in enumerate(manifest)
        if i > 0 and m.get("machine") != manifest[i - 1].get("machine")
    ]
    doc = {
        "order": order,
        "n_arrivals": len(manifest),
        "window": args.window,
        "train_frac": args.train_frac,
        "gap": args.gap,
        "margin": args.margin,
        "machine_change_points": change_points,
        "note": (
            "Only ood_d3d is genuinely held out; kstar and the third device are "
            "different machines that were nonetheless in pre-training. Do not "
            "name the third device in write-ups."
        ),
        "arrivals": manifest,
    }
    with (out_root / "stream_manifest.json").open("w") as fh:
        json.dump(doc, fh, indent=2)

    print(f"\n{len(manifest)} arrivals staged under {out_root}")
    print(f"machine change points at arrival index: {change_points}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
