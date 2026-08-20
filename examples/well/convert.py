"""Materialize Well trajectories into a committed ``WindowStore``.

Each committed window is a contiguous block of timesteps turned into **next-step
prediction** pairs ``(field_t -> field_{t+1})`` -- the standard neural-surrogate
task. Windows are ordered by the simulation parameter (``tcool``), so streaming
through the store walks the physical regime from fast- to slow-cooling: a real,
monotonic concept drift the detector and CL loop have to cope with.

The core (:func:`convert_files`) is network-free and is what the tests and the
benchmark run against a fixture. :func:`download_well` is an optional helper that
pulls real Well HDF5 from the HuggingFace Hub.

Run::

    # fixture (no download):
    python -m examples.well.fixture /tmp/wellsrc
    python -m examples.well.convert --files /tmp/wellsrc/*.hdf5 --out /tmp/wellstore

    # real data (downloads N smallest files of a split):
    python -m examples.well.convert --dataset turbulent_radiative_layer_2D \
        --split test --max-files 4 --out /data/wellstore
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

from apeiron.data.window_store import WindowStore
from examples.well.wellio import infer_param, read_all_trajectories

WELL_META = "well_meta.json"


def _chunk_ranges(n_time: int, window_steps: int) -> Iterator[tuple[int, int, int]]:
    """Yield ``(chunk_idx, t_lo, t_hi)`` time chunks over ``n_time`` steps.

    A chunk spans ``window_steps`` timesteps and yields ``window_steps - 1``
    consecutive ``(field_t, field_{t+1})`` pairs per trajectory.
    """
    chunk = 0
    for lo in range(0, n_time - 1, max(1, window_steps - 1)):
        hi = min(lo + window_steps, n_time)
        if hi - lo < 2:
            break
        yield chunk, lo, hi
        chunk += 1


def convert_files(
    files: Sequence[str | Path],
    out_store: str | Path,
    *,
    window_steps: int = 8,
    val_fraction: float = 0.25,
    max_trajectories: int | None = None,
    order_by_param: bool = True,
) -> dict:
    """Convert Well HDF5 files into a committed WindowStore ordered by drift.

    Each committed window stacks **all trajectories** of a file over one time
    chunk, so a window holds ``n_trajectories * (window_steps - 1)`` next-step
    pairs -- big enough to feed many ranks -- while staying within one simulation
    regime (a file is one ``tcool``). ``max_trajectories`` caps how many are used
    (None = all). Returns the store metadata dict (also written to
    ``well_meta.json``).
    """
    paths = [Path(p) for p in files]
    if not paths:
        raise ValueError("no input files")

    # Order by the simulation parameter so the stream drifts monotonically.
    if order_by_param:
        keyed = []
        for p in paths:
            pinfo = infer_param(p)
            keyed.append((pinfo[1] if pinfo else 0.0, p))
        paths = [p for _, p in sorted(keyed, key=lambda kp: kp[0])]

    bundles = [read_all_trajectories(p, max_trajectories) for p in paths]
    channels = bundles[0].channels
    for b in bundles:
        if b.channels != channels:
            raise ValueError(
                f"channel layout differs: {b.source} has {b.channels}, "
                f"expected {channels}"
            )

    # Per-channel normalization stats over every trajectory of every file.
    n_ch = len(channels)
    csum = np.zeros(n_ch, dtype=np.float64)
    csqsum = np.zeros(n_ch, dtype=np.float64)
    count = 0
    for b in bundles:
        for traj in b.trajectories:
            flat = traj.reshape(traj.shape[0], n_ch, -1)
            csum += flat.sum(axis=(0, 2))
            csqsum += (flat.astype(np.float64) ** 2).sum(axis=(0, 2))
            count += flat.shape[0] * flat.shape[2]
    mean = (csum / count).astype(np.float32)
    var = (csqsum / count - (csum / count) ** 2).clip(min=1e-12)
    std = np.sqrt(var).astype(np.float32)

    store = WindowStore(out_store, catalog=True)
    param_name = (infer_param(paths[0]) or ("param", 0.0))[0]

    skipped = 0
    for regime_idx, b in enumerate(bundles):
        pv = float(b.params.get(param_name, regime_idx))
        prov = {
            "source": b.source,
            "params": b.params,
            "n_trajectories": b.n_trajectories,
            "grid": list(b.trajectories[0].shape[2:]),
        }
        n_time = b.trajectories[0].shape[0]
        for chunk_idx, t_lo, t_hi in _chunk_ranges(n_time, window_steps):
            # Stack every trajectory's next-step pairs for this time chunk into
            # one window (same regime, more samples). Contiguous, so val_fraction
            # holds out the trailing trajectories.
            x = np.concatenate([tr[t_lo : t_hi - 1] for tr in b.trajectories], axis=0)
            y = np.concatenate([tr[t_lo + 1 : t_hi] for tr in b.trajectories], axis=0)

            # Drop a chunk too small to form a non-empty train AND val split.
            n = x.shape[0]
            n_val = round(n * val_fraction)
            if n_val < 1 or n - n_val < 1:
                skipped += 1
                continue
            store.commit(
                x,
                y,
                window_id=f"{regime_idx:03d}_{chunk_idx:02d}",
                val_fraction=val_fraction,
                t_start=f"{param_name}={pv:.4f}:t{t_lo:04d}",
                t_end=f"{param_name}={pv:.4f}:t{t_hi:04d}",
                extra={
                    "provenance": prov,
                    param_name: pv,
                    "regime_idx": regime_idx,
                    "chunk_idx": chunk_idx,
                },
            )

    meta = {
        "dataset": "the_well",
        "task": "next_step_regression",
        "channels": list(channels),
        "n_channels": n_ch,
        "grid": list(bundles[0].trajectories[0].shape[2:]),
        "param_name": param_name,
        "window_steps": window_steps,
        "val_fraction": val_fraction,
        "trajectories_per_window": bundles[0].n_trajectories,
        "norm_mean": mean.tolist(),
        "norm_std": std.tolist(),
        "n_windows": len(store),
        "skipped_small_windows": skipped,
        "regime_order": [
            float(b.params.get(param_name, i)) for i, b in enumerate(bundles)
        ],
    }
    (Path(out_store) / WELL_META).write_text(json.dumps(meta, indent=2))
    store.close()
    return meta


def download_well(
    dataset: str,
    split: str = "test",
    max_files: Optional[int] = 4,
    cache_dir: Optional[str] = None,
) -> list[str]:
    """Download up to ``max_files`` HDF5 files of a Well split from HuggingFace."""
    from huggingface_hub import HfApi, hf_hub_download

    repo = dataset if "/" in dataset else f"polymathic-ai/{dataset}"
    api = HfApi()
    info = api.repo_info(repo, repo_type="dataset", files_metadata=True)
    files = [
        (s.rfilename, s.size or 0)
        for s in (getattr(info, "siblings", None) or [])
        if s.rfilename.endswith((".hdf5", ".h5")) and f"/{split}/" in s.rfilename
    ]
    files.sort(key=lambda x: x[1])  # smallest first
    if max_files:
        files = files[:max_files]
    out = []
    for name, _ in files:
        out.append(
            hf_hub_download(
                repo_id=repo, filename=name, repo_type="dataset", cache_dir=cache_dir
            )
        )
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--files", nargs="+", help="Well HDF5 files (or globs)")
    src.add_argument("--dataset", help="Well dataset name to download from HF")
    p.add_argument("--split", default="test", help="HF split (with --dataset)")
    p.add_argument("--max-files", type=int, default=4, help="cap downloaded files")
    p.add_argument("--out", required=True, help="output WindowStore directory")
    p.add_argument("--window-steps", type=int, default=8)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument(
        "--max-trajectories",
        type=int,
        default=0,
        help="cap trajectories stacked per window (0 = all in the file)",
    )
    p.add_argument("--cache-dir", default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.dataset:
        files: list[str] = download_well(
            args.dataset, args.split, args.max_files, args.cache_dir
        )
    else:
        files = []
        for pat in args.files:
            files.extend(sorted(glob.glob(pat)) or [pat])
    meta = convert_files(
        files,
        args.out,
        window_steps=args.window_steps,
        val_fraction=args.val_fraction,
        max_trajectories=args.max_trajectories or None,
    )
    print(
        f"committed {meta['n_windows']} windows to {args.out}\n"
        f"  channels={meta['channels']} grid={meta['grid']} "
        f"trajectories/window={meta['trajectories_per_window']} "
        f"regimes={meta['regime_order']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
