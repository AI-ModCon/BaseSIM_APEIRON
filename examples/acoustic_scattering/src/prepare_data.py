#!/usr/bin/env python
"""Convert The Well acoustic_scattering_maze HDF5 files to a .pt tensor.

Reads HDF5 files produced by ``the-well-download``, extracts the pressure
field, optionally subsamples trajectories and spatial resolution, and saves
the result as a single ``.pt`` file ready for the Apeiron harness.

Memory-efficient: trajectories are selected and downsampled *per file*
so only the kept (and possibly resized) data is ever resident in memory.

The Well organises data as::

    <base>/acoustic_scattering_maze/<split>/
        ├── file_0000.hdf5   # n_trajectories per file
        ├── file_0001.hdf5
        └── ...

Each HDF5 file has (at minimum)::

    /t0_fields/pressure          (B, T, H, W)   float32
    /t0_fields/c                 (B, 1, H, W)   float32   speed of sound (encodes walls)
    /scalars/<param>             (B,) or scalar
    @n_trajectories              int
    @simulation_parameters       list[str]

Output
------
A dictionary saved via ``torch.save`` containing:

- ``"pressure"``    : Tensor (N, T, H', W')  — the main data tensor
- ``"metadata"``    : dict with per-trajectory scalars and derived complexity
- ``"complexity"``  : Tensor (N,) — pre-computed complexity score per trajectory
- ``"source_info"`` : dict with provenance (files, sampling, resolution)

Usage
-----
::

    # Full dataset, 256x256
    python -m examples.acoustic_scattering.src.prepare_data \\
        --data-dir /path/to/acoustic_scattering_maze/train \\
        --output data/acoustic_scattering.pt

    # Subsample 200 trajectories, downsample to 128x128
    python -m examples.acoustic_scattering.src.prepare_data \\
        --data-dir /path/to/acoustic_scattering_maze/train \\
        --output data/acoustic_scattering_200x128.pt \\
        --n-trajectories 200 \\
        --spatial-size 128 \\
        --seed 42

    # Include validation split too
    python -m examples.acoustic_scattering.src.prepare_data \\
        --data-dir /path/to/acoustic_scattering_maze/train \\
        --data-dir /path/to/acoustic_scattering_maze/valid \\
        --output data/acoustic_scattering_all.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import interpolate


# ------------------------------------------------------------------
# HDF5 discovery
# ------------------------------------------------------------------


def discover_hdf5_files(dirs: list[Path]) -> list[Path]:
    """Return sorted list of .hdf5 / .h5 files across all directories."""
    files: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"Directory not found: {d}")
        files.extend(sorted(d.glob("*.hdf5")))
        files.extend(sorted(d.glob("*.h5")))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in: {[str(d) for d in dirs]}")
    return files


# ------------------------------------------------------------------
# Two-pass processing
# ------------------------------------------------------------------


def survey_files(files: list[Path]) -> list[int]:
    """Return the number of trajectories in each HDF5 file (without loading data)."""
    counts: list[int] = []
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            n = f["t0_fields"]["pressure"].shape[0]
            counts.append(n)
    return counts


def choose_indices(
    file_counts: list[int],
    n_keep: int,
    seed: int,
) -> list[list[int]]:
    """Decide which local trajectory indices to keep from each file.

    Returns a list (one per file) of sorted local indices.
    """
    n_total = sum(file_counts)
    n_keep = min(n_keep, n_total)

    # Draw n_keep global indices
    rng = np.random.RandomState(seed)
    if n_keep < n_total:
        global_idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))
    else:
        global_idx = np.arange(n_total)

    # Map global → (file_id, local_index)
    per_file: list[list[int]] = [[] for _ in file_counts]
    cumsum = 0
    file_id = 0
    for gi in global_idx:
        while gi >= cumsum + file_counts[file_id]:
            cumsum += file_counts[file_id]
            file_id += 1
        per_file[file_id].append(int(gi - cumsum))

    return per_file


def downsample_trajectory(traj: np.ndarray, target_size: int) -> Tensor:
    """Bilinear downsample a single trajectory (T, H, W) → (T, S, S) Tensor."""
    t = torch.from_numpy(traj).unsqueeze(0).float()  # (1, T, H, W)
    t = interpolate(
        t, size=(target_size, target_size), mode="bilinear", align_corners=False
    )
    return t.squeeze(0)  # (T, S, S)


def read_selected(
    fpath: Path,
    local_indices: list[int],
    spatial_size: int,
) -> tuple[list[Tensor], list[np.ndarray], dict[str, list[np.ndarray]]]:
    """Read only the selected trajectories from one HDF5 file.

    Downsamples spatially on the fly so the full-res data is never
    fully resident in memory.

    Returns
    -------
    pressures : list of Tensor, each (T, H', W')
    c_fields  : list of ndarray, each (1, H, W) — original resolution (small)
    scalars   : dict of param → list of ndarray per trajectory
    """
    pressures: list[Tensor] = []
    c_fields: list[np.ndarray] = []
    scalars: dict[str, list[np.ndarray]] = {}

    with h5py.File(fpath, "r") as f:
        pressure_ds = f["t0_fields"]["pressure"]  # (B, T, H, W)
        has_c = "c" in f["t0_fields"]
        c_ds = f["t0_fields"]["c"] if has_c else None

        scalar_keys: list[str] = []
        if "scalars" in f:
            scalar_keys = list(f["scalars"].keys())

        for li in local_indices:
            # Read one trajectory at a time — h5py loads only the slice
            p = pressure_ds[li]  # (T, H, W)

            if spatial_size > 0 and p.shape[-1] != spatial_size:
                pressures.append(downsample_trajectory(p, spatial_size))
            else:
                pressures.append(torch.from_numpy(p))

            if c_ds is not None:
                c_fields.append(c_ds[li])  # (1, H, W) — small, keep as-is

            for sk in scalar_keys:
                scalars.setdefault(sk, []).append(f["scalars"][sk][li])

    return pressures, c_fields, scalars


# ------------------------------------------------------------------
# Complexity from metadata
# ------------------------------------------------------------------


def compute_metadata_complexity(
    c_fields: np.ndarray | None,
    scalars: dict[str, np.ndarray],
    n_traj: int,
) -> np.ndarray:
    """Derive a per-trajectory complexity score from metadata.

    Strategy (in priority order):
    1. If speed-of-sound field ``c`` is available, use the fraction of wall
       pixels (high-density regions where c differs from the path value).
       More walls → more complex scattering.
    2. If scalar ``n_pressure_rings`` or similar is available, use it directly.
    3. Fall back to zeros (the harness will use spatial-variance instead).

    Returns
    -------
    complexity : ndarray (N,)
    """
    # --- Strategy 1: wall fraction from c field ---
    if c_fields is not None and c_fields.size > 0:
        # c has shape (N, 1, H, W).  Wall pixels have a very different c
        # value from the path.  The path c is sqrt(K/rho_path) = sqrt(4/3).
        # Wall c is sqrt(K/rho_wall) = sqrt(4/1e6) ≈ 0.002.
        # We threshold: pixels where c < median(c) are walls.
        c_squeezed = c_fields.squeeze(1)  # (N, H, W)
        per_traj_median = np.median(c_squeezed, axis=(1, 2), keepdims=True)
        wall_mask = c_squeezed < per_traj_median
        wall_fraction = wall_mask.mean(axis=(1, 2))  # (N,)
        return wall_fraction.astype(np.float32)

    # --- Strategy 2: scalar parameters ---
    # Common names in The Well acoustic scattering metadata
    for ring_key in ("n_pressure_rings", "num_rings", "n_rings"):
        if ring_key in scalars:
            vals = scalars[ring_key].flatten()[:n_traj]
            return vals.astype(np.float32)

    # --- Fallback ---
    return np.zeros(n_traj, dtype=np.float32)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert The Well acoustic_scattering_maze HDF5 → .pt"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        required=True,
        help="Directory containing HDF5 files (repeatable for multiple splits)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .pt file path",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=0,
        help="Max trajectories to keep (0 = all). Sampled uniformly.",
    )
    parser.add_argument(
        "--spatial-size",
        type=int,
        default=0,
        help="Downsample to this spatial resolution (0 = keep original).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for trajectory subsampling.",
    )
    args = parser.parse_args(argv)

    # 1. Discover files and survey trajectory counts (no data loaded)
    files = discover_hdf5_files(args.data_dir)
    file_counts = survey_files(files)
    n_total = sum(file_counts)
    print(f"Found {len(files)} HDF5 file(s), {n_total} total trajectories")
    for fpath, cnt in zip(files, file_counts):
        print(f"  {fpath.name}: {cnt} trajectories")

    # 2. Decide which trajectories to keep
    n_keep = args.n_trajectories if args.n_trajectories > 0 else n_total
    n_keep = min(n_keep, n_total)
    per_file_indices = choose_indices(file_counts, n_keep, args.seed)
    print(f"Keeping {n_keep} / {n_total} trajectories")

    # 3. Read selected trajectories, downsample on the fly
    all_pressure: list[Tensor] = []
    all_c: list[np.ndarray] = []
    all_scalars: dict[str, list[np.ndarray]] = {}

    for fpath, local_idx in zip(files, per_file_indices):
        if not local_idx:
            continue
        print(
            f"  Reading {len(local_idx)} trajectories from {fpath.name} ...",
            flush=True,
        )
        pressures, c_fields, scalars = read_selected(
            fpath, local_idx, args.spatial_size
        )
        all_pressure.extend(pressures)
        all_c.extend(c_fields)
        for k, v in scalars.items():
            all_scalars.setdefault(k, []).extend(v)

    # 4. Stack into tensors
    pressure_t = torch.stack(all_pressure)  # (N, T, H', W')
    print(f"Pressure tensor shape: {pressure_t.shape}")

    c_np: np.ndarray | None = None
    if all_c:
        c_np = np.stack(all_c)  # (N, 1, H, W) — original resolution

    merged_scalars: dict[str, np.ndarray] = {}
    for k, v in all_scalars.items():
        merged_scalars[k] = np.array(v)

    # 5. Compute complexity from metadata
    complexity = compute_metadata_complexity(c_np, merged_scalars, n_keep)
    print(f"Complexity range: [{complexity.min():.4f}, {complexity.max():.4f}]")

    # 6. Build metadata dict
    metadata: dict[str, Any] = {}
    for k, v in merged_scalars.items():
        metadata[k] = v.tolist()

    source_info = {
        "files": [str(f) for f in files],
        "n_original": n_total,
        "n_kept": n_keep,
        "spatial_size": int(pressure_t.shape[-1]),
        "seed": args.seed,
    }

    # 7. Save
    output_dict = {
        "pressure": pressure_t,
        "metadata": metadata,
        "complexity": torch.from_numpy(complexity),
        "source_info": source_info,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_dict, args.output)
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Saved to {args.output} ({size_mb:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
