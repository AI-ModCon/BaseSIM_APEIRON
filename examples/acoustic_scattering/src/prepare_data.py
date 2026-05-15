#!/usr/bin/env python
"""Convert The Well acoustic_scattering_maze HDF5 files to a .pt tensor.

Reads HDF5 files produced by ``the-well-download``, extracts the pressure
field, optionally subsamples trajectories and spatial resolution, and saves
the result as a single ``.pt`` file ready for the Apeiron harness.

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

The harness's ``load_tensor`` call expects shape ``(seq, len, x, y)``, so
we also provide a thin wrapper that unpacks just the ``"pressure"`` key.

Usage
-----
::

    # Full dataset, 256x256
    python -m examples.acoustic_scattering.src.prepare_data \
        --data-dir /path/to/acoustic_scattering_maze/train \
        --output data/acoustic_scattering.pt

    # Subsample 200 trajectories, downsample to 128x128
    python -m examples.acoustic_scattering.src.prepare_data \
        --data-dir /path/to/acoustic_scattering_maze/train \
        --output data/acoustic_scattering_200x128.pt \
        --n-trajectories 200 \
        --spatial-size 128 \
        --seed 42

    # Include validation split too
    python -m examples.acoustic_scattering.src.prepare_data \
        --data-dir /path/to/acoustic_scattering_maze/train \
        --data-dir /path/to/acoustic_scattering_maze/valid \
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
# HDF5 reading
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


def read_hdf5_file(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read one HDF5 file, returning pressure, c, and scalar metadata.

    Returns
    -------
    pressure : ndarray (B, T, H, W)
    c_field  : ndarray (B, 1, H, W) or empty if not present
    scalars  : dict of param_name → ndarray (B,)
    """
    with h5py.File(path, "r") as f:
        # Pressure field — required
        pressure = f["t0_fields"]["pressure"][:]  # (B, T, H, W)

        # Speed of sound — encodes maze geometry (optional)
        c_field = np.empty(0)
        if "c" in f["t0_fields"]:
            c_field = f["t0_fields"]["c"][:]  # (B, 1, H, W)

        # Scalar metadata
        scalars: dict[str, np.ndarray] = {}
        if "scalars" in f:
            for key in f["scalars"]:
                val = f["scalars"][key][:]
                scalars[key] = val

    return pressure, c_field, scalars


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
# Spatial downsampling
# ------------------------------------------------------------------


def downsample_spatial(data: Tensor, target_size: int) -> Tensor:
    """Bilinear downsample (N, T, H, W) → (N, T, target_size, target_size).

    Processes one trajectory at a time to limit peak memory.
    """
    if data.shape[-1] == target_size and data.shape[-2] == target_size:
        return data

    N, T = data.shape[:2]
    out = torch.empty(N, T, target_size, target_size, dtype=data.dtype)

    for i in range(N):
        # interpolate expects (B, C, H, W) — treat T as the channel dim
        frame = data[i].unsqueeze(0)  # (1, T, H, W)
        frame = interpolate(
            frame, size=(target_size, target_size), mode="bilinear", align_corners=False
        )
        out[i] = frame.squeeze(0)

    return out


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

    # 1. Discover and read HDF5 files
    files = discover_hdf5_files(args.data_dir)
    print(f"Found {len(files)} HDF5 file(s)")

    all_pressure: list[np.ndarray] = []
    all_c: list[np.ndarray] = []
    all_scalars: dict[str, list[np.ndarray]] = {}

    for fpath in files:
        print(f"  Reading {fpath.name} ...", end=" ", flush=True)
        pressure, c_field, scalars = read_hdf5_file(fpath)
        print(f"pressure {pressure.shape}", flush=True)

        all_pressure.append(pressure)
        if c_field.size > 0:
            all_c.append(c_field)

        for k, v in scalars.items():
            all_scalars.setdefault(k, []).append(v)

    # Concatenate along batch dim
    pressure_np = np.concatenate(all_pressure, axis=0)  # (N, T, H, W)
    c_np = np.concatenate(all_c, axis=0) if all_c else None
    merged_scalars = {k: np.concatenate(v, axis=0) for k, v in all_scalars.items()}

    n_total = pressure_np.shape[0]
    print(f"Total trajectories: {n_total}, shape: {pressure_np.shape}")

    # 2. Subsample trajectories
    rng = np.random.RandomState(args.seed)
    n_keep = args.n_trajectories if args.n_trajectories > 0 else n_total
    n_keep = min(n_keep, n_total)

    if n_keep < n_total:
        idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))
        pressure_np = pressure_np[idx]
        if c_np is not None:
            c_np = c_np[idx]
        merged_scalars = {k: v[idx] for k, v in merged_scalars.items()}
        print(f"Subsampled to {n_keep} trajectories")
    else:
        idx = np.arange(n_total)

    # 3. Compute complexity from metadata
    complexity = compute_metadata_complexity(c_np, merged_scalars, n_keep)
    print(f"Complexity range: [{complexity.min():.4f}, {complexity.max():.4f}]")

    # 4. Convert to torch and optionally downsample
    pressure_t = torch.from_numpy(pressure_np)  # (N, T, H, W)

    if args.spatial_size > 0:
        print(f"Downsampling {pressure_t.shape[-1]} → {args.spatial_size} ...")
        pressure_t = downsample_spatial(pressure_t, args.spatial_size)

    print(f"Final tensor shape: {pressure_t.shape}")

    # 5. Build metadata dict
    metadata: dict[str, Any] = {}
    for k, v in merged_scalars.items():
        metadata[k] = v.tolist()

    source_info = {
        "files": [str(f) for f in files],
        "n_original": n_total,
        "n_kept": n_keep,
        "subsampled_indices": idx.tolist(),
        "spatial_size": pressure_t.shape[-1],
        "original_spatial_size": pressure_np.shape[-1],
        "seed": args.seed,
    }

    # 6. Save
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
