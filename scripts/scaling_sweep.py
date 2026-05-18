#!/usr/bin/env python
"""Neural scaling sweep: model size × data size, Track 1 budget → Track 2.

For each (model_size, n_trajectories) cell:
  1. Generate a data subsample via ``prepare_data`` (if not already cached).
  2. Generate a shared init checkpoint for that model size.
  3. Run Track 1 (offline, 1 epoch) — measure its total FLOPs.
  4. Run Track 2 (Apeiron CL) with Track 1's FLOPs as its budget.
  5. Collect both ``results.json`` into a summary CSV.
  6. Generate scaling plots.

Usage::

    python scripts/scaling_sweep.py \
        --cl-config examples/acoustic_scattering/acoustic_scattering_cl.toml \
        --offline-config examples/acoustic_scattering/acoustic_scattering_offline.toml \
        --data-dir /path/to/acoustic_scattering_maze/train \
        --models vit_dense_small,vit_dense_base,vit_dense_large \
        --n-trajectories 50,100,200,500 \
        --output-dir output/scaling_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Neural scaling sweep: model size × data size"
    )
    p.add_argument(
        "--cl-config",
        type=Path,
        required=True,
        help="TOML config for Track 2 (Apeiron CL)",
    )
    p.add_argument(
        "--offline-config",
        type=Path,
        required=True,
        help="TOML config for Track 1 (offline supervised)",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to raw HDF5 directory (for prepare_data)",
    )
    p.add_argument(
        "--models",
        type=str,
        default="vit_dense_small,vit_dense_base,vit_dense_large",
        help="Comma-separated model names (default: small,base,large)",
    )
    p.add_argument(
        "--n-trajectories",
        type=str,
        default="50,100,200,500",
        help="Comma-separated trajectory counts (default: 50,100,200,500)",
    )
    p.add_argument(
        "--spatial-size",
        type=int,
        default=256,
        help="Spatial resolution for subsamples (default: 256)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data subsampling and init (default: 42)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/scaling_sweep"),
        help="Root output directory (default: output/scaling_sweep)",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Subsample generation
# ---------------------------------------------------------------------------


def prepare_subsample(
    data_dir: str,
    n_traj: int,
    spatial_size: int,
    seed: int,
    output_dir: Path,
) -> Path:
    """Run prepare_data to create a .pt subsample (skip if cached)."""
    out_path = output_dir / "data" / f"acoustic_{n_traj}x{spatial_size}.pt"
    if out_path.exists():
        print(f"  [cached] {out_path}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "examples.acoustic_scattering.src.prepare_data",
        "--data-dir",
        data_dir,
        "--output",
        str(out_path),
        "--n-trajectories",
        str(n_traj),
        "--spatial-size",
        str(spatial_size),
        "--seed",
        str(seed),
    ]
    print(
        f"  Generating subsample: {n_traj} trajectories @ {spatial_size}x{spatial_size}"
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: prepare_data failed:\n{result.stderr}")
        raise RuntimeError(f"prepare_data failed for n_traj={n_traj}")
    return out_path


# ---------------------------------------------------------------------------
# Init checkpoint generation
# ---------------------------------------------------------------------------


def generate_init_checkpoint(
    model_name: str,
    seed: int,
    output_dir: Path,
) -> Path:
    """Generate a deterministic init checkpoint for a model size (skip if cached)."""
    out_path = output_dir / "init_weights" / f"{model_name}_seed{seed}.pt"
    if out_path.exists():
        print(f"  [cached] {out_path}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-c",
        (
            "from examples.acoustic_scattering.model import generate_init_checkpoint; "
            f"generate_init_checkpoint('{model_name}', '{out_path}', {seed})"
        ),
    ]
    print(f"  Generating init checkpoint: {model_name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: init checkpoint failed:\n{result.stderr}")
        raise RuntimeError(f"generate_init_checkpoint failed for {model_name}")
    return out_path


# ---------------------------------------------------------------------------
# Track runners
# ---------------------------------------------------------------------------


def run_offline(
    config_path: Path,
    model_name: str,
    data_path: Path,
    init_ckpt: Path,
    run_dir: Path,
    seed: int,
) -> dict | None:
    """Run Track 1: offline supervised, 1 epoch, no early stopping."""
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "src.offline_train",
        "--config",
        str(config_path),
        "--set",
        f"model.name={model_name}",
        "--set",
        f"model.pretrained_path={init_ckpt}",
        "--set",
        f"model.ckpts_path={run_dir}",
        "--set",
        f"data.path={data_path}",
        "--set",
        f"seed={seed}",
        "--set",
        "train.max_iter=1",
    ]

    print(f"  Track 1 (offline, 1 epoch): {run_dir.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: offline training failed:\n{result.stderr[-500:]}")
        return None

    return _read_results(run_dir)


def run_cl(
    config_path: Path,
    model_name: str,
    data_path: Path,
    init_ckpt: Path,
    run_dir: Path,
    seed: int,
) -> dict | None:
    """Run Track 2: Apeiron CL, runs to natural completion."""
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "src.main",
        "--config",
        str(config_path),
        "--set",
        f"model.name={model_name}",
        "--set",
        f"model.pretrained_path={init_ckpt}",
        "--set",
        f"model.ckpts_path={run_dir}",
        "--set",
        f"data.path={data_path}",
        "--set",
        f"seed={seed}",
    ]

    print(f"  Track 2 (CL): {run_dir.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: CL training failed:\n{result.stderr[-500:]}")
        return None

    return _read_results(run_dir)


def _read_results(run_dir: Path) -> dict | None:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        print(f"  WARNING: {results_path} not found")
        return None
    with results_path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CSV & plotting
# ---------------------------------------------------------------------------


def collect_to_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary CSV: {output_path}")


def plot_scaling(csv_path: Path, output_dir: Path) -> None:
    """Generate scaling plots from the summary CSV."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("matplotlib/pandas not available; skipping plot.")
        return

    df = pd.read_csv(csv_path)

    # --- Plot 1: Test VRMSE vs. total FLOPs (log-log), one curve per track ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for track, group in df.groupby("track"):
        group = group.sort_values("total_flops")
        ax.plot(
            group["total_flops"],
            group["test_vrmse"],
            "o-",
            label=str(track),
            markersize=6,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total FLOPs")
    ax.set_ylabel("Test VRMSE")
    ax.set_title("Neural Scaling: Test VRMSE vs. Total FLOPs")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    path = output_dir / "scaling_flops.png"
    fig.savefig(path, dpi=150)
    print(f"Plot saved: {path}")
    plt.close(fig)

    # --- Plot 2: faceted by model, x=n_trajectories ---
    models = df["model"].unique()
    if len(models) > 1:
        fig, axes = plt.subplots(
            1, len(models), figsize=(5 * len(models), 4), sharey=True
        )
        if len(models) == 1:
            axes = [axes]
        for ax, model in zip(axes, sorted(models)):
            sub = df[df["model"] == model]
            for track, group in sub.groupby("track"):
                group = group.sort_values("n_trajectories")
                ax.plot(
                    group["n_trajectories"],
                    group["test_vrmse"],
                    "o-",
                    label=str(track),
                    markersize=6,
                )
            ax.set_xlabel("Trajectories")
            ax.set_ylabel("Test VRMSE")
            ax.set_title(model)
            ax.legend(fontsize=8)
            ax.grid(True, ls="--", alpha=0.5)
        fig.suptitle("Test VRMSE vs. Data Size (by Model)", y=1.02)
        fig.tight_layout()
        path = output_dir / "scaling_by_model.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    models = [m.strip() for m in args.models.split(",")]
    n_traj_list = [int(n.strip()) for n in args.n_trajectories.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for model_name in models:
        print(f"\n{'=' * 70}")
        print(f"Model: {model_name}")
        print(f"{'=' * 70}")

        # Shared init checkpoint for this model size
        init_ckpt = generate_init_checkpoint(model_name, args.seed, args.output_dir)

        for n_traj in n_traj_list:
            print(f"\n--- {model_name} | {n_traj} trajectories ---")

            # 1. Generate data subsample
            data_path = prepare_subsample(
                args.data_dir, n_traj, args.spatial_size, args.seed, args.output_dir
            )

            cell_dir = args.output_dir / f"{model_name}_n{n_traj}"

            # 2. Track 1: offline, 1 epoch
            t1_result = run_offline(
                args.offline_config,
                model_name,
                data_path,
                init_ckpt,
                cell_dir / "offline",
                args.seed,
            )
            if t1_result is None:
                print("  Skipping cell (Track 1 failed)")
                continue

            t1_flops = t1_result.get("flops", {}).get("total", 0.0)
            t1_metrics = t1_result.get("test_metrics", {})
            print(f"  Track 1 FLOPs: {t1_flops:.2e}")

            rows.append(
                _make_row("offline", model_name, n_traj, t1_flops, None, t1_result)
            )

            # 3. Track 2: CL (runs to completion, compared against Track 1 FLOPs)
            t2_result = run_cl(
                args.cl_config,
                model_name,
                data_path,
                init_ckpt,
                cell_dir / "cl",
                args.seed,
            )
            if t2_result is None:
                print("  Track 2 failed")
                continue

            t2_metrics = t2_result.get("test_metrics", {})
            t2_flops = t2_result.get("flops", {}).get("total", 0.0)
            print(f"  Track 2 FLOPs: {t2_flops:.2e} (Track 1 ref: {t1_flops:.2e})")

            rows.append(
                _make_row("apeiron", model_name, n_traj, t2_flops, t1_flops, t2_result)
            )

            # Quick comparison
            v1 = t1_metrics.get("vrmse", float("nan"))
            v2 = t2_metrics.get("vrmse", float("nan"))
            print(f"  VRMSE — offline: {v1:.4f}  CL: {v2:.4f}")

    # 4. Save CSV
    csv_path = args.output_dir / "scaling_summary.csv"
    collect_to_csv(rows, csv_path)

    # 5. Plot
    if not args.no_plot and rows:
        plot_scaling(csv_path, args.output_dir)

    return 0


def _make_row(
    track: str,
    model: str,
    n_traj: int,
    total_flops: float,
    flop_budget: float | None,
    result: dict,
) -> dict:
    test_metrics = result.get("test_metrics", {})
    return {
        "track": track,
        "model": model,
        "n_trajectories": n_traj,
        "total_flops": total_flops,
        "flop_budget": flop_budget,
        "test_vrmse": test_metrics.get("vrmse", float("nan")),
        "test_loss": test_metrics.get("loss", float("nan")),
        "epochs": result.get("epochs", None),
        "drift_events": result.get("drift_events", None),
        "data_budget": result.get("data_budget", None),
    }


if __name__ == "__main__":
    sys.exit(main())
