#!/usr/bin/env python3
"""Plot matey inference drift CSV with eval, drift-check, and stream-boundary markers."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _series(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = df[df.metric == metric].copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.sort_values("step")


def _pick_drift_metric_series(
    df: pd.DataFrame, metric_index: int | None = None
) -> tuple[pd.DataFrame, str]:
    """Return drift/metric_* series; prefer explicit metric_index when set."""
    if metric_index is not None:
        name = f"drift/metric_{metric_index}"
        series = _series(df, name)
        if not series.empty:
            return series, name

    drift_metrics = sorted(
        m for m in df["metric"].unique() if isinstance(m, str) and m.startswith("drift/metric_")
    )
    if not drift_metrics:
        return pd.DataFrame(), "drift/metric_0"
    name = drift_metrics[-1]
    return _series(df, name), name


def _infer_stream_boundary_steps(
    eval_df: pd.DataFrame,
    *,
    batches_per_stream: int | None,
    num_streams: int | None,
) -> list[tuple[int, str]]:
    """Return (global_step, label) for each stream reload after the first."""
    n_eval = len(eval_df)
    if n_eval == 0:
        return []

    if batches_per_stream is None and num_streams is not None and num_streams > 1:
        batches_per_stream = max(1, n_eval // num_streams)
    if batches_per_stream is None:
        batches_per_stream = 200

    boundaries: list[tuple[int, str]] = []
    idx = batches_per_stream
    stream_num = 2
    max_boundaries = (num_streams - 1) if num_streams and num_streams > 1 else None
    while idx < n_eval:
        if max_boundaries is not None and len(boundaries) >= max_boundaries:
            break
        step = int(eval_df.iloc[idx]["step"])
        domain = "baseline" if (stream_num - 1) % 2 == 0 else "shift"
        boundaries.append((step, f"stream #{stream_num} ({domain})"))
        idx += batches_per_stream
        stream_num += 1
    return boundaries


def plot_dashboard(
    csv_path: Path,
    output_path: Path,
    *,
    batches_per_stream: int | None = None,
    num_streams: int | None = None,
    metric_index: int | None = None,
) -> None:
    df = pd.read_csv(csv_path)

    eval_nrmse_mean = _series(df, "eval/nrmse_mean")
    eval_nrmse = _series(df, "eval/nrmse")
    eval_loss = _series(df, "eval/loss")
    drift_metric, drift_metric_name = _pick_drift_metric_series(df, metric_index)
    drift_score = _series(df, "drift/score")
    drift_detected = _series(df, "drift/detected")

    eval_primary = (
        eval_nrmse_mean
        if not eval_nrmse_mean.empty
        else (eval_nrmse if not eval_nrmse.empty else eval_loss)
    )
    eval_primary_name = (
        "eval/nrmse_mean"
        if not eval_nrmse_mean.empty
        else ("eval/nrmse" if not eval_nrmse.empty else "eval/loss")
    )

    if eval_primary.empty and drift_metric.empty:
        raise ValueError(f"No eval or drift rows found in {csv_path}")

    eval_ref = eval_primary
    stream_bounds = _infer_stream_boundary_steps(
        eval_ref.reset_index(drop=True),
        batches_per_stream=batches_per_stream,
        num_streams=num_streams,
    )

    eval_steps = eval_primary["step"].tolist()
    drift_steps = drift_metric["step"].tolist()

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    panels: list[tuple] = [
        (axes[0], eval_primary, eval_primary_name, "Per-batch monitored NRMSE", "#1f77b4"),
        (axes[1], eval_loss, "eval/loss", "Per-batch loss", "#2ca02c"),
        (
            axes[2],
            drift_metric,
            drift_metric_name,
            f"ADWIN input ({drift_metric_name})",
            "#ff7f0e",
        ),
        (
            axes[3],
            drift_score,
            "drift/score",
            "Recent drift rate (0 until ADWIN fires)",
            "#9467bd",
        ),
    ]

    for ax, series, _name, ylabel, color in panels:
        if series.empty:
            ax.set_ylabel(ylabel)
            ax.text(0.01, 0.5, f"no {ylabel} data", transform=ax.transAxes)
            continue
        ax.plot(series["step"], series["value"], color=color, linewidth=0.9, zorder=3)
        ax.set_ylabel(ylabel)
        if _name == "drift/score" and not series.empty:
            if float(series["value"].max()) <= 0.0:
                ax.text(
                    0.99,
                    0.95,
                    "Score = recent ADWIN fire rate\n(stays 0 until drift_detected=True)",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                )

        for step in eval_steps:
            ax.axvline(step, color="#1f77b4", alpha=0.06, linewidth=0.6, zorder=1)
        for step in drift_steps:
            ax.axvline(step, color="#ff7f0e", alpha=0.06, linewidth=0.6, zorder=1)
        for step, label in stream_bounds:
            ax.axvline(step, color="#d62728", alpha=0.85, linewidth=1.4, linestyle="--", zorder=2)

    if not drift_detected.empty:
        det = drift_detected.copy()
        det["flag"] = det["value"].map({"True": 1, "False": 0, True: 1, False: 0})
        det = det.dropna(subset=["flag"])
        if not det.empty and det["flag"].sum() > 0:
            axes[3].scatter(
                det["step"],
                det["flag"],
                c="red",
                s=12,
                alpha=0.7,
                label="drift detected",
                zorder=4,
            )

    axes[-1].set_xlabel("Global logger step")
    axes[0].set_title("MATEY inference drift dashboard")

    if stream_bounds:
        for step, label in stream_bounds:
            axes[0].annotate(
                label,
                xy=(step, 1.0),
                xycoords=("data", "axes fraction"),
                rotation=90,
                va="bottom",
                ha="right",
                fontsize=7,
                color="#d62728",
            )

    legend_lines = [
        plt.Line2D([0], [0], color="#1f77b4", alpha=0.35, linewidth=2, label="eval measurement step"),
        plt.Line2D([0], [0], color="#ff7f0e", alpha=0.35, linewidth=2, label="drift check step"),
        plt.Line2D(
            [0],
            [0],
            color="#d62728",
            alpha=0.85,
            linewidth=1.4,
            linestyle="--",
            label="stream reload (inferred)",
        ),
    ]
    fig.legend(handles=legend_lines, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995))

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved {output_path}")
    print(f"  eval steps: {len(eval_steps)}, drift checks: {len(drift_steps)}")
    print(f"  stream boundaries (inferred): {[s for s, _ in stream_bounds]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("output/matey_inference_drift.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/matey_inference_drift_dashboard.png"),
    )
    parser.add_argument(
        "--batches-per-stream",
        type=int,
        default=None,
        help="Val batches per stream pass (~epoch_size from checkpoint).",
    )
    parser.add_argument(
        "--num-streams",
        type=int,
        default=None,
        help="Optional stream count; overrides batches-per-stream if set.",
    )
    parser.add_argument(
        "--metric-index",
        type=int,
        default=None,
        help="ADWIN metric_index used in the run (e.g. 3 for nrmse_mean / drift/metric_3).",
    )
    args = parser.parse_args()
    plot_dashboard(
        args.csv,
        args.output,
        batches_per_stream=args.batches_per_stream,
        num_streams=args.num_streams,
        metric_index=args.metric_index,
    )


if __name__ == "__main__":
    main()
