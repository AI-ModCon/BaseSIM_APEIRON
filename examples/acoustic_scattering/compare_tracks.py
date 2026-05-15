#!/usr/bin/env python
"""Compare Track 1 (Offline) vs Track 2 (Apeiron) results.

Reads ``results.json`` from both output directories and prints a
comparison table.  Optionally saves a bar chart (requires matplotlib).

Usage::

    poetry run python examples/acoustic_scattering/compare_tracks.py \
        [--offline output/acoustic_scattering_offline] \
        [--cl output/acoustic_scattering_cl] \
        [--plot comparison.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_flops(f: float) -> str:
    if f >= 1e15:
        return f"{f / 1e15:.2f}e15"
    if f >= 1e12:
        return f"{f / 1e12:.2f}e12"
    if f >= 1e9:
        return f"{f / 1e9:.2f}e9"
    if f >= 1e6:
        return f"{f / 1e6:.2f}e6"
    return f"{f:.0f}"


def load_results(path: Path) -> dict:
    results_file = path / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"No results.json found at {results_file}")
    return json.loads(results_file.read_text())


def print_table(offline: dict, cl: dict) -> None:
    w = 60
    print("=" * w)
    print(f"{'Metric':<25} {'Track 1 (Offline)':>16} {'Track 2 (Apeiron)':>16}")
    print("-" * w)

    # Test metrics
    all_keys = set(offline.get("test_metrics", {})) | set(cl.get("test_metrics", {}))
    for key in sorted(all_keys):
        v1 = offline.get("test_metrics", {}).get(key)
        v2 = cl.get("test_metrics", {}).get(key)
        s1 = f"{v1:.4f}" if v1 is not None else "\u2014"
        s2 = f"{v2:.4f}" if v2 is not None else "\u2014"
        print(f"  Test {key:<19} {s1:>16} {s2:>16}")

    print("-" * w)

    # FLOPs breakdown
    f1 = offline.get("flops", {})
    f2 = cl.get("flops", {})
    for tag in ("train", "scoring", "inference", "total"):
        v1 = f1.get(tag, 0.0)
        v2 = f2.get(tag, 0.0)
        s1 = _fmt_flops(v1) if v1 else "\u2014"
        s2 = _fmt_flops(v2) if v2 else "\u2014"
        label = f"{tag.capitalize()} FLOPs"
        print(f"  {label:<23} {s1:>16} {s2:>16}")

    print("-" * w)

    # Data budget
    b1 = offline.get("data_budget", "\u2014")
    b2 = cl.get("data_budget", "\u2014")
    print(f"  {'Data Budget':<23} {str(b1):>16} {str(b2):>16}")

    # CL-specific stats
    drift = cl.get("drift_events", "\u2014")
    updates = cl.get("stream_updates", "\u2014")
    epochs = offline.get("epochs", "\u2014")
    print(f"  {'Epochs (offline)':<23} {str(epochs):>16} {'\u2014':>16}")
    print(f"  {'Drift Events':<23} {'\u2014':>16} {str(drift):>16}")
    print(f"  {'Stream Updates':<23} {'\u2014':>16} {str(updates):>16}")

    print("=" * w)


def plot_comparison(offline: dict, cl: dict, save_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: Test metrics
    ax = axes[0]
    keys = sorted(
        set(offline.get("test_metrics", {})) | set(cl.get("test_metrics", {}))
    )
    x_pos = range(len(keys))
    v1 = [offline.get("test_metrics", {}).get(k, 0) for k in keys]
    v2 = [cl.get("test_metrics", {}).get(k, 0) for k in keys]
    bar_w = 0.35
    ax.bar([p - bar_w / 2 for p in x_pos], v1, bar_w, label="Offline")
    ax.bar([p + bar_w / 2 for p in x_pos], v2, bar_w, label="Apeiron")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(keys)
    ax.set_ylabel("Value")
    ax.set_title("Test Metrics")
    ax.legend()

    # Panel 2: FLOPs
    ax = axes[1]
    tags = ["train", "scoring", "inference", "total"]
    f1 = [offline.get("flops", {}).get(t, 0) for t in tags]
    f2 = [cl.get("flops", {}).get(t, 0) for t in tags]
    x_pos = range(len(tags))
    ax.bar([p - bar_w / 2 for p in x_pos], f1, bar_w, label="Offline")
    ax.bar([p + bar_w / 2 for p in x_pos], f2, bar_w, label="Apeiron")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(tags)
    ax.set_ylabel("FLOPs")
    ax.set_title("Computational Cost")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare offline vs Apeiron results")
    parser.add_argument(
        "--offline",
        type=Path,
        default=Path("output/acoustic_scattering_offline"),
        help="Path to offline results directory",
    )
    parser.add_argument(
        "--cl",
        type=Path,
        default=Path("output/acoustic_scattering_cl"),
        help="Path to Apeiron CL results directory",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Save comparison bar chart to this path (requires matplotlib)",
    )
    args = parser.parse_args()

    offline = load_results(args.offline)
    cl = load_results(args.cl)
    print_table(offline, cl)

    if args.plot:
        plot_comparison(offline, cl, args.plot)


if __name__ == "__main__":
    main()
