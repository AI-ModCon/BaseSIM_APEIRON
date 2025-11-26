#!/usr/bin/env python3
"""
Drift Metric Visualization Script

Generates comprehensive drift detection dashboards from continual learning metrics.
Reads configuration from TOML file and creates visualizations for:
- Test and historical accuracy
- Loss metrics
- Computational performance (FLOPs)
- Execution time analysis

Configuration in TOML file ([visualization] section):
    baseline: Accuracy threshold for drift detection (default: 95.0)
    input: Path to CSV file with metrics (default: "output/cl_only.csv")
    output: Path to save dashboard image (default: "output/drift_dashboard.png")

Usage:
    # Run from BaseSim_Framework directory
    python src/visualization/drift_metric_viz.py --config examples/mnist/mnist.toml
    # Override baseline threshold
    python src/visualization/drift_metric_viz.py --config examples/mnist/mnist.toml --set visualization.baseline=85
    # Override input file
    python src/visualization/drift_metric_viz.py --config examples/mnist/mnist.toml --set visualization.input=output/custom.csv
"""

import sys
from pathlib import Path

# Add project root to Python path to enable imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import pandas as pd

from src.config.configuration import build_config, Config


def load_data(csv_path):
    """Load metrics data from CSV file.
    
    Args:
        csv_path: Path to the CSV file containing metrics data
        
    Returns:
        DataFrame with metrics data
    """
    df = pd.read_csv(csv_path)
    return df


def create_pivot_table(df):
    """Create pivot table from metrics dataframe.
    
    Args:
        df: DataFrame with metrics in long format (step, metric, value)
        
    Returns:
        Pivoted DataFrame with metrics as columns
    """
    pivot_df = df.pivot(index="step", columns="metric", values="value")
    return pivot_df


def create_drift_dashboard(pivot_df, show_plot=True, save_path=None, baseline=95.0):
    """
    Create comprehensive drift detection dashboard.

    Args:
        pivot_df: Pivoted dataframe with metrics
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure
        baseline: Baseline accuracy threshold for drift detection (default: 95.0)
    """
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle("Drift Detection Dashboard", fontsize=16, fontweight="bold")

    # ----------------------------------------------------------
    # 1. Test Accuracy
    # ----------------------------------------------------------
    ax0 = axes[0, 0]
    if "test/acc" in pivot_df.columns:
        ax0.plot(
            pivot_df.index,
            pivot_df["test/acc"],
            marker="o",
            linewidth=2,
            color="#2E86AB",
            label="Test Accuracy",
        )

        # Add baseline/threshold
        ax0.axhline(
            y=baseline,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Baseline ({baseline}%)",
        )

        # Add drift zone
        ax0.fill_between(
            pivot_df.index,
            baseline - 2,
            baseline,
            alpha=0.2,
            color="orange",
            label="Warning Zone",
        )
        ax0.fill_between(
            pivot_df.index,
            0,
            baseline - 2,
            alpha=0.2,
            color="red",
            label="Critical Zone",
        )

    ax0.set_xlabel("Step")
    ax0.set_ylabel("Accuracy (%)")
    ax0.set_title("Test Accuracy")
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.3)

    # ----------------------------------------------------------
    # 2. Hist Test Accuracy
    # ----------------------------------------------------------
    ax1 = axes[0, 1]
    if "hist_test/acc" in pivot_df.columns:
        ax1.plot(
            pivot_df.index,
            pivot_df["hist_test/acc"],
            marker="o",
            linewidth=2,
            color="#2E86AB",
            label="Hist Test Accuracy",
        )

        # Add baseline/threshold
        ax1.axhline(
            y=baseline,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Baseline ({baseline}%)",
        )

        # Add drift zone
        ax1.fill_between(
            pivot_df.index,
            baseline - 2,
            baseline,
            alpha=0.2,
            color="orange",
            label="Warning Zone",
        )
        ax1.fill_between(
            pivot_df.index,
            0,
            baseline - 2,
            alpha=0.2,
            color="red",
            label="Critical Zone",
        )

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Hist Test Accuracy")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # ----------------------------------------------------------
    # 3. Loss Drift Detection
    # ----------------------------------------------------------
    ax2 = axes[1, 0]
    loss_cols = [col for col in pivot_df.columns if "loss" in col]
    colors = ["#d7191c", "#fdae61", "#abd9e9", "#2c7bb6"]

    for i, col in enumerate(loss_cols):
        if col in pivot_df.columns:
            data_clean = pivot_df[col].dropna()
            ax2.plot(
                data_clean.index,
                data_clean.values,
                marker="o",
                linewidth=2,
                color=colors[i % len(colors)],
                label=col,
                alpha=0.6,
            )

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss Metrics")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ----------------------------------------------------------
    # 4. Computational Performance: FLOP
    # ----------------------------------------------------------
    ax3 = axes[1, 1]
    flops_cols = [
        col
        for col in pivot_df.columns
        if "flop" in col.lower() and "flops" not in col.lower()
    ]
    colors_flops = ["#7b3294", "#c2a5cf", "#a6dba0", "#008837"]
    for i, col in enumerate(flops_cols):
        if col in pivot_df.columns:
            data_clean = pivot_df[col].dropna()
            ax3.plot(
                data_clean.index,
                data_clean.values / 1e9,  # Convert to GFLOPS
                marker="o",
                linewidth=2.5,
                markersize=8,
                color=colors_flops[i % len(colors_flops)],
                label=col,
                alpha=0.8,
            )
    ax3.set_xlim(left=0, right=None)
    ax3.set_xlabel("Step", fontsize=10)
    ax3.set_ylabel("GFLOP (×10⁹)", fontsize=10)
    ax3.set_title("Computational Performance: FLOP", fontsize=12)
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ----------------------------------------------------------
    # 5. Computational Performance: FLOPS
    # ----------------------------------------------------------
    ax4 = axes[2, 0]
    flops_cols_perf = [col for col in pivot_df.columns if "flops" in col.lower()]
    colors_flops = ["#7b3294", "#c2a5cf", "#a6dba0", "#008837"]
    for i, col in enumerate(flops_cols_perf):
        if col in pivot_df.columns:
            data_clean = pivot_df[col].dropna()
            ax4.plot(
                data_clean.index,
                data_clean.values / 1e9,  # Convert to GFLOPS
                marker="o",
                linewidth=2.5,
                markersize=8,
                color=colors_flops[i % len(colors_flops)],
                label=col,
                alpha=0.8,
            )
    ax4.set_xlim(left=0, right=None)
    ax4.set_xlabel("Step", fontsize=10)
    ax4.set_ylabel("GFLOPS (×10⁹)", fontsize=10)
    ax4.set_title("Computational Performance: FLOPS", fontsize=12)
    ax4.legend(loc="best", fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ----------------------------------------------------------
    # 6. Execution Time
    # ----------------------------------------------------------
    ax5 = axes[2, 1]

    # Get timing columns
    time_cols = [col for col in pivot_df.columns if "time" in col.lower()]
    colors_time = ["#80cdc1", "#018571", "#a6611a", "#dfc27d"]

    # Plot each timing metric
    for i, col in enumerate(time_cols):
        if col in pivot_df.columns:
            data_clean = pivot_df[col].dropna()
            ax5.plot(
                data_clean.index,
                data_clean.values * 1000,  # Convert to ms
                marker="o",
                linewidth=2.5,
                markersize=8,
                color=colors_time[i % len(colors_time)],
                label=col,
                alpha=0.8,
            )
    ax5.set_xlim(left=0, right=None)
    ax5.set_xlabel("Step", fontsize=12)
    ax5.set_ylabel("Time (ms)", fontsize=12)
    ax5.set_title("Execution Time by Step", fontsize=14)
    ax5.legend(loc="best", fontsize=10)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Dashboard saved to: {save_path}")

    if show_plot:
        plt.show()

    return fig, axes


def print_timing_analysis(pivot_df):
    """Print execution time statistics.
    
    Args:
        pivot_df: Pivoted dataframe with metrics
    """
    time_cols = [col for col in pivot_df.columns if "time" in col.lower()]

    print("\n=== Execution Time Analysis ===")
    for col in time_cols:
        if col in pivot_df.columns:
            data_clean = pivot_df[col].dropna()
            if len(data_clean) > 0:
                print(f"\n{col}:")
                print(f"  Mean: {data_clean.mean() * 1000:.3f} ms")
                print(f"  Min:  {data_clean.min() * 1000:.3f} ms")
                print(f"  Max:  {data_clean.max() * 1000:.3f} ms")


def print_drift_summary(pivot_df, baseline=95.0):
    """Print drift detection summary.

    Args:
        pivot_df: Pivoted dataframe with metrics
        baseline: Baseline accuracy threshold (default: 95.0)
    """
    loss_cols = [col for col in pivot_df.columns if "loss" in col]

    print("\n=== Drift Detection Summary ===")
    print(f"Steps analyzed: {pivot_df.index.min()} - {pivot_df.index.max()}")

    if "test/acc" in pivot_df.columns:
        acc_vals = pivot_df["test/acc"].dropna()
        if len(acc_vals) > 0:
            print(f"\nAccuracy range: {acc_vals.min():.2f}% - {acc_vals.max():.2f}%")
            if acc_vals.min() < baseline:
                print(f"WARNING: Accuracy below baseline ({baseline}%) detected!")

    if len(loss_cols) > 0:
        print(f"\nLoss metrics tracked: {len(loss_cols)}")
        for col in loss_cols:
            if col in pivot_df.columns:
                loss_vals = pivot_df[col].dropna()
                if len(loss_vals) > 0:
                    print(f"  {col}: {loss_vals.min():.3f} - {loss_vals.max():.3f}")


def print_flops_analysis(pivot_df):
    """Print FLOPS performance statistics.
    
    Args:
        pivot_df: Pivoted dataframe with metrics
    """
    flops_cols = [col for col in pivot_df.columns if "flops" in col.lower()]

    print("\n=== FLOPS Analysis ===")
    for col in flops_cols:
        if col in pivot_df.columns:
            data_clean = pivot_df[col].dropna()
            if len(data_clean) > 0:
                print(f"\n{col}:")
                print(f"  Mean: {data_clean.mean() / 1e9:.2f} GFLOPS")
                print(f"  Min:  {data_clean.min() / 1e9:.2f} GFLOPS")
                print(f"  Max:  {data_clean.max() / 1e9:.2f} GFLOPS")


def print_unique_metrics(pivot_df):
    """Print information about available metrics.
    
    Args:
        pivot_df: Pivoted dataframe with metrics
    """
    print("\n=== Available Metrics ===")
    print(f"Total unique metrics: {len(pivot_df.columns)}")
    print("\nMetric categories:")

    categories = {
        "Accuracy": [col for col in pivot_df.columns if "acc" in col],
        "Loss": [col for col in pivot_df.columns if "loss" in col],
        "FLOP": [
            col
            for col in pivot_df.columns
            if "flop" in col.lower() and "flops" not in col.lower()
        ],
        "FLOPS": [col for col in pivot_df.columns if "flops" in col.lower()],
        "Time": [col for col in pivot_df.columns if "time" in col.lower()],
    }

    for category, metrics in categories.items():
        if metrics:
            print(f"\n{category} metrics ({len(metrics)}):")
            for metric in metrics:
                print(f"  - {metric}")


def get_visualization_config(cfg):
    """Extract visualization configuration with defaults.
    
    Args:
        cfg: Config object from build_config
        
    Returns:
        tuple: (baseline, csv_path, output_path) with default values if not specified
        
    Raises:
        ValueError: If [visualization] section is missing from config
    """
    if cfg.visualization is not None:
        baseline = cfg.visualization.baseline
        csv_path = cfg.visualization.input
        output_path = cfg.visualization.output
        print(f"Using baseline from config: {baseline}%")
        print(f"Using input file from config: {csv_path}")
        print(f"Using output path from config: {output_path}")
    else:
        raise ValueError(
            "Missing [visualization] section in config file. "
            "Please add the following to your TOML config:\n"
            "[visualization]\n"
            "baseline = 95.0\n"
            'input = "output/cl_only.csv"\n'
            'output = "output/drift_dashboard.png"'
        )

    return baseline, csv_path, output_path


def main(argv=None) -> int:
    """Main execution function.
    
    Args:
        argv: Command-line arguments (for testing)
        
    Returns:
        int: Exit code (0 for success)
    """
    # Load configuration from TOML file
    cfg: Config = build_config(argv)
    baseline, csv_path, output_path = get_visualization_config(cfg)

    # Load and process data
    print(f"\nLoading metrics data from {csv_path}...")
    df = load_data(csv_path)
    print(f"Loaded {len(df)} rows of data")
    print(f"Unique metrics: {df['metric'].nunique()}")

    # Create pivot table
    pivot_df = create_pivot_table(df)

    # Print available metrics
    print_unique_metrics(pivot_df)

    # Create dashboard
    print(f"\nGenerating drift detection dashboard...")
    print(f"  Dataset: {cfg.data.name}")
    print(f"  Baseline: {baseline}%")
    create_drift_dashboard(
        pivot_df, show_plot=True, save_path=output_path, baseline=baseline
    )

    # Print detailed analyses
    print_timing_analysis(pivot_df)
    print_drift_summary(pivot_df, baseline=baseline)
    print_flops_analysis(pivot_df)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
