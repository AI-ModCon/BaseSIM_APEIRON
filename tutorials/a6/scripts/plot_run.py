"""Plot one or more Apeiron runs: the monitored metric over the stream, drift detections, adaptations.

    python tutorials/a6/scripts/plot_run.py output/a6/lab1.csv
    python tutorials/a6/scripts/plot_run.py output/a6/lab3_*.csv --out output/a6/lab3.png

Reads the long-format metrics CSV that Apeiron writes to [logging] metrics_output_path
(columns: step, metric, value). For each run it draws:
  * the metric the detector watches (drift/metric_<i>, one point per drift check),
  * a smoothed per-batch stream accuracy/score (eval/<name>) when available,
  * a dashed vertical line wherever the detector fired (drift/detected == 1).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def series(df: pd.DataFrame, metric: str) -> pd.Series:
    s = df[df.metric == metric].set_index("step")["value"]
    return s[~s.index.duplicated(keep="last")].sort_index()


def monitored_metric(df: pd.DataFrame) -> str | None:
    names = sorted(
        m for m in df.metric.unique() if re.fullmatch(r"drift/metric_\d+", m)
    )
    return names[0] if names else None


def plot(csvs: list[Path], out: Path, smooth: int) -> None:
    fig, axes = plt.subplots(
        len(csvs), 1, figsize=(11, 2.6 * len(csvs) + 0.6), sharex=False, squeeze=False
    )
    for ax, csv in zip(axes[:, 0], csvs):
        df = load(csv)
        mm = monitored_metric(df)
        if mm is None:
            ax.text(
                0.5,
                0.5,
                f"{csv.name}: no drift/metric_* rows",
                ha="center",
                transform=ax.transAxes,
            )
            continue
        watched = series(df, mm)
        ax.plot(
            watched.index,
            watched.values,
            color="#303585",
            lw=1.4,
            label=f"{mm} (what the detector sees)",
        )

        # Per-batch stream score, smoothed, for context (eval/accuracy for MNIST).
        stream_names = [
            m for m in ("eval/accuracy", "eval/mae", "eval/loss") if m in set(df.metric)
        ]
        if stream_names:
            s = series(df, stream_names[0])
            if len(s) > smooth:
                ax.plot(
                    s.index,
                    s.rolling(smooth, min_periods=1).mean().values,
                    color="#9AA3AB",
                    lw=1,
                    alpha=0.9,
                    label=f"{stream_names[0]} (rolling mean)",
                )

        detected = series(df, "drift/detected")
        fired = detected[detected >= 1].index
        for i, step in enumerate(fired):
            ax.axvline(
                step,
                color="#D6243A",
                ls="--",
                lw=1.1,
                label="drift detected" if i == 0 else None,
            )

        n_cl = int((df.metric == "eval/fwt").sum())
        mean_watched = watched.mean()
        ax.set_title(
            f"{csv.stem}:  {len(fired)} detections, {n_cl} adaptations, mean {mm} = {mean_watched:.2f}",
            fontsize=10,
            loc="left",
        )
        ax.set_ylabel(mm.split("/")[-1])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    axes[-1, 0].set_xlabel("logging step (stream time)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("csv", nargs="+", type=Path)
    p.add_argument("--out", type=Path, help="PNG path (default: next to the first CSV)")
    p.add_argument(
        "--smooth", type=int, default=50, help="rolling window for per-batch scores"
    )
    a = p.parse_args()
    out = a.out or a.csv[0].with_suffix(".png")
    plot(a.csv, out, a.smooth)


if __name__ == "__main__":
    main()
