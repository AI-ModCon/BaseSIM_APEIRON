"""Summarize several Apeiron runs side by side.

    python tutorials/a6/scripts/compare_runs.py output/a6/lab3_*.csv
    python tutorials/a6/scripts/compare_runs.py output/a6/lab2_*.csv --markdown

Columns
  checks        drift checks performed
  detections    times the detector fired (drift/detected == 1)
  adaptations   continual-learning updates that ran (one per detection in src.main runs)
  watched       which metric the detector monitored (drift/metric_<metric_index>)
  stream_mean   mean of the monitored metric over the whole stream  <- the headline number
  final_mean    mean of the monitored metric over the final 20% of drift checks
  hist_acc      mean score on replayed past windows after each update (retention)
  fwt           mean gain on the current window from each update (post - pre)
  bwt           backward transfer at the last update (negative accuracy = forgetting)
  seconds       wall-clock time, if the sweep scripts recorded it in <run>.seconds
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def summarize(csv: Path) -> dict:
    df = pd.read_csv(csv)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    by = {m: g.sort_values("step")["value"] for m, g in df.groupby("metric")}

    watched = sorted(m for m in by if re.fullmatch(r"drift/metric_\d+", m))
    row: dict = {"run": csv.stem}
    if watched:
        w = by[watched[0]]
        tail = w.iloc[int(len(w) * 0.8) :] if len(w) >= 5 else w
        row["watched"] = watched[0].split("/")[-1]
        row["checks"] = len(w)
        row["stream_mean"] = round(w.mean(), 2)
        row["final_mean"] = round(tail.mean(), 2)
    row["detections"] = int(
        (by.get("drift/detected", pd.Series(dtype=float)) >= 1).sum()
    )
    row["adaptations"] = len(by.get("eval/fwt", []))
    if "eval/test_hist_acc" in by:
        row["hist_acc"] = round(by["eval/test_hist_acc"].mean(), 2)
    if "eval/fwt" in by:
        row["fwt"] = round(by["eval/fwt"].mean(), 2)
    if "eval/bwt" in by:
        row["bwt"] = round(by["eval/bwt"].iloc[-1], 2)
    seconds = csv.with_suffix(".seconds")
    if seconds.exists():
        row["seconds"] = int(float(seconds.read_text().strip() or 0))
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("csv", nargs="+", type=Path)
    p.add_argument("--markdown", action="store_true", help="print a Markdown table")
    a = p.parse_args()
    table = pd.DataFrame([summarize(c) for c in a.csv]).set_index("run")
    order = [
        c
        for c in (
            "watched",
            "checks",
            "detections",
            "adaptations",
            "stream_mean",
            "final_mean",
            "hist_acc",
            "fwt",
            "bwt",
            "seconds",
        )
        if c in table
    ]
    table = table[order]
    if a.markdown:
        try:
            print(table.to_markdown())
        except ImportError:  # pandas needs the optional `tabulate` package for Markdown
            print(table.to_string(na_rep="-"))
    else:
        with pd.option_context("display.width", 160, "display.max_columns", 20):
            print(table.to_string(na_rep="-"))


if __name__ == "__main__":
    main()
