#!/usr/bin/env python3
"""Compare two generations of one claim, and say whether the move is real.

Reads a `paired.sh` ledger and reports each side's kept-ratio spread, the
difference between their medians, and -- the only part that matters -- whether
that difference clears the noise floor the draws themselves establish.

The test is deliberately crude and stated in full rather than dressed up as
statistics on six samples: the move counts only if the two sides' observed
ranges do not overlap. With three draws a side that is a weak test, and it
is the honest one; a t-test on n=3 would read as more than it is.

    python3 claims/scripts/paired_report.py claims/measurements/paired-*.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys


def load(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    return rows


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip().split("    python3")[-1], file=sys.stderr)
        return 2
    rows = load(argv[1:])
    if not rows:
        print("no draws", file=sys.stderr)
        return 1

    sides = {}
    for row in rows:
        sides.setdefault(row.get("side", "?"), []).append(row)

    summary = {}
    print(f"{'side':<8} {'draws':>5} {'built':>5} {'quota':>5} "
          f"{'min':>6} {'median':>7} {'max':>6}  root")
    print("-" * 68)
    for side in ("before", "after"):
        draws = sides.get(side, [])
        if not draws:
            continue
        counted = [d for d in draws if d.get("status") != "quota"]
        built = [d for d in counted if d.get("status") == "rebuilt"
                 and isinstance(d.get("kept"), (int, float))]
        ratios = sorted(d["kept"] for d in built)
        root = (draws[0].get("claim_root") or "")[:12]
        if ratios:
            summary[side] = ratios
            print(f"{side:<8} {len(counted):>5} {len(built):>5} "
                  f"{len(draws) - len(counted):>5} "
                  f"{ratios[0]:>5.0%} {statistics.median(ratios):>6.0%} "
                  f"{ratios[-1]:>5.0%}  {root}")
        else:
            print(f"{side:<8} {len(counted):>5} {len(built):>5} "
                  f"{len(draws) - len(counted):>5} {'-':>6} {'-':>7} "
                  f"{'-':>6}  {root}")

    print()
    if len(summary) < 2:
        print("note: only one generation produced a usable draw, so there is "
              "nothing to compare yet.")
        return 0

    b, a = summary["before"], summary["after"]
    move = statistics.median(a) - statistics.median(b)
    overlap = not (min(a) > max(b) or min(b) > max(a))
    print(f"median moves {move:+.0%}  "
          f"(before {statistics.median(b):.0%} -> after {statistics.median(a):.0%})")
    print(f"ranges       before [{min(b):.0%}, {max(b):.0%}]  "
          f"after [{min(a):.0%}, {max(a):.0%}]")
    print()
    if overlap:
        print("VERDICT: the ranges overlap. On this evidence the criteria "
              "change did not move\n         the kept-ratio beyond the "
              "spread of drawing twice. Either the move is\n         smaller "
              "than this many draws can see, or it is not there.")
    else:
        print("VERDICT: the ranges are disjoint. Strengthening the criteria "
              "moved what an\n         independent reconstruction was forced "
              "to build, by more than\n         repeated drawing moves it.")

    models = {d.get("model") for d in rows if d.get("status") == "rebuilt"}
    if len(models) > 1:
        print(f"\nnote: draws used more than one model ({models}) -- the "
              f"comparison mixes a criteria change with a producer change.")
    if None in models:
        print("\nnote: at least one draw recorded no model, so it cannot be "
              "attributed to a producer.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
