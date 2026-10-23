#!/usr/bin/env python3
"""Summarize repeated reconstruction draws: per layer, the spread.

Reads the JSONL a `variance.sh` run writes and reports, per layer, how many
draws rebuilt at all and what range of kept-ratios they landed in. The point
of the exercise is the SPREAD, not the mean: a single draw reported as a
scalar cannot be compared with a later single draw, so this is what turns
"51%" into a statement someone can check.

Two reporting rules, both load-bearing:

  * A `quota` draw (the producer refused — usage limit, rate limit) is not a
    spec failure and is excluded from every statistic. Counting it as a zero
    would manufacture variance that is really rate limiting.
  * A layer whose draws used more than one model is reported as such rather
    than pooled, because then the spread mixes sampling noise with a change
    of producer.

    python3 claims/scripts/variance_report.py claims/measurements/*.jsonl

Several files pool into one table, which is how an interrupted run is
continued: each draw is one line, written as it completes, so a run that
dies mid-layer loses nothing already measured.
"""

from __future__ import annotations

import json
import statistics
import sys


def load(path: str) -> list[dict]:
    rows = []
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
    rows = [row for path in argv[1:] for row in load(path)]
    if not rows:
        print(f"no draws in {', '.join(argv[1:])}", file=sys.stderr)
        return 1

    layers: dict[str, list[dict]] = {}
    for row in rows:
        layers.setdefault(row.get("claim", "?"), []).append(row)

    width = max(len(name) for name in layers)
    print(f"{'layer':<{width}}  {'draws':>5} {'built':>5} {'quota':>5} "
          f"{'kept min':>9} {'kept max':>9} {'spread':>7} {'median':>7}")
    print("-" * (width + 56))

    pooled_min, pooled_max, notes = [], [], []
    for name, draws in layers.items():
        counted = [d for d in draws if d.get("status") != "quota"]
        built = [d for d in counted if d.get("status") == "rebuilt"
                 and isinstance(d.get("kept"), (int, float))]
        quota = len(draws) - len(counted)
        ratios = sorted(d["kept"] for d in built)
        if ratios:
            lo, hi = ratios[0], ratios[-1]
            mid = statistics.median(ratios)
            pooled_min.append(lo)
            pooled_max.append(hi)
            print(f"{name:<{width}}  {len(counted):>5} {len(built):>5} "
                  f"{quota:>5} {lo:>8.0%} {hi:>8.0%} "
                  f"{hi - lo:>6.0%} {mid:>6.0%}")
        else:
            print(f"{name:<{width}}  {len(counted):>5} {len(built):>5} "
                  f"{quota:>5} {'-':>9} {'-':>9} {'-':>7} {'-':>7}")
        models = {d.get("model") for d in built}
        if len(models) > 1:
            notes.append(f"{name}: draws used more than one model "
                         f"({', '.join(sorted(str(m) for m in models))}) -- "
                         f"the spread mixes sampling with a producer change")
        unrecorded = [d for d in built if not d.get("model")]
        if unrecorded:
            notes.append(f"{name}: {len(unrecorded)} draw(s) recorded no "
                         f"model; they cannot be attributed to a producer")
        failed = [d for d in counted if d.get("status") == "failed"]
        if failed:
            notes.append(f"{name}: {len(failed)} draw(s) failed to rebuild "
                         f"at all -- a kept-ratio over survivors only is "
                         f"conditioned on success, and says so")

    print("-" * (width + 56))
    if pooled_min:
        print(f"{'per-layer worst / best':<{width}}  "
              f"{'':>5} {'':>5} {'':>5} "
              f"{min(pooled_min):>8.0%} {max(pooled_max):>8.0%}")

    # The headline figure, weighted by code volume rather than averaged over
    # layers -- a 40-line layer and a 600-line one are not one vote each.
    # The bounds take every layer at its worst and then at its best
    # simultaneously, which is deliberately conservative: independent draws
    # do not all land on their extreme at once, so a true interval is
    # narrower than this. It is the honest envelope, not an error bar.
    src_total = lo_total = mid_total = hi_total = 0
    mixed_src = []
    for name, draws in layers.items():
        built = [d for d in draws if d.get("status") == "rebuilt"
                 and isinstance(d.get("regrown_lines"), int)
                 and isinstance(d.get("src_lines"), int)]
        if not built:
            continue
        sizes = {d["src_lines"] for d in built}
        if len(sizes) > 1:
            # the layer's own code changed between draws, so its draws are
            # not of one claim and must not be pooled
            mixed_src.append(name)
            continue
        grown = sorted(d["regrown_lines"] for d in built)
        src_total += built[0]["src_lines"]
        lo_total += grown[0]
        hi_total += grown[-1]
        mid_total += statistics.median(grown)
    if src_total:
        print()
        print(f"aggregate over {len(layers) - len(mixed_src)} layer(s), "
              f"weighted by logical lines ({src_total} in src):")
        print(f"  kept {mid_total / src_total:.0%}  "
              f"envelope [{lo_total / src_total:.0%}, "
              f"{hi_total / src_total:.0%}]  "
              f"(every layer at its worst, then at its best)")
    for name in mixed_src:
        print(f"\nnote: {name}: draws measured different amounts of src -- "
              f"the criteria moved mid-run, so these draws are not of one "
              f"claim and are left out of the aggregate")
    for note in notes:
        print(f"\nnote: {note}")
    if not notes:
        print("\nnote: every counted draw used one recorded model and rebuilt.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
