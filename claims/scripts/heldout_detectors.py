#!/usr/bin/env python3
"""Differential probe: a domesticated detector against river, OFF the pinned set.

A claim's verdict is evidence bounded by what its criteria exercise. For a
claim whose purpose is to make a dependency replaceable, that bound is the
whole question -- an implementation can satisfy every pinned stream and still
be a different algorithm, agreeing only in the regimes the criteria happen to
visit. This script is what separates the two: it runs the reconstruction and
river over streams NO claim pins and reports where they part.

It earned its place. `apeiron-page-hinkley` sealed on six streams with three
independent redos all agreeing with river on every one. Run here, it fired at
342 on a slow sine where river fires at 340 -- because Page-Hinkley has two
inequivalent spellings (clamp the accumulator at zero, or measure its
distance from its own running minimum) that agree on every stream rising
from a floor and disagree on every stream that dips and recovers. All six
pinned streams were the former. See claims/LOG.md A37.

    python3 claims/scripts/heldout_detectors.py <detector> <claim-dir>

`<detector>` is one of page_hinkley, adwin, kswin. The river side needs an
interpreter with river installed; the reconstruction side must not have it,
which is the point -- pass RIVER_PYTHON to name the former. Exit 1 if any
held-out stream disagrees, so this can gate a reseal.

KSWIN is compared differently and necessarily so: it draws its reference
window at random, so river's own first-fire index moves with the seed. The
comparison is over the DISTRIBUTION across 40 seeds -- detection counts and
the first-fire envelope -- which is the strongest statement the algorithm
supports. See the criteria file in the kswin claim.
"""
import json
import math
import os
import random
import subprocess
import sys

RIVER_PYTHON = os.environ.get("RIVER_PYTHON", sys.executable)
LOCAL_PYTHON = os.environ.get("LOCAL_PYTHON", sys.executable)
SEEDS = 40

PARAMS = {
    "page_hinkley": {"min_instances": 30, "delta": 0.005, "threshold": 50.0,
                     "alpha": 0.9999},
    "adwin": {"delta": 0.002},
    "kswin": {"alpha": 0.005, "window_size": 100, "stat_size": 30},
}
CLASSES = {"page_hinkley": "PageHinkley", "adwin": "ADWIN", "kswin": "KSWIN"}
MODULES = {"page_hinkley": "page_hinkley", "adwin": "adwin", "kswin": "kswin"}


def streams() -> dict:
    """Streams deliberately unlike the pinned ones, in the regimes that
    discriminate: slow oscillation, excursion-and-return, repeated change,
    noise, and outliers. Deterministic -- a probe that moves between runs
    cannot be cited.
    """
    rng = random.Random(4242)
    return {
        # Slow oscillation: the statistic goes well negative and recovers.
        # This is the family that caught the Page-Hinkley spelling, and the
        # three below are CONFIRMED discriminators -- each separates the
        # clamped form from river's running-minimum form, verified by running
        # both against river (A37). Do not drop them without re-establishing
        # that whatever replaces them discriminates something; the first
        # version of this probe used plausible-looking slow streams instead
        # of measured ones and passed the very implementation it was written
        # to catch.
        "sine_p60_a30":  [0.5 + 0.30 * math.sin(i / 60) for i in range(400)],
        "sine_p45_a40":  [0.5 + 0.40 * math.sin(i / 45) for i in range(400)],
        "sine_p90_a20":  [0.5 + 0.20 * math.sin(i / 90) for i in range(600)],
        # a second firing after a reset, in the same slow regime
        "sine_p75_a40":  [0.5 + 0.40 * math.sin(i / 75) for i in range(600)],
        "decay_osc":    [0.5 + 0.4 * math.exp(-i / 200) * math.sin(i / 30)
                         for i in range(500)],
        "slow_v":       [0.7 - 0.004 * i for i in range(120)]
                        + [0.22 + 0.004 * i for i in range(120)],
        # excursion and return: a change that undoes itself
        "double_dip":   [0.6] * 80 + [0.15] * 80 + [0.6] * 80 + [0.15] * 80
                        + [0.6] * 80,
        "plateau":      [0.1] * 60 + [0.5] * 60 + [0.9] * 60 + [0.5] * 60
                        + [0.1] * 60,
        # repeated change: exercises the reset path, and whether a second
        # change is still found after the first
        "staircase":    [0.2 + 0.1 * (i // 70) for i in range(490)],
        "down_stair":   [0.9 - 0.1 * (i // 70) for i in range(490)],
        "two_changes":  [0.2] * 150 + [0.7] * 150 + [0.35] * 150,
        "sawtooth":     [0.3 + 0.25 * ((i // 40) % 2) for i in range(320)],
        # noise: a detector that fires on variance alone shows up here
        "noisy_flat":   [0.5 + rng.uniform(-0.2, 0.2) for _ in range(400)],
        "noisy_drift":  [0.3 + 0.001 * i + rng.uniform(-0.1, 0.1)
                         for i in range(400)],
        # magnitude extremes: a threshold scaled wrongly shows up here
        "micro_step":   [0.500] * 250 + [0.515] * 250,
        "big_step":     [0.0] * 120 + [5.0] * 120,
        "outlier_pair": [0.4] * 100 + [-8.0] + [0.4] * 50 + [8.0] + [0.4] * 100,
        "alternating":  [0.1 if i % 2 else 0.9 for i in range(200)],
    }


_RUNNER = r'''
import json, sys
spec = json.loads(sys.stdin.read())
name, params, data, seeds = (spec["name"], spec["params"], spec["streams"],
                             spec["seeds"])
if spec["side"] == "river":
    from river import drift
    factory = {"page_hinkley": lambda **k: drift.PageHinkley(**k),
               "adwin": lambda **k: drift.ADWIN(**k),
               "kswin": lambda **k: drift.KSWIN(**k)}[name]
else:
    sys.path.insert(0, spec["claim"])
    mod = __import__(spec["module"])
    factory = getattr(mod, spec["cls"])

def firings(values, **over):
    d = factory(**{**params, **over})
    out = []
    for i, x in enumerate(values):
        d.update(x)
        if d.drift_detected:
            out.append(i)
    return out

out = {}
for stream, values in data.items():
    if name == "kswin":
        counts, firsts = set(), []
        for seed in range(seeds):
            fired = firings(values, seed=seed)
            counts.add(len(fired))
            if fired:
                firsts.append(fired[0])
        out[stream] = {"counts": sorted(counts),
                       "first": [min(firsts), max(firsts)] if firsts else None}
    else:
        out[stream] = firings(values)
print(json.dumps(out))
'''


def _side(side: str, name: str, claim: str, data: dict) -> dict:
    spec = {"side": side, "name": name, "params": PARAMS[name], "claim": claim,
            "module": MODULES[name], "cls": CLASSES[name], "streams": data,
            "seeds": SEEDS}
    python = RIVER_PYTHON if side == "river" else LOCAL_PYTHON
    proc = subprocess.run([python, "-c", _RUNNER], input=json.dumps(spec),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"the {side} side failed:\n{proc.stderr.strip()}")
    return json.loads(proc.stdout)


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in PARAMS:
        sys.exit(f"usage: heldout_detectors.py "
                 f"<{'|'.join(PARAMS)}> <claim-dir>")
    name, claim = argv[0], os.path.abspath(argv[1])

    data = streams()
    oracle = _side("river", name, claim, data)
    mine = _side("mine", name, claim, data)

    disagree = []
    print(f"{name}: {len(data)} held-out streams, none of them pinned"
          + (f", {SEEDS} seeds each" if name == "kswin" else ""))
    for stream in data:
        want, got = oracle[stream], mine[stream]
        same = want == got
        if not same:
            disagree.append(stream)
        print(f"  {'ok  ' if same else 'DIFF'} {stream:<14} "
              f"river {str(want)[:40]:<40} this {str(got)[:40]}")

    if disagree:
        print(f"\n{len(disagree)} of {len(data)} held-out streams disagree: "
              f"{disagree}")
        print("The criteria admit an implementation that is not equivalent to "
              "river in these regimes. Either pin a stream from each, or say "
              "in the claim that the equivalence is bounded and where.")
        return 1
    print(f"\nall {len(data)} agree — the criteria carry the behaviour past "
          f"the streams they pin")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
