"""Acceptance criteria for KSWIN: river's behaviour, pinned as far as it IS pinnable.

KSWIN is the one detector of the three that cannot be pinned the way
Page-Hinkley and ADWIN were, and the reason is in the algorithm rather than
in us. It draws its reference window at random. River's own firing index
moves with the seed -- measured over 40 seeds, the step stream fires
anywhere in [209, 213] -- so an exact index is not a property of KSWIN at
all; it is a property of KSWIN plus one particular draw. Asserting one index
would pin river's RNG consumption order, which is an implementation detail,
and would reject correct implementations for having a different random
stream.

What IS invariant over those 40 seeds is stronger than it looks:

  * the NUMBER of detections per stream never varies -- step fires once,
    ramp twice, constant and spike never, down_step once
  * on ramp, even the index is stable at 99
  * the first-fire index stays inside a narrow envelope

So the criteria pin counts exactly, pin the index where it is stable, and
bound it where it is not -- with bounds taken from the reference
implementation's own spread, not invented. And they test the WHOLE envelope
rather than a single seed: an implementation is admitted only if its
behaviour across many seeds matches river's behaviour across many seeds.
That is a weaker claim than ADWIN's, and it is the strongest claim this
algorithm supports.

Standard library only: river 0.23.0 ships no cp314 wheels (A8).
"""

import math
import sys

sys.path.insert(0, ".")
from kswin import KSWIN  # noqa: E402

PARAMS = {"alpha": 0.005, "window_size": 100, "stat_size": 30}
SEEDS = range(40)

#: Measured from river.drift.KSWIN 0.23.0 over SEEDS. `count` is exact for
#: every seed; `first` is the inclusive envelope the index fell within.
REFERENCE = {
    "step":      {"count": 1, "first": (209, 213)},
    "constant":  {"count": 0, "first": None},
    "ramp":      {"count": 2, "first": (99, 99)},
    "spike":     {"count": 0, "first": None},
    "down_step": {"count": 1, "first": (205, 213)},
}


def streams():
    wobble = lambda i: 0.03 * math.sin(i * 0.7) + 0.01 * math.cos(i * 2.3)
    return {
        "step": [(0.20 if i < 200 else 0.80) + wobble(i) for i in range(400)],
        "constant": [0.5] * 200,
        "ramp": [0.1 + 0.004 * i for i in range(250)],
        "spike": [0.3] * 100 + [9.0] + [0.3] * 99,
        "down_step": [(0.80 if i < 200 else 0.20) + wobble(i) for i in range(400)],
    }


def firings(values, seed):
    detector = KSWIN(seed=seed, **PARAMS)
    out = []
    for i, value in enumerate(values):
        detector.update(value)
        if detector.drift_detected:
            out.append(i)
    return out


def main():
    data = streams()

    for name, expect in REFERENCE.items():
        counts, firsts = set(), []
        for seed in SEEDS:
            fired = firings(data[name], seed)
            counts.add(len(fired))
            if fired:
                firsts.append(fired[0])

        # 1. the count is invariant -- the part of KSWIN that does not depend
        #    on which reference window it happened to draw
        assert counts == {expect["count"]}, (
            f"{name}: river detects exactly {expect['count']} time(s) on every "
            f"seed; this implementation produced counts {sorted(counts)}")

        # 2. the index, bounded by what river itself spans
        if expect["first"] is None:
            assert not firsts, f"{name}: river never fires here; this did"
        else:
            low, high = expect["first"]
            assert firsts, f"{name}: river fires on every seed; this never did"
            assert min(firsts) >= low and max(firsts) <= high, (
                f"{name}: river's first detection lies in [{low}, {high}] "
                f"across {len(list(SEEDS))} seeds; this implementation spans "
                f"[{min(firsts)}, {max(firsts)}]")

    # 3. the same seed twice is the same run: randomness must be seeded, not
    #    ambient, or nothing above is reproducible
    assert firings(data["step"], 7) == firings(data["step"], 7), (
        "seeded runs must be deterministic")

    # 4. different seeds really do draw different reference windows -- an
    #    implementation that ignores the seed would pass everything above
    indices = {tuple(firings(data["step"], s)) for s in (1, 2, 3, 5, 8, 13, 21)}
    assert len(indices) > 1, (
        "every seed gave an identical answer; the seed is being ignored, so "
        "this is not KSWIN's sampled reference window")

    # 5. the constructor surface callers depend on
    bare = KSWIN()
    assert abs(bare.alpha - 0.005) < 1e-12
    assert bare.window_size == 100 and bare.stat_size == 30
    assert bare.drift_detected is False

    with open("KSWIN_OK", "w", encoding="utf-8") as f:
        f.write("ok\n")
    print(f"kswin: river's envelope reproduced over {len(list(SEEDS))} seeds "
          f"on {len(REFERENCE)} streams")
    return 0


if __name__ == "__main__":
    sys.exit(main())
