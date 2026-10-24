"""Acceptance criteria for Page-Hinkley: river's observable behaviour, pinned.

This is a DOMESTICATION claim. The reference behaviour below was recorded
from `river.drift.PageHinkley` 0.23.0 and is now the specification; any
implementation that reproduces it is admitted, and river itself is no longer
a dependency. The point is not to describe the algorithm -- prose does not
constrain a reconstruction (claims/LOG.md A36) -- but to execute it, because
only a gate does.

TWELVE streams, in four groups, every group earned by a falsification:

  sharp changes      step, down_step, two_regimes
  no change          constant, spike
  monotone drift     ramp, down_ramp
  dip and recover    slow_sine, slow_cosine, dip_return, rise_return, v_shape

The first version of these criteria had only upward and flat streams. The
downward accumulator was therefore never exercised and three implementations
passed: river's two-sided form, an algebraic respelling of it, and -- the one
that proved the gap was real -- a detector that ignored downward drift
entirely. That bought `down_step` and `down_ramp`.

The second version, with those added, was still too wide, and a held-out
probe is what found it. Page-Hinkley has two standard spellings that are NOT
equivalent:

    clamped          up = max(0, alpha*up + dev - delta)
                     fire when up > threshold

    running minimum  sum += alpha*sum + dev - delta      (never clamped)
                     min_sum = min(min_sum, sum)
                     fire when sum - min_sum > threshold

They agree whenever the statistic rises from a floor near zero, which is
every sharp or monotone stream. They diverge when the statistic dips well
below zero and then recovers, because the clamped form has forgotten the dip
and the running-minimum form measures from the bottom of it. River uses the
running-minimum form. An implementation of the clamped form passed version
two of these criteria on all six streams and then missed river by two samples
on `slow_sine` -- the only one of twelve streams that separates them. It is
pinned here for that reason; the other dip-and-recover streams are included
because they cover the same regime without firing, which is a different and
also necessary constraint.

River also initialises its downward extremum to `-1` rather than `-inf` or
`0`. That is arbitrary -- almost certainly incidental rather than intended --
and it is observable, so pinning behaviour pins it. This is the whole
argument for domestication in one line: a specification written in prose
("a Page-Hinkley change detector") admits every spelling and every
initialisation, and would never have caught either of the two gaps above.

Firing INDICES are asserted, not merely counts, because detection latency is
what a change detector is judged on -- and latency is exactly where an
independent reconstruction of ADWIN diverged from river by ten samples while
agreeing on everything else (A36).

Standard library only, by design: river 0.23.0 ships no cp314 wheels, which
is why every claim in the APEIRON ladder carries a python3.13 pin (A8). A
domesticated detector removes that constraint.
"""

import sys

sys.path.insert(0, ".")

from page_hinkley import PageHinkley  # noqa: E402
from ph_streams import streams  # noqa: E402

PARAMS = {"min_instances": 30, "delta": 0.005, "threshold": 50.0,
          "alpha": 0.9999}

#: Recorded from river.drift.PageHinkley 0.23.0 under PARAMS.
REFERENCE = {
    "step": [305],
    "constant": [],
    "ramp": [227],
    "spike": [],
    "down_step": [305],
    "down_ramp": [227],
    "slow_sine": [340],
    "slow_cosine": [],
    "dip_return": [],
    "rise_return": [],
    "v_shape": [],
    "two_regimes": [297],
}

#: The streams whose statistic goes well negative before recovering. These
#: are what separate the clamped spelling from the running-minimum one; an
#: implementation may not pass by covering only the sharp-change group.
DISCRIMINATING = ("slow_sine", "slow_cosine", "dip_return", "rise_return",
                  "v_shape")


def firings(values, **over):
    detector = PageHinkley(**{**PARAMS, **over})
    out = []
    for i, value in enumerate(values):
        detector.update(value)
        if detector.drift_detected:
            out.append(i)
    return out


def main():
    data = streams()

    missing = set(REFERENCE) - set(data)
    assert not missing, f"streams() must supply every pinned stream: {missing}"

    # 1. the behaviour river exhibits, step for step, on all twelve
    for name, expected in REFERENCE.items():
        got = firings(data[name])
        assert got == expected, (
            f"{name}: river fires at {expected}, this implementation at {got}. "
            f"Detection latency is the property under test; a detector that "
            f"finds the same change at a different time is a different "
            f"detector."
            + ("\n\nThis stream dips below zero and recovers. The clamped "
               "spelling of Page-Hinkley (`up = max(0, ...)`) forgets the dip "
               "and fires late here; river measures from the running minimum "
               "of an unclamped sum. See this file's docstring."
               if name in DISCRIMINATING else ""))

    # 2. the running minimum is actually being tracked, stated directly
    #    rather than only as a consequence of stream 'slow_sine'. A long
    #    negative excursion followed by a rise of just over the threshold
    #    must fire; under the clamped spelling the excursion is discarded
    #    and the same rise does not reach the bar.
    descend = [0.0] * 40 + [-3.0] * 60 + [0.0] * 120
    assert firings(descend, min_instances=30, delta=0.0, threshold=20.0,
                   alpha=1.0), (
        "a sustained negative excursion followed by a return to baseline must "
        "be detected: the statistic's distance from its running MINIMUM is "
        "the test, not its distance from zero")

    # 3. the constructor surface callers depend on, with river's defaults
    bare = PageHinkley()
    assert bare.min_instances == 30 and bare.delta == 0.005
    assert bare.threshold == 50.0 and abs(bare.alpha - 0.9999) < 1e-12
    assert bare.drift_detected is False, "a fresh detector has detected nothing"

    # 4. the statistic resets after firing, so one change is reported once
    #    rather than latching on forever
    detector = PageHinkley(**PARAMS)
    fired = [i for i, v in enumerate(data["step"])
             if (detector.update(v), detector.drift_detected)[1]]
    assert len(fired) == 1, f"one change, one report -- got {len(fired)}"
    assert detector.width < len(data["step"]), (
        "the window must reset after a detection; an implementation that "
        "keeps accumulating cannot detect a second change")

    # 5. a second change after a reset is still found -- two_regimes has two,
    #    and river reports the first; the reset must leave the detector live
    second = PageHinkley(**PARAMS)
    for v in data["two_regimes"]:
        second.update(v)
    assert second.width > 0, "the detector must still be accumulating at the end"

    # 6. min_instances is respected: nothing fires before it, however
    #    violent the stream
    violent = [0.0, 100.0] * 30
    seen = firings(violent, min_instances=50, threshold=0.0)
    assert all(i >= 49 for i in seen), (
        f"fired before min_instances=50 at {[i for i in seen if i < 49]}")

    # 7. width reports how many samples the current window holds
    counter = PageHinkley(**PARAMS)
    for v in data["constant"][:17]:
        counter.update(v)
    assert counter.width == 17, f"width {counter.width}, expected 17"

    with open("PH_OK", "w", encoding="utf-8") as f:
        f.write("ok\n")
    print(f"page-hinkley: river's behaviour reproduced on {len(REFERENCE)} "
          f"streams, {len(DISCRIMINATING)} of them dip-and-recover")
    return 0


if __name__ == "__main__":
    sys.exit(main())
