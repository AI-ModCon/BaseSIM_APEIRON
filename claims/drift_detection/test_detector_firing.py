"""WHEN each statistical detector fires, on streams with no randomness in them.

The suite covers what a `DriftSignal` carries, how `drift_score` is
averaged, and which regime a score maps to. It does not cover the question
the detectors exist to answer: given this sequence of metric values, at
which step does drift fire? That decides when `ContinuousMonitor`
dispatches continual learning, so it decides which batches the model
trains on and therefore the weights the run ends with.

Two reconstructions of this layer show why it needs pinning. One of them
does not import river at all -- it reimplemented ADWIN, KSWIN and
Page-Hinkley from scratch in 115 lines against the source's 316 -- and it
passed every test here. On the step stream below it fires at 206 where the
original fires at 223: seventeen steps of stream consumed before
adaptation, same regime label, same drift score, and nothing in the suite
notices. That is A37's finding generalised from Page-Hinkley to the whole
layer: a detector wrapper is not specified by the shape of its output.

The streams are closed-form -- no RNG, no sampling, no clock -- so the
expected tables below are computations rather than recordings, and anyone
can re-derive them from the generator. KSWIN is constructed with an
explicit `seed`, because river's KSWIN draws its reference window and the
default `seed=None` leaves the firing steps unreproducible from one
process to the next.

Two of the six streams are NEGATIVE CONTROLS. `flat` never varies and
`spike` moves for three samples out of four hundred; no detector fires on
either. Without them the table could be satisfied by something that fires
constantly, which is the shape of mistake a firing-step assertion invites.

One asymmetry pinned here deliberately, because it is surprising and
unasserted: the three detectors average their drift history over DIFFERENT
windows. ADWIN uses the last 100 samples, KSWIN and Page-Hinkley the last
50 (`statistical_detectors.py:80`, `:183`, `:287`), so a single detection
scores 0.01 on ADWIN and 0.02 on the other two. A reconstruction that
picked one window for all three would pass `test_drift_severity.py` and
fail here.
"""

from __future__ import annotations

import pytest

from apeiron.drift_detection.detectors.statistical_detectors import (
    ADWINDetector,
    KSWINDetector,
    PageHinkleyDetector,
)

N = 400

#: Every stream is a pure function of its index. No RNG anywhere.
STREAMS: dict[str, list[float]] = {
    "step": [0.10 if i < N // 2 else 0.60 for i in range(N)],
    "ramp": [0.10 + 0.5 * max(0, i - 200) / 200 for i in range(N)],
    "flat": [0.25] * N,
    "sawtooth": [0.1 + 0.4 * ((i % 50) / 50) for i in range(N)],
    "spike": [0.9 if i in (150, 151, 152) else 0.1 for i in range(N)],
    "two_steps": [0.1 if i < 150 else (0.5 if i < 300 else 0.9)
                  for i in range(N)],
}

#: (stream, detector) -> the exact firings, as (step, regime, drift_score).
#: Measured against `src/` and re-derivable by running the detector over
#: `STREAMS[name]`.
EXPECTED: dict[tuple[str, str], list[tuple[int, str, float]]] = {
    ("step", "ADWIN"): [(223, "continual_learning", 0.01)],
    ("step", "KSWIN"): [(213, "continual_learning", 0.02)],
    ("step", "PageHinkley"): [(333, "continual_learning", 0.02)],

    ("ramp", "ADWIN"): [(319, "continual_learning", 0.01)],
    ("ramp", "KSWIN"): [(214, "continual_learning", 0.02),
                        (314, "continual_learning", 0.02)],
    ("ramp", "PageHinkley"): [],

    # Negative control: a constant stream has no drift to find.
    ("flat", "ADWIN"): [],
    ("flat", "KSWIN"): [],
    ("flat", "PageHinkley"): [],

    # Only the distribution test fires on a periodic sawtooth, and it fires
    # at the end of each period. ADWIN's mean is unchanged by it, which is
    # the behavioural difference between the two families.
    ("sawtooth", "ADWIN"): [],
    ("sawtooth", "KSWIN"): [(99, "continual_learning", 0.02),
                            (199, "continual_learning", 0.02),
                            (299, "continual_learning", 0.02),
                            (399, "continual_learning", 0.02)],
    ("sawtooth", "PageHinkley"): [],

    # Negative control: three samples out of four hundred is not drift.
    ("spike", "ADWIN"): [],
    ("spike", "KSWIN"): [],
    ("spike", "PageHinkley"): [],

    ("two_steps", "ADWIN"): [(191, "continual_learning", 0.01),
                             (351, "continual_learning", 0.01)],
    ("two_steps", "KSWIN"): [(163, "continual_learning", 0.02),
                             (313, "continual_learning", 0.02)],
    ("two_steps", "PageHinkley"): [(316, "continual_learning", 0.02)],
}

MAKERS = {
    "ADWIN": ADWINDetector,
    "KSWIN": lambda: KSWINDetector(seed=7),
    "PageHinkley": PageHinkleyDetector,
}


def firings(detector_name: str, stream_name: str):
    detector = MAKERS[detector_name]()
    out = []
    for step, value in enumerate(STREAMS[stream_name]):
        signal = detector.update(value)
        if signal.drift_detected:
            out.append((step, signal.regime.value, round(signal.drift_score, 6)))
    return out


@pytest.mark.parametrize(("stream", "detector"), sorted(EXPECTED))
def test_the_firing_steps_are_exact(stream, detector):
    assert firings(detector, stream) == EXPECTED[(stream, detector)]


class TestTheControlsDiscriminate:
    """A firing table is only evidence if something fails it."""

    def test_at_least_one_stream_fires_for_every_detector(self):
        """Otherwise a detector that never fires would pass the whole
        table, negative controls included."""
        for detector in MAKERS:
            total = sum(len(v) for (_s, d), v in EXPECTED.items()
                        if d == detector)
            assert total > 0, f"{detector} never fires in this table"

    def test_at_least_one_stream_is_silent_for_every_detector(self):
        """And a detector that fires on everything would pass the positive
        rows."""
        for detector in MAKERS:
            silent = [s for (s, d), v in EXPECTED.items()
                      if d == detector and not v]
            assert silent, f"{detector} has no silent stream in this table"

    def test_the_detectors_do_not_all_agree(self):
        """If every detector fired identically the table would be pinning
        one algorithm three times."""
        per_detector = {
            d: sorted((s, tuple(step for step, _r, _sc in v))
                      for (s, dd), v in EXPECTED.items() if dd == d)
            for d in MAKERS
        }
        assert len(set(map(str, per_detector.values()))) == len(MAKERS)


class TestScoreWindowsDiffer:
    """ADWIN averages over 100 samples; KSWIN and Page-Hinkley over 50."""

    def test_a_single_detection_scores_differently_per_detector(self):
        adwin = firings("ADWIN", "step")
        kswin = firings("KSWIN", "step")
        page = firings("PageHinkley", "step")
        assert adwin[0][2] == pytest.approx(1 / 100)
        assert kswin[0][2] == pytest.approx(1 / 50)
        assert page[0][2] == pytest.approx(1 / 50)


#: The step stream run long enough for a detection's window to expire.
LONG_STEP = [0.10 if i < 200 else 0.60 for i in range(500)]


def _score_sequence(detector_name: str, values: list[float]):
    detector = MAKERS[detector_name]()
    return [detector.update(value).drift_score for value in values]


class TestTheScoreIsATrailingWindow:
    """`drift_score` persists after a detection, and reaches the CSV.

    The score is not only a regime input. `ContinuousMonitor._log_metrics`
    writes `drift_signal.drift_score` as the drift stage's `score` column
    on every check, so the whole sequence lands in the metrics file at
    `[logging] metrics_output_path` -- which makes the values on
    NON-firing steps contract too, by the same reasoning as the profiler's
    reported FLOP figures.

    A reconstruction that delegated to river and got every firing step
    right still failed here: its score dropped straight back to 0.0 the
    step after a detection instead of carrying it for the window. Ninety
    nine of four hundred steps differed, every one of them a different
    cell in the CSV, with the firing steps and regime labels identical.
    A probe that only looked at firings would have called that layer clean.

    Whether that difference can also change a REGIME is unproven: it needs
    two detections inside one window, and no stream tried here produces
    them -- ADWIN's firings on an oscillating stream sit 160 steps apart,
    and tightening the oscillation stops it firing at all. So this is
    pinned as a difference in logged output, which it demonstrably is, and
    not as a difference in adaptation decisions, which was not shown.
    """

    def test_adwin_carries_a_detection_for_one_hundred_steps(self):
        scores = _score_sequence("ADWIN", LONG_STEP)
        nonzero = [i for i, s in enumerate(scores) if s > 0]
        assert nonzero[0] == 223
        assert nonzero[-1] == 322
        assert len(nonzero) == 100
        assert scores[323] == 0.0

    def test_kswin_carries_a_detection_for_fifty_steps(self):
        scores = _score_sequence("KSWIN", LONG_STEP)
        nonzero = [i for i, s in enumerate(scores) if s > 0]
        assert nonzero[0] == 213
        assert len(nonzero) == 50
        assert scores[nonzero[-1] + 1] == 0.0

    def test_the_score_does_not_reset_the_step_after_a_detection(self):
        """The single assertion that separates a trailing window from a
        per-step flag, stated on its own because it is the one a
        reconstruction got wrong."""
        scores = _score_sequence("ADWIN", LONG_STEP)
        assert scores[224] == pytest.approx(scores[223])
        assert scores[224] > 0


def _kswin_steps(seed: int, stream_name: str) -> list[int]:
    """One detector instance over a whole stream; the steps it fires at."""
    detector = KSWINDetector(seed=seed)
    return [step for step, value in enumerate(STREAMS[stream_name])
            if detector.update(value).drift_detected]


class TestKswinNeedsItsSeed:
    """This layer's reproducibility rests on it.

    River's KSWIN draws its reference window, so the firing steps are a
    function of the RNG as well as of the data. `KSWINDetector`'s `seed`
    parameter defaults to `None`, which means a run that does not set
    `drift_detection.kswin_seed` is not reproducible from one process to
    the next -- the same class of defect as A41, in the component that
    decides when training happens.
    """

    @pytest.mark.parametrize("seed", [7, 11, 101])
    def test_the_same_seed_fires_at_the_same_steps(self, seed):
        assert _kswin_steps(seed, "step") == _kswin_steps(seed, "step")

    def test_the_seed_is_what_fixes_them(self):
        assert _kswin_steps(7, "sawtooth") == [99, 199, 299, 399]
