"""Which regime a drift score maps to, including when the thresholds cross.

The regime is the detector's instruction to the rest of the framework:
`STABLE` means carry on, `CONTINUAL_LEARNING`, `FINE_TUNING` and `RETRAIN`
ask for progressively more adaptation. It is the output the monitor acts
on, so a different regime for the same drift is a different training run.

`test_drift_severity.py` covers the ordinary case -- thresholds in
ascending order, a score placed between them. It does not cover what
happens when the two thresholds are equal or inverted, and nothing else
does either. The ladder is a plain if/elif chain
(`statistical_detectors.py:84`-`:91`):

    if not drift_detected:                       STABLE
    elif drift_score < minor_threshold:          CONTINUAL_LEARNING
    elif drift_score < moderate_threshold:       FINE_TUNING
    else:                                        RETRAIN

so `minor` is tested FIRST whatever its relation to `moderate`. With
`minor=0.02, moderate=0.01` and a score of 0.01, the first comparison
already succeeds and the answer is `CONTINUAL_LEARNING` -- the *least*
adaptation, from the configuration that on its face asks for a low bar.
A reconstruction that normalised the thresholds, or tested `moderate`
first, answered `RETRAIN` for the same input: the most adaptation instead
of the least, on a stream the original would have barely touched.

Both answers are defensible readings of an inverted configuration. What is
not defensible is leaving it unspecified, because the two differ by the
whole width of the ladder. Pinned as the measured table, with the
comparisons stated as strict `<` at the boundaries, where equal thresholds
land on `RETRAIN`.
"""

from __future__ import annotations

import pytest

from apeiron.drift_detection.detectors.base import LearningRegime
from apeiron.drift_detection.detectors.statistical_detectors import ADWINDetector

#: A step stream ADWIN fires on exactly once, at index 223, scoring 1/100.
STREAM = [0.10 if i < 200 else 0.60 for i in range(400)]
SCORE = 0.01


def _regime_at_the_detection(minor: float, moderate: float) -> str:
    detector = ADWINDetector(minor_threshold=minor, moderate_threshold=moderate)
    fired = [
        (step, signal.regime.value, round(signal.drift_score, 4))
        for step, value in enumerate(STREAM)
        if (signal := detector.update(value)).drift_detected
    ]
    assert len(fired) == 1, f"expected one detection, got {fired}"
    step, regime, score = fired[0]
    assert step == 223
    assert score == pytest.approx(SCORE)
    return regime


class TestTheLadder:
    """The measured table. The score is 0.01 in every row."""

    @pytest.mark.parametrize(
        ("minor", "moderate", "expected"),
        [
            # Ordinary: the score is below both, so the lightest response.
            (0.3, 0.6, "continual_learning"),
            # The score sits between them.
            (0.005, 0.015, "fine_tuning"),
            (0.01, 0.02, "fine_tuning"),
            # Both thresholds at or below the score: the heaviest response.
            (0.0, 0.0, "retrain"),
            (0.0, 0.01, "retrain"),
            # Equal thresholds, score equal to them: `<` is strict, so
            # neither branch is taken and it falls through to RETRAIN.
            (0.01, 0.01, "retrain"),
            # INVERTED, and the case a reconstruction answered differently:
            # `minor` is tested first regardless, so a score under it means
            # the LIGHTEST response even though `moderate` is lower.
            (0.02, 0.01, "continual_learning"),
            # A minor threshold below every possible score, moderate above:
            # the first test fails, the second succeeds.
            (-1.0, 2.0, "fine_tuning"),
        ],
    )
    def test_the_regime_for_a_threshold_pair(self, minor, moderate, expected):
        assert _regime_at_the_detection(minor, moderate) == expected

    def test_the_minor_threshold_is_tested_first(self):
        """Stated on its own because it is the whole of the inverted case:
        swapping the thresholds does NOT swap the answer, it changes which
        comparison decides."""
        assert _regime_at_the_detection(0.02, 0.01) == "continual_learning"
        assert _regime_at_the_detection(0.01, 0.02) == "fine_tuning"

    def test_the_boundary_is_strict(self):
        """A score exactly equal to a threshold is NOT below it."""
        assert _regime_at_the_detection(SCORE, 1.0) == "fine_tuning"
        assert _regime_at_the_detection(SCORE + 1e-9, 1.0) == "continual_learning"


class TestNoDetectionIsAlwaysStable:
    """The one row no threshold configuration can change."""

    @pytest.mark.parametrize(
        ("minor", "moderate"), [(0.3, 0.6), (0.0, 0.0), (0.02, 0.01)]
    )
    def test_a_flat_stream_stays_stable(self, minor, moderate):
        detector = ADWINDetector(minor_threshold=minor,
                                 moderate_threshold=moderate)
        regimes = {detector.update(0.25).regime for _ in range(200)}
        assert regimes == {LearningRegime.STABLE}
