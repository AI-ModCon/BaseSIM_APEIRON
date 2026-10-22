"""Boundary tests for the severity-to-regime mapping of the statistical detectors.

The behavioral tests in test_drift_detection.py drive real river detectors;
these pin the mapping itself — the comparisons against `minor_threshold` and
`moderate_threshold`, the no-detection short-circuit, and the recent-window
truncation of the drift score — by substituting a stub for the river
detector, so each case lands exactly on the boundary it asserts.
"""

from __future__ import annotations

import pytest

from apeiron.drift_detection.detectors.base import LearningRegime
from apeiron.drift_detection.detectors.statistical_detectors import (
    ADWINDetector,
    KSWINDetector,
    PageHinkleyDetector,
)


class _ForcedRiverDetector:
    """Stands in for the river detector: a fixed verdict, no statistics."""

    def __init__(self, detected: bool):
        self.drift_detected = detected
        self.width = 7  # ADWIN metadata reads it

    def update(self, value):
        pass


DETECTORS = [
    pytest.param(ADWINDetector, 100, id="adwin"),
    pytest.param(KSWINDetector, 50, id="kswin"),
    pytest.param(PageHinkleyDetector, 50, id="page-hinkley"),
]


def _make(cls, minor=0.25, moderate=0.75):
    return cls(minor_threshold=minor, moderate_threshold=moderate)


def _signal(det, history, detected):
    det.detector = _ForcedRiverDetector(detected)
    det._drift_history = list(history)
    return det.update(0.0)


@pytest.mark.parametrize("cls,window", DETECTORS)
class TestSeverityBoundaries:
    def test_no_detection_is_stable_whatever_the_history(self, cls, window):
        # the short-circuit: a quiet step is STABLE even after a loud past
        sig = _signal(_make(cls), [1] * 10, detected=False)
        assert sig.regime == LearningRegime.STABLE
        assert not sig.drift_detected

    def test_score_below_minor_is_continual_learning(self, cls, window):
        # history [0]*7 + the appended 1: score 1/8 = 0.125 < 0.25
        sig = _signal(_make(cls), [0] * 7, detected=True)
        assert sig.regime == LearningRegime.CONTINUAL_LEARNING
        assert sig.drift_score == pytest.approx(0.125)

    def test_score_at_minor_threshold_is_fine_tuning(self, cls, window):
        # [0, 0, 0] + 1: score exactly 0.25 — the minor comparison is strict
        sig = _signal(_make(cls), [0, 0, 0], detected=True)
        assert sig.drift_score == pytest.approx(0.25)
        assert sig.regime == LearningRegime.FINE_TUNING

    def test_score_between_thresholds_is_fine_tuning(self, cls, window):
        # [1, 0, 0] + 1: score 0.5 in [0.25, 0.75)
        sig = _signal(_make(cls), [1, 0, 0], detected=True)
        assert sig.drift_score == pytest.approx(0.5)
        assert sig.regime == LearningRegime.FINE_TUNING

    def test_score_at_moderate_threshold_is_retrain(self, cls, window):
        # [1, 1, 0] + 1: score exactly 0.75 — the moderate comparison is strict
        sig = _signal(_make(cls), [1, 1, 0], detected=True)
        assert sig.drift_score == pytest.approx(0.75)
        assert sig.regime == LearningRegime.RETRAIN

    def test_score_is_the_mean_over_the_recent_window_only(self, cls, window):
        # a loud distant past behind a quiet recent window must not count:
        # last `window` entries are [0]*(window-1) + the appended 1
        history = [1] * 200 + [0] * (window - 1)
        sig = _signal(_make(cls, minor=0.1), history, detected=True)
        assert sig.drift_score == pytest.approx(1 / window)
        assert sig.regime == LearningRegime.CONTINUAL_LEARNING
