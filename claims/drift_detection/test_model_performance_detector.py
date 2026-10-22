"""Tests for the evidently-backed detectors.

Two tiers. The integration tier drives real evidently reports over frames
with known drift, so the contract with the library — the preset used, the
metric looked for, and the shape of its value — is checked against the
installed version rather than against a guess. The parsing tier stubs
`evidently.Report` to reach the result-shape refusals and to place the
rolling drift score exactly on each regime boundary, which real data cannot
do reliably.

This pipeline was the loosest boundary the reconstruction measured: an
81-line stand-in satisfied every criterion the suite previously stated
(claims/LOG.md, A11).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from apeiron.drift_detection.detectors.base import DriftSignal, LearningRegime
from apeiron.drift_detection.detectors.model_performance_detector import (
    EnsembleDetector,
    ModelEvalDetector,
    ModelPerformanceDetector,
)

N = 200


def _frame(seed: int, *, shift_f1: float = 0.0, shift_f2: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "f1": rng.normal(shift_f1, 1.0, N),
            "f2": rng.normal(shift_f2, 1.0, N),
        }
    )


def _detector(**kwargs: Any) -> ModelPerformanceDetector:
    defaults: dict[str, Any] = dict(reference_data=_frame(0), drift_share_threshold=0.5)
    defaults.update(kwargs)
    return ModelPerformanceDetector(**defaults)


# ---------------------------------------------------------------------------
# Against real evidently
# ---------------------------------------------------------------------------
class TestAgainstEvidently:
    def test_identically_distributed_data_is_stable(self):
        detector = _detector()
        signal = detector.update(data=_frame(1))

        assert isinstance(signal, DriftSignal)
        assert signal.drift_detected is False
        assert signal.regime == LearningRegime.STABLE
        assert signal.metadata["drift_share"] == 0.0
        assert signal.metadata["n_drifted_columns"] == 0
        assert signal.metadata["n_columns"] == 2
        assert signal.confidence == 0.95

    def test_every_column_shifted_is_detected(self):
        detector = _detector()
        signal = detector.update(data=_frame(1, shift_f1=5.0, shift_f2=5.0))

        assert signal.drift_detected is True
        assert signal.metadata["drift_share"] == 1.0
        assert signal.metadata["n_drifted_columns"] == 2
        assert signal.regime == LearningRegime.RETRAIN

    def test_the_share_threshold_is_strict(self):
        # one of two columns drifted is a share of exactly 0.5, and the
        # comparison is `share > threshold`, so the default 0.5 does NOT fire
        at_threshold = _detector(drift_share_threshold=0.5)
        signal = at_threshold.update(data=_frame(1, shift_f1=5.0))
        assert signal.metadata["drift_share"] == 0.5
        assert signal.drift_detected is False

        below_threshold = _detector(drift_share_threshold=0.4)
        signal = below_threshold.update(data=_frame(1, shift_f1=5.0))
        assert signal.metadata["drift_share"] == 0.5
        assert signal.drift_detected is True

    def test_predictions_and_targets_join_the_compared_columns(self):
        rng = np.random.default_rng(7)
        detector = ModelPerformanceDetector(
            reference_data=_frame(0),
            reference_predictions=rng.integers(0, 2, N),
            reference_targets=rng.integers(0, 2, N),
        )
        signal = detector.update(
            data=_frame(1),
            predictions=np.ones(N, dtype=int),  # a collapsed predictor
            targets=rng.integers(0, 2, N),
        )
        # four columns are now compared, and the collapsed prediction drifts
        assert signal.metadata["n_columns"] == 4
        assert signal.metadata["n_drifted_columns"] >= 1

    def test_history_accumulates_across_updates(self):
        detector = _detector()
        detector.update(data=_frame(1))
        detector.update(data=_frame(2, shift_f1=5.0, shift_f2=5.0))
        assert detector._drift_history == [0.0, 1.0]


# ---------------------------------------------------------------------------
# Reference data lifecycle
# ---------------------------------------------------------------------------
class TestReferenceData:
    def test_an_uninitialized_detector_refuses(self):
        detector = ModelPerformanceDetector()
        assert detector._is_initialized is False
        with pytest.raises(ValueError, match="not initialized"):
            detector.update(value=0.5)

    def test_set_reference_initializes_and_copies(self):
        detector = ModelPerformanceDetector()
        frame = _frame(0)
        predictions = np.zeros(N, dtype=int)
        detector.set_reference(frame, predictions=predictions, targets=predictions)

        assert detector._is_initialized is True
        assert detector.reference_predictions is predictions
        # a copy, so later mutation of the caller's frame cannot change the
        # reference the detector compares against
        frame["f1"] = 999.0
        assert detector.reference_data["f1"].iloc[0] != 999.0

    def test_set_reference_replaces_an_earlier_reference(self):
        detector = _detector()
        drifted = _frame(9, shift_f1=5.0, shift_f2=5.0)
        detector.set_reference(drifted)
        # now the drifted distribution IS the reference, so like data is stable
        signal = detector.update(data=_frame(10, shift_f1=5.0, shift_f2=5.0))
        assert signal.drift_detected is False

    def test_reset_clears_history_and_keeps_the_reference(self):
        detector = _detector()
        detector.update(data=_frame(1))
        assert detector._drift_history
        detector.reset()
        assert detector._drift_history == []
        assert detector.reference_data is not None
        assert detector._is_initialized is True


# ---------------------------------------------------------------------------
# Result parsing and its refusals
# ---------------------------------------------------------------------------
def _payload(count=1.0, share=0.5, name="DriftedColumnsCount(drift_share=0.5)"):
    return {
        "metrics": [{"metric_name": name, "value": {"count": count, "share": share}}]
    }


def _stub_report(payload):
    """Patch evidently.Report so `run().dict()` yields `payload`."""

    class _Report:
        def __init__(self, metrics=None):
            self.metrics = metrics

        def run(self, reference_data=None, current_data=None):
            return SimpleNamespace(dict=lambda: payload)

    return patch("evidently.Report", _Report)


class TestResultParsing:
    def _update(self, payload, detector=None, **update_kwargs):
        detector = detector or _detector()
        with _stub_report(payload):
            return detector.update(data=_frame(1), **update_kwargs)

    def test_a_result_without_metrics_refuses(self):
        with pytest.raises(KeyError, match="No metrics found"):
            self._update({})

    def test_an_empty_metrics_list_refuses(self):
        with pytest.raises(KeyError, match="No metrics found"):
            self._update({"metrics": []})

    def test_a_missing_drifted_columns_count_refuses(self):
        with pytest.raises(KeyError, match="DriftedColumnsCount metric not found"):
            self._update({"metrics": [{"metric_name": "ValueDrift(column=f1)"}]})

    def test_a_non_dict_value_refuses(self):
        payload = {"metrics": [{"metric_name": "DriftedColumnsCount", "value": 0.5}]}
        with pytest.raises(ValueError, match="Expected dict"):
            self._update(payload)

    def test_the_metric_is_found_among_others(self):
        payload = {
            "metrics": [
                {"metric_name": "ValueDrift(column=f1)", "value": 0.01},
                _payload(count=2.0, share=1.0)["metrics"][0],
                {"metric_name": "ValueDrift(column=f2)", "value": 0.9},
            ]
        }
        signal = self._update(payload)
        assert signal.metadata["n_drifted_columns"] == 2
        assert signal.metadata["drift_share"] == 1.0

    def test_absent_count_and_share_default_to_zero(self):
        payload = {"metrics": [{"metric_name": "DriftedColumnsCount", "value": {}}]}
        signal = self._update(payload)
        assert signal.metadata["n_drifted_columns"] == 0
        assert signal.metadata["drift_share"] == 0.0
        assert signal.drift_detected is False


class TestRegimeLadder:
    """The rolling drift score decides the regime; the share decides detection."""

    def _feed(self, detector, shares):
        signal = None
        for share in shares:
            with _stub_report(_payload(share=share)):
                signal = detector.update(data=_frame(1))
        return signal

    def test_quiet_stream_is_stable(self):
        detector = _detector(minor_threshold=0.3, moderate_threshold=0.6)
        signal = self._feed(detector, [0.0])
        assert signal.regime == LearningRegime.STABLE

    def test_a_first_spike_recommends_continual_learning(self):
        # detected (0.6 > 0.5) but the ten-window mean is still low
        detector = _detector(minor_threshold=0.3, moderate_threshold=0.6)
        signal = self._feed(detector, [0.0] * 9 + [0.6])
        assert signal.drift_detected is True
        assert signal.drift_score == pytest.approx(0.06)
        assert signal.regime == LearningRegime.CONTINUAL_LEARNING

    def test_a_sustained_middling_score_recommends_fine_tuning(self):
        detector = _detector(minor_threshold=0.3, moderate_threshold=0.6)
        signal = self._feed(detector, [0.45, 0.45])
        assert signal.drift_score == pytest.approx(0.45)
        assert signal.regime == LearningRegime.FINE_TUNING

    def test_a_high_score_recommends_retraining(self):
        detector = _detector(minor_threshold=0.3, moderate_threshold=0.6)
        signal = self._feed(detector, [0.9])
        assert signal.regime == LearningRegime.RETRAIN

    def test_an_elevated_history_outweighs_a_quiet_check(self):
        # worth knowing: a check that does NOT fire can still recommend
        # fine-tuning, because the regime reads the rolling score
        detector = _detector(minor_threshold=0.3, moderate_threshold=0.6)
        signal = self._feed(detector, [0.45, 0.45, 0.0])
        assert signal.drift_detected is False
        assert signal.regime == LearningRegime.FINE_TUNING

    def test_the_score_is_the_mean_of_the_last_ten_windows(self):
        detector = _detector()
        signal = self._feed(detector, [1.0] + [0.0] * 10)
        # the opening spike has fallen out of the window
        assert signal.drift_score == pytest.approx(0.0)
        assert len(detector._drift_history) == 11


# ---------------------------------------------------------------------------
# The value-only fallback
# ---------------------------------------------------------------------------
class TestSimpleValueFallback:
    def test_a_value_without_data_takes_the_fallback(self):
        detector = _detector()
        signal = detector.update(value=0.1)
        # the fallback reports no metadata, unlike the evidently path
        assert signal.metadata == {}
        assert detector._drift_history == [0.1]

    def test_neither_data_nor_value_refuses(self):
        with pytest.raises(ValueError, match="Either 'data' DataFrame or 'value'"):
            _detector().update()

    @pytest.mark.parametrize(
        "value,regime",
        [
            (0.1, LearningRegime.STABLE),
            (0.45, LearningRegime.CONTINUAL_LEARNING),
            (0.7, LearningRegime.FINE_TUNING),
            (0.9, LearningRegime.RETRAIN),
        ],
    )
    def test_the_fallback_ladder(self, value, regime):
        # a different ladder from the data path: STABLE below minor, then
        # continual learning, then fine-tuning below a hardcoded 0.8
        detector = _detector(minor_threshold=0.3, moderate_threshold=0.6)
        assert detector.update(value=value).regime == regime

    def test_the_fallback_detects_above_the_share_threshold(self):
        detector = _detector(drift_share_threshold=0.5)
        assert detector.update(value=0.9).drift_detected is True
        detector.reset()
        assert detector.update(value=0.1).drift_detected is False


# ---------------------------------------------------------------------------
# ModelEvalDetector
# ---------------------------------------------------------------------------
class _StubHarness:
    def __init__(self, metrics):
        self._metrics = metrics

    def eval(self):
        return self._metrics


class TestModelEvalDetector:
    def test_a_missing_harness_refuses(self):
        with pytest.raises(ValueError, match="modelHarness must be provided"):
            ModelEvalDetector().update(0.0)

    def test_a_drop_in_a_higher_is_better_metric_is_drift(self):
        signal = ModelEvalDetector().update(
            0.0,
            modelHarness=_StubHarness([80.0]),
            reference_validation_metrics=[90.0],
            higher_is_better={"accuracy": True},
        )
        assert signal.drift_detected is True
        assert signal.regime == LearningRegime.CONTINUAL_LEARNING

    def test_a_rise_in_a_lower_is_better_metric_is_drift(self):
        signal = ModelEvalDetector().update(
            0.0,
            modelHarness=_StubHarness([2.0]),
            reference_validation_metrics=[1.0],
            higher_is_better={"loss": False},
        )
        assert signal.drift_detected is True

    def test_metrics_holding_their_reference_are_stable(self):
        signal = ModelEvalDetector().update(
            0.0,
            modelHarness=_StubHarness([95.0, 0.5]),
            reference_validation_metrics=[90.0, 1.0],
            higher_is_better={"accuracy": True, "loss": False},
        )
        assert signal.drift_detected is False
        assert signal.regime == LearningRegime.STABLE

    def test_mismatched_metric_counts_refuse(self):
        with pytest.raises(AssertionError):
            ModelEvalDetector().update(
                0.0,
                modelHarness=_StubHarness([1.0, 2.0]),
                reference_validation_metrics=[1.0],
                higher_is_better={"accuracy": True},
            )

    def test_reset_is_a_no_op(self):
        detector = ModelEvalDetector()
        detector.reset()  # stateless: nothing to clear, nothing to raise


# ---------------------------------------------------------------------------
# EnsembleDetector guards
# ---------------------------------------------------------------------------
class _FixedDetector(ModelEvalDetector):
    """A sub-detector with a fixed verdict, for voting arithmetic."""

    def __init__(self, detected, name, score=1.0, confidence=0.5):
        super().__init__(name=name)
        self._detected = detected
        self._score = score
        self._confidence = confidence
        self._is_initialized = True

    def update(self, value, **kwargs):
        return DriftSignal(
            regime=LearningRegime.STABLE,
            drift_detected=self._detected,
            drift_score=self._score,
            confidence=self._confidence,
        )


class TestEnsembleGuards:
    def test_a_corrupted_voting_strategy_refuses(self):
        ensemble = EnsembleDetector([_FixedDetector(True, "a")], voting="any")
        ensemble.voting = "telepathy"  # bypassing the constructor's validation
        with pytest.raises(ValueError, match="unknown voting strategy"):
            ensemble.update(1.0)

    def test_confidence_averages_over_the_sub_detectors(self):
        ensemble = EnsembleDetector(
            [
                _FixedDetector(True, "a", confidence=0.4),
                _FixedDetector(False, "b", confidence=0.8),
            ],
            voting="any",
        )
        signal = ensemble.update(1.0)
        assert signal.confidence == pytest.approx(0.6)
        assert signal.metadata["n_votes"] == 1

    def test_absent_confidences_leave_the_field_empty(self):
        a = _FixedDetector(True, "a")
        a._confidence = None
        ensemble = EnsembleDetector([a], voting="any")
        assert ensemble.update(1.0).confidence is None

    def test_a_checkpoint_without_sub_detectors_refuses(self):
        ensemble = EnsembleDetector([_FixedDetector(True, "a")], voting="any")
        state = ensemble.state_dict()
        del state["sub_detectors"]
        with pytest.raises(KeyError, match="missing 'sub_detectors'"):
            ensemble.load_state_dict(state)

    def test_a_checkpoint_of_the_wrong_width_refuses(self):
        two = EnsembleDetector(
            [_FixedDetector(True, "a"), _FixedDetector(False, "b")], voting="any"
        )
        one = EnsembleDetector([_FixedDetector(True, "a")], voting="any")
        with pytest.raises(ValueError, match="holds 2 sub-detectors"):
            one.load_state_dict(two.state_dict())
