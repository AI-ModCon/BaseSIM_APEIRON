"""Tests for drift detection: base classes, statistical detectors, and load factory."""

from __future__ import annotations

import numpy as np
import pytest

from drift_detection.detectors.base import (
    DriftSignal,
    LearningRegime,
)
from drift_detection.detectors.statistical_detectors import (
    ADWINDetector,
    KSWINDetector,
    PageHinkleyDetector,
)
from drift_detection.detectors.model_performance_detector import (
    EnsembleDetector,
    ModelEvalDetector,
    ModelPerformanceDetector,
)
from config.configuration import DriftDetectionCfg
from drift_detection.load_drift_detector import load_drift_detector


# ---------------------------------------------------------------------------
# DriftSignal
# ---------------------------------------------------------------------------
class TestDriftSignal:
    def test_construction(self):
        sig = DriftSignal(
            regime=LearningRegime.STABLE,
            drift_detected=False,
            drift_score=0.0,
        )
        assert sig.regime == LearningRegime.STABLE
        assert sig.drift_detected is False
        assert sig.drift_score == 0.0
        assert sig.confidence is None
        assert sig.metadata == {}

    def test_with_metadata(self):
        sig = DriftSignal(
            regime=LearningRegime.RETRAIN,
            drift_detected=True,
            drift_score=0.9,
            confidence=0.95,
            metadata={"key": "value"},
        )
        assert sig.metadata == {"key": "value"}
        assert sig.confidence == 0.95

    def test_repr(self):
        sig = DriftSignal(
            regime=LearningRegime.CONTINUAL_LEARNING,
            drift_detected=True,
            drift_score=0.5,
        )
        r = repr(sig)
        assert "continual_learning" in r
        assert "detected=True" in r


# ---------------------------------------------------------------------------
# LearningRegime
# ---------------------------------------------------------------------------
class TestLearningRegime:
    def test_values(self):
        assert LearningRegime.STABLE.value == "stable"
        assert LearningRegime.CONTINUAL_LEARNING.value == "continual_learning"
        assert LearningRegime.FINE_TUNING.value == "fine_tuning"
        assert LearningRegime.RETRAIN.value == "retrain"


# ---------------------------------------------------------------------------
# ADWINDetector
# ---------------------------------------------------------------------------
class TestADWINDetector:
    def test_init(self):
        d = ADWINDetector(delta=0.01)
        assert d.name == "ADWIN"
        assert d._is_initialized is True

    def test_stable_stream(self):
        d = ADWINDetector(delta=0.002)
        for _ in range(50):
            signal = d.update(0.5)
        assert signal.regime == LearningRegime.STABLE
        assert signal.drift_detected is False

    def test_drift_on_abrupt_shift(self):
        d = ADWINDetector(delta=0.002)
        # Feed a stable stream, then an abrupt shift
        for _ in range(200):
            d.update(0.5)
        detected = False
        for _ in range(200):
            signal = d.update(5.0)
            if signal.drift_detected:
                detected = True
                break
        assert detected

    def test_reset_clears_history(self):
        d = ADWINDetector()
        for _ in range(20):
            d.update(1.0)
        assert len(d._drift_history) > 0
        d.reset()
        assert d._drift_history == []
        assert d._value_history == []

    def test_confidence_equals_one_minus_delta(self):
        d = ADWINDetector(delta=0.01)
        signal = d.update(1.0)
        assert signal.confidence == pytest.approx(0.99)

    def test_metadata_keys(self):
        d = ADWINDetector()
        signal = d.update(1.0)
        assert "window_size" in signal.metadata
        assert "n_detections" in signal.metadata
        assert "recent_drift_rate" in signal.metadata


# ---------------------------------------------------------------------------
# KSWINDetector
# ---------------------------------------------------------------------------
class TestKSWINDetector:
    def test_init(self):
        d = KSWINDetector(alpha=0.01)
        assert d.name == "KSWIN"
        assert d._is_initialized is True

    def test_stable_stream(self):
        d = KSWINDetector(alpha=0.005, window_size=50, stat_size=20)
        for _ in range(100):
            signal = d.update(np.random.normal(0, 0.01))
        assert signal.drift_detected is False

    def test_reset(self):
        d = KSWINDetector()
        for _ in range(20):
            d.update(1.0)
        assert len(d._drift_history) == 20
        d.reset()
        assert d._drift_history == []

    def test_confidence(self):
        d = KSWINDetector(alpha=0.01)
        signal = d.update(1.0)
        assert signal.confidence == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# PageHinkleyDetector
# ---------------------------------------------------------------------------
class TestPageHinkleyDetector:
    def test_init(self):
        d = PageHinkleyDetector(min_instances=10)
        assert d.name == "PageHinkley"
        assert d._is_initialized is True

    def test_stable_stream(self):
        d = PageHinkleyDetector(min_instances=5, threshold=50)
        for _ in range(50):
            signal = d.update(0.5)
        assert signal.drift_detected is False

    def test_drift_on_large_shift(self):
        d = PageHinkleyDetector(min_instances=5, delta=0.005, threshold=10)
        for _ in range(50):
            d.update(0.5)
        detected = False
        for _ in range(200):
            signal = d.update(100.0)
            if signal.drift_detected:
                detected = True
                break
        assert detected

    def test_reset(self):
        d = PageHinkleyDetector()
        for _ in range(10):
            d.update(1.0)
        d.reset()
        assert d._drift_history == []


# ---------------------------------------------------------------------------
# ModelPerformanceDetector (simple value path)
# ---------------------------------------------------------------------------
class TestModelPerformanceDetector:
    def test_not_initialized_raises(self):
        d = ModelPerformanceDetector()
        with pytest.raises(ValueError, match="not initialized"):
            d.update(data=None, value=None)

    def test_simple_value_stable(self):
        d = ModelPerformanceDetector(drift_share_threshold=0.5)
        d._is_initialized = True
        signal = d.update(value=0.1)
        assert signal.drift_detected is False
        assert signal.regime == LearningRegime.STABLE

    def test_simple_value_drift(self):
        d = ModelPerformanceDetector(drift_share_threshold=0.5)
        d._is_initialized = True
        signal = d.update(value=0.9)
        assert signal.drift_detected is True

    def test_reset_clears_history(self):
        d = ModelPerformanceDetector()
        d._is_initialized = True
        d.update(value=0.5)
        d.update(value=0.6)
        assert len(d._drift_history) == 2
        d.reset()
        assert d._drift_history == []


# ---------------------------------------------------------------------------
# ModelEvalDetector
# ---------------------------------------------------------------------------
class TestModelEvalDetector:
    def test_raises_without_harness(self):
        d = ModelEvalDetector()
        with pytest.raises(ValueError, match="modelHarness must be provided"):
            d.update(0.0)

    def test_no_drift_when_metrics_same(self):
        from unittest.mock import MagicMock

        harness = MagicMock()
        harness.eval.return_value = [95.0]

        d = ModelEvalDetector()
        signal = d.update(
            0.0,
            modelHarness=harness,
            reference_validation_metrics=[95.0],
            higher_is_better={"acc": True},
        )
        assert signal.drift_detected is False

    def test_drift_when_metric_drops(self):
        from unittest.mock import MagicMock

        harness = MagicMock()
        harness.eval.return_value = [80.0]

        d = ModelEvalDetector()
        signal = d.update(
            0.0,
            modelHarness=harness,
            reference_validation_metrics=[95.0],
            higher_is_better={"acc": True},
        )
        assert signal.drift_detected is True
        assert signal.regime == LearningRegime.CONTINUAL_LEARNING


# ---------------------------------------------------------------------------
# EnsembleDetector
# ---------------------------------------------------------------------------
class TestEnsembleDetector:
    def _make_detectors(self, n=3):
        return [ADWINDetector(delta=0.002) for _ in range(n)]

    def test_majority_voting(self):
        detectors = self._make_detectors(3)
        ensemble = EnsembleDetector(detectors, voting="majority")
        signal = ensemble.update(1.0)
        assert isinstance(signal, DriftSignal)
        assert signal.metadata["n_detectors"] == 3

    def test_any_voting(self):
        detectors = self._make_detectors(3)
        ensemble = EnsembleDetector(detectors, voting="any")
        signal = ensemble.update(1.0)
        assert isinstance(signal, DriftSignal)

    def test_unanimous_voting(self):
        detectors = self._make_detectors(2)
        ensemble = EnsembleDetector(detectors, voting="unanimous")
        signal = ensemble.update(1.0)
        assert isinstance(signal, DriftSignal)

    def test_reset_resets_all(self):
        detectors = self._make_detectors(2)
        ensemble = EnsembleDetector(detectors, voting="majority")
        for _ in range(10):
            ensemble.update(1.0)
        ensemble.reset()
        for d in detectors:
            assert d._drift_history == []


# ---------------------------------------------------------------------------
# load_drift_detector factory
# ---------------------------------------------------------------------------
class TestLoadDriftDetector:
    def test_adwin(self, default_cfg):
        from dataclasses import replace

        cfg = replace(
            default_cfg,
            drift_detection=DriftDetectionCfg(detector_name="ADWINDetector"),
        )
        d = load_drift_detector(cfg)
        assert isinstance(d, ADWINDetector)

    def test_kswin(self, default_cfg):
        from dataclasses import replace

        cfg = replace(
            default_cfg,
            drift_detection=DriftDetectionCfg(detector_name="KSWINDetector"),
        )
        d = load_drift_detector(cfg)
        assert isinstance(d, KSWINDetector)

    def test_page_hinkley(self, default_cfg):
        from dataclasses import replace

        cfg = replace(
            default_cfg,
            drift_detection=DriftDetectionCfg(detector_name="PageHinkleyDetector"),
        )
        d = load_drift_detector(cfg)
        assert isinstance(d, PageHinkleyDetector)

    def test_model_performance(self, default_cfg):
        from dataclasses import replace

        cfg = replace(
            default_cfg,
            drift_detection=DriftDetectionCfg(detector_name="ModelPerformanceDetector"),
        )
        d = load_drift_detector(cfg)
        assert isinstance(d, ModelPerformanceDetector)

    def test_eval_detector(self, default_cfg):
        from dataclasses import replace

        cfg = replace(
            default_cfg,
            drift_detection=DriftDetectionCfg(detector_name="EvalDetector"),
        )
        d = load_drift_detector(cfg)
        assert isinstance(d, ModelEvalDetector)

    def test_ensemble_not_implemented(self, default_cfg):
        from dataclasses import replace

        cfg = replace(
            default_cfg,
            drift_detection=DriftDetectionCfg(detector_name="EnsembleDetector"),
        )
        with pytest.raises(NotImplementedError):
            load_drift_detector(cfg)

    def test_unknown_detector_raises(self, default_cfg):
        from dataclasses import replace

        cfg = replace(
            default_cfg,
            drift_detection=DriftDetectionCfg(detector_name="NonExistent"),
        )
        with pytest.raises(ValueError, match="Unknown drift detector"):
            load_drift_detector(cfg)

    def test_adwin_params_propagated(self, default_cfg):
        from dataclasses import replace

        cfg = replace(
            default_cfg,
            drift_detection=DriftDetectionCfg(
                detector_name="ADWINDetector",
                adwin_delta=0.05,
                adwin_minor_threshold=0.4,
                adwin_moderate_threshold=0.7,
            ),
        )
        d = load_drift_detector(cfg)
        assert d.delta == 0.05
        assert d.minor_threshold == 0.4
        assert d.moderate_threshold == 0.7
