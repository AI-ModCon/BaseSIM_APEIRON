"""Tests for drift detection: base classes, statistical detectors, and load factory."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from apeiron.drift_detection.detectors.base import (
    BaseDriftDetector,
    DriftSignal,
    LearningRegime,
)
from apeiron.drift_detection.detectors.statistical_detectors import (
    ADWINDetector,
    KSWINDetector,
    PageHinkleyDetector,
)
from apeiron.config.configuration import DriftDetectionCfg
from apeiron.drift_detection.load_drift_detector import load_drift_detector

# Evidently has no wheel for every deployment target. Import them behind a
# guard so the statistical-detector tests -- which need nothing beyond river --
# still run there, instead of the whole module erroring out at collection.
try:
    from apeiron.drift_detection.detectors.model_performance_detector import (
        EnsembleDetector,
        ModelEvalDetector,
        ModelPerformanceDetector,
    )

    HAS_EVIDENTLY = True
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    HAS_EVIDENTLY = False

requires_evidently = pytest.mark.skipif(
    not HAS_EVIDENTLY, reason="evidently is not installed in this environment"
)


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
def _shifting_stream() -> list[float]:
    """Deterministic stream: 100 samples at mean 0, then 100 at mean 5."""
    rng = np.random.default_rng(0)
    return [
        *rng.normal(0.0, 1.0, 100).tolist(),
        *rng.normal(5.0, 1.0, 100).tolist(),
    ]


def _drift_steps(detector: KSWINDetector, stream: list[float]) -> list[int]:
    return [i for i, v in enumerate(stream) if detector.update(v).drift_detected]


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

    def test_same_seed_gives_the_same_detections(self):
        stream = _shifting_stream()
        first = _drift_steps(
            KSWINDetector(window_size=60, stat_size=20, seed=1337), stream
        )
        second = _drift_steps(
            KSWINDetector(window_size=60, stat_size=20, seed=1337), stream
        )
        assert first == second
        assert first, "expected at least one detection on a mean shift of 5 sigma"

    def test_reset_restores_the_seeded_sequence(self):
        stream = _shifting_stream()
        d = KSWINDetector(window_size=60, stat_size=20, seed=1337)
        first = _drift_steps(d, stream)
        d.reset()
        assert _drift_steps(d, stream) == first


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
@requires_evidently
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
@requires_evidently
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
class FakeDetector(BaseDriftDetector):
    """Detector that always reports a fixed verdict, for voting tests."""

    def __init__(self, detected: bool, name: str = "Fake"):
        super().__init__(name)
        self.detected = detected

    def update(self, value: float, **kwargs) -> DriftSignal:
        return DriftSignal(
            regime=LearningRegime.CONTINUAL_LEARNING
            if self.detected
            else LearningRegime.STABLE,
            drift_detected=self.detected,
            drift_score=1.0 if self.detected else 0.0,
        )

    def reset(self) -> None:
        pass


class TestEnsembleDetector:
    def _make_detectors(self, n=3):
        return [ADWINDetector(delta=0.002) for _ in range(n)]

    def _votes(self, *detected: bool):
        return [FakeDetector(d) for d in detected]

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

    @pytest.mark.parametrize(
        "voting,votes,expected",
        [
            ("majority", (True, True, False), True),
            ("majority", (True, False, False), False),
            ("any", (True, False, False), True),
            ("any", (False, False, False), False),
            ("unanimous", (True, True, True), True),
            ("unanimous", (True, True, False), False),
        ],
    )
    def test_voting_strategies(self, voting, votes, expected):
        ensemble = EnsembleDetector(self._votes(*votes), voting=voting)
        signal = ensemble.update(1.0)
        assert signal.drift_detected is expected
        assert signal.metadata["voting"] == voting
        assert signal.metadata["n_votes"] == sum(votes)

    @pytest.mark.parametrize(
        "alias,canonical",
        [("all", "unanimous"), ("and", "unanimous"), ("or", "any")],
    )
    def test_voting_aliases(self, alias, canonical):
        ensemble = EnsembleDetector(self._votes(True, False), voting=alias)
        assert ensemble.voting == canonical

    def test_unknown_voting_raises(self):
        with pytest.raises(ValueError, match="voting"):
            EnsembleDetector(self._votes(True), voting="plurality")

    def test_empty_detectors_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            EnsembleDetector([], voting="any")

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

    def test_kswin_seed_is_forwarded_from_config(self, default_cfg):
        from dataclasses import replace

        cfg = replace(
            default_cfg,
            drift_detection=DriftDetectionCfg(
                detector_name="KSWINDetector", kswin_seed=1337
            ),
        )
        d = load_drift_detector(cfg)
        assert d.seed == 1337

    def test_page_hinkley(self, default_cfg):
        from dataclasses import replace

        cfg = replace(
            default_cfg,
            drift_detection=DriftDetectionCfg(detector_name="PageHinkleyDetector"),
        )
        d = load_drift_detector(cfg)
        assert isinstance(d, PageHinkleyDetector)

    @requires_evidently
    def test_model_performance(self, default_cfg):
        from dataclasses import replace

        cfg = replace(
            default_cfg,
            drift_detection=DriftDetectionCfg(detector_name="ModelPerformanceDetector"),
        )
        d = load_drift_detector(cfg)
        assert isinstance(d, ModelPerformanceDetector)

    @requires_evidently
    def test_eval_detector(self, default_cfg):
        from dataclasses import replace

        cfg = replace(
            default_cfg,
            drift_detection=DriftDetectionCfg(detector_name="EvalDetector"),
        )
        d = load_drift_detector(cfg)
        assert isinstance(d, ModelEvalDetector)

    def test_ensemble(self, default_cfg):
        from dataclasses import replace

        cfg = replace(
            default_cfg,
            drift_detection=DriftDetectionCfg(
                detector_name="EnsembleDetector",
                ensemble_detectors=["ADWINDetector", "KSWINDetector"],
                ensemble_voting="any",
            ),
        )
        d = load_drift_detector(cfg)
        assert isinstance(d, EnsembleDetector)
        assert d.voting == "any"
        assert [type(sub) for sub in d.detectors] == [ADWINDetector, KSWINDetector]

    def test_ensemble_without_sub_detectors_raises(self, default_cfg):
        from dataclasses import replace

        cfg = replace(
            default_cfg,
            drift_detection=DriftDetectionCfg(detector_name="EnsembleDetector"),
        )
        with pytest.raises(ValueError, match="ensemble_detectors"):
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


# ---------------------------------------------------------------------------
# Checkpointing: state_dict / load_state_dict
# ---------------------------------------------------------------------------
DETECTORS = [
    pytest.param(lambda: ADWINDetector(delta=0.002), id="adwin"),
    pytest.param(lambda: KSWINDetector(seed=7), id="kswin"),
    pytest.param(lambda: PageHinkleyDetector(threshold=10.0), id="page_hinkley"),
]


def _stream() -> list[float]:
    """Deterministic stream whose mean shifts at the halfway point."""
    rng = np.random.default_rng(1337)
    return [
        float(v) for v in np.concatenate([rng.normal(0, 1, 150), rng.normal(5, 1, 150)])
    ]


def _verdicts(detector: BaseDriftDetector, values: list[float]) -> list[tuple]:
    """Feed values through a detector, recording its decision for each."""
    return [(s.drift_detected, s.drift_score) for s in map(detector.update, values)]


class TestDetectorCheckpointing:
    """Restoring a detector must make a resumed run equivalent to an
    uninterrupted one, not merely cheaper than restarting."""

    @pytest.mark.parametrize("make_detector", DETECTORS)
    def test_resume_matches_uninterrupted_run(self, make_detector):
        """Restoring mid-stream reproduces the verdicts of a detector that never stopped."""
        warmup, tail = _stream()[:150], _stream()[150:]

        uninterrupted_detector = make_detector()
        _verdicts(uninterrupted_detector, warmup)
        expected = _verdicts(uninterrupted_detector, tail)

        checkpointed_detector = make_detector()
        _verdicts(checkpointed_detector, warmup)
        resumed_detector = make_detector()
        resumed_detector.load_state_dict(
            pickle.loads(pickle.dumps(checkpointed_detector.state_dict()))
        )

        assert _verdicts(resumed_detector, tail) == expected
        assert any(detected for detected, _ in expected), "stream never drifted"

    def test_fresh_detector_diverges(self):
        """Without the restore, the assertion above would pass trivially."""
        warmup, tail = _stream()[:150], _stream()[150:]
        warmed = ADWINDetector()
        _verdicts(warmed, warmup)

        assert _verdicts(warmed, tail) != _verdicts(ADWINDetector(), tail)

    def test_state_is_copied_not_shared(self):
        """Snapshots must not alias the detector, in either direction."""
        source = ADWINDetector()
        _verdicts(source, _stream()[:50])
        state = source.state_dict()

        _verdicts(source, _stream()[:10])  # must not leak into `state`
        assert len(state["_value_history"]) == 50

        first, second = ADWINDetector(), ADWINDetector()
        first.load_state_dict(state)
        second.load_state_dict(state)
        _verdicts(first, _stream()[:10])  # must not reach `second`
        assert len(second._value_history) == 50

    def test_cross_detector_load_raises(self):
        """Loading one detector type's checkpoint into another is rejected."""
        with pytest.raises(ValueError, match="written by"):
            PageHinkleyDetector().load_state_dict(ADWINDetector().state_dict())

    def test_missing_state_raises(self):
        """A checkpoint missing a declared attribute is rejected, not half-applied."""
        state = ADWINDetector().state_dict()
        del state["_drift_history"]
        with pytest.raises(KeyError, match="_drift_history"):
            ADWINDetector().load_state_dict(state)

    def test_undeclared_detector_raises(self):
        """Opting in is explicit, so a custom detector cannot silently save nothing."""

        class Undeclared(BaseDriftDetector):
            def update(self, value: float, **kwargs) -> DriftSignal:
                return DriftSignal(LearningRegime.STABLE, False, 0.0)

            def reset(self) -> None:
                pass

        with pytest.raises(NotImplementedError, match="_STATE_ATTRS"):
            Undeclared(name="undeclared").state_dict()


@requires_evidently
class TestEnsembleCheckpointing:
    """An ensemble's state is its sub-detectors' state, positionally aligned."""

    def _ensemble(self) -> EnsembleDetector:
        return EnsembleDetector([ADWINDetector(), PageHinkleyDetector()], voting="any")

    def test_resume_matches_uninterrupted_run(self):
        """Restoring an ensemble reproduces the verdicts of one that never stopped."""
        warmup, tail = _stream()[:150], _stream()[150:]

        uninterrupted_detector = self._ensemble()
        _verdicts(uninterrupted_detector, warmup)
        expected = _verdicts(uninterrupted_detector, tail)

        checkpointed_detector = self._ensemble()
        _verdicts(checkpointed_detector, warmup)
        resumed_detector = self._ensemble()
        resumed_detector.load_state_dict(checkpointed_detector.state_dict())

        assert _verdicts(resumed_detector, tail) == expected

    def test_sub_detector_count_mismatch_raises(self):
        """An ensemble rebuilt with a different number of sub-detectors is rejected."""
        state = self._ensemble().state_dict()
        smaller = EnsembleDetector([ADWINDetector()], voting="any")
        with pytest.raises(ValueError, match="sub-detectors"):
            smaller.load_state_dict(state)

    def test_stateless_detector_round_trips(self):
        """A detector declaring no state saves and loads a bare class tag."""
        state = ModelEvalDetector().state_dict()
        assert state == {"detector_class": "ModelEvalDetector"}
        ModelEvalDetector().load_state_dict(state)
