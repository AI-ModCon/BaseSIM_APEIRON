"""Tests for trigger policies, actions, and schedules (the unified strategies)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from apeiron.drift_detection.detectors.base import DriftSignal, LearningRegime
from apeiron.driver.schedules import (
    FixedSchedule,
    NeverSchedule,
    PeriodicSchedule,
    RandomSchedule,
)
from apeiron.driver.stream_engine import StreamEngine
from apeiron.driver.trigger_action import AdaptAction, RecordOnlyAction
from apeiron.driver.trigger_policy import DetectorPolicy, SchedulePolicy


class TestDetectorPolicy:
    def test_interval_cadence_and_profiling(self, default_cfg):
        pol = DetectorPolicy(default_cfg)
        assert pol.cadence == "interval"
        assert pol.profile_decision is True
        assert pol.label == default_cfg.drift_detection.detector_name

    def test_decide_delegates_to_detector(self, default_cfg):
        pol = DetectorPolicy(default_cfg)
        sig = DriftSignal(
            regime=LearningRegime.STABLE, drift_detected=False, drift_score=0.0
        )
        with patch.object(pol.detector, "update", return_value=sig) as upd:
            out = pol.decide(42.0, 0)
        assert out is sig
        assert upd.call_args[0][0] == 42.0

    def test_reset_resets_detector(self, default_cfg):
        pol = DetectorPolicy(default_cfg)
        with patch.object(pol.detector, "reset") as rst:
            pol.reset()
        rst.assert_called_once()


class TestSchedulePolicy:
    def test_window_cadence_no_profiling(self):
        pol = SchedulePolicy(PeriodicSchedule(2))
        assert pol.cadence == "window"
        assert pol.profile_decision is False

    def test_fires_per_schedule(self):
        pol = SchedulePolicy(PeriodicSchedule(2))
        assert pol.decide(0.0, 0).drift_detected is False
        assert pol.decide(0.0, 1).drift_detected is True
        assert pol.decide(0.0, 3).drift_detected is True

    def test_signal_has_no_regime_and_zero_score(self):
        sig = SchedulePolicy(PeriodicSchedule(1)).decide(99.0, 0)
        assert sig.regime is None
        assert sig.drift_score == 0.0

    def test_annotate_log_marks_scheduled(self):
        fields = SchedulePolicy(NeverSchedule()).annotate_log(
            {"regime": "N/A", "confidence": "x"}
        )
        assert fields["regime"] == "scheduled"
        assert fields["confidence"] == "N/A"

    def test_headline_mentions_decision_point(self):
        assert "decision point 4" in SchedulePolicy(NeverSchedule()).fire_headline(1, 4)


class TestSchedules:
    def test_never(self):
        s = NeverSchedule()
        assert all(not s.should_fire(i) for i in range(10))

    def test_periodic_offset_to_period_end(self):
        s = PeriodicSchedule(3)
        fired = [i for i in range(9) if s.should_fire(i)]
        assert fired == [2, 5, 8]  # floor(9/3) = 3 fires

    def test_fixed(self):
        s = FixedSchedule([1, 4, 7])
        assert [i for i in range(9) if s.should_fire(i)] == [1, 4, 7]

    def test_random_budget_exact_count(self):
        s = RandomSchedule(seed=0, budget=3, horizon=20)
        fired = [i for i in range(20) if s.should_fire(i)]
        assert len(fired) == 3

    def test_random_budget_over_horizon_raises(self):
        with pytest.raises(ValueError):
            RandomSchedule(seed=0, budget=30, horizon=20)

    def test_random_needs_prob_or_budget(self):
        with pytest.raises(ValueError):
            RandomSchedule(seed=0)


class TestTriggerActions:
    def _engine(self, cfg, harness, action):
        return StreamEngine(
            cfg, harness, DetectorPolicy(cfg), action, logger=MagicMock()
        )

    def test_adapt_action_builds_trainer_and_runs_loop(
        self, default_cfg, dummy_harness
    ):
        engine = self._engine(default_cfg, dummy_harness, AdaptAction())
        assert engine.trainer is not None
        sig = DriftSignal(regime=None, drift_detected=True, drift_score=0.5)
        with patch.object(
            engine.trainer, "outer_cl_training_loop", return_value=0
        ) as loop:
            engine.action.on_fire(engine, sig, 3, 12.0)
        assert engine.fire_count == 1
        assert engine.fire_points == [3]
        loop.assert_called_once_with(drift_event_id=1)

    def test_adapt_action_checkpoints_when_enabled(self, default_cfg, dummy_harness):
        engine = self._engine(default_cfg, dummy_harness, AdaptAction())
        sig = DriftSignal(regime=None, drift_detected=True, drift_score=0.5)
        with (
            patch.object(engine.trainer, "outer_cl_training_loop", return_value=0),
            patch.object(
                type(dummy_harness), "ckpts_enabled", property(lambda self: True)
            ),
            patch.object(dummy_harness, "save_ckpt", return_value="/tmp/ck.pt") as save,
        ):
            engine.action.on_fire(engine, sig, 0, 1.0)
        save.assert_called_once_with(event=1)

    def test_record_only_action_no_trainer(self, default_cfg, dummy_harness):
        engine = self._engine(default_cfg, dummy_harness, RecordOnlyAction())
        assert engine.trainer is None
        sig = DriftSignal(
            regime=LearningRegime.STABLE, drift_detected=True, drift_score=0.3
        )
        engine.action.on_fire(engine, sig, 2, 55.0)
        assert engine.fire_count == 1
        assert engine.fire_points == [2]
        event = engine.events[-1]
        assert event["stream_idx"] == engine.stream_update_count
        assert event["metric_0"] == 55.0
