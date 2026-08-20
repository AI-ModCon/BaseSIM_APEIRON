"""Tests for ContinuousMonitor and the StreamEngine loop mechanics.

ContinuousMonitor is now thin wiring of StreamEngine (DetectorPolicy +
AdaptAction); these cover the loop machinery it inherits. The logger is injected
(a MagicMock) rather than patched from a global.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
import torch

from apeiron.drift_detection.detectors.base import DriftSignal, LearningRegime
from apeiron.driver.continuous_monitor import ContinuousMonitor


def _monitor(cfg, harness):
    return ContinuousMonitor(cfg=cfg, modelHarness=harness, logger=MagicMock())


class TestContinuousMonitorInit:
    def test_initialization(self, default_cfg, dummy_harness):
        mon = _monitor(default_cfg, dummy_harness)
        assert mon.stream_update_count == 0
        assert mon.batch_count == 0
        assert mon.fire_count == 0
        assert mon.metric_buffer == []
        assert mon.detection_interval == default_cfg.drift_detection.detection_interval
        assert mon.max_stream_updates == default_cfg.drift_detection.max_stream_updates
        assert mon.cadence == "interval"
        assert mon.detector is mon.policy.detector

    def test_adapt_run_builds_trainer(self, default_cfg, dummy_harness):
        mon = _monitor(default_cfg, dummy_harness)
        assert mon.trainer is not None  # AdaptAction.needs_trainer


class TestShouldStop:
    def test_false_initially(self, default_cfg, dummy_harness):
        assert _monitor(default_cfg, dummy_harness)._should_stop() is False

    def test_true_at_max(self, default_cfg, dummy_harness):
        mon = _monitor(default_cfg, dummy_harness)
        mon.stream_update_count = default_cfg.drift_detection.max_stream_updates
        assert mon._should_stop() is True


class TestExtendStream:
    def test_increments_counter(self, default_cfg, dummy_harness):
        mon = _monitor(default_cfg, dummy_harness)
        with patch.object(dummy_harness, "update_data_stream") as mock_update:
            mon._extend_stream()
        assert mon.stream_update_count == 1
        mock_update.assert_called_once()


class TestAggregateBuffer:
    def test_raises_on_empty_buffer(self, default_cfg, dummy_harness):
        mon = _monitor(default_cfg, dummy_harness)
        with pytest.raises(RuntimeError, match="requires evaluation metrics"):
            mon._aggregate_buffer()

    def _agg(self, default_cfg, dummy_harness, method):
        cfg = replace(
            default_cfg,
            drift_detection=replace(default_cfg.drift_detection, aggregation=method),
        )
        mon = _monitor(cfg, dummy_harness)
        mon.metric_buffer = [[90.0], [70.0], [60.0]]
        value = mon._aggregate_buffer()
        assert mon.metric_buffer == []
        return value

    def test_mean(self, default_cfg, dummy_harness):
        assert self._agg(default_cfg, dummy_harness, "mean") == pytest.approx(
            73.33, abs=0.01
        )

    def test_last(self, default_cfg, dummy_harness):
        assert self._agg(default_cfg, dummy_harness, "last") == 60.0

    def test_median(self, default_cfg, dummy_harness):
        assert self._agg(default_cfg, dummy_harness, "median") == 70.0


class TestDecisionPoint:
    def test_feeds_aggregate_to_policy_and_fires(self, default_cfg, dummy_harness):
        mon = _monitor(default_cfg, dummy_harness)
        mon.metric_buffer = [[50.0]]
        fired = DriftSignal(
            regime=LearningRegime.CONTINUAL_LEARNING,
            drift_detected=True,
            drift_score=0.9,
        )
        with (
            patch.object(mon.detector, "update", return_value=fired) as upd,
            patch.object(mon.trainer, "outer_cl_training_loop", return_value=0) as loop,
        ):
            mon._decision_point()
        assert upd.call_args[0][0] == 50.0  # aggregated metric reached the detector
        assert mon.decision_count == 1
        assert mon.fire_count == 1
        assert mon.fire_points == [0]
        loop.assert_called_once()

    def test_no_fire_does_not_adapt(self, default_cfg, dummy_harness):
        mon = _monitor(default_cfg, dummy_harness)
        mon.metric_buffer = [[50.0]]
        calm = DriftSignal(
            regime=LearningRegime.STABLE, drift_detected=False, drift_score=0.0
        )
        with (
            patch.object(mon.detector, "update", return_value=calm),
            patch.object(mon.trainer, "outer_cl_training_loop", return_value=0) as loop,
        ):
            mon._decision_point()
        assert mon.fire_count == 0
        loop.assert_not_called()


class TestEvaluateBatch:
    def test_returns_metrics(self, default_cfg, dummy_harness):
        mon = _monitor(default_cfg, dummy_harness)
        batch = (torch.randn(4, 4), torch.randint(0, 3, (4,)))
        metrics = mon._evaluate_batch(batch)
        assert isinstance(metrics, list)
        assert len(metrics) == 1  # one eval metric (accuracy)
