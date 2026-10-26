"""Tests for src/apeiron/driver/continuous_monitor.py"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch, MagicMock

import pytest

from apeiron.drift_detection.detectors.base import DriftSignal, LearningRegime
from apeiron.driver.continuous_monitor import ContinuousMonitor


# We patch get_logger globally for these tests since ContinuousMonitor calls it in __init__
@pytest.fixture(autouse=True)
def _patch_logger():
    mock_logger = MagicMock()
    mock_logger.step = 0
    with patch(
        "apeiron.driver.continuous_monitor.get_logger", return_value=mock_logger
    ):
        with patch(
            "apeiron.training.continuous_trainer.get_logger", return_value=mock_logger
        ):
            yield mock_logger


class TestContinuousMonitorInit:
    def test_initialization(self, default_cfg, dummy_harness):
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        assert mon.stream_update_count == 0
        assert mon.batch_count == 0
        assert mon.drift_event_count == 0
        assert mon.metric_buffer == []
        assert mon.detection_interval == default_cfg.drift_detection.detection_interval
        assert mon.max_stream_updates == default_cfg.drift_detection.max_stream_updates


class TestShouldStop:
    def test_false_initially(self, default_cfg, dummy_harness):
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        assert mon._should_stop() is False

    def test_true_at_max(self, default_cfg, dummy_harness):
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        mon.stream_update_count = default_cfg.drift_detection.max_stream_updates
        assert mon._should_stop() is True


class TestExtendStream:
    def test_increments_counter(self, default_cfg, dummy_harness):
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        assert mon.stream_update_count == 0
        with patch.object(dummy_harness, "update_data_stream") as mock_update:
            mon._extend_stream()
        assert mon.stream_update_count == 1
        mock_update.assert_called_once()


class TestCheckDrift:
    def test_raises_on_empty_buffer(self, default_cfg, dummy_harness):
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        with pytest.raises(RuntimeError, match="requires evaluation metrics"):
            mon._check_drift()

    def test_mean_aggregation(self, default_cfg, dummy_harness):
        cfg = replace(
            default_cfg,
            drift_detection=replace(default_cfg.drift_detection, aggregation="mean"),
        )
        mon = ContinuousMonitor(cfg=cfg, modelHarness=dummy_harness)
        mon.metric_buffer = [[90.0], [70.0], [60.0]]
        mock_signal = DriftSignal(
            regime=LearningRegime.STABLE,
            drift_detected=False,
            drift_score=0.0,
        )
        with patch.object(
            mon.detector, "update", return_value=mock_signal
        ) as mock_update:
            signal = mon._check_drift()
        agg_value = mock_update.call_args[0][0]
        assert agg_value == pytest.approx(73.33, abs=0.01)
        assert mon.metric_buffer == []
        assert signal is mock_signal

    def test_last_aggregation(self, default_cfg, dummy_harness):
        cfg = replace(
            default_cfg,
            drift_detection=replace(default_cfg.drift_detection, aggregation="last"),
        )
        mon = ContinuousMonitor(cfg=cfg, modelHarness=dummy_harness)
        mon.metric_buffer = [[90.0], [70.0], [60.0]]
        mock_signal = DriftSignal(
            regime=LearningRegime.STABLE,
            drift_detected=False,
            drift_score=0.0,
        )
        with patch.object(
            mon.detector, "update", return_value=mock_signal
        ) as mock_update:
            signal = mon._check_drift()
        agg_value = mock_update.call_args[0][0]
        assert agg_value == 60.0
        assert mon.metric_buffer == []
        assert signal is mock_signal

    def test_median_aggregation(self, default_cfg, dummy_harness):
        cfg = replace(
            default_cfg,
            drift_detection=replace(default_cfg.drift_detection, aggregation="median"),
        )
        mon = ContinuousMonitor(cfg=cfg, modelHarness=dummy_harness)
        mon.metric_buffer = [[90.0], [70.0], [60.0]]
        mock_signal = DriftSignal(
            regime=LearningRegime.STABLE,
            drift_detected=False,
            drift_score=0.0,
        )
        with patch.object(
            mon.detector, "update", return_value=mock_signal
        ) as mock_update:
            signal = mon._check_drift()
        agg_value = mock_update.call_args[0][0]
        assert agg_value == 70.0
        assert mon.metric_buffer == []
        assert signal is mock_signal


class TestHandleDrift:
    def test_increments_drift_event_count(self, default_cfg, dummy_harness):
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        signal = DriftSignal(
            regime=LearningRegime.CONTINUAL_LEARNING,
            drift_detected=True,
            drift_score=0.5,
        )
        with patch.object(mon.trainer, "outer_cl_training_loop", return_value=0):
            mon._handle_drift(signal)
        assert mon.drift_event_count == 1


class TestEvaluateBatch:
    def test_returns_metrics(self, default_cfg, dummy_harness):
        import torch

        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        batch = (torch.randn(4, 4), torch.randint(0, 3, (4,)))
        metrics = mon._evaluate_batch(batch)
        assert isinstance(metrics, list)
        assert len(metrics) == 1  # one eval metric (accuracy)
        # TODO: assert on the actual metric values, not just shape


class TestRun:
    """The monitoring loop itself.

    `run()` is what the whole framework exists to execute, and no test in this
    layer drove it: the three test files called `_check_drift`,
    `_handle_drift` and the constructor, so a regrowth of this layer omitted
    the loop entirely and passed (claims/LOG.md, A29). It was pinned only one
    layer up, by the entry points. The layer that owns a behaviour should be
    the layer that pins it.
    """

    def _cfg(self, default_cfg, **drift):
        from apeiron.config.configuration import DriftDetectionCfg

        return replace(default_cfg, drift_detection=DriftDetectionCfg(**drift))

    def test_the_loop_runs_to_its_stream_budget(self, default_cfg, dummy_harness):
        cfg = self._cfg(default_cfg, detection_interval=2, max_stream_updates=3)
        mon = ContinuousMonitor(cfg=cfg, modelHarness=dummy_harness)
        mon.run()

        # the loop ends by exhausting its budget, not by running out of data:
        # each exhausted window extends the stream and counts
        assert mon.stream_update_count == 3
        assert mon.batch_count > 0
        assert mon._should_stop() is True

    def test_the_loop_evaluates_every_batch_it_sees(self, default_cfg, dummy_harness):
        cfg = self._cfg(default_cfg, detection_interval=2, max_stream_updates=2)
        mon = ContinuousMonitor(cfg=cfg, modelHarness=dummy_harness)
        with patch.object(
            mon, "_evaluate_batch", wraps=mon._evaluate_batch
        ) as evaluated:
            mon.run()
        assert evaluated.call_count == mon.batch_count

    def test_the_loop_checks_drift_on_the_interval(self, default_cfg, dummy_harness):
        interval = 2
        cfg = self._cfg(default_cfg, detection_interval=interval, max_stream_updates=2)
        mon = ContinuousMonitor(cfg=cfg, modelHarness=dummy_harness)
        mon.run()
        # one check per `interval` batches, and the buffer is left empty
        assert mon.check_count == mon.batch_count // interval
        assert mon.metric_buffer == []

    def test_a_zero_interval_disables_checking(self, default_cfg, dummy_harness):
        cfg = self._cfg(default_cfg, detection_interval=0, max_stream_updates=2)
        mon = ContinuousMonitor(cfg=cfg, modelHarness=dummy_harness)
        mon.run()
        assert mon.batch_count > 0
        assert mon.check_count == 0
        assert mon.drift_event_count == 0

    def test_the_loop_extends_the_stream_for_each_window(
        self, default_cfg, dummy_harness
    ):
        cfg = self._cfg(default_cfg, detection_interval=2, max_stream_updates=3)
        mon = ContinuousMonitor(cfg=cfg, modelHarness=dummy_harness)
        with patch.object(
            dummy_harness, "update_data_stream", wraps=dummy_harness.update_data_stream
        ) as extended:
            mon.run()
        # once to prime the first window, then once per exhausted window
        assert extended.call_count == mon.stream_update_count + 1

    def test_a_detection_during_the_loop_dispatches_adaptation(
        self, default_cfg, dummy_harness
    ):
        cfg = self._cfg(default_cfg, detection_interval=2, max_stream_updates=2)
        mon = ContinuousMonitor(cfg=cfg, modelHarness=dummy_harness)
        fired = DriftSignal(
            regime=LearningRegime.CONTINUAL_LEARNING,
            drift_detected=True,
            drift_score=1.0,
        )
        with patch.object(mon, "_check_drift", return_value=fired):
            with patch.object(mon, "_handle_drift") as handled:
                mon.run()
        assert handled.call_count == mon.batch_count // 2
        assert handled.call_args.args[0] is fired

    def test_a_quiet_stream_never_adapts(self, default_cfg, dummy_harness):
        cfg = self._cfg(default_cfg, detection_interval=2, max_stream_updates=2)
        mon = ContinuousMonitor(cfg=cfg, modelHarness=dummy_harness)
        quiet = DriftSignal(
            regime=LearningRegime.STABLE, drift_detected=False, drift_score=0.0
        )
        with patch.object(mon, "_check_drift", return_value=quiet):
            with patch.object(mon, "_handle_drift") as handled:
                mon.run()
        handled.assert_not_called()
        assert mon.drift_event_count == 0


class TestLazyConstruction:
    """The detector and the trainer are built on first use, not on init.

    This is what lets the two side runners inherit this class: a drift-only
    run never adapts, so building a trainer would construct an optimizer over
    the model for nothing and oblige a frozen-model harness to supply one; a
    scheduled run never consults a detector, so it should not need a valid
    `detector_name`. Both are `cached_property`, so the saving is real and the
    object is built at most once.
    """

    def test_init_builds_no_trainer(self, default_cfg, dummy_harness):
        with patch.object(dummy_harness, "get_optimizer") as optimizer:
            ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        optimizer.assert_not_called()

    def test_init_builds_no_detector(self, default_cfg, dummy_harness):
        with patch("apeiron.driver.continuous_monitor.load_drift_detector") as loader:
            ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        loader.assert_not_called()

    def test_the_trainer_is_built_once_on_demand(self, default_cfg, dummy_harness):
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        assert mon.trainer is mon.trainer

    def test_the_detector_is_built_once_on_demand(self, default_cfg, dummy_harness):
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        with patch("apeiron.driver.continuous_monitor.load_drift_detector") as loader:
            first = mon.detector
            second = mon.detector
        assert first is second
        loader.assert_called_once_with(default_cfg)
