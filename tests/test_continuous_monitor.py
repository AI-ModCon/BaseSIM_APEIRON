"""Tests for src/driver/continuous_monitor.py"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch, MagicMock

import pytest

from apeiron.config.configuration import EvalCfg
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
        assert mon.drift_check_count == 0
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

    def test_drift_detection_counter_increments(self, default_cfg, dummy_harness):
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        mon.metric_buffer = [[0.1], [0.2]]
        mock_signal = DriftSignal(
            regime=LearningRegime.CONTINUAL_LEARNING,
            drift_detected=True,
            drift_score=0.5,
        )
        with patch.object(mon.detector, "update", return_value=mock_signal):
            _ = mon._check_drift()
        assert mon.drift_check_count == 1


class TestRunSummary:
    def test_logs_no_drift_summary_message(self, default_cfg, dummy_harness):
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        with patch.object(mon, "_should_stop", side_effect=[True]):
            mon.run()
        mon.logger.info.assert_any_call(
            "CL dispatch did not run because no drift was detected.",
            level=0,
        )


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


class TestMaxValBatches:
    """``eval.max_val_batches`` caps the per-window evaluation.

    Models whose validation pass costs far more than the drift decision it feeds
    need to stop early; ``0`` keeps the historical behaviour of consuming the
    whole loader.
    """

    def _run_window(self, default_cfg, dummy_harness, cap):
        """Drive one window and report how many batches it consumed.

        ``dummy_harness.get_stream_dataloader`` returns a 1-tuple, which
        ``_process_stream`` does not expect, so point it at a real loader.
        """
        from torch.utils.data import DataLoader

        cfg = replace(default_cfg, eval=EvalCfg(max_val_batches=cap))
        mon = ContinuousMonitor(cfg=cfg, modelHarness=dummy_harness)
        loader = DataLoader(dummy_harness._val_ds, batch_size=4)
        with patch.object(dummy_harness, "get_stream_dataloader", return_value=loader):
            with pytest.raises(StopIteration):
                mon._process_stream()
        return mon, len(loader)

    def test_cap_stops_the_window_early(self, default_cfg, dummy_harness):
        mon, total = self._run_window(default_cfg, dummy_harness, cap=2)
        assert mon.batch_count == 2
        assert total > 2, "fixture must have more batches than the cap to be a test"

    def test_zero_consumes_the_whole_loader(self, default_cfg, dummy_harness):
        mon, total = self._run_window(default_cfg, dummy_harness, cap=0)
        assert mon.batch_count == total

    def test_absent_eval_section_is_uncapped(self, default_cfg, dummy_harness):
        assert default_cfg.eval is None
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        assert mon.max_val_batches == 0
