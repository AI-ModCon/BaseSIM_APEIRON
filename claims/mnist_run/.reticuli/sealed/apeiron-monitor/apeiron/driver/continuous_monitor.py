"""Continuous drift monitoring system.

This module implements a continuous monitoring architecture that:
- Maintains detector state across data stream updates
- Processes batches continuously until drift is detected
- Pauses for learning when drift is detected
- Resumes monitoring with updated model weights
- Automatically extends the data stream when exhausted
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
import numpy as np

from apeiron.config.configuration import Config
from apeiron.drift_detection.load_drift_detector import load_drift_detector
from apeiron.drift_detection.detectors.base import DriftSignal
from apeiron.profilers import FLOPSProfiler
from apeiron.logger import get_logger
from apeiron.training import ContinuousTrainer
from tqdm import tqdm

if TYPE_CHECKING:
    from apeiron.model.torch_model_harness import BaseModelHarness


class ContinuousMonitor:
    """Continuous drift monitoring system.

    This class manages the continuous monitoring of a data stream, detecting
    drift and dispatching learning modules as needed.

    Attributes:
        cfg: Configuration object
        modelHarness: Model harness containing model and data loaders
        logger: Logger for metrics
        detector: Persistent drift detector instance
        metric_idx: Index of metric to monitor
        detection_interval: Number of batches between drift checks
        max_stream_updates: Maximum number of stream extensions
        stream_update_count: Number of times stream has been extended
        batch_count: Total number of batches processed
        metric_buffer: Buffer for accumulating metrics between checks
    """

    def __init__(
        self,
        cfg: Config,
        modelHarness: BaseModelHarness,
    ):
        """Initialize continuous monitor.

        Args:
            cfg: Configuration object
            modelHarness: Model harness containing model and data loaders
            logger: Logger for metrics
        """
        self.cfg = cfg
        self.modelHarness = modelHarness
        self.logger = get_logger()

        # Create performance profiler
        self.flops_profiler = FLOPSProfiler()

        # Configuration
        self.metric_idx = cfg.drift_detection.metric_index
        self.detection_interval = cfg.drift_detection.detection_interval
        self.max_stream_updates = cfg.drift_detection.max_stream_updates
        self.aggregation = cfg.drift_detection.aggregation

        # What this loop calls itself in its own log lines; a subclass that
        # monitors without adapting says something else.
        self.label = "Continuous Monitoring"

        # State tracking
        self.stream_update_count = 0
        self.batch_count = 0
        self.check_count = 0
        self.drift_event_count = 0
        # The aggregate the last drift check fed the detector. Kept so a
        # subclass handling a detection can record WHAT fired without
        # re-deriving it or re-plumbing _check_drift's return type.
        self.last_agg_metric: float | None = None

        # Metrics accumulation
        self.metric_buffer: list[list[float]] = []

        self.logger.info("==== ContinuousMonitor initialized ====", level=0)
        self.logger.info(f"\tDetector: {cfg.drift_detection.detector_name}", level=1)
        self.logger.info(f"\tMonitoring metric index: {self.metric_idx}", level=1)
        self.logger.info(
            f"\tDetection interval: {self.detection_interval} batches", level=1
        )
        self.logger.info(f"\tAggregation method: {self.aggregation}", level=1)
        self.logger.info(f"\tMax stream updates: {self.max_stream_updates}", level=1)

    @functools.cached_property
    def detector(self):
        """The drift detector, built on first use.

        Lazy for the same reason as the trainer: a subclass that adapts on a
        schedule bypasses detection entirely, and should not need a valid
        ``[drift_detection] detector_name`` to run.
        """
        return load_drift_detector(self.cfg)

    @functools.cached_property
    def trainer(self) -> ContinuousTrainer:
        """The continual-learning trainer, built on first use.

        Lazy because a drift-only run never adapts: building a trainer would
        construct an optimizer over the model for nothing, and would oblige a
        frozen-model harness to supply one.
        """
        return ContinuousTrainer(
            cfg=self.cfg,
            modelHarness=self.modelHarness,
            logger=self.logger,
            profiler=self.flops_profiler,
        )

    def run(self) -> None:
        """Main continuous monitoring loop.

        This loop continues until max_stream_updates is reached. It processes
        batches from the data stream, checks for drift at regular intervals,
        and dispatches learning modules when drift is detected.
        """
        self.logger.info(f"==== Starting {self.label} ====", level=0)

        # Initialize first data stream
        self.logger.info("\tInitializing first data stream...", level=1)
        self.modelHarness.update_data_stream()

        while not self._should_stop():
            try:
                self._process_stream()
            except StopIteration:
                # Stream exhausted, extend it
                self._extend_stream()

        self.logger.info(f"==== {self.label} Complete ====", level=0)
        self.logger.info(f"\tTotal batches processed: {self.batch_count}", level=1)
        self.logger.info(f"\tTotal stream updates: {self.stream_update_count}", level=1)

    def _process_stream(self) -> None:
        """Process batches from current data stream.

        Iterates through the current data loader, evaluating batches and
        checking for drift at regular intervals. When drift is detected,
        pauses monitoring and dispatches the learning module.

        Raises:
            StopIteration: When the data loader is exhausted
        """
        val_loader = self.modelHarness.get_stream_dataloader()

        for batch_idx, batch in tqdm(
            enumerate(val_loader),
            desc="Processing batches",
            leave=False,
        ):
            # Evaluate batch and compute all metrics
            metrics = self._evaluate_batch(batch)
            self.metric_buffer.append(metrics)
            self.batch_count += 1

            # Check drift at specified interval
            if (
                self.detection_interval > 0
                and self.batch_count % self.detection_interval == 0
            ):
                drift_signal = self._check_drift()

                if drift_signal.drift_detected:
                    self._handle_drift(drift_signal)

        raise StopIteration()

    def _evaluate_batch(self, batch: tuple[torch.Tensor, torch.Tensor]) -> list[float]:
        """Evaluate model on a single batch and compute all metrics.

        Args:
            batch: Tuple of (inputs, targets)

        Returns:
            List of metric values (one per eval_metric)
        """
        self.modelHarness.model.eval()

        if self.batch_count > self.flops_profiler.warmup_iters:
            # Profile inference after warmup
            with self.flops_profiler.measure_flops(tag="infer"):
                with torch.no_grad():
                    x, y = self.modelHarness._unpack(batch)
                    x, y = x.to(self.cfg.device), y.to(self.cfg.device)

                    # Forward pass
                    y_hat = self.modelHarness.model(x)

                    # Compute all metrics
                    metrics = []
                    eval_metrics_log = {}
                    for key, metric_fn in self.modelHarness.eval_metrics.items():
                        value = self.modelHarness._to_scalar(metric_fn(y_hat, y))
                        metrics.append(value)
                        eval_metrics_log[key] = value

                    # Log all eval metrics in one call
                    self.logger.stage("eval")
                    self.logger.log(eval_metrics_log)
        else:
            # Skip profiling during warmup
            with torch.no_grad():
                x, y = self.modelHarness._unpack(batch)
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)

                # Forward pass
                y_hat = self.modelHarness.model(x)

                # Compute all metrics
                metrics = []
                for key, metric_fn in self.modelHarness.eval_metrics.items():
                    value = self.modelHarness._to_scalar(metric_fn(y_hat, y))
                    metrics.append(value)

        return metrics

    def _aggregate_buffer(self) -> float:
        """Reduce the buffered per-batch metrics to the monitored scalar.

        Clears the buffer. Separate from :meth:`_check_drift` because a
        subclass that decides on a schedule still needs the window's
        aggregate -- logged at every decision point so a schedule run's
        metric trace stays comparable to a detector run's -- without
        consulting a detector to get it.
        """
        if not self.metric_buffer:
            # Edge case: no metrics buffered
            raise RuntimeError(
                "Model Harness requires evaluation metrics"
            )  # Todo: This should be checked in model harness

        metric_values = [m[self.metric_idx] for m in self.metric_buffer]
        self.metric_buffer = []

        if self.aggregation == "median":
            return float(np.median(metric_values))
        if self.aggregation == "last":
            return float(metric_values[-1])
        # mean, and the fallback for an unrecognised method
        return float(np.mean(metric_values))

    def _check_drift(self) -> DriftSignal:
        """Aggregate buffered metrics and check for drift.

        Takes the buffered metrics, aggregates them according to the
        configured aggregation method, extracts the monitored metric,
        and updates the detector.

        Returns:
            DriftSignal from the detector
        """
        agg_metric = self._aggregate_buffer()
        self.check_count += 1
        self.last_agg_metric = agg_metric

        # Update detector with aggregated metric
        # NOTE: Profiler only covers Pytorch operations
        # Even so, we measure runtime to see if it's a potential bottleneck
        if self.batch_count > self.flops_profiler.warmup_iters:
            with self.flops_profiler.measure_flops(tag="detector"):
                drift_signal = self.detector.update(agg_metric)
        else:
            drift_signal = self.detector.update(agg_metric)

        # Log drift metrics
        self._log_metrics(drift_signal, agg_metric)

        return drift_signal

    def _handle_drift(self, drift_signal: DriftSignal) -> None:
        """Handle detected drift by dispatching learning module.

        PAUSES monitoring, dispatches the continual learning loop,
        then RESUMES monitoring with updated model weights.

        Args:
            drift_signal: The drift signal from the detector
        """
        self.drift_event_count += 1
        self.logger.info(
            f"==== DRIFT DETECTED (Event #{self.drift_event_count})! ====", level=0
        )
        # Log data timestamp range if the harness tracks it
        timerange = getattr(self.modelHarness, "current_window_timerange", None)
        if timerange is not None:
            self.logger.info(
                f"\tData time range: {timerange[0]} → {timerange[1]}", level=1
            )
        self.logger.info(
            f"\tRegime: {drift_signal.regime.value if drift_signal.regime else 'N/A'}",
            level=1,
        )
        self.logger.info(f"\tDrift Score: {drift_signal.drift_score:.4f}", level=1)
        self.logger.info(
            f"\tConfidence: {drift_signal.confidence if drift_signal.confidence else 'N/A'}",
            level=1,
        )

        self._adapt(self.drift_event_count)

        # Optionally reset detector after learning
        if self.cfg.drift_detection.reset_after_learning:
            self.logger.debug("Resetting detector state...")
            self.detector.reset()

        self.logger.info("==== RESUMING MONITORING! ====", level=0)

    def _adapt(self, event_id: int) -> None:
        """Pause monitoring, run the CL loop, checkpoint, and resume.

        The adaptation itself, with nothing about *why* it was triggered: a
        subclass that adapts on a schedule rather than on a detection calls
        this with its own event numbering.
        """
        # Log profiler performance summary
        self.flops_profiler.print_performance(logger=self.logger, level=2)

        self.logger.info("-> Dispatching continual learning module...", level=0)

        self.trainer.outer_cl_training_loop(drift_event_id=event_id)

        if self.modelHarness.ckpts_enabled:
            ckptpath = self.modelHarness.save_ckpt(event=event_id)
            self.logger.info(f"* Checkpoint saved to: {ckptpath}", level=0)

        self.logger.info("<- Continual learning complete.", level=0)

    def _extend_stream(self) -> None:
        """Extend the data stream when exhausted.

        Calls update_data_stream() to load the next buffer of data.
        This does NOT necessarily mean drift occurred - drift is detected
        by the statistical detector based on metric changes.
        """
        self.stream_update_count += 1

        self.logger.info(
            f"\tStream exhausted. Loading next data buffer. {self.stream_update_count}/{self.max_stream_updates}",
            level=1,
        )

        # Load next data buffer
        self.modelHarness.update_data_stream()

    def _should_stop(self) -> bool:
        """Check if monitoring should stop.

        Returns:
            True if max_stream_updates has been reached
        """
        return self.stream_update_count >= self.max_stream_updates

    def _log_metrics(self, drift_signal: DriftSignal, metric_value: float) -> None:
        """Log drift detection metrics.

        Args:
            drift_signal: The drift signal from the detector
            metric_value: The aggregated metric value
        """
        flops_perf = self.flops_profiler.get_performance()

        # Log all drift metrics in a single call.
        # detected=0 is sampled at 10% to reduce log volume; detected=1 is
        # always included so no true drift events are dropped.
        log_detected = drift_signal.drift_detected or (np.random.random() <= 0.1)
        self.logger.stage("drift")
        # Include data timestamp range if available from the harness
        timerange = getattr(self.modelHarness, "current_window_timerange", None)
        ts_fields = {}
        if timerange is not None:
            ts_fields["data_time_start"] = timerange[0]
            ts_fields["data_time_end"] = timerange[1]
        self.logger.log(
            {
                **(
                    {"detected": int(drift_signal.drift_detected)}
                    if log_detected
                    else {}
                ),
                "score": drift_signal.drift_score,
                "regime": (drift_signal.regime.value if drift_signal.regime else "N/A"),
                "confidence": (
                    drift_signal.confidence if drift_signal.confidence else "N/A"
                ),
                f"metric_{self.metric_idx}": metric_value,
                **ts_fields,
                **{f"cperf_{k}": v for k, v in flops_perf.items()},
            },
        )
