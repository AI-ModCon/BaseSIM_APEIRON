"""Continuous drift monitoring system.

This module implements a continuous monitoring architecture that:
- Maintains detector state across data stream updates
- Processes batches continuously until drift is detected
- Pauses for learning when drift is detected
- Resumes monitoring with updated model weights
- Automatically extends the data stream when exhausted
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import numpy as np

from config.configuration import Config
from drift_detection.load_drift_detector import load_drift_detector
from drift_detection.detectors.base import DriftSignal
from training.continual_learning import continual_learning_loop

if TYPE_CHECKING:
    from model.torch_model_harness import BaseModelHarness


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
        global_step: Global training step counter
        stream_update_count: Number of times stream has been extended
        batch_count: Total number of batches processed
        metric_buffer: Buffer for accumulating metrics between checks
    """

    def __init__(
        self,
        cfg: Config,
        modelHarness: BaseModelHarness,
        logger,
    ):
        """Initialize continuous monitor.

        Args:
            cfg: Configuration object
            modelHarness: Model harness containing model and data loaders
            logger: Logger for metrics
        """
        self.cfg = cfg
        self.modelHarness = modelHarness
        self.logger = logger

        # Create persistent detector instance
        self.detector = load_drift_detector(cfg)

        # Configuration
        self.metric_idx = cfg.drift_detection.metric_index
        self.detection_interval = cfg.drift_detection.detection_interval
        self.max_stream_updates = cfg.drift_detection.max_stream_updates
        self.aggregation = cfg.drift_detection.aggregation

        # State tracking
        self.global_step = 0
        self.stream_update_count = 0
        self.batch_count = 0

        # Metrics accumulation
        self.metric_buffer: list[list[float]] = []

        print("ContinuousMonitor initialized:")
        print(f"  Detector: {cfg.drift_detection.detector_name}")
        print(f"  Monitoring metric index: {self.metric_idx}")
        print(f"  Detection interval: {self.detection_interval} batches")
        print(f"  Aggregation method: {self.aggregation}")
        print(f"  Max stream updates: {self.max_stream_updates}")

    def run(self) -> None:
        """Main continuous monitoring loop.

        This loop continues until max_stream_updates is reached. It processes
        batches from the data stream, checks for drift at regular intervals,
        and dispatches learning modules when drift is detected.
        """
        print("\n" + "=" * 60)
        print("Starting Continuous Monitoring")
        print("=" * 60 + "\n")

        # Initialize first data stream
        print("Initializing first data stream...")
        self.modelHarness.update_data_stream()

        while not self._should_stop():
            try:
                self._process_stream()
            except StopIteration:
                # Stream exhausted, extend it
                self._extend_stream()

        print("\n" + "=" * 60)
        print("Continuous Monitoring Complete")
        print(f"Total batches processed: {self.batch_count}")
        print(f"Total stream updates: {self.stream_update_count}")
        print("=" * 60 + "\n")

    def _process_stream(self) -> None:
        """Process batches from current data stream.

        Iterates through the current data loader, evaluating batches and
        checking for drift at regular intervals. When drift is detected,
        pauses monitoring and dispatches the learning module.

        Raises:
            StopIteration: When the data loader is exhausted
        """
        train_loader, _ = self.modelHarness.get_cur_data_loaders()

        for batch_idx, batch in enumerate(train_loader):
            # Evaluate batch and compute all metrics
            metrics = self._evaluate_batch(batch)
            self.metric_buffer.append(metrics)
            self.batch_count += 1

            # Check drift at specified interval
            if self.batch_count % self.detection_interval == 0:
                drift_signal = self._check_drift()

                if drift_signal.drift_detected:
                    print(f"\n{'!' * 60}")
                    print("DRIFT DETECTED")
                    print(f"{'!' * 60}\n")
                    self._handle_drift(drift_signal)

        # Stream exhausted - check drift one last time if we have buffered metrics
        # This ensures we don't miss drift when stream has fewer than detection_interval batches
        if self.metric_buffer:
            drift_signal = self._check_drift()
            if drift_signal.drift_detected:
                print(f"\n{'!' * 60}")
                print("DRIFT DETECTED")
                print(f"{'!' * 60}\n")
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

        with torch.no_grad():
            x, y = self.modelHarness._unpack(batch)
            x, y = x.to(self.cfg.device), y.to(self.cfg.device)

            # Forward pass
            y_hat = self.modelHarness.model(x)

            # Compute all metrics
            metrics = []
            for metric_fn in self.modelHarness.eval_metrics:
                value = self.modelHarness._to_scalar(metric_fn(y_hat, y))
                metrics.append(value)

        return metrics

    def _check_drift(self) -> DriftSignal:
        """Aggregate buffered metrics and check for drift.

        Takes the buffered metrics, aggregates them according to the
        configured aggregation method, extracts the monitored metric,
        and updates the detector.

        Returns:
            DriftSignal from the detector
        """
        if not self.metric_buffer:
            # Edge case: no metrics buffered
            print("Warning: Empty metric buffer, skipping drift check")
            return DriftSignal(
                regime=(
                    self.detector._get_regime(0.0)
                    if hasattr(self.detector, "_get_regime")
                    else None
                ),
                drift_detected=False,
                drift_score=0.0,
                confidence=None,
            )

        # Extract the monitored metric from all buffered metrics
        metric_values = [m[self.metric_idx] for m in self.metric_buffer]

        # Aggregate according to configured method
        if self.aggregation == "mean":
            agg_metric = float(np.mean(metric_values))
        elif self.aggregation == "median":
            agg_metric = float(np.median(metric_values))
        elif self.aggregation == "last":
            agg_metric = float(metric_values[-1])
        else:
            # Default to mean
            agg_metric = float(np.mean(metric_values))

        # Clear buffer
        self.metric_buffer = []

        # Update detector with aggregated metric
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
        print(f"Drift Score: {drift_signal.drift_score:.4f}")
        print(f"Regime: {drift_signal.regime.value if drift_signal.regime else 'N/A'}")
        print(
            f"Confidence: {drift_signal.confidence if drift_signal.confidence else 'N/A'}"
        )
        print(f"Global Step: {self.global_step}")
        print("\nDispatching continual learning module...")

        # PAUSE monitoring, dispatch learning module
        continual_learning_loop(
            cfg=self.cfg,
            modelHarness=self.modelHarness,  # Model weights will be updated
            logger=self.logger,
            global_step=self.global_step,
            basic_only=False,
        )

        # Update global step (add 1 extra to avoid step conflicts with CL loop's final log)
        self.global_step += self.cfg.continuous_learning.max_iter + 1

        print(f"Continual learning complete. New global step: {self.global_step}")

        # Optionally reset detector after learning
        if self.cfg.drift_detection.reset_after_learning:
            print("Resetting detector state...")
            self.detector.reset()

        print(f"\n{'!' * 60}")
        print("RESUMING MONITORING")
        print(f"{'!' * 60}\n")

    def _extend_stream(self) -> None:
        """Extend the data stream when exhausted.

        Calls update_data_stream() to load the next buffer of data.
        This does NOT necessarily mean drift occurred - drift is detected
        by the statistical detector based on metric changes.
        """
        self.stream_update_count += 1

        print(f"\n{'-' * 60}")
        print("Stream exhausted. Loading next data buffer...")
        print(
            f"Stream update count: {self.stream_update_count}/{self.max_stream_updates}"
        )
        print(f"{'-' * 60}\n")

        # Load next data buffer
        self.modelHarness.update_data_stream()

        # Log stream update at current global_step (will be committed with next drift check)
        self.logger.log(
            {"monitor/stream_update": self.stream_update_count},
            step=self.global_step,
            commit=False,
        )

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
        self.logger.log(
            {"drift/detected": drift_signal.drift_detected},
            step=self.global_step,
            commit=False,
        )
        self.logger.log(
            {"drift/score": drift_signal.drift_score},
            step=self.global_step,
            commit=False,
        )

        if drift_signal.regime:
            self.logger.log(
                {"drift/regime": drift_signal.regime.value},
                step=self.global_step,
                commit=False,
            )

        if drift_signal.confidence is not None:
            self.logger.log(
                {"drift/confidence": drift_signal.confidence},
                step=self.global_step,
                commit=False,
            )

        # Log the monitored metric value
        self.logger.log(
            {f"drift/metric_{self.metric_idx}": metric_value},
            step=self.global_step,
            commit=False,
        )

        # Log batch count
        self.logger.log(
            {"monitor/batch_count": self.batch_count},
            step=self.global_step,
            commit=True,  # Commit all pending logs
        )
