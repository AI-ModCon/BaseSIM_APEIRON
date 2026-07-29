"""Detection-only monitoring: run a drift detector without any adaptation.

Streams data past a frozen model, feeds the monitored metric to the configured
detector, and records every time it fires without triggering continual learning.
The weights that go in are the weights that come out. Drift is checked every
``drift_detection.detection_interval`` batches, so the metric/score trace lines
up batch-for-batch with a full run.

A ``main.py`` run with ``[continual_learning] update_mode = "none"`` also leaves
the weights untouched, but it still dispatches the whole CL path on every
detection -- iterating the train and historic loaders, stepping an optimizer
that has no gradients to apply, and checkpointing -- so use it to measure that
pipeline with training made inert, and this file when the detection trace is the
point.

Every check lands in the run's metrics CSV (``[visualization] input``) under the
``drift/`` stage -- detected, score, regime, confidence, the monitored metric,
check_idx and stream_idx -- so the full detection trace is already on disk and
this file writes no output of its own.

Examples
--------
    python -m src.drift_only --config examples/aeris/aeris.toml

Takes the same flags as ``main.py``: ``--config``, ``--set key=val``,
``--device``, ``--multi-gpu``.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from apeiron.config.configuration import build_config, Config
from apeiron.drift_detection.detectors.base import DriftSignal
from apeiron.drift_detection.load_drift_detector import load_drift_detector
from apeiron.logger import get_logger, configure_backend
from apeiron.model.torch_model_harness import BaseModelHarness
from apeiron.profilers import FLOPSProfiler

from examples.utils import get_example


class DriftOnlyMonitor:
    """Stream monitor that detects drift and then does nothing about it.

    Mirrors ``ContinuousMonitor``'s loop -- per-batch evaluation, interval
    drift checks, the same aggregation, the same FLOPs profiling, the same
    logged field names -- with the ``_handle_drift`` branch replaced by
    recording the event. No trainer is constructed and no checkpoint is
    written, since the weights never change.
    """

    def __init__(
        self,
        cfg: Config,
        modelHarness: BaseModelHarness,
    ) -> None:
        self.cfg = cfg
        self.modelHarness = modelHarness
        # No learning happens here, so this reads as "re-arm after a detection".
        # Sharing the key with ContinuousMonitor keeps one TOML driving both.
        self.reset_on_detect = cfg.drift_detection.reset_after_learning
        self.logger = get_logger()

        self.detector = load_drift_detector(cfg)
        self.flops_profiler = FLOPSProfiler()

        self.metric_idx = cfg.drift_detection.metric_index
        self.detection_interval = cfg.drift_detection.detection_interval
        self.max_stream_updates = cfg.drift_detection.max_stream_updates
        self.aggregation = cfg.drift_detection.aggregation

        # State
        self.stream_update_count = 0
        self.batch_count = 0
        self.check_count = 0
        self.drift_event_count = 0
        self.events: list[dict[str, Any]] = []
        self.metric_buffer: list[list[float]] = []

        self.logger.info("==== DriftOnlyMonitor initialized ====", level=0)
        self.logger.info(f"\tDetector: {cfg.drift_detection.detector_name}", level=1)
        self.logger.info("\tContinual learning: BYPASSED (model stays frozen)", level=1)
        self.logger.info(f"\tMonitoring metric index: {self.metric_idx}", level=1)
        self.logger.info(
            f"\tDetection interval: {self.detection_interval} batches", level=1
        )
        self.logger.info(f"\tAggregation method: {self.aggregation}", level=1)
        self.logger.info(f"\tMax stream updates: {self.max_stream_updates}", level=1)
        self.logger.info(
            f"\tReset detector on detection: {self.reset_on_detect}", level=1
        )

    # -- main loop ---------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the detection-only loop and return a run summary."""
        self.logger.info("==== Starting Drift-Only Monitoring ====", level=0)
        self.logger.info("\tInitializing first data stream...", level=1)
        self.modelHarness.update_data_stream()

        while self.stream_update_count < self.max_stream_updates:
            try:
                self._process_stream()
            except StopIteration:
                self._extend_stream()

        self.logger.info("==== Drift-Only Monitoring Complete ====", level=0)
        self.logger.info(f"\tTotal batches processed: {self.batch_count}", level=1)
        self.logger.info(f"\tTotal stream updates: {self.stream_update_count}", level=1)
        self.logger.info(f"\tDrift checks: {self.check_count}", level=1)
        self.logger.info(f"\tDrift events: {self.drift_event_count}", level=1)

        self.flops_profiler.print_performance(logger=self.logger, level=1)

        final_metrics = self.modelHarness.eval()
        self.logger.info(f"\tFinal eval metrics: {final_metrics}", level=1)

        return {
            "detector": self.cfg.drift_detection.detector_name,
            "batches": self.batch_count,
            "stream_updates": self.stream_update_count,
            "drift_checks": self.check_count,
            "drift_events": self.drift_event_count,
            "drift_windows": self.drift_windows(),
            "events": list(self.events),
            "final_metrics": final_metrics,
        }

    def drift_windows(self) -> list[int]:
        """Stream windows in which drift fired, deduplicated and sorted.

        Coarser than the per-check event list: several detections inside one
        window collapse to a single entry, so this counts *windows the detector
        flagged* rather than how insistently it flagged them.
        """
        return sorted({int(e["stream_idx"]) for e in self.events})

    def _process_stream(self) -> None:
        """Evaluate one stream window, checking for drift at each interval.

        Raises:
            StopIteration: Once the window is exhausted, to hand control back
                to :meth:`run` for the next stream update.
        """
        stream_loader = self.modelHarness.get_stream_dataloader()

        for _, batch in tqdm(
            enumerate(stream_loader),
            desc="Processing batches",
            leave=False,
        ):
            metrics = self._evaluate_batch(batch)
            self.metric_buffer.append(metrics)
            self.batch_count += 1

            if (
                self.detection_interval > 0
                and self.batch_count % self.detection_interval == 0
            ):
                drift_signal, agg_metric = self._check_drift()
                if drift_signal.drift_detected:
                    self._record_drift(drift_signal, agg_metric)

        raise StopIteration()

    def _check_drift(self) -> tuple[DriftSignal, float]:
        """Aggregate the buffered metrics and push them through the detector."""
        if not self.metric_buffer:
            raise RuntimeError("Model Harness requires evaluation metrics")

        metric_values = [m[self.metric_idx] for m in self.metric_buffer]
        self.metric_buffer = []

        if self.aggregation == "median":
            agg_metric = float(np.median(metric_values))
        elif self.aggregation == "last":
            agg_metric = float(metric_values[-1])
        else:
            agg_metric = float(np.mean(metric_values))

        self.check_count += 1

        # NOTE: Profiler only covers PyTorch operations. Even so, we measure
        # runtime to see if the detector is a potential bottleneck.
        if self.batch_count > self.flops_profiler.warmup_iters:
            with self.flops_profiler.measure_flops(tag="detector"):
                drift_signal = self.detector.update(agg_metric)
        else:
            drift_signal = self.detector.update(agg_metric)

        self._log_metrics(drift_signal, agg_metric)

        return drift_signal, agg_metric

    def _record_drift(self, drift_signal: DriftSignal, agg_metric: float) -> None:
        """Record a detection instead of dispatching a learning module."""
        self.drift_event_count += 1

        timerange = getattr(self.modelHarness, "current_window_timerange", None)

        event: dict[str, Any] = {
            "event": self.drift_event_count,
            "batch": self.batch_count,
            "check": self.check_count,
            "stream_idx": self.stream_update_count,
            "score": float(drift_signal.drift_score),
            "regime": drift_signal.regime.value if drift_signal.regime else "N/A",
            "confidence": drift_signal.confidence,
            f"metric_{self.metric_idx}": agg_metric,
        }
        if timerange is not None:
            event["data_time_start"] = timerange[0]
            event["data_time_end"] = timerange[1]
        self.events.append(event)

        self.logger.info(
            f"==== DRIFT DETECTED (Event #{self.drift_event_count})! ====", level=0
        )
        self.logger.info(
            f"\tBatch {self.batch_count}, stream window {self.stream_update_count}",
            level=1,
        )
        if timerange is not None:
            self.logger.info(
                f"\tData time range: {timerange[0]} -> {timerange[1]}", level=1
            )
        self.logger.info(f"\tRegime: {event['regime']}", level=1)
        self.logger.info(f"\tDrift Score: {drift_signal.drift_score:.4f}", level=1)
        self.logger.info(
            f"\tConfidence: {drift_signal.confidence if drift_signal.confidence else 'N/A'}",
            level=1,
        )
        self.logger.info("-> No adaptation (drift-only run); model unchanged.", level=0)

        if self.reset_on_detect:
            self.logger.debug("Resetting detector state...")
            self.detector.reset()

    def _extend_stream(self) -> None:
        """Load the next data buffer when the current stream is exhausted."""
        self.stream_update_count += 1
        self.logger.info(
            f"\tStream exhausted. Loading next data buffer. "
            f"{self.stream_update_count}/{self.max_stream_updates}",
            level=1,
        )
        self.modelHarness.update_data_stream()

    # -- evaluation / logging ---------------------------------------------

    def _evaluate_batch(self, batch: tuple[torch.Tensor, torch.Tensor]) -> list[float]:
        """Evaluate the frozen model on one streaming batch, returning all metrics."""
        self.modelHarness.model.eval()

        profile = self.batch_count > self.flops_profiler.warmup_iters

        with torch.no_grad():
            if profile:
                with self.flops_profiler.measure_flops(tag="infer"):
                    metrics, named = self._forward_metrics(batch)
            else:
                metrics, named = self._forward_metrics(batch)

        if profile:
            self.logger.stage("eval")
            self.logger.log(named)

        return metrics

    def _forward_metrics(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[list[float], dict[str, float]]:
        """Run the forward pass and compute every harness eval metric."""
        x, y = self.modelHarness._unpack(batch)
        x, y = x.to(self.cfg.device), y.to(self.cfg.device)

        y_hat = self.modelHarness.model(x)

        metrics: list[float] = []
        named: dict[str, float] = {}
        for key, metric_fn in self.modelHarness.eval_metrics.items():
            value = self.modelHarness._to_scalar(metric_fn(y_hat, y))
            metrics.append(value)
            named[key] = value
        return metrics, named

    def _log_metrics(self, drift_signal: DriftSignal, metric_value: float) -> None:
        """Log one drift check.

        Field names match ``ContinuousMonitor._log_metrics`` so drift-only runs
        and full runs land in the same CSV schema. Unlike there, every check is
        logged rather than sampling ``detected=0`` at 10% -- the complete
        score trace is the entire product of this run, and a sampled trace
        cannot be used to re-tune a threshold offline.
        """
        flops_perf = self.flops_profiler.get_performance()

        timerange = getattr(self.modelHarness, "current_window_timerange", None)
        ts_fields = {}
        if timerange is not None:
            ts_fields["data_time_start"] = timerange[0]
            ts_fields["data_time_end"] = timerange[1]

        self.logger.stage("drift")
        self.logger.log(
            {
                "detected": int(drift_signal.drift_detected),
                "score": drift_signal.drift_score,
                "regime": (drift_signal.regime.value if drift_signal.regime else "N/A"),
                "confidence": (
                    drift_signal.confidence if drift_signal.confidence else "N/A"
                ),
                f"metric_{self.metric_idx}": metric_value,
                "check_idx": self.check_count,
                "stream_idx": self.stream_update_count,
                **ts_fields,
                **{f"cperf_{k}": v for k, v in flops_perf.items()},
            },
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_drift_only(
    cfg: Config,
    modelHarness: BaseModelHarness,
) -> dict[str, Any]:
    """Run drift detection over the stream without ever adapting the model.

    Args:
        cfg: Resolved apeiron configuration.
        modelHarness: Harness supplying the model and stream loader.

    Returns:
        Summary dict with the detection count, the per-event details, the
        stream windows that fired, and the final eval metrics.
    """
    return DriftOnlyMonitor(cfg=cfg, modelHarness=modelHarness).run()


def main(argv: list[str] | None = None) -> int:
    cfg: Config = build_config(argv)
    modelHarness = get_example(cfg=cfg)

    backend = configure_backend(cfg)
    logger = get_logger(
        verbosity=cfg.verbosity,
        backend=backend,
        csv_path=cfg.visualization.input if cfg.visualization else None,
    )

    project_name = "basesim-framework"
    if cfg.logging and cfg.logging.experiment_name:
        project_name = cfg.logging.experiment_name

    logger.init(cfg, project=project_name)

    summary = run_drift_only(cfg=cfg, modelHarness=modelHarness)

    logger.finish()

    print("\n==== Drift-only summary ====")
    for key in (
        "detector",
        "batches",
        "stream_updates",
        "drift_checks",
        "drift_events",
        "drift_windows",
        "final_metrics",
    ):
        print(f"  {key}: {summary[key]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
