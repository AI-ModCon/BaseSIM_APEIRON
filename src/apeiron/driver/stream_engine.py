"""The single stream-monitoring loop.

`StreamEngine` replaces the three hand-synced loops (`ContinuousMonitor`,
`DriftOnlyMonitor`, `ScheduledCLRunner`) with one implementation parameterized by
a :class:`~apeiron.driver.trigger_policy.TriggerPolicy` (when/whether to fire) and
a :class:`~apeiron.driver.trigger_action.TriggerAction` (what to do on a fire).
Everything the three used to duplicate lives here exactly once: per-batch
evaluation, metric buffering + aggregation, FLOPs profiling, stream extension,
the decision point, and the drift-stage log schema. Because there is only one
loop, the distributed choreography a later phase adds (shard the window, gather
metrics to rank 0, broadcast the fire decision) has a single home.

The two cadences correspond to the two ways a decision point arises:

* ``"interval"`` -- decide every ``detection_interval`` batches, mid-window; the
  metric buffer carries across window boundaries (a detector sees a continuous
  metric stream). This is the old ``ContinuousMonitor`` / ``DriftOnlyMonitor``.
* ``"window"`` -- decide once at each window's end; the buffer is aggregated over
  the whole window. This is the old ``ScheduledCLRunner``.

The logger is injected, not fetched from a global, so a later multi-node phase
can give rank 0 a real logger and every other rank a no-op one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from apeiron.config.configuration import Config
from apeiron.driver.trigger_action import TriggerAction
from apeiron.driver.trigger_policy import TriggerPolicy
from apeiron.logger import Logger, get_logger
from apeiron.profilers import FLOPSProfiler

if TYPE_CHECKING:
    from apeiron.model.torch_model_harness import BaseModelHarness
    from apeiron.training import ContinuousTrainer


class StreamEngine:
    """Stream monitor parameterized by a trigger policy and a trigger action."""

    def __init__(
        self,
        cfg: Config,
        modelHarness: "BaseModelHarness",
        policy: TriggerPolicy,
        action: TriggerAction,
        *,
        logger: Optional[Logger] = None,
        profiler: Optional[FLOPSProfiler] = None,
        detected_sample_rate: float = 1.0,
    ) -> None:
        self.cfg = cfg
        self.modelHarness = modelHarness
        self.policy = policy
        self.action = action
        self.logger = logger if logger is not None else get_logger()
        self.flops_profiler = profiler if profiler is not None else FLOPSProfiler()

        # Detector-driven runs sample non-fire ``detected`` rows to cut log
        # volume; detection-only / schedule runs keep the full trace (rate 1.0).
        self.detected_sample_rate = detected_sample_rate

        # A trainer is built only when the action needs one (adaptation runs);
        # detection-only runs leave the model, and the optimizer, untouched.
        self.trainer: Optional["ContinuousTrainer"] = None
        if action.needs_trainer:
            from apeiron.training import ContinuousTrainer

            self.trainer = ContinuousTrainer(
                cfg=cfg,
                modelHarness=modelHarness,
                logger=self.logger,
                profiler=self.flops_profiler,
            )

        # Configuration mirrored from cfg for the loop.
        self.metric_idx = cfg.drift_detection.metric_index
        self.detection_interval = cfg.drift_detection.detection_interval
        self.max_stream_updates = cfg.drift_detection.max_stream_updates
        self.aggregation = cfg.drift_detection.aggregation

        # State.
        self.stream_update_count = 0
        self.batch_count = 0
        self.decision_count = 0
        self.fire_count = 0
        self.fire_points: list[int] = []
        self.events: list[dict[str, Any]] = []
        self.metric_buffer: list[list[float]] = []

    @property
    def cadence(self) -> str:
        return self.policy.cadence

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the monitoring loop and return a run summary."""
        self._log_startup()
        self.logger.info("\tInitializing first data stream...", level=1)
        self.modelHarness.update_data_stream()

        while not self._should_stop():
            try:
                self._process_stream()
            except StopIteration:
                self._extend_stream()

        self.logger.info("==== Monitoring Complete ====", level=0)
        self.logger.info(f"\tTotal batches processed: {self.batch_count}", level=1)
        self.logger.info(f"\tTotal stream updates: {self.stream_update_count}", level=1)
        self.logger.info(f"\tDecision points: {self.decision_count}", level=1)
        self.logger.info(f"\tTriggers fired: {self.fire_count}", level=1)
        if self.fire_points:
            self.logger.info(f"\tTrigger points: {self.fire_points}", level=1)

        final_metrics = self.modelHarness.eval()
        self.logger.info(f"\tFinal eval metrics: {final_metrics}", level=1)

        return {
            "policy": self.policy.describe(),
            "action": self.action.describe(),
            "batches": self.batch_count,
            "stream_updates": self.stream_update_count,
            "decision_points": self.decision_count,
            "fires": self.fire_count,
            "fire_points": list(self.fire_points),
            "events": list(self.events),
            "final_metrics": final_metrics,
        }

    def _log_startup(self) -> None:
        self.logger.info("==== StreamEngine initialized ====", level=0)
        self.logger.info(f"\tTrigger policy: {self.policy.describe()}", level=1)
        self.logger.info(f"\tOn fire: {self.action.describe()}", level=1)
        self.logger.info(f"\tDecision cadence: {self.cadence}", level=1)
        self.logger.info(f"\tMonitoring metric index: {self.metric_idx}", level=1)
        if self.cadence == "interval":
            self.logger.info(
                f"\tDetection interval: {self.detection_interval} batches", level=1
            )
        self.logger.info(f"\tAggregation method: {self.aggregation}", level=1)
        self.logger.info(f"\tMax stream updates: {self.max_stream_updates}", level=1)

    def _should_stop(self) -> bool:
        return self.stream_update_count >= self.max_stream_updates

    def _extend_stream(self) -> None:
        """Load the next data buffer when the current stream is exhausted."""
        self.stream_update_count += 1
        self.logger.info(
            f"\tStream exhausted. Loading next data buffer. "
            f"{self.stream_update_count}/{self.max_stream_updates}",
            level=1,
        )
        self.modelHarness.update_data_stream()

    # -- stream processing -------------------------------------------------

    def _process_stream(self) -> None:
        """Evaluate one stream window, reaching decision points per the cadence.

        Raises:
            StopIteration: Always, once the window is exhausted, to hand control
                back to :meth:`run` for the next stream update.
        """
        stream_loader = self.modelHarness.get_stream_dataloader()

        for _, batch in tqdm(
            enumerate(stream_loader), desc="Processing batches", leave=False
        ):
            metrics = self._evaluate_batch(batch)
            self.metric_buffer.append(metrics)
            self.batch_count += 1

            if (
                self.cadence == "interval"
                and self.detection_interval > 0
                and self.batch_count % self.detection_interval == 0
            ):
                self._decision_point()

        if self.cadence == "window":
            self._decision_point()

        raise StopIteration()

    def _decision_point(self) -> None:
        """Aggregate the buffer, consult the policy, log, and fire if triggered."""
        agg_metric = self._aggregate_buffer()

        decision_idx = self.decision_count
        self.decision_count += 1

        if (
            self.policy.profile_decision
            and self.batch_count > self.flops_profiler.warmup_iters
        ):
            with self.flops_profiler.measure_flops(tag="detector"):
                signal = self.policy.decide(agg_metric, decision_idx)
        else:
            signal = self.policy.decide(agg_metric, decision_idx)

        self._log_decision(decision_idx, agg_metric, signal)

        if signal.drift_detected:
            self.action.on_fire(self, signal, decision_idx, agg_metric)

    def _aggregate_buffer(self) -> float:
        """Reduce the buffered per-batch metrics to the monitored scalar."""
        if not self.metric_buffer:
            raise RuntimeError("Model Harness requires evaluation metrics")

        metric_values = [m[self.metric_idx] for m in self.metric_buffer]
        self.metric_buffer = []

        if self.aggregation == "median":
            return float(np.median(metric_values))
        if self.aggregation == "last":
            return float(metric_values[-1])
        return float(np.mean(metric_values))

    # -- evaluation --------------------------------------------------------

    def _evaluate_batch(self, batch: tuple[torch.Tensor, torch.Tensor]) -> list[float]:
        """Evaluate the model on one streaming batch, returning all metrics."""
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

    # -- logging -----------------------------------------------------------

    def _log_decision(self, decision_idx: int, agg_metric: float, signal: Any) -> None:
        """Log one decision point under the unified ``drift`` schema.

        All runs share these fields: ``detected`` (sampled for non-fires only
        when ``detected_sample_rate < 1``), ``score``, ``regime``, ``confidence``,
        ``metric_<idx>``, ``decision_idx``, ``trigger_count``, ``stream_idx``, the
        data timestamp range if the harness tracks it, and ``cperf_*``. The policy
        may stamp its own fields (a schedule marks ``regime="scheduled"``).
        """
        fire = bool(signal.drift_detected)
        flops_perf = self.flops_profiler.get_performance()

        fields: dict[str, Any] = {
            "score": float(signal.drift_score),
            "regime": signal.regime.value if signal.regime else "N/A",
            "confidence": signal.confidence if signal.confidence else "N/A",
            f"metric_{self.metric_idx}": agg_metric,
            "decision_idx": decision_idx,
            "trigger_count": self.fire_count + int(fire),
            "stream_idx": self.stream_update_count,
        }

        # Always record fires; sample non-fires only when a rate < 1 is set.
        include_detected = (
            fire
            or self.detected_sample_rate >= 1.0
            or float(np.random.random()) <= self.detected_sample_rate
        )
        if include_detected:
            fields["detected"] = int(fire)

        timerange = getattr(self.modelHarness, "current_window_timerange", None)
        if timerange is not None:
            fields["data_time_start"] = timerange[0]
            fields["data_time_end"] = timerange[1]

        fields.update({f"cperf_{k}": v for k, v in flops_perf.items()})
        fields = self.policy.annotate_log(fields)

        self.logger.stage("drift")
        self.logger.log(fields)
