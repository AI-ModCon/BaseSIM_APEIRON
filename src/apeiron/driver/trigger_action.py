"""Trigger actions: what the engine does when a decision point fires.

The second axis that separates monitoring runs. Given a fire, either:

* :class:`AdaptAction` -- pause monitoring, dispatch the continual-learning loop,
  optionally checkpoint, then resume (the full run and the schedule-driven
  control arm); or
* :class:`RecordOnlyAction` -- record the detection and leave the model frozen
  (the detection-only run: the weights that go in are the weights that come out).

An action operates on the engine, reading its counters/harness/trainer/logger and
appending to its ``events``/``fire_points`` so the run summary is uniform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from apeiron.drift_detection.detectors.base import DriftSignal

if TYPE_CHECKING:
    from apeiron.driver.stream_engine import StreamEngine


class TriggerAction:
    """Strategy invoked once per fired decision point."""

    #: Whether the engine must build a ContinuousTrainer for this action.
    needs_trainer: bool = False
    label: str = "action"

    def on_fire(
        self,
        engine: "StreamEngine",
        signal: DriftSignal,
        decision_idx: int,
        agg_metric: float,
    ) -> None:
        raise NotImplementedError

    def describe(self) -> str:
        return self.label


class AdaptAction(TriggerAction):
    """Dispatch continual learning, then optionally checkpoint, then resume."""

    needs_trainer = True
    label = "adapt"

    def on_fire(
        self,
        engine: "StreamEngine",
        signal: DriftSignal,
        decision_idx: int,
        agg_metric: float,
    ) -> None:
        engine.fire_count += 1
        engine.fire_points.append(decision_idx)
        logger = engine.logger

        logger.info(
            f"==== {engine.policy.fire_headline(engine.fire_count, decision_idx)} ====",
            level=0,
        )
        timerange = getattr(engine.modelHarness, "current_window_timerange", None)
        if timerange is not None:
            logger.info(f"\tData time range: {timerange[0]} -> {timerange[1]}", level=1)
        regime = signal.regime.value if signal.regime else "N/A"
        logger.info(f"\tRegime: {regime}", level=1)
        logger.info(f"\tDrift Score: {signal.drift_score:.4f}", level=1)
        logger.info(
            f"\tConfidence: {signal.confidence if signal.confidence else 'N/A'}",
            level=1,
        )

        engine.flops_profiler.print_performance(logger=logger, level=2)

        logger.info("-> Dispatching continual learning module...", level=0)
        assert engine.trainer is not None  # needs_trainer guarantees this
        engine.trainer.outer_cl_training_loop(drift_event_id=engine.fire_count)
        logger.info("<- Continual learning complete.", level=0)

        if engine.modelHarness.ckpts_enabled:
            ckptpath = engine.modelHarness.save_ckpt(event=engine.fire_count)
            logger.info(f"* Checkpoint saved to: {ckptpath}", level=0)

        if engine.cfg.drift_detection.reset_after_learning:
            logger.debug("Resetting trigger policy state...")
            engine.policy.reset()

        engine.events.append(
            {
                "event": engine.fire_count,
                "decision_idx": decision_idx,
                "stream_idx": engine.stream_update_count,
                "score": float(signal.drift_score),
            }
        )
        logger.info("==== RESUMING MONITORING! ====", level=0)


class RecordOnlyAction(TriggerAction):
    """Record the detection and leave the model frozen (detection-only run)."""

    needs_trainer = False
    label = "record-only"

    def on_fire(
        self,
        engine: "StreamEngine",
        signal: DriftSignal,
        decision_idx: int,
        agg_metric: float,
    ) -> None:
        engine.fire_count += 1
        engine.fire_points.append(decision_idx)
        logger = engine.logger

        timerange = getattr(engine.modelHarness, "current_window_timerange", None)
        regime = signal.regime.value if signal.regime else "N/A"
        event: dict[str, Any] = {
            "event": engine.fire_count,
            "batch": engine.batch_count,
            "decision_idx": decision_idx,
            "stream_idx": engine.stream_update_count,
            "score": float(signal.drift_score),
            "regime": regime,
            "confidence": signal.confidence,
            f"metric_{engine.metric_idx}": agg_metric,
        }
        if timerange is not None:
            event["data_time_start"] = timerange[0]
            event["data_time_end"] = timerange[1]
        engine.events.append(event)

        logger.info(
            f"==== {engine.policy.fire_headline(engine.fire_count, decision_idx)} ====",
            level=0,
        )
        logger.info(
            f"\tBatch {engine.batch_count}, stream window {engine.stream_update_count}",
            level=1,
        )
        if timerange is not None:
            logger.info(f"\tData time range: {timerange[0]} -> {timerange[1]}", level=1)
        logger.info(f"\tRegime: {regime}", level=1)
        logger.info(f"\tDrift Score: {signal.drift_score:.4f}", level=1)
        logger.info(
            f"\tConfidence: {signal.confidence if signal.confidence else 'N/A'}",
            level=1,
        )
        logger.info("-> No adaptation (drift-only run); model unchanged.", level=0)

        if engine.cfg.drift_detection.reset_after_learning:
            logger.debug("Resetting trigger policy state...")
            engine.policy.reset()
