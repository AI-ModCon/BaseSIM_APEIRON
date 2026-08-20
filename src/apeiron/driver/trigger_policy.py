"""Trigger policies: the one thing that differs between monitoring runs.

Every stream-monitoring run -- detector-driven adaptation, detection-only, and
schedule-driven control arms -- shares the same loop (evaluate the window, buffer
metrics, aggregate, decide, maybe fire, extend). The *only* thing that varies in
the decision step is the predicate connecting the aggregated metric to a
fire/no-fire verdict. A ``TriggerPolicy`` is that predicate, expressed as a
:class:`~apeiron.drift_detection.detectors.base.DriftSignal` so the engine logs
every run in one schema.

Two axes a policy controls:

* **cadence** -- ``"interval"`` decides every ``detection_interval`` batches
  (mid-window; a detector consumes a running metric stream); ``"window"`` decides
  once at each window's end (a schedule fires per window).
* **profiling** -- whether the decision call is worth wrapping in the FLOPs
  profiler (a detector update, yes; a schedule lookup, no).
"""

from __future__ import annotations

from typing import Any, Optional

from apeiron.config.configuration import Config
from apeiron.drift_detection.detectors.base import BaseDriftDetector, DriftSignal
from apeiron.drift_detection.load_drift_detector import load_drift_detector
from apeiron.driver.schedules import TriggerSchedule


class TriggerPolicy:
    """Decides whether a decision point fires, and how it is logged."""

    #: "interval" -> decide every detection_interval batches; "window" -> once per window.
    cadence: str = "interval"
    #: Whether to wrap the decision call in the FLOPs profiler.
    profile_decision: bool = False
    #: Short human-readable label for run summaries.
    label: str = "policy"

    def decide(self, agg_metric: float, decision_idx: int) -> DriftSignal:
        """Return the verdict for one decision point."""
        raise NotImplementedError

    def reset(self) -> None:
        """Re-arm after a fire (e.g. reset a detector). Default: no-op."""

    def describe(self) -> str:
        return self.label

    def fire_headline(self, count: int, decision_idx: int) -> str:
        """Console banner for a fire event (count is 1-based)."""
        return f"TRIGGER (#{count}) at decision point {decision_idx}"

    def annotate_log(self, fields: dict[str, Any]) -> dict[str, Any]:
        """Stamp policy-specific fields onto the drift-stage log record."""
        return fields


class DetectorPolicy(TriggerPolicy):
    """Fire when a streaming drift detector reports drift.

    Interval cadence: the detector consumes the aggregated metric every
    ``detection_interval`` batches, exactly as ``ContinuousMonitor`` did.
    """

    cadence = "interval"
    profile_decision = True

    def __init__(
        self, cfg: Config, detector: Optional[BaseDriftDetector] = None
    ) -> None:
        self.detector: BaseDriftDetector = detector or load_drift_detector(cfg)
        self.label = cfg.drift_detection.detector_name

    def decide(self, agg_metric: float, decision_idx: int) -> DriftSignal:
        return self.detector.update(agg_metric)

    def reset(self) -> None:
        self.detector.reset()

    def fire_headline(self, count: int, decision_idx: int) -> str:
        return f"DRIFT DETECTED (Event #{count})!"


class SchedulePolicy(TriggerPolicy):
    """Fire on a fixed :class:`TriggerSchedule`, ignoring the metric.

    Window cadence: one decision per stream window. The verdict carries no drift
    statistic, so ``score`` stays 0.0 and the log is stamped ``regime="scheduled"``
    / ``confidence="N/A"`` -- a placeholder severity would be indistinguishable
    from a real one during analysis.
    """

    cadence = "window"
    profile_decision = False

    def __init__(self, schedule: TriggerSchedule) -> None:
        self.schedule = schedule
        self.label = schedule.describe()

    def decide(self, agg_metric: float, decision_idx: int) -> DriftSignal:
        fire = self.schedule.should_fire(decision_idx)
        return DriftSignal(
            regime=None, drift_detected=fire, drift_score=0.0, confidence=None
        )

    def fire_headline(self, count: int, decision_idx: int) -> str:
        return f"SCHEDULED TRIGGER (#{count}) at decision point {decision_idx}"

    def annotate_log(self, fields: dict[str, Any]) -> dict[str, Any]:
        fields["regime"] = "scheduled"
        fields["confidence"] = "N/A"
        return fields
