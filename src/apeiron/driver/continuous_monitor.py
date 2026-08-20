"""Detector-driven adaptation -- the default full monitoring run.

``ContinuousMonitor`` is now thin wiring over
:class:`~apeiron.driver.stream_engine.StreamEngine`: a
:class:`~apeiron.driver.trigger_policy.DetectorPolicy` (fire when the configured
drift detector reports drift) plus an
:class:`~apeiron.driver.trigger_action.AdaptAction` (dispatch continual learning
on a fire). The detection-only and schedule-driven control arms are the same
engine with a different policy/action (see ``src/drift_only.py`` and
``src/cl_only.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from apeiron.config.configuration import Config
from apeiron.drift_detection.detectors.base import BaseDriftDetector
from apeiron.driver.stream_engine import StreamEngine
from apeiron.driver.trigger_action import AdaptAction
from apeiron.driver.trigger_policy import DetectorPolicy
from apeiron.logger import Logger
from apeiron.profilers import FLOPSProfiler

if TYPE_CHECKING:
    from apeiron.model.torch_model_harness import BaseModelHarness

# Non-fire ``detected`` rows are sampled at this rate to cut log volume; fires
# are always logged. Detection-only / schedule runs keep the full trace instead.
_DETECTED_SAMPLE_RATE = 0.1


class ContinuousMonitor(StreamEngine):
    """Monitor a stream, detect drift, and adapt the model when it fires."""

    def __init__(
        self,
        cfg: Config,
        modelHarness: "BaseModelHarness",
        *,
        logger: Optional[Logger] = None,
        detector: Optional[BaseDriftDetector] = None,
        profiler: Optional[FLOPSProfiler] = None,
    ) -> None:
        policy = DetectorPolicy(cfg, detector=detector)
        super().__init__(
            cfg,
            modelHarness,
            policy,
            AdaptAction(),
            logger=logger,
            profiler=profiler,
            detected_sample_rate=_DETECTED_SAMPLE_RATE,
        )
        # Back-compat convenience: expose the detector directly.
        self.detector: BaseDriftDetector = policy.detector
