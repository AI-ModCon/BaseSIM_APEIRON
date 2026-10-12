"""
Base classes for drift detection in continual learning systems.

This module provides abstract interfaces for drift detectors that can signal
when to switch between different learning regimes (continual learning, fine-tuning, retraining).
"""

import copy
from enum import Enum
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LearningRegime(Enum):
    """
    Learning regime recommendations based on drift severity.
    - Stable: No significant drift, continue current strategy
    - Continual Learning: Minor drift, use CL with replay
    - Fine-Tuning: Moderate drift, fine-tune model
    - Retrain: Severe drift, retrain from scratch
    """

    STABLE = "stable"
    CONTINUAL_LEARNING = "continual_learning"
    FINE_TUNING = "fine_tuning"
    RETRAIN = "retrain"


class DriftSignal:
    """Container for drift detection results."""

    def __init__(
        self,
        regime: LearningRegime,
        drift_detected: bool,
        drift_score: float,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize drift signal.

        Args:
            regime: Recommended learning regime
            drift_detected: Whether drift was detected
            drift_score: Numeric drift severity score (higher = more drift)
            confidence: Optional confidence in the detection (0-1)
            metadata: Optional additional information
        """
        self.regime = regime
        self.drift_detected = drift_detected
        self.drift_score = drift_score
        self.confidence = confidence
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"DriftSignal(regime={self.regime.value}, "
            f"detected={self.drift_detected}, "
            f"score={self.drift_score:.4f}, "
            f"confidence={self.confidence}, )"
        )


class BaseDriftDetector(ABC):
    """Abstract base class for all drift detectors."""

    _STATE_ATTRS: Optional[tuple[str, ...]] = None

    def __init__(self, name: str):
        """
        Initialize drift detector.

        Args:
            name: Human-readable name for this detector
        """
        self.name = name
        self._is_initialized = False

    @abstractmethod
    def update(self, value: float, **kwargs) -> DriftSignal:
        """
        Update detector with new observation and check for drift.

        Args:
            value: Numeric value to monitor (e.g., loss, accuracy, prediction error)
            **kwargs: Additional detector-specific parameters

        Returns:
            DriftSignal indicating whether drift occurred and recommended regime
        """
        raise NotImplementedError(f"Method not implemented for {self.name}")

    @abstractmethod
    def reset(self) -> None:
        """Reset detector to initial state."""
        raise NotImplementedError(f"Method not implemented for {self.name}")

    def state_dict(self) -> Dict[str, Any]:
        """Snapshot the detector's evolving state so a run can be resumed.

        A detector is the one component whose verdict depends on everything it
        has already seen, so restarting it from scratch mid-stream changes the
        run's results rather than merely repeating work. This returns a
        picklable snapshot of the attributes named in :attr:`_STATE_ATTRS`.

        Returns:
            Picklable mapping of state, tagged with the detector class name so
            :meth:`load_state_dict` can reject a mismatched checkpoint.

        Raises:
            NotImplementedError: If the subclass has not declared
                :attr:`_STATE_ATTRS`.
        """
        attrs = self._state_attrs()
        state: Dict[str, Any] = {
            attr: copy.deepcopy(getattr(self, attr)) for attr in attrs
        }
        state["detector_class"] = type(self).__name__
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`.

        The detector must already be constructed from the same config that
        produced the checkpoint; this restores only what the stream accumulated.

        Args:
            state: A mapping previously returned by :meth:`state_dict`.

        Raises:
            NotImplementedError: If the subclass has not declared
                :attr:`_STATE_ATTRS`.
            ValueError: If the checkpoint was written by a different detector
                class.
            KeyError: If the checkpoint is missing an expected attribute.
        """
        attrs = self._state_attrs()

        expected = type(self).__name__
        found = state.get("detector_class")
        if found != expected:
            raise ValueError(
                f"Checkpoint was written by {found!r} but is being loaded into "
                f"{expected!r}. Detector state is not portable across detector "
                f"types -- check that [drift_detection] detector_name matches "
                f"the run that produced this checkpoint."
            )

        missing = [attr for attr in attrs if attr not in state]
        if missing:
            raise KeyError(
                f"Checkpoint for {expected} is missing expected state: {missing}"
            )

        # Deep-copy on the way in too, so loading one checkpoint into two
        # detectors does not leave them sharing a single mutable object.
        for attr in attrs:
            setattr(self, attr, copy.deepcopy(state[attr]))

    def _state_attrs(self) -> tuple[str, ...]:
        """Return the declared state attributes, or explain that there are none."""
        if self._STATE_ATTRS is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support checkpointing. Declare "
                f"_STATE_ATTRS on the class -- an empty tuple if the detector "
                f"keeps no state across update() calls (i.e., it is genuinely stateless)."
            )
        return self._STATE_ATTRS

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
