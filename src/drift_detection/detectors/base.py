"""
Base classes for drift detection in continual learning systems.

This module provides abstract interfaces for drift detectors that can signal
when to switch between different learning regimes (continual learning, fine-tuning, retraining).
"""

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
