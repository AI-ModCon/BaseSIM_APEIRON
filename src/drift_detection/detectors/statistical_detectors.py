"""
Statistical drift detectors using river library.

These detectors are lightweight, online algorithms suitable for streaming data.
They monitor statistical properties of data streams to detect distribution shifts.

References:
- river library: https://github.com/online-ml/river
"""

import numpy as np
from river import drift as river_drift
from .base import BaseDriftDetector, DriftSignal, LearningRegime


class ADWINDetector(BaseDriftDetector):
    """
    Adaptive Windowing (ADWIN) drift detector.

    ADWIN maintains a variable-length sliding window and detects change when
    there's a significant difference between the means of two sub-windows.

    Best for: Detecting gradual and abrupt changes in mean values.
    Use case: Monitor model loss, accuracy, or prediction errors over time.
    """

    def __init__(
        self,
        delta: float = 0.002,
        minor_threshold: float = 0.3,
        moderate_threshold: float = 0.6,
        name: str = "ADWIN",
    ):
        """
        Initialize ADWIN detector.

        Args:
            delta: Confidence level (lower = more sensitive, more false alarms)
            minor_threshold: Drift score threshold for continual learning regime
            moderate_threshold: Drift score threshold for fine-tuning regime
            name: Detector name
        """
        super().__init__(name)
        self.delta = delta
        self.minor_threshold = minor_threshold
        self.moderate_threshold = moderate_threshold
        self.detector = river_drift.ADWIN(delta=delta)
        self._drift_history: list[int] = []
        self._value_history: list[float] = []
        self._is_initialized = True

    def update(self, value: float, **kwargs) -> DriftSignal:
        """
        Update ADWIN with new value.

        Args:
            value: Metric to monitor (e.g., loss, error rate)

        Returns:
            DriftSignal with regime recommendation
        """
        self._value_history.append(value)

        # Update detector
        self.detector.update(value)
        drift_detected = self.detector.drift_detected

        # Track drift occurrences
        self._drift_history.append(1 if drift_detected else 0)

        # Compute drift score based on recent drift frequency
        recent_window = min(100, len(self._drift_history))
        recent_drifts = self._drift_history[-recent_window:]
        drift_score = np.mean(recent_drifts)

        # Determine regime based on drift severity
        if not drift_detected:
            regime = LearningRegime.STABLE
        elif drift_score < self.minor_threshold:
            regime = LearningRegime.CONTINUAL_LEARNING
        elif drift_score < self.moderate_threshold:
            regime = LearningRegime.FINE_TUNING
        else:
            regime = LearningRegime.RETRAIN

        metadata = {
            "window_size": self.detector.width,
            "n_detections": sum(self._drift_history),
            "recent_drift_rate": drift_score,
        }

        return DriftSignal(
            regime=regime,
            drift_detected=drift_detected,
            drift_score=float(drift_score),
            confidence=1 - self.delta,
            metadata=metadata,
        )

    def reset(self) -> None:
        """Reset detector to initial state."""
        self.detector = river_drift.ADWIN(delta=self.delta)
        self._drift_history = []
        self._value_history = []


class KSWINDetector(BaseDriftDetector):
    """
    Kolmogorov-Smirnov Windowing (KSWIN) drift detector.

    KSWIN uses the two-sample Kolmogorov-Smirnov test to compare recent data
    against a reference window. Detects when distributions differ significantly.

    Best for: Detecting changes in data distributions (not just mean).
    Use case: Monitor prediction distributions, feature statistics.
    """

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        minor_threshold: float = 0.3,
        moderate_threshold: float = 0.6,
        name: str = "KSWIN",
    ):
        """
        Initialize KSWIN detector.

        Args:
            alpha: Significance level for KS test (lower = more sensitive)
            window_size: Size of reference window
            stat_size: Size of recent window for comparison
            minor_threshold: Drift score threshold for continual learning
            moderate_threshold: Drift score threshold for fine-tuning
            name: Detector name
        """
        super().__init__(name)
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.minor_threshold = minor_threshold
        self.moderate_threshold = moderate_threshold
        self.detector = river_drift.KSWIN(
            alpha=alpha, window_size=window_size, stat_size=stat_size
        )
        self._drift_history: list[int] = []
        self._is_initialized = True

    def update(self, value: float, **kwargs) -> DriftSignal:
        """
        Update KSWIN with new value.

        Args:
            value: Metric to monitor

        Returns:
            DriftSignal with regime recommendation
        """
        self.detector.update(value)
        drift_detected = self.detector.drift_detected

        self._drift_history.append(1 if drift_detected else 0)

        # Compute drift score
        recent_window = min(50, len(self._drift_history))
        recent_drifts = self._drift_history[-recent_window:]
        drift_score = np.mean(recent_drifts)

        # Determine regime
        if not drift_detected:
            regime = LearningRegime.STABLE
        elif drift_score < self.minor_threshold:
            regime = LearningRegime.CONTINUAL_LEARNING
        elif drift_score < self.moderate_threshold:
            regime = LearningRegime.FINE_TUNING
        else:
            regime = LearningRegime.RETRAIN

        metadata = {
            "p_value": getattr(self.detector, "p_value", None),
            "n_detections": sum(self._drift_history),
        }

        return DriftSignal(
            regime=regime,
            drift_detected=drift_detected,
            drift_score=float(drift_score),
            confidence=1 - self.alpha,
            metadata=metadata,
        )

    def reset(self) -> None:
        """Reset detector to initial state."""
        self.detector = river_drift.KSWIN(
            alpha=self.alpha, window_size=self.window_size, stat_size=self.stat_size
        )
        self._drift_history = []


class PageHinkleyDetector(BaseDriftDetector):
    """
    Page-Hinkley test for drift detection.

    Page-Hinkley detects changes in the mean of a Gaussian signal. It's fast
    and works well for detecting abrupt changes.

    Best for: Fast detection of abrupt changes in mean values.
    Use case: Real-time monitoring of model performance metrics.
    """

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 1 - 0.0001,
        minor_threshold: float = 0.3,
        moderate_threshold: float = 0.6,
        name: str = "PageHinkley",
    ):
        """
        Initialize Page-Hinkley detector.

        Args:
            min_instances: Minimum instances before detection
            delta: Magnitude threshold for change
            threshold: Detection threshold
            alpha: Forgetting factor (weight of recent values)
            minor_threshold: Drift score threshold for continual learning
            moderate_threshold: Drift score threshold for fine-tuning
            name: Detector name
        """
        super().__init__(name)
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.minor_threshold = minor_threshold
        self.moderate_threshold = moderate_threshold
        self.detector = river_drift.PageHinkley(
            min_instances=min_instances, delta=delta, threshold=threshold, alpha=alpha
        )
        self._drift_history: list[int] = []
        self._is_initialized = True

    def update(self, value: float, **kwargs) -> DriftSignal:
        """
        Update Page-Hinkley test with new value.

        Args:
            value: Metric to monitor

        Returns:
            DriftSignal with regime recommendation
        """
        self.detector.update(value)
        drift_detected = self.detector.drift_detected

        self._drift_history.append(1 if drift_detected else 0)

        # Compute drift score
        recent_window = min(50, len(self._drift_history))
        recent_drifts = self._drift_history[-recent_window:]
        drift_score = np.mean(recent_drifts)

        # Determine regime
        if not drift_detected:
            regime = LearningRegime.STABLE
        elif drift_score < self.minor_threshold:
            regime = LearningRegime.CONTINUAL_LEARNING
        elif drift_score < self.moderate_threshold:
            regime = LearningRegime.FINE_TUNING
        else:
            regime = LearningRegime.RETRAIN

        return DriftSignal(
            regime=regime,
            drift_detected=drift_detected,
            drift_score=float(drift_score),
            metadata={"n_detections": sum(self._drift_history)},
        )

    def reset(self) -> None:
        """Reset detector to initial state."""
        self.detector = river_drift.PageHinkley(
            min_instances=self.min_instances,
            delta=self.delta,
            threshold=self.threshold,
            alpha=self.alpha,
        )
        self._drift_history = []
