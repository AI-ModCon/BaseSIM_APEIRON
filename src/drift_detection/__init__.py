"""
Drift detection module for continual learning systems.

This module provides various drift detection algorithms that can signal when to switch between different learning regimes:
- Stable: No significant drift
- Continual Learning: Minor drift, use CL with replay
- Fine-Tuning: Moderate drift, fine-tune model
- Retrain: Severe drift, retrain from scratch

Available detectors:
- ADWINDetector: Adaptive windowing
- KSWINDetector: Kolmogorov-Smirnov windowing
- PageHinkleyDetector: Page-Hinkley test
- ModelPerformanceDetector: Model-based using evidently
- EnsembleDetector: Combine multiple detectors
"""

from drift_detection.detectors.base import (
    BaseDriftDetector,
    DriftSignal,
    LearningRegime,
)
from drift_detection.detectors.statistical_detectors import (
    ADWINDetector,
    KSWINDetector,
    PageHinkleyDetector,
)
from drift_detection.detectors.model_performance_detector import (
    ModelPerformanceDetector,
    EnsembleDetector,
)

__all__ = [
    "BaseDriftDetector",
    "DriftSignal",
    "LearningRegime",
    "ADWINDetector",
    "KSWINDetector",
    "PageHinkleyDetector",
    "ModelPerformanceDetector",
    "EnsembleDetector",
]
