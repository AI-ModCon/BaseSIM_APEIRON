"""
Model performance-based drift detector using evidently.

This detector monitors model predictions and performance metrics to detect drift.
It uses evidently's data drift detection capabilities for comprehensive analysis.

Reference: https://github.com/evidentlyai/evidently
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from evidently import Report
from evidently.presets import DataDriftPreset
from drift_detection.detectors.base import (
    BaseDriftDetector,
    DriftSignal,
    LearningRegime,
)

from model.torch_model_harness import BaseModelHarness


class ModelEvalDetector(BaseDriftDetector):
    def __init__(
        self,
        name: str = "ModelEval",
    ):
        super().__init__(name=name)

    def update(
        self,
        modelHarness: BaseModelHarness,
        reference_validation_metrics: list[float] = [],
        higher_is_better: list[bool] = [],
    ) -> DriftSignal:

        validation_metrics = (
            modelHarness.eval()
        )  # need to find away to explicitly match the metrics to reference values

        assert (
            len(reference_validation_metrics)
            == len(validation_metrics)
            == len(higher_is_better)
        )

        for metric, ref_metric, higher_is_better in zip(
            validation_metrics, reference_validation_metrics, higher_is_better
        ):
            print(
                "metric:",
                metric,
                "ref_metric:",
                ref_metric,
                "higher_is_better:",
                higher_is_better,
            )
            if higher_is_better:
                if metric < ref_metric:
                    return DriftSignal(
                        regime=LearningRegime.CONTINUAL_LEARNING,
                        drift_detected=True,
                        drift_score=1,  # dummy
                        confidence=0.95,  # dummy
                        metadata=None,
                    )
            else:
                if metric > ref_metric:
                    return DriftSignal(
                        regime=LearningRegime.CONTINUAL_LEARNING,
                        drift_detected=True,
                        drift_score=1,  # dummy
                        confidence=0.95,  # dummy
                        metadata=None,
                    )

        return DriftSignal(
            regime=LearningRegime.STABLE,
            drift_detected=False,
            drift_score=1,  # dummy
            confidence=0.95,  # dummy
            metadata=None,
        )

    def reset(self):
        pass


class ModelPerformanceDetector(BaseDriftDetector):
    """
    Drift detector based on model predictions and performance.

    Uses evidently to detect drift in:
    - Prediction distributions
    - Feature distributions (if features provided)
    - Target distributions

    This is more comprehensive than statistical detectors but requires
    reference data and batched updates.
    """

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        reference_predictions: Optional[np.ndarray] = None,
        reference_targets: Optional[np.ndarray] = None,
        drift_share_threshold: float = 0.5,
        minor_threshold: float = 0.3,
        moderate_threshold: float = 0.6,
        name: str = "ModelPerformance",
    ):
        """
        Initialize model performance detector.

        Args:
            reference_data: Reference dataset (features as DataFrame)
            reference_predictions: Reference model predictions
            reference_targets: Reference ground truth labels
            drift_share_threshold: Share of drifted features to trigger detection
            minor_threshold: Drift share for continual learning regime
            moderate_threshold: Drift share for fine-tuning regime
            name: Detector name
        """
        super().__init__(name)
        self.reference_data = reference_data
        self.reference_predictions = reference_predictions
        self.reference_targets = reference_targets
        self.drift_share_threshold = drift_share_threshold
        self.minor_threshold = minor_threshold
        self.moderate_threshold = moderate_threshold

        self._drift_history: list[float] = []
        self._is_initialized = reference_data is not None

    def set_reference(
        self,
        data: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set or update reference data.

        Args:
            data: Reference features
            predictions: Reference predictions
            targets: Reference targets
        """
        self.reference_data = data.copy()
        self.reference_predictions = predictions
        self.reference_targets = targets
        self._is_initialized = True

    def update(
        self,
        value: Optional[float] = None,
        data: Optional[pd.DataFrame] = None,
        predictions: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
        **kwargs,
    ) -> DriftSignal:
        """
        Update detector with new batch of data.

        This detector requires batch updates (not single values like statistical detectors).

        Args:
            value: Optional single metric value (e.g., average loss)
            data: Current data batch (features)
            predictions: Current predictions
            targets: Current targets
            **kwargs: Additional parameters

        Returns:
            DriftSignal with regime recommendation
        """
        if not self._is_initialized:
            raise ValueError(
                "Detector not initialized. Call set_reference() first or provide "
                "reference data in constructor."
            )

        # If no data provided, fall back to simple value-based detection
        if data is None and value is not None:
            return self._simple_value_detection(value)

        if data is None:
            raise ValueError(
                "Either 'data' DataFrame or 'value' must be provided for update()"
            )

        # Prepare current data
        current_data = data.copy()
        if predictions is not None:
            current_data["prediction"] = predictions
        if targets is not None:
            current_data["target"] = targets

        # Prepare reference data
        if self.reference_data is not None:
            reference_data = self.reference_data.copy()
        if self.reference_predictions is not None:
            reference_data["prediction"] = self.reference_predictions
        if self.reference_targets is not None:
            reference_data["target"] = self.reference_targets

        # Run drift detection
        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=reference_data, current_data=current_data)
        result_dict = snapshot.dict()

        if "metrics" not in result_dict or len(result_dict["metrics"]) == 0:
            raise KeyError("No metrics found in snapshot results")

        drift_count_metric = None
        for metric in result_dict["metrics"]:
            metric_name = metric.get("metric_name", "")
            if "DriftedColumnsCount" in metric_name:
                drift_count_metric = metric
                break

        if drift_count_metric is None:
            raise KeyError("DriftedColumnsCount metric not found in results")

        # Extract drift info from the value field
        value = drift_count_metric.get("value", {})
        if not isinstance(value, dict):
            raise ValueError(
                f"Expected dict for DriftedColumnsCount value, got {type(value)}"
            )

        # Extract metrics
        n_drifted_columns = int(value.get("count", 0))
        drift_share = float(value.get("share", 0.0))

        # Determine if dataset has drift (if any columns drifted)
        dataset_drift = drift_share > self.drift_share_threshold

        # Get total number of columns from current data
        n_columns = len(current_data.columns)

        self._drift_history.append(drift_share)

        # Compute average drift score
        recent_window = min(10, len(self._drift_history))
        drift_score = np.mean(self._drift_history[-recent_window:])

        # Determine regime
        if not dataset_drift and drift_score < self.minor_threshold:
            regime = LearningRegime.STABLE
        elif drift_score < self.minor_threshold:
            regime = LearningRegime.CONTINUAL_LEARNING
        elif drift_score < self.moderate_threshold:
            regime = LearningRegime.FINE_TUNING
        else:
            regime = LearningRegime.RETRAIN

        metadata = {
            "dataset_drift": dataset_drift,
            "drift_share": drift_share,
            "n_drifted_columns": n_drifted_columns,
            "n_columns": n_columns,
        }

        return DriftSignal(
            regime=regime,
            drift_detected=dataset_drift,
            drift_score=float(drift_score),
            confidence=0.95,  # Evidently uses statistical tests with high confidence
            metadata=metadata,
        )

    def _simple_value_detection(self, value: float) -> DriftSignal:
        """Fallback simple value-based detection."""
        self._drift_history.append(value)
        recent_window = min(10, len(self._drift_history))
        drift_score = np.mean(self._drift_history[-recent_window:])

        if drift_score < self.minor_threshold:
            regime = LearningRegime.STABLE
        elif drift_score < self.moderate_threshold:
            regime = LearningRegime.CONTINUAL_LEARNING
        elif drift_score < 0.8:
            regime = LearningRegime.FINE_TUNING
        else:
            regime = LearningRegime.RETRAIN

        return DriftSignal(
            regime=regime,
            drift_detected=bool(drift_score > self.drift_share_threshold),
            drift_score=float(drift_score),
        )

    def reset(self) -> None:
        """Reset detector (keeps reference data, clears history)."""
        self._drift_history = []


class EnsembleDetector(BaseDriftDetector):
    """
    Ensemble of multiple drift detectors with voting mechanism.

    Combines signals from multiple detectors to make more robust decisions.
    """

    def __init__(
        self,
        detectors: List[BaseDriftDetector],
        voting: str = "majority",
        name: str = "Ensemble",
    ):
        """
        Initialize ensemble detector.

        Args:
            detectors: List of individual detectors
            voting: Voting strategy ('majority', 'unanimous', 'any', 'weighted')
            name: Detector name
        """
        super().__init__(name)
        self.detectors = detectors
        self.voting = voting
        self._is_initialized = all(d._is_initialized for d in detectors)

    def update(self, value: float, **kwargs) -> DriftSignal:
        """
        Update all detectors and combine signals.

        Args:
            value: Metric value
            **kwargs: Additional parameters passed to individual detectors

        Returns:
            Combined DriftSignal
        """
        # Update all detectors
        signals = []
        detector_names = []
        for detector in self.detectors:
            signal = detector.update(value, **kwargs)
            signals.append(signal)
            detector_names.append(detector.name)

        # Combine signals based on voting strategy
        if self.voting == "majority":
            drift_detected = sum(s.drift_detected for s in signals) > len(signals) / 2
        elif self.voting == "unanimous":
            drift_detected = all(s.drift_detected for s in signals)
        elif self.voting == "any":
            drift_detected = any(s.drift_detected for s in signals)
        else:  # weighted - use average drift score
            drift_detected = bool(np.mean([s.drift_score for s in signals]) > 0.5)

        # Average drift scores
        avg_drift_score = np.mean([s.drift_score for s in signals])

        # Determine regime by majority vote
        regime_votes = [s.regime for s in signals]
        regime = max(set(regime_votes), key=regime_votes.count)

        # Combine metadata
        metadata = {
            "n_detectors": len(signals),
            "individual_signals": [
                {"detector": name, "detected": signal.drift_detected}
                for name, signal in zip(detector_names, signals)
            ],
        }

        return DriftSignal(
            regime=regime,
            drift_detected=drift_detected,
            drift_score=float(avg_drift_score),
            confidence=float(
                np.mean([s.confidence for s in signals if s.confidence is not None])
            ),
            metadata=metadata,
        )

    def reset(self) -> None:
        """Reset all detectors."""
        for detector in self.detectors:
            detector.reset()
