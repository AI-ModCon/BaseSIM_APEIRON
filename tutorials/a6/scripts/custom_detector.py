"""A drift detector you can read in one minute, as a starting point for your own.

ThresholdDetector learns a baseline from the first `warmup` checks, then fires when
the monitored metric stays worse than `baseline - k * std` for `patience` checks in
a row. After it fires it re-learns the baseline, so it measures drift relative to
the adapted model rather than the original one.

Use it on the MNIST stream (from the Apeiron repo root):

    python tutorials/a6/scripts/a6_run.py --config tutorials/a6/configs/a6_mnist.toml \
        --detector tutorials/a6/scripts/custom_detector.py:ThresholdDetector \
        --set logging.metrics_output_path=output/a6/lab2_threshold.csv

Any detector only needs two methods: update(value) -> DriftSignal, and reset().
"""

from __future__ import annotations

import statistics

from apeiron.drift_detection.detectors.base import (
    BaseDriftDetector,
    DriftSignal,
    LearningRegime,
)


class ThresholdDetector(BaseDriftDetector):
    def __init__(
        self,
        warmup: int = 20,
        k: float = 3.0,
        patience: int = 3,
        higher_is_better: bool = True,  # accuracy: True; error metrics such as MAE: False
    ):
        super().__init__(name="ThresholdDetector")
        self.warmup, self.k, self.patience = warmup, k, patience
        self.sign = 1.0 if higher_is_better else -1.0
        self.reset()

    def reset(self) -> None:
        self.history: list[float] = []
        self.baseline: float | None = None
        self.spread: float = 0.0
        self.bad_streak = 0

    def update(self, value: float, **kwargs) -> DriftSignal:
        if self.baseline is None:
            self.history.append(value)
            if len(self.history) >= self.warmup:
                self.baseline = statistics.fmean(self.history)
                self.spread = statistics.pstdev(self.history) or 1e-8
            return DriftSignal(
                LearningRegime.STABLE, False, 0.0, metadata={"phase": "warmup"}
            )

        # How many standard deviations worse than the baseline is this check?
        score = self.sign * (self.baseline - value) / self.spread
        self.bad_streak = self.bad_streak + 1 if score > self.k else 0
        detected = self.bad_streak >= self.patience

        if score > 3 * self.k:
            regime = LearningRegime.RETRAIN
        elif score > 2 * self.k:
            regime = LearningRegime.FINE_TUNING
        elif detected:
            regime = LearningRegime.CONTINUAL_LEARNING
        else:
            regime = LearningRegime.STABLE

        signal = DriftSignal(
            regime=regime if detected else LearningRegime.STABLE,
            drift_detected=detected,
            drift_score=max(score, 0.0),
            metadata={"baseline": self.baseline, "streak": self.bad_streak},
        )
        if detected:
            self.reset()  # re-learn the baseline on the adapted model
        return signal
