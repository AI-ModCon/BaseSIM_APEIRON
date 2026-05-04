"""Apeiron: A PyTorch continual learning framework for real-time concept drift detection and model adaptation."""

from apeiron.config.configuration import (
    Config,
    ModelCfg,
    DataCfg,
    TrainCfg,
    ContinualLearningCfg,
    DriftDetectionCfg,
    VisualizationCfg,
    LoggingCfg,
    build_config,
)
from apeiron.model.torch_model_harness import BaseModelHarness
from apeiron.driver.continuous_monitor import ContinuousMonitor
from apeiron.drift_detection import (
    BaseDriftDetector,
    DriftSignal,
    LearningRegime,
    ADWINDetector,
    KSWINDetector,
    PageHinkleyDetector,
    ModelPerformanceDetector,
    EnsembleDetector,
    ModelEvalDetector,
)
from apeiron.training import ContinuousTrainer
from apeiron.training.updater import BaseUpdater
from apeiron.logger import Logger, get_logger

__all__ = [
    "Config",
    "ModelCfg",
    "DataCfg",
    "TrainCfg",
    "ContinualLearningCfg",
    "DriftDetectionCfg",
    "VisualizationCfg",
    "LoggingCfg",
    "build_config",
    "BaseModelHarness",
    "ContinuousMonitor",
    "BaseDriftDetector",
    "DriftSignal",
    "LearningRegime",
    "ADWINDetector",
    "KSWINDetector",
    "PageHinkleyDetector",
    "ModelPerformanceDetector",
    "EnsembleDetector",
    "ModelEvalDetector",
    "ContinuousTrainer",
    "BaseUpdater",
    "Logger",
    "get_logger",
]
