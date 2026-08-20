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
from apeiron.data import (
    WindowStore,
    WindowHandle,
    WindowManifest,
    WindowSplit,
    WindowCatalog,
    WindowedHarness,
)
from apeiron.driver.continuous_monitor import ContinuousMonitor
from apeiron.driver.stream_engine import StreamEngine
from apeiron.driver.trigger_policy import (
    TriggerPolicy,
    DetectorPolicy,
    SchedulePolicy,
)
from apeiron.driver.trigger_action import (
    TriggerAction,
    AdaptAction,
    RecordOnlyAction,
)
from apeiron.driver.schedules import (
    TriggerSchedule,
    NeverSchedule,
    PeriodicSchedule,
    RandomSchedule,
    FixedSchedule,
)
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
from apeiron.distributed import comm

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
    "WindowStore",
    "WindowHandle",
    "WindowManifest",
    "WindowSplit",
    "WindowCatalog",
    "WindowedHarness",
    "ContinuousMonitor",
    "StreamEngine",
    "TriggerPolicy",
    "DetectorPolicy",
    "SchedulePolicy",
    "TriggerAction",
    "AdaptAction",
    "RecordOnlyAction",
    "TriggerSchedule",
    "NeverSchedule",
    "PeriodicSchedule",
    "RandomSchedule",
    "FixedSchedule",
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
    "comm",
]
