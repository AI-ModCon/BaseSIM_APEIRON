"""The package's public import surface.

`import apeiron` is how an external harness author reaches the framework, and
`apeiron.__all__` is the promise that goes with it — but nothing asserted
either, so a name could be dropped from the re-export list without a test
noticing. The eager imports behind this surface are also the subject of a
known cost: pulling in `Config` currently pulls in river, evidently, wandb
and mlflow too (claims/LOG.md, A1).
"""

from __future__ import annotations

import importlib

import pytest

EXPECTED = {
    # configuration
    "Config",
    "ModelCfg",
    "DataCfg",
    "TrainCfg",
    "ContinualLearningCfg",
    "DriftDetectionCfg",
    "LoggingCfg",
    "build_config",
    # extension points
    "BaseModelHarness",
    "BaseUpdater",
    "BaseDriftDetector",
    # drift detection
    "DriftSignal",
    "LearningRegime",
    "ADWINDetector",
    "KSWINDetector",
    "PageHinkleyDetector",
    "ModelPerformanceDetector",
    "ModelEvalDetector",
    "EnsembleDetector",
    # drivers
    "ContinuousMonitor",
    "ContinuousTrainer",
    # logging
    "Logger",
    "get_logger",
}


@pytest.fixture(scope="module")
def apeiron():
    return importlib.import_module("apeiron")


def test_all_lists_exactly_the_documented_surface(apeiron):
    assert set(apeiron.__all__) == EXPECTED


def test_every_exported_name_resolves(apeiron):
    missing = [name for name in apeiron.__all__ if not hasattr(apeiron, name)]
    assert missing == [], f"__all__ promises names the package does not bind: {missing}"


def test_no_duplicate_exports(apeiron):
    assert len(apeiron.__all__) == len(set(apeiron.__all__))


@pytest.mark.parametrize(
    "name,module",
    [
        ("Config", "apeiron.config.configuration"),
        ("build_config", "apeiron.config.configuration"),
        ("BaseModelHarness", "apeiron.model.torch_model_harness"),
        ("BaseUpdater", "apeiron.training.updater.base"),
        ("ContinuousTrainer", "apeiron.training.continuous_trainer"),
        ("ContinuousMonitor", "apeiron.driver.continuous_monitor"),
        ("ADWINDetector", "apeiron.drift_detection.detectors.statistical_detectors"),
        (
            "EnsembleDetector",
            "apeiron.drift_detection.detectors.model_performance_detector",
        ),
        ("Logger", "apeiron.logger.logger"),
    ],
)
def test_exports_are_the_same_objects_as_their_modules(apeiron, name, module):
    # re-exports, not copies: a harness author importing from either path must
    # get the same class, or isinstance checks across the boundary break
    assert getattr(apeiron, name) is getattr(importlib.import_module(module), name)


def test_the_model_package_re_exports_the_harness():
    import apeiron.model as model
    from apeiron.model.torch_model_harness import BaseModelHarness

    assert model.BaseModelHarness is BaseModelHarness


def test_the_driver_package_re_exports_the_monitor():
    import apeiron.driver as driver
    from apeiron.driver.continuous_monitor import ContinuousMonitor

    assert driver.ContinuousMonitor is ContinuousMonitor


def test_the_training_package_re_exports_the_trainer():
    import apeiron.training as training
    from apeiron.training.continuous_trainer import ContinuousTrainer

    assert training.ContinuousTrainer is ContinuousTrainer
