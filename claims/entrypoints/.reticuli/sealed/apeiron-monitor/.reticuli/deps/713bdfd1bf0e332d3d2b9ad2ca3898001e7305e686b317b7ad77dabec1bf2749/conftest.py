"""The one fixture these tests use, authored for this claim.

The repository's shared conftest also defines harness fixtures, which pull
the model layer into every room; this claim pins only what its tests
consume. The fixture body matches the repository's `default_cfg` exactly.
"""

from __future__ import annotations

import pytest

from apeiron.config.configuration import (
    Config,
    ContinualLearningCfg,
    DataCfg,
    DriftDetectionCfg,
    ModelCfg,
    TrainCfg,
)


@pytest.fixture()
def default_cfg() -> Config:
    """A minimal Config suitable for CPU-only unit tests."""
    return Config(
        model=ModelCfg(name="tiny", pretrained_path=""),
        data=DataCfg(name="test", path="/tmp"),
        train=TrainCfg(batch_size=8, num_workers=0, init_lr=0.01, max_iter=2),
        continual_learning=ContinualLearningCfg(),
        drift_detection=DriftDetectionCfg(),
        seed=42,
        device="cpu",
        multi_gpu=False,
    )
