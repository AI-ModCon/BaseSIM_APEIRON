"""Shared fixtures for BaseSim tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from apeiron.config.configuration import (
    Config,
    ContinualLearningCfg,
    DataCfg,
    DriftDetectionCfg,
    ModelCfg,
    TrainCfg,
)
from apeiron.model.torch_model_harness import BaseModelHarness


# ---------------------------------------------------------------------------
# Tiny model for tests
# ---------------------------------------------------------------------------
class TinyModel(nn.Module):
    """Minimal model for unit tests."""

    def __init__(self, in_features: int = 4, num_classes: int = 3):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TinyCNN(nn.Module):
    """Minimal CNN for updater tests that need Conv2d layers."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Concrete harness for tests
# ---------------------------------------------------------------------------
class DummyHarness(BaseModelHarness):
    """Concrete harness wrapping TinyModel for unit tests."""

    def __init__(
        self,
        cfg: Config,
        model: nn.Module | None = None,
        train_data: TensorDataset | None = None,
        val_data: TensorDataset | None = None,
        hist_train_data: TensorDataset | None = None,
        hist_val_data: TensorDataset | None = None,
    ):
        _model = model or TinyModel()
        super().__init__(cfg, _model)

        n = 32
        in_f = 4
        nc = 3
        self._train_ds = train_data or TensorDataset(
            torch.randn(n, in_f), torch.randint(0, nc, (n,))
        )
        self._val_ds = val_data or TensorDataset(
            torch.randn(n, in_f), torch.randint(0, nc, (n,))
        )
        self._hist_train_ds = hist_train_data
        self._hist_val_ds = hist_val_data

        from apeiron.evaluation.metrics import accuracy

        self.eval_metrics = {"accuracy": accuracy}

    def get_optmizer(self):
        return SGD(self.model.parameters(), lr=self.cfg.train.init_lr)

    def update_data_stream(self):
        pass

    def get_train_dataloaders(self):
        bs = self.cfg.train.batch_size
        return (
            DataLoader(self._train_ds, batch_size=bs),
            DataLoader(self._val_ds, batch_size=bs),
        )

    def get_hist_dataloaders(self):
        if self._hist_train_ds is None:
            return (None, None)
        bs = self.cfg.train.batch_size
        return (
            DataLoader(self._hist_train_ds, batch_size=bs),
            DataLoader(self._hist_val_ds, batch_size=bs) if self._hist_val_ds else None,
        )

    def get_criterion(self):
        return nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
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


@pytest.fixture()
def tiny_model() -> TinyModel:
    return TinyModel()


@pytest.fixture()
def tiny_cnn() -> TinyCNN:
    return TinyCNN()


@pytest.fixture()
def dummy_harness(default_cfg: Config) -> DummyHarness:
    """Harness with current data only (no history)."""
    return DummyHarness(default_cfg)


@pytest.fixture()
def make_harness() -> Callable[..., DummyHarness]:
    """Factory fixture: call with a Config to get a fresh DummyHarness."""
    return lambda cfg, **kwargs: DummyHarness(cfg, **kwargs)


@pytest.fixture()
def dummy_harness_with_history(default_cfg: Config) -> DummyHarness:
    """Harness with both current and historical data."""
    n, in_f, nc = 32, 4, 3
    return DummyHarness(
        default_cfg,
        hist_train_data=TensorDataset(torch.randn(n, in_f), torch.randint(0, nc, (n,))),
        hist_val_data=TensorDataset(torch.randn(n, in_f), torch.randint(0, nc, (n,))),
    )
