"""Tests for src/apeiron/determinism.py and the property it exists to provide:
two runs of one config produce the same metrics file.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
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
from apeiron.determinism import seed_everything, stable_seed
from apeiron.driver.continuous_monitor import ContinuousMonitor
from apeiron.drift_detection.load_drift_detector import load_drift_detector
from apeiron.evaluation.metrics import accuracy
from apeiron.logger import get_logger, reset_logger
from apeiron.model.torch_model_harness import BaseModelHarness


# ---------------------------------------------------------------------------
# stable_seed
# ---------------------------------------------------------------------------
class TestStableSeed:
    def test_same_parts_same_seed(self):
        assert stable_seed(42, "window", 3) == stable_seed(42, "window", 3)

    def test_different_parts_different_seed(self):
        assert stable_seed(42, "train") != stable_seed(42, "val")

    def test_order_matters(self):
        assert stable_seed(1, 2) != stable_seed(2, 1)

    def test_no_arithmetic_collision(self):
        # seed + window collides: (40, 2) and (41, 1) share a stream under
        # addition. Hashing keeps them apart.
        assert stable_seed(40, 2) != stable_seed(41, 1)

    def test_fits_torch_manual_seed(self):
        # Non-negative and within torch's accepted range.
        s = stable_seed("anything", 0)
        assert 0 <= s < 2**63
        torch.Generator().manual_seed(s)


# ---------------------------------------------------------------------------
# seed_everything
# ---------------------------------------------------------------------------
class TestSeedEverything:
    def test_repeats_every_stream(self):
        seed_everything(123)
        first = (
            torch.rand(4).tolist(),
            np.random.random(4).tolist(),
            [random.random() for _ in range(4)],
        )
        seed_everything(123)
        second = (
            torch.rand(4).tolist(),
            np.random.random(4).tolist(),
            [random.random() for _ in range(4)],
        )
        assert first == second

    def test_different_seeds_diverge(self):
        seed_everything(123)
        a = torch.rand(4)
        seed_everything(124)
        b = torch.rand(4)
        assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# KSWIN seed derivation
# ---------------------------------------------------------------------------
def _cfg(seed: int = 42, **dd_overrides: Any) -> Config:
    dd: dict[str, Any] = dict(
        detector_name="PageHinkleyDetector",
        detection_interval=2,
        aggregation="mean",
        metric_index=1,
        max_stream_updates=3,
        ph_min_instances=2,
        ph_delta=0.0,
        ph_threshold=0.1,
    )
    dd.update(dd_overrides)
    return Config(
        model=ModelCfg(name="tiny", pretrained_path=""),
        data=DataCfg(name="synthetic", path="/tmp", batch_size=8),
        train=TrainCfg(batch_size=8, num_workers=0, init_lr=0.05, max_iter=2),
        continual_learning=ContinualLearningCfg(),
        drift_detection=DriftDetectionCfg(**dd),
        seed=seed,
        device="cpu",
        multi_gpu=False,
    )


class TestKswinSeed:
    def test_unset_seed_derives_from_run_seed(self):
        cfg = _cfg(seed=7, detector_name="KSWINDetector")
        detector = load_drift_detector(cfg)
        assert detector.seed == stable_seed(7, "kswin")

    def test_derivation_is_stable(self):
        cfg = _cfg(seed=7, detector_name="KSWINDetector")
        assert load_drift_detector(cfg).seed == load_drift_detector(cfg).seed

    def test_explicit_seed_wins(self):
        cfg = _cfg(seed=7, detector_name="KSWINDetector", kswin_seed=99)
        assert load_drift_detector(cfg).seed == 99


# ---------------------------------------------------------------------------
# End to end: same config, same metrics file
# ---------------------------------------------------------------------------
class ShiftingStreamHarness(BaseModelHarness):
    """Harness whose stream distribution shifts each window, so drift fires
    and the continual-learning path runs. Every random draw (data synthesis,
    model init, shuffling, dropout) comes from the global generators that
    seed_everything() seeds.
    """

    def __init__(self, cfg: Config):
        model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 3),
        )
        super().__init__(cfg=cfg, model=model)
        self.eval_metrics = {"accuracy": accuracy, "loss": self.get_criterion()}
        self.window = 0
        self._stream: DataLoader | None = None
        self._train: DataLoader | None = None
        self._val: DataLoader | None = None

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self):
        return SGD(self.model.parameters(), lr=self.cfg.train.init_lr)

    def update_data_stream(self) -> None:
        n = 64
        shift = 2.0 * self.window
        x_train = torch.randn(n, 4) + shift
        y_train = torch.randint(0, 3, (n,))
        x_val = torch.randn(n, 4) + shift
        y_val = torch.randint(0, 3, (n,))
        self._train = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
        )
        self._val = DataLoader(
            TensorDataset(x_val, y_val), batch_size=self.cfg.train.batch_size
        )
        self._stream = DataLoader(
            TensorDataset(x_val, y_val),
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
        )
        self.window += 1

    def get_stream_dataloader(self):
        return self._stream

    def get_train_dataloaders(self):
        return self._train, self._val

    def get_hist_dataloaders(self):
        return None, None


def _run_once(seed: int, csv_path: Path) -> str:
    """One full monitoring run, the way an entry point sets it up."""
    reset_logger()
    try:
        seed_everything(seed)
        cfg = _cfg(seed=seed)
        logger = get_logger(verbosity="ERROR", backend="none", csv_path=str(csv_path))
        harness = ShiftingStreamHarness(cfg)
        logger.init(cfg, project="determinism-test")
        ContinuousMonitor(cfg=cfg, modelHarness=harness).run()
        logger.finish()
    finally:
        reset_logger()
    return csv_path.read_text()


def _behavioral_rows(csv_text: str) -> list[str]:
    """Drop profiler rows: cperf_* values are wall-clock measurements, which
    differ between two runs that did exactly the same work.
    """
    return [line for line in csv_text.splitlines() if "cperf" not in line]


class TestRunsAreDeterministic:
    def test_same_seed_same_metrics(self, tmp_path: Path):
        first = _run_once(42, tmp_path / "a.csv")
        second = _run_once(42, tmp_path / "b.csv")
        assert _behavioral_rows(first) == _behavioral_rows(second)

    def test_run_exercises_drift_and_learning(self, tmp_path: Path):
        # The equality above proves nothing if the run was trivial: make sure
        # drift checks happened and the continual-learning path actually ran.
        text = _run_once(42, tmp_path / "c.csv")
        assert "drift/" in text
        assert "cl/" in text

    def test_different_seed_different_metrics(self, tmp_path: Path):
        # The oracle can fail: a different seed must change the rows.
        first = _run_once(42, tmp_path / "d.csv")
        second = _run_once(43, tmp_path / "e.csv")
        assert _behavioral_rows(first) != _behavioral_rows(second)
