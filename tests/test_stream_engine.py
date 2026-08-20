"""End-to-end tests: all three monitoring modes through one StreamEngine.run().

Drives the unified loop over a real WindowedHarness (memmap windows -> real
loaders) with a deterministic fire policy, exercising interval vs window cadence
and the adapt vs record-only actions.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch.nn as nn
from torch.optim import SGD
from unittest.mock import MagicMock

from apeiron.data.window_store import WindowStore
from apeiron.data.windowed_harness import WindowedHarness
from apeiron.drift_detection.detectors.base import DriftSignal, LearningRegime
from apeiron.driver.schedules import PeriodicSchedule
from apeiron.driver.stream_engine import StreamEngine
from apeiron.driver.trigger_action import AdaptAction, RecordOnlyAction
from apeiron.driver.trigger_policy import SchedulePolicy, TriggerPolicy
from apeiron.evaluation.metrics import accuracy


class FixedFirePolicy(TriggerPolicy):
    """Deterministic policy: fire at a fixed set of decision indices."""

    profile_decision = False

    def __init__(self, fire_at, cadence="interval"):
        self.fire_at = set(fire_at)
        self.cadence = cadence
        self.label = f"fixed-fire{sorted(self.fire_at)}"

    def decide(self, agg_metric, decision_idx):
        return DriftSignal(
            regime=LearningRegime.CONTINUAL_LEARNING,
            drift_detected=decision_idx in self.fire_at,
            drift_score=1.0,
        )


def _store(root, n_windows=3, n=20, seed=0):
    rng = np.random.default_rng(seed)
    store = WindowStore(root, catalog=False)
    for i in range(n_windows):
        x = rng.standard_normal((n, 4)).astype(np.float32)
        y = rng.integers(0, 3, n).astype(np.int64)
        store.commit(
            x,
            y,
            val_fraction=0.5,
            t_start=f"2026-03-{i + 1:02d}",
            t_end=f"2026-03-{i + 1:02d}T12",
        )
    return store


def _harness(cfg, store):
    return WindowedHarness(
        cfg,
        nn.Linear(4, 4),
        store,
        criterion=nn.CrossEntropyLoss(),
        optimizer_factory=lambda m: SGD(m.parameters(), lr=0.05),
        eval_metrics={"accuracy": accuracy},
    )


def _cfg(default_cfg, **dd):
    return replace(
        default_cfg,
        data=replace(default_cfg.data, batch_size=5),
        drift_detection=replace(default_cfg.drift_detection, **dd),
    )


class TestIntervalCadence:
    def test_detector_adapt_run_processes_all_windows(self, default_cfg, tmp_path):
        # 3 windows x 20 samples, stream batch 5 -> 4 batches/window; interval 4
        # -> one decision at each window boundary (3 decisions total).
        cfg = _cfg(default_cfg, detection_interval=4, max_stream_updates=3)
        store = _store(tmp_path / "s")
        engine = StreamEngine(
            cfg,
            _harness(cfg, store),
            FixedFirePolicy(fire_at={1}, cadence="interval"),
            AdaptAction(),
            logger=MagicMock(),
        )
        summary = engine.run()
        assert summary["stream_updates"] == 3
        assert summary["batches"] == 12
        assert summary["decision_points"] == 3
        assert summary["fires"] == 1
        assert summary["fire_points"] == [1]

    def test_record_only_run_leaves_model_frozen(self, default_cfg, tmp_path):
        cfg = _cfg(default_cfg, detection_interval=4, max_stream_updates=3)
        store = _store(tmp_path / "s")
        harness = _harness(cfg, store)
        before = [p.clone() for p in harness.model.parameters()]

        engine = StreamEngine(
            cfg,
            harness,
            FixedFirePolicy(fire_at={0, 2}, cadence="interval"),
            RecordOnlyAction(),
            logger=MagicMock(),
        )
        summary = engine.run()
        assert engine.trainer is None
        assert summary["fires"] == 2
        assert len(summary["events"]) == 2  # one recorded event per fire
        after = list(harness.model.parameters())
        assert all((b == a).all() for b, a in zip(before, after))  # frozen


class TestWindowCadence:
    def test_schedule_decides_once_per_window(self, default_cfg, tmp_path):
        cfg = _cfg(default_cfg, detection_interval=2, max_stream_updates=3)
        store = _store(tmp_path / "s")
        engine = StreamEngine(
            cfg,
            _harness(cfg, store),
            SchedulePolicy(PeriodicSchedule(1)),  # fire every window
            AdaptAction(),
            logger=MagicMock(),
        )
        summary = engine.run()
        # Window cadence: exactly one decision per stream window regardless of
        # detection_interval, and period 1 fires on all of them.
        assert summary["decision_points"] == 3
        assert summary["fires"] == 3
        assert summary["fire_points"] == [0, 1, 2]

    def test_periodic_budget_matches(self, default_cfg, tmp_path):
        cfg = _cfg(default_cfg, max_stream_updates=4)
        store = _store(tmp_path / "s", n_windows=4)
        engine = StreamEngine(
            cfg,
            _harness(cfg, store),
            SchedulePolicy(PeriodicSchedule(2)),  # every 2nd window
            AdaptAction(),
            logger=MagicMock(),
        )
        summary = engine.run()
        assert summary["decision_points"] == 4
        assert summary["fire_points"] == [1, 3]


class TestSmallWindowRegression:
    def test_adapt_on_split_smaller_than_batch_size_terminates(
        self, default_cfg, tmp_path
    ):
        # Windows of 10 samples -> train/val splits of 5, smaller than
        # train.batch_size (8). Adapting used to spin forever in _safe_next;
        # it must now complete (degrading to a smaller step).
        cfg = _cfg(default_cfg, max_stream_updates=2)
        store = _store(tmp_path / "s", n_windows=2, n=10)
        engine = StreamEngine(
            cfg,
            _harness(cfg, store),
            SchedulePolicy(PeriodicSchedule(1)),  # adapt on every window
            AdaptAction(),
            logger=MagicMock(),
        )
        summary = engine.run()
        assert summary["fires"] == 2  # completed instead of hanging
