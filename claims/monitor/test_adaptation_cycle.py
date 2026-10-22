"""The branches of one full adaptation cycle: drift, CL, checkpoint, resume.

`_handle_drift` and `inner_cl_training_loop` carry optional work that only
runs when a config asks for it — a checkpoint after adapting, a detector
reset, and the FLOPs profiler around the forward/backward and the optimizer
step. None of it ran under test before, so the features were reachable only
from a real experiment.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from apeiron.config.configuration import DriftDetectionCfg, ModelCfg
from apeiron.driver.continuous_monitor import ContinuousMonitor
from apeiron.profilers import FLOPSProfiler
from apeiron.training.continuous_trainer import ContinuousTrainer


def _loader(n: int = 16, batch_size: int = 8) -> DataLoader:
    g = torch.Generator().manual_seed(0)
    return DataLoader(
        TensorDataset(
            torch.randn(n, 4, generator=g), torch.randint(0, 3, (n,), generator=g)
        ),
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# ContinuousMonitor._handle_drift
# ---------------------------------------------------------------------------
class TestHandleDrift:
    def _monitor(self, make_harness, default_cfg, *, tmp_path=None, reset=False):
        cfg = replace(
            default_cfg,
            drift_detection=DriftDetectionCfg(
                detector_name="PageHinkleyDetector", reset_after_learning=reset
            ),
            model=ModelCfg(
                name="tiny",
                pretrained_path="",
                max_ckpts=2 if tmp_path else 0,
                ckpts_path=str(tmp_path / "ckpts") if tmp_path else "",
            ),
            train=replace(default_cfg.train, max_iter=1),
        )
        harness = make_harness(cfg)
        monitor = ContinuousMonitor(cfg=cfg, modelHarness=harness)
        # the CL loop itself is covered by the trainer's own tests
        monitor.trainer.outer_cl_training_loop = MagicMock()
        return monitor, harness

    def _signal(self, monitor):
        monitor.metric_buffer = [[1.0]]
        return monitor._check_drift()

    def test_counts_the_event_and_runs_the_cl_loop(self, make_harness, default_cfg):
        monitor, _ = self._monitor(make_harness, default_cfg)
        monitor._handle_drift(self._signal(monitor))
        assert monitor.drift_event_count == 1
        monitor.trainer.outer_cl_training_loop.assert_called_once_with(drift_event_id=1)

    def test_checkpoints_when_enabled(self, make_harness, default_cfg, tmp_path):
        monitor, _ = self._monitor(make_harness, default_cfg, tmp_path=tmp_path)
        monitor._handle_drift(self._signal(monitor))
        saved = list((tmp_path / "ckpts").glob("drift_adaptation_*.pt"))
        assert [p.name for p in saved] == ["drift_adaptation_1.pt"]

    def test_no_checkpoint_when_disabled(self, make_harness, default_cfg, tmp_path):
        monitor, _ = self._monitor(make_harness, default_cfg)
        monitor._handle_drift(self._signal(monitor))
        assert not (tmp_path / "ckpts").exists()

    def test_reset_after_learning_resets_the_detector(self, make_harness, default_cfg):
        monitor, _ = self._monitor(make_harness, default_cfg, reset=True)
        signal = self._signal(monitor)
        assert monitor.detector._drift_history, "the detector recorded a check"
        monitor._handle_drift(signal)
        assert monitor.detector._drift_history == []

    def test_detector_state_survives_when_reset_is_off(self, make_harness, default_cfg):
        monitor, _ = self._monitor(make_harness, default_cfg, reset=False)
        signal = self._signal(monitor)
        history = list(monitor.detector._drift_history)
        monitor._handle_drift(signal)
        assert monitor.detector._drift_history == history

    def test_an_optional_window_timerange_is_reported(self, make_harness, default_cfg):
        monitor, harness = self._monitor(make_harness, default_cfg)
        harness.current_window_timerange = ("2026-01-01", "2026-01-02")
        monitor._handle_drift(self._signal(monitor))  # must not raise
        assert monitor.drift_event_count == 1


# ---------------------------------------------------------------------------
# ContinuousTrainer: the profiled paths and the batch-shape fallback
# ---------------------------------------------------------------------------
class TestProfiledTrainingStep:
    def _trainer(self, cfg, harness, profiler):
        return ContinuousTrainer(
            cfg=cfg, modelHarness=harness, logger=MagicMock(), profiler=profiler
        )

    def test_unprofiled_step_trains(self, default_cfg, dummy_harness):
        trainer = self._trainer(default_cfg, dummy_harness, None)
        loader = _loader()
        loss, reg, train_iter, _ = trainer.inner_cl_training_loop(
            iter_count=1, cur_train_loader=loader, train_iter=iter(loader)
        )
        assert isinstance(loss, float)
        assert isinstance(reg, float)
        assert train_iter is not None

    def test_warmed_up_profiler_measures_fwd_bwd_and_the_optimizer(
        self, default_cfg, dummy_harness
    ):
        # warmup_iters=0 with iter_count=1 puts both profiled branches in play
        profiler = FLOPSProfiler(warmup_iters=0)
        trainer = self._trainer(default_cfg, dummy_harness, profiler)
        loader = _loader()
        trainer.inner_cl_training_loop(
            iter_count=1, cur_train_loader=loader, train_iter=iter(loader)
        )
        assert "update_fwd_bwd" in profiler.profiles
        assert "optimizer" in profiler.profiles
        assert profiler.profiles["update_fwd_bwd"]["flop"]
        assert profiler.profiles["optimizer"]["flop"]

    def test_during_warmup_nothing_is_profiled(self, default_cfg, dummy_harness):
        profiler = FLOPSProfiler(warmup_iters=10)
        trainer = self._trainer(default_cfg, dummy_harness, profiler)
        loader = _loader()
        trainer.inner_cl_training_loop(
            iter_count=1, cur_train_loader=loader, train_iter=iter(loader)
        )
        assert "update_fwd_bwd" not in profiler.profiles
        assert "optimizer" not in profiler.profiles

    def test_an_uninspectable_batch_is_accepted(self, default_cfg, dummy_harness):
        # _safe_next enforces min_batch by reading batch[1].shape; a batch
        # whose second element has no shape must be accepted rather than
        # spinning the loader forever
        trainer = self._trainer(default_cfg, dummy_harness, None)

        class _ShapelessBatch(list):
            def __getitem__(self, index):
                if index == 1:
                    raise TypeError("not indexable the way you hoped")
                return super().__getitem__(index)

        batch = _ShapelessBatch([torch.randn(8, 4), torch.randint(0, 3, (8,))])

        class _OneBatchLoader:
            def __iter__(self):
                return iter([batch])

        loader = _OneBatchLoader()
        _, returned = trainer._safe_next(
            iter(loader), loader, min_batch=default_cfg.train.batch_size
        )
        assert len(returned) == 2

    def test_a_window_smaller_than_one_batch_refuses(self, default_cfg, dummy_harness):
        trainer = self._trainer(default_cfg, dummy_harness, None)
        tiny = _loader(n=2, batch_size=2)
        with pytest.raises(ValueError, match="smaller than one batch"):
            trainer._safe_next(iter(tiny), tiny, min_batch=64)
