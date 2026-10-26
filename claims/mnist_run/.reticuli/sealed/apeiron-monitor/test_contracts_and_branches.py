"""The extension-point contracts and the conditional branches around them.

Three groups: what the abstract base classes refuse when a subclass does not
implement a hook, the guard clauses the updaters use to survive being driven
out of order, and the optional branches the monitor takes (checkpointing after
adaptation, detector reset, an optional harness time range, and the
aggregation fallback).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn as nn

from apeiron.config.configuration import ContinualLearningCfg, DriftDetectionCfg
from apeiron.drift_detection.detectors.base import BaseDriftDetector
from apeiron.model.torch_model_harness import BaseModelHarness
from apeiron.training.updater.ewc import OnlineEWCUpdater
from apeiron.training.updater.kfac import OnlineKFACUpdater


# ---------------------------------------------------------------------------
# BaseModelHarness: the contract a subclass must honour
# ---------------------------------------------------------------------------
class TestHarnessContract:
    def _bare(self, default_cfg):
        """A subclass that declares the abstract hooks but delegates to super."""

        class Bare(BaseModelHarness):
            def get_optimizer(self):
                return super().get_optimizer()

            def update_data_stream(self):
                return super().update_data_stream()

            def get_stream_dataloader(self):
                return super().get_stream_dataloader()

            def get_train_dataloaders(self):
                return super().get_train_dataloaders()

            def get_hist_dataloaders(self):
                return super().get_hist_dataloaders()

            def get_criterion(self):
                return super().get_criterion()

        return Bare(default_cfg, nn.Linear(4, 3))

    @pytest.mark.parametrize(
        "hook",
        [
            "get_optimizer",
            "update_data_stream",
            "get_stream_dataloader",
            "get_train_dataloaders",
            "get_hist_dataloaders",
            "get_criterion",
        ],
    )
    def test_unimplemented_hooks_refuse(self, default_cfg, hook):
        harness = self._bare(default_cfg)
        with pytest.raises(NotImplementedError):
            getattr(harness, hook)()

    def test_empty_loader_refuses_rather_than_dividing_by_zero(
        self, make_harness, default_cfg
    ):
        from torch.utils.data import DataLoader, TensorDataset

        empty = TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long))
        harness = make_harness(default_cfg, val_data=empty)
        with pytest.raises(RuntimeError, match="Empty loader"):
            harness._eval_loader(DataLoader(empty, batch_size=4))


# ---------------------------------------------------------------------------
# BaseDriftDetector: the contract and its repr
# ---------------------------------------------------------------------------
class TestDetectorContract:
    def _bare(self):
        class Bare(BaseDriftDetector):
            def update(self, value, **kwargs):
                return super().update(value)

            def reset(self):
                return super().reset()

        return Bare("bare")

    def test_update_refuses_and_names_the_detector(self):
        with pytest.raises(NotImplementedError, match="bare"):
            self._bare().update(1.0)

    def test_reset_refuses_and_names_the_detector(self):
        with pytest.raises(NotImplementedError, match="bare"):
            self._bare().reset()

    def test_repr_names_the_class_and_the_detector(self):
        assert repr(self._bare()) == "Bare(name='bare')"


# ---------------------------------------------------------------------------
# Updater guard clauses: driven out of order, they must not corrupt state
# ---------------------------------------------------------------------------
class TestUpdaterGuards:
    def _cfg(self, default_cfg, mode):
        return replace(
            default_cfg, continual_learning=ContinualLearningCfg(update_mode=mode)
        )

    def test_ewc_commit_without_a_cl_loop_is_a_no_op(self, make_harness, default_cfg):
        cfg = self._cfg(default_cfg, "ewc_online")
        upd = OnlineEWCUpdater(cfg, make_harness(cfg))
        before = {n: f.clone() for n, f in upd.fisher.items()}
        upd.cl_postprocessing()  # never preprocessed: nothing to commit
        for name, fisher in upd.fisher.items():
            assert torch.equal(fisher, before[name])

    def test_ewc_commit_without_steps_is_a_no_op(self, make_harness, default_cfg):
        cfg = self._cfg(default_cfg, "ewc_online")
        upd = OnlineEWCUpdater(cfg, make_harness(cfg))
        upd.cl_preprocessing()
        upd.cl_postprocessing()  # preprocessed but zero steps taken
        assert all(float(f.abs().sum()) == 0.0 for f in upd.fisher.values())

    def test_ewc_skips_parameters_without_gradients(self, make_harness, default_cfg):
        cfg = self._cfg(default_cfg, "ewc_online")
        harness = make_harness(cfg)
        upd = OnlineEWCUpdater(cfg, harness)
        upd.cl_preprocessing()
        upd.update_pre_fwd_bwd()
        # no backward pass ran, so every p.grad is None
        assert upd.update_post_fwd_bwd() == 0.0

    def test_kfac_commit_without_a_cl_loop_is_a_no_op(self, make_harness, default_cfg):
        cfg = self._cfg(default_cfg, "kfac_online")
        upd = OnlineKFACUpdater(cfg, make_harness(cfg))
        upd.cl_postprocessing()
        assert all(float(a.abs().sum()) == 0.0 for a in upd.A.values())

    def test_kfac_commit_without_steps_is_a_no_op(self, make_harness, default_cfg):
        cfg = self._cfg(default_cfg, "kfac_online")
        upd = OnlineKFACUpdater(cfg, make_harness(cfg))
        upd.cl_preprocessing()
        upd.cl_postprocessing()
        assert all(float(a.abs().sum()) == 0.0 for a in upd.A.values())

    def test_kfac_skips_layers_it_never_saw(self, make_harness, default_cfg):
        cfg = self._cfg(default_cfg, "kfac_online")
        upd = OnlineKFACUpdater(cfg, make_harness(cfg))
        upd.cl_preprocessing()
        # no forward/backward ran, so the activation and grad caches are empty
        assert upd.update_post_fwd_bwd() == 0.0


# ---------------------------------------------------------------------------
# ContinuousMonitor's optional branches
# ---------------------------------------------------------------------------
class TestMonitorBranches:
    def _monitor(self, make_harness, default_cfg, **dd):
        from apeiron.driver.continuous_monitor import ContinuousMonitor

        cfg = replace(default_cfg, drift_detection=DriftDetectionCfg(**dd))
        harness = make_harness(cfg)
        return ContinuousMonitor(cfg=cfg, modelHarness=harness), harness, cfg

    def test_unknown_aggregation_falls_back_to_the_mean(
        self, make_harness, default_cfg
    ):
        # worth knowing: a misspelled aggregation is not refused, it becomes
        # the mean (LOG A20)
        monitor, _, _ = self._monitor(
            make_harness, default_cfg, aggregation="geometric-mean"
        )
        monitor.metric_buffer = [[1.0], [2.0], [6.0]]
        signal = monitor._check_drift()
        assert signal is not None
        assert monitor.metric_buffer == []

    def _aggregate(self, monitor, buffer):
        """Run one drift check, returning the value handed to the detector."""
        seen = []
        real = monitor.detector.update
        monitor.detector.update = lambda value, **kw: (  # type: ignore[method-assign]
            seen.append(value),
            real(value, **kw),
        )[1]
        monitor.metric_buffer = list(buffer)
        monitor._check_drift()
        assert len(seen) == 1
        return seen[0]

    @pytest.mark.parametrize(
        "aggregation,expected", [("mean", 3.0), ("median", 2.0), ("last", 6.0)]
    )
    def test_declared_aggregations(
        self, make_harness, default_cfg, aggregation, expected
    ):
        monitor, _, _ = self._monitor(
            make_harness, default_cfg, aggregation=aggregation
        )
        value = self._aggregate(monitor, [[1.0], [2.0], [6.0]])
        assert value == pytest.approx(expected)

    def test_the_fallback_aggregates_as_the_mean(self, make_harness, default_cfg):
        monitor, _, _ = self._monitor(
            make_harness, default_cfg, aggregation="geometric-mean"
        )
        assert self._aggregate(monitor, [[1.0], [2.0], [6.0]]) == pytest.approx(3.0)

    def test_empty_buffer_refuses(self, make_harness, default_cfg):
        monitor, _, _ = self._monitor(make_harness, default_cfg)
        monitor.metric_buffer = []
        with pytest.raises(RuntimeError, match="evaluation metrics"):
            monitor._check_drift()

    def test_an_optional_window_timerange_rides_into_the_log(
        self, make_harness, default_cfg
    ):
        monitor, harness, _ = self._monitor(make_harness, default_cfg)
        rows: list[dict] = []
        monitor.logger.log = lambda metrics, *a, **kw: rows.append(metrics)  # type: ignore[method-assign]

        monitor.metric_buffer = [[1.0]]
        monitor._check_drift()
        assert rows and "data_time_start" not in rows[0]

        harness.current_window_timerange = ("2026-01-01", "2026-01-02")
        rows.clear()
        monitor.metric_buffer = [[1.0]]
        monitor._check_drift()
        assert rows[0]["data_time_start"] == "2026-01-01"
        assert rows[0]["data_time_end"] == "2026-01-02"


def test_numpy_aggregation_helpers_agree_with_the_monitor():
    # the monitor aggregates with numpy; this pins the arithmetic the
    # fallback and the declared paths share
    values = [1.0, 2.0, 6.0]
    assert float(np.mean(values)) == pytest.approx(3.0)
    assert float(np.median(values)) == pytest.approx(2.0)
