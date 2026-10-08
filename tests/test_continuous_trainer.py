"""Tests for src/apeiron/training/continuous_trainer.py"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import count
from unittest.mock import patch, MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from apeiron.training.continuous_trainer import ContinuousTrainer


@pytest.fixture(autouse=True)
def _patch_logger():
    mock_logger = MagicMock()
    mock_logger.step = 0
    with patch(
        "apeiron.training.continuous_trainer.get_logger", return_value=mock_logger
    ):
        yield mock_logger


class TestSafeNext:
    def test_returns_batch(self, default_cfg, dummy_harness):
        trainer = ContinuousTrainer(
            cfg=default_cfg,
            modelHarness=dummy_harness,
            logger=MagicMock(),
            profiler=None,
        )
        loader = DataLoader(
            TensorDataset(torch.randn(16, 4), torch.randint(0, 3, (16,))),
            batch_size=8,
        )
        it = iter(loader)
        it, batch = trainer._safe_next(it, loader)
        assert len(batch) == 2
        assert batch[0].shape[0] == 8

    def test_restarts_on_exhaustion(self, default_cfg, dummy_harness):
        trainer = ContinuousTrainer(
            cfg=default_cfg,
            modelHarness=dummy_harness,
            logger=MagicMock(),
            profiler=None,
        )
        loader = DataLoader(
            TensorDataset(torch.randn(8, 4), torch.randint(0, 3, (8,))),
            batch_size=8,
        )
        it = iter(loader)
        # Exhaust the loader
        next(it)
        # _safe_next should restart
        it, batch = trainer._safe_next(it, loader)
        assert len(batch) == 2

    def test_min_batch_enforcement(self, default_cfg, dummy_harness):
        trainer = ContinuousTrainer(
            cfg=default_cfg,
            modelHarness=dummy_harness,
            logger=MagicMock(),
            profiler=None,
        )
        loader = DataLoader(
            TensorDataset(torch.randn(16, 4), torch.randint(0, 3, (16,))),
            batch_size=8,
        )
        it = iter(loader)
        it, batch = trainer._safe_next(it, loader, min_batch=4)
        assert batch[1].shape[0] >= 4

    def test_an_unsatisfiable_min_batch_fails_instead_of_spinning(
        self, default_cfg, dummy_harness
    ):
        """``_safe_next`` rebuilds the loader until a batch meets ``min_batch``.

        When the window holds fewer samples than one batch, no batch ever can,
        and the loop used to have no exit: it rebuilt DataLoaders forever.
        """
        trainer = ContinuousTrainer(
            cfg=default_cfg,
            modelHarness=dummy_harness,
            logger=MagicMock(),
            profiler=None,
        )
        short = DataLoader(
            TensorDataset(torch.randn(5, 4), torch.randint(0, 3, (5,))), batch_size=8
        )

        rebuilds = count()
        real_iter = DataLoader.__iter__

        def bounded_iter(self):
            if next(rebuilds) > 4:
                raise AssertionError(
                    "_safe_next rebuilt the loader five times looking for a "
                    "batch that cannot exist; it is spinning, not failing"
                )
            return real_iter(self)

        with patch.object(DataLoader, "__iter__", bounded_iter):
            with pytest.raises(ValueError, match="min_batch"):
                trainer._safe_next(iter(short), short, min_batch=8)


class TestInnerCLLoop:
    def test_hook_call_order(self, default_cfg, dummy_harness):
        trainer = ContinuousTrainer(
            cfg=default_cfg,
            modelHarness=dummy_harness,
            logger=MagicMock(),
            profiler=None,
        )
        mock_updater = MagicMock()
        mock_updater.fwd_bwd.return_value = 0.5
        mock_updater.update_post_fwd_bwd.return_value = 0.1
        trainer.cl_updater = mock_updater

        cur_train, _ = dummy_harness.get_train_dataloaders()
        train_iter = iter(cur_train)

        trainer.inner_cl_training_loop(
            iter_count=0,
            cur_train_loader=cur_train,
            train_iter=train_iter,
        )

        call_names = [c[0] for c in mock_updater.method_calls]
        assert call_names == [
            "update_pre_fwd_bwd",
            "fwd_bwd",
            "update_post_fwd_bwd",
            "update_post_optimizer_call",
        ]

    def test_no_hist_passes_none(self, default_cfg, dummy_harness):
        trainer = ContinuousTrainer(
            cfg=default_cfg,
            modelHarness=dummy_harness,
            logger=MagicMock(),
            profiler=None,
        )
        mock_updater = MagicMock()
        mock_updater.fwd_bwd.return_value = 0.5
        mock_updater.update_post_fwd_bwd.return_value = 0.1
        trainer.cl_updater = mock_updater

        cur_train, _ = dummy_harness.get_train_dataloaders()
        train_iter = iter(cur_train)

        trainer.inner_cl_training_loop(
            iter_count=0,
            cur_train_loader=cur_train,
            train_iter=train_iter,
        )

        _, hist_batch = mock_updater.fwd_bwd.call_args[0]
        assert hist_batch is None

    def test_grad_accumulation(self, default_cfg, make_harness):
        cfg = replace(
            default_cfg,
            train=replace(default_cfg.train, grad_accumulation_steps=3),
        )
        harness = make_harness(cfg)
        trainer = ContinuousTrainer(
            cfg=cfg,
            modelHarness=harness,
            logger=MagicMock(),
            profiler=None,
        )
        mock_updater = MagicMock()
        mock_updater.fwd_bwd.return_value = 0.5
        mock_updater.update_post_fwd_bwd.return_value = 0.1
        trainer.cl_updater = mock_updater

        cur_train, _ = harness.get_train_dataloaders()
        train_iter = iter(cur_train)

        trainer.inner_cl_training_loop(
            iter_count=0,
            cur_train_loader=cur_train,
            train_iter=train_iter,
        )

        assert mock_updater.fwd_bwd.call_count == 3

    def test_parameters_update(self, default_cfg, dummy_harness):
        trainer = ContinuousTrainer(
            cfg=default_cfg,
            modelHarness=dummy_harness,
            logger=MagicMock(),
            profiler=None,
        )
        params_before = {
            name: p.clone() for name, p in dummy_harness.model.named_parameters()
        }

        cur_train, _ = dummy_harness.get_train_dataloaders()
        train_iter = iter(cur_train)

        trainer.inner_cl_training_loop(
            iter_count=0,
            cur_train_loader=cur_train,
            train_iter=train_iter,
        )

        any_changed = any(
            not torch.equal(p.data, params_before[name])
            for name, p in dummy_harness.model.named_parameters()
        )
        assert any_changed, "Model parameters should change after training"

    def test_returns_losses(self, default_cfg, dummy_harness):
        trainer = ContinuousTrainer(
            cfg=default_cfg,
            modelHarness=dummy_harness,
            logger=MagicMock(),
            profiler=None,
        )
        cur_train, _ = dummy_harness.get_train_dataloaders()
        train_iter = iter(cur_train)

        gen_loss, reg_loss, used_iter, hist_iter = trainer.inner_cl_training_loop(
            iter_count=0,
            cur_train_loader=cur_train,
            train_iter=train_iter,
        )
        assert isinstance(gen_loss, float)
        assert isinstance(reg_loss, float)
        # Where the step left off, so the next one can carry on from it.
        assert used_iter is not None
        assert hist_iter is None


class TestOuterCLLoop:
    def test_updater_lifecycle(self, default_cfg, make_harness):
        cfg = replace(
            default_cfg,
            train=replace(default_cfg.train, max_iter=2),
        )
        harness = make_harness(cfg)
        trainer = ContinuousTrainer(
            cfg=cfg,
            modelHarness=harness,
            logger=MagicMock(),
            profiler=None,
        )
        mock_updater = MagicMock()
        mock_updater.fwd_bwd.return_value = 0.5
        mock_updater.update_post_fwd_bwd.return_value = 0.1
        trainer.cl_updater = mock_updater

        trainer.outer_cl_training_loop(drift_event_id=1)

        mock_updater.cl_preprocessing.assert_called_once()
        mock_updater.cl_postprocessing.assert_called_once()

        # Verify preprocessing comes before any fwd_bwd, postprocessing after all
        call_names = [c[0] for c in mock_updater.method_calls]
        pre_idx = call_names.index("cl_preprocessing")
        post_idx = call_names.index("cl_postprocessing")
        fwd_indices = [i for i, name in enumerate(call_names) if name == "fwd_bwd"]
        assert pre_idx < min(fwd_indices)
        assert post_idx > max(fwd_indices)

    def test_runs_max_iter_iterations(self, default_cfg, make_harness):
        cfg = replace(
            default_cfg,
            train=replace(default_cfg.train, max_iter=3),
        )
        harness = make_harness(cfg)
        trainer = ContinuousTrainer(
            cfg=cfg,
            modelHarness=harness,
            logger=MagicMock(),
            profiler=None,
        )
        mock_updater = MagicMock()
        mock_updater.fwd_bwd.return_value = 0.5
        mock_updater.update_post_fwd_bwd.return_value = 0.1
        trainer.cl_updater = mock_updater

        trainer.outer_cl_training_loop(drift_event_id=1)

        assert mock_updater.fwd_bwd.call_count == 3

    def test_evaluates_before_and_after(self, default_cfg, make_harness):
        cfg = replace(
            default_cfg,
            train=replace(default_cfg.train, max_iter=2),
        )
        harness = make_harness(cfg)
        trainer = ContinuousTrainer(
            cfg=cfg,
            modelHarness=harness,
            logger=MagicMock(),
            profiler=None,
        )

        with (
            patch.object(harness, "eval", wraps=harness.eval) as mock_eval,
            patch.object(
                harness, "history_eval", wraps=harness.history_eval
            ) as mock_hist_eval,
        ):
            trainer.outer_cl_training_loop(drift_event_id=1)

        assert mock_eval.call_count == 2
        assert mock_hist_eval.call_count == 2

    def test_parameters_update(self, default_cfg, make_harness):
        cfg = replace(
            default_cfg,
            train=replace(default_cfg.train, max_iter=2),
        )
        harness = make_harness(cfg)
        trainer = ContinuousTrainer(
            cfg=cfg,
            modelHarness=harness,
            logger=MagicMock(),
            profiler=None,
        )
        params_before = {
            name: p.clone() for name, p in harness.model.named_parameters()
        }

        trainer.outer_cl_training_loop(drift_event_id=1)

        any_changed = any(
            not torch.equal(p.data, params_before[name])
            for name, p in harness.model.named_parameters()
        )
        assert any_changed, "Model parameters should change after training"

    def test_passes_history_batches(self, default_cfg, make_harness):
        cfg = replace(
            default_cfg,
            train=replace(default_cfg.train, max_iter=2),
        )
        harness = make_harness(
            cfg,
            hist_train_data=TensorDataset(
                torch.randn(32, 4), torch.randint(0, 3, (32,))
            ),
            hist_val_data=TensorDataset(torch.randn(32, 4), torch.randint(0, 3, (32,))),
        )
        trainer = ContinuousTrainer(
            cfg=cfg,
            modelHarness=harness,
            logger=MagicMock(),
            profiler=None,
        )
        mock_updater = MagicMock()
        mock_updater.fwd_bwd.return_value = 0.5
        mock_updater.update_post_fwd_bwd.return_value = 0.1
        trainer.cl_updater = mock_updater

        trainer.outer_cl_training_loop(drift_event_id=1)

        for c in mock_updater.fwd_bwd.call_args_list:
            _, hist_batch = c[0]
            assert hist_batch is not None


class TestTrainingLoaderPosition:
    """A round of learning walks its window; it does not restart it."""

    def test_a_round_walks_its_window_evenly(self, default_cfg, make_harness):
        """Twelve steps over a four-batch window is three passes, so every
        sample in the window is trained on exactly three times.

        ``inner_cl_training_loop`` advances the training iterator but used to
        return only the losses, so the outer loop handed the original iterator
        back on the next step. Once that iterator is exhausted, every later
        step rebuilds the loader and takes only its first batch: the first
        batch is then trained on nine times and the rest once each.
        """
        samples, batch, steps = 32, 8, 12
        cfg = replace(
            default_cfg,
            train=replace(default_cfg.train, batch_size=batch, max_iter=steps),
            continual_learning=replace(
                default_cfg.continual_learning, update_mode="base"
            ),
        )
        harness = make_harness(cfg)

        # x[i, 0] == i, so a batch reports exactly which samples it carried.
        features = torch.zeros(samples, 4)
        features[:, 0] = torch.arange(samples, dtype=torch.float32)
        window = TensorDataset(features, torch.randint(0, 3, (samples,)))
        # Unshuffled, and a whole number of batches: a restart then shows up as
        # a repeat rather than as noise.
        train = DataLoader(window, batch_size=batch)
        harness.get_train_dataloaders = lambda: (train, train)  # type: ignore[method-assign]
        harness.get_hist_dataloaders = lambda: (None, None)  # type: ignore[method-assign]

        trainer = ContinuousTrainer(
            cfg=cfg, modelHarness=harness, logger=MagicMock(), profiler=None
        )

        visits: Counter[int] = Counter()
        fwd_bwd = trainer.cl_updater.fwd_bwd

        def recording_fwd_bwd(cur_batch, hist_batch=None):
            visits.update(int(i) for i in cur_batch[0][:, 0].tolist())
            return fwd_bwd(cur_batch, hist_batch)

        trainer.cl_updater.fwd_bwd = recording_fwd_bwd  # type: ignore[method-assign]
        trainer.outer_cl_training_loop(drift_event_id=1)

        passes = steps * batch // samples
        assert visits.total() == steps * batch, "a step delivered the wrong batch size"
        assert set(visits) == set(range(samples)), (
            f"the round trained on {len(visits)} of {samples} samples; "
            "part of the window was never reached"
        )
        assert set(visits.values()) == {passes}, (
            f"samples were trained on between {min(visits.values())} and "
            f"{max(visits.values())} times instead of {passes} each -- the round "
            "is restarting its loader rather than advancing it"
        )
