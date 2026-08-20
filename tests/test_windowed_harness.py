"""End-to-end tests for apeiron.data.windowed_harness.WindowedHarness."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch.nn as nn
from torch.optim import SGD

from apeiron.data.window_store import WindowStore
from apeiron.data.windowed_harness import WindowedHarness
from apeiron.evaluation.metrics import accuracy
from apeiron.model.task_record import WindowEvalSetRef


def _make_store(root, n_windows=3, n=10, seed=0):
    rng = np.random.default_rng(seed)
    store = WindowStore(root, catalog=True)
    for i in range(n_windows):
        x = rng.standard_normal((n, 4)).astype(np.float32)
        y = rng.integers(0, 3, size=n).astype(np.int64)
        store.commit(
            x,
            y,
            val_fraction=0.5,
            t_start=f"2026-03-{i + 1:02d}T00:00",
            t_end=f"2026-03-{i + 1:02d}T06:00",
            detected=(i % 2 == 1),
        )
    return store


def _harness(cfg, model, store):
    return WindowedHarness(
        cfg,
        model,
        store,
        criterion=nn.CrossEntropyLoss(),
        optimizer_factory=lambda m: SGD(m.parameters(), lr=0.01),
        eval_metrics={"accuracy": accuracy},
    )


def _count(loader):
    return sum(y.shape[0] for _, y in loader)


class TestWindowNavigation:
    def test_windows_seen_in_order(self, default_cfg, tiny_model, tmp_path):
        store = _make_store(tmp_path / "s")
        h = _harness(default_cfg, tiny_model, store)
        assert h.n_windows == 3

        seen = []
        for _ in range(3):
            h.update_data_stream()
            seen.append(h.current_window_id)
        assert seen == store.window_ids()

    def test_timerange_exposed(self, default_cfg, tiny_model, tmp_path):
        store = _make_store(tmp_path / "s")
        h = _harness(default_cfg, tiny_model, store)
        h.update_data_stream()
        assert h.current_window_timerange == ("2026-03-01T00:00", "2026-03-01T06:00")

    def test_read_before_advance_raises(self, default_cfg, tiny_model, tmp_path):
        store = _make_store(tmp_path / "s")
        h = _harness(default_cfg, tiny_model, store)
        with pytest.raises(RuntimeError):
            h.get_train_dataloaders()

    def test_over_advance_clamps_and_flags(self, default_cfg, tiny_model, tmp_path):
        store = _make_store(tmp_path / "s", n_windows=2)
        h = _harness(default_cfg, tiny_model, store)
        for _ in range(5):
            h.update_data_stream()
        assert h.stream_exhausted is True
        assert h.current_window_id == store.window_ids()[-1]


class TestStreams:
    def test_train_val_split_sizes(self, default_cfg, tiny_model, tmp_path):
        store = _make_store(tmp_path / "s", n=10)
        h = _harness(default_cfg, tiny_model, store)
        h.update_data_stream()
        train, val = h.get_train_dataloaders()
        assert _count(train) == 5
        assert _count(val) == 5

    def test_stream_loader_covers_whole_window(self, default_cfg, tiny_model, tmp_path):
        store = _make_store(tmp_path / "s", n=10)
        h = _harness(default_cfg, tiny_model, store)
        h.update_data_stream()
        assert _count(h.get_stream_dataloader()) == 10

    def test_history_is_none_on_first_window(self, default_cfg, tiny_model, tmp_path):
        store = _make_store(tmp_path / "s")
        h = _harness(default_cfg, tiny_model, store)
        h.update_data_stream()
        assert h.get_hist_dataloaders() == (None, None)

    def test_history_accumulates_prior_windows(self, default_cfg, tiny_model, tmp_path):
        store = _make_store(tmp_path / "s", n=10)
        h = _harness(default_cfg, tiny_model, store)
        h.update_data_stream()  # window 0
        h.update_data_stream()  # window 1
        hist_train, hist_val = h.get_hist_dataloaders()
        # Exactly the prior window (0): its train (5) + val (5) splits.
        assert _count(hist_train) == 5
        assert _count(hist_val) == 5

        h.update_data_stream()  # window 2 -> history is windows 0 and 1
        hist_train2, _ = h.get_hist_dataloaders()
        assert _count(hist_train2) == 10


class TestPointerTaskRecords:
    def test_freeze_returns_window_pointer(self, default_cfg, tiny_model, tmp_path):
        store = _make_store(tmp_path / "s")
        h = _harness(default_cfg, tiny_model, store)
        h.update_data_stream()
        ref = h._freeze_task_evalset(h.current_window_id)
        assert isinstance(ref, WindowEvalSetRef)
        assert ref.window_id == h.current_window_id

    def test_register_and_eval_past_tasks_pointer_path(
        self, default_cfg, tiny_model, tmp_path
    ):
        store = _make_store(tmp_path / "s")
        h = _harness(default_cfg, tiny_model, store)
        h.model.eval()
        h.update_data_stream()

        diag = h.eval()
        h.register_task(diag)
        rec = h._task_records[-1]
        assert isinstance(rec.eval_ref, WindowEvalSetRef)
        assert rec.window_id == h.current_window_id
        # No-copy pointer still reproduces the diagonal under a frozen model.
        assert h.eval_past_tasks()[0] == pytest.approx(h.task_diagonals[0])

    def test_windowed_records_persist_and_reload(
        self, default_cfg, tiny_model, tmp_path
    ):
        store = _make_store(tmp_path / "s")
        cfg = replace(
            default_cfg,
            model=replace(
                default_cfg.model, ckpts_path=str(tmp_path / "ck"), max_ckpts=2
            ),
        )
        h1 = _harness(cfg, tiny_model, store)
        h1.update_data_stream()
        h1.register_task(h1.eval())

        h2 = _harness(cfg, tiny_model, store)
        assert h2.load_task_records() == 1
        assert isinstance(h2._task_records[0].eval_ref, WindowEvalSetRef)
        assert len(h2.eval_past_tasks()) == 1


class TestConfigPathConstruction:
    def test_builds_store_from_config_path(self, default_cfg, tiny_model, tmp_path):
        _make_store(tmp_path / "s")  # materialize windows on disk
        cfg = replace(
            default_cfg,
            data=replace(default_cfg.data, window_store_path=str(tmp_path / "s")),
        )
        h = WindowedHarness(
            cfg,
            tiny_model,
            criterion=nn.CrossEntropyLoss(),
            optimizer_factory=lambda m: SGD(m.parameters(), lr=0.01),
            eval_metrics={"accuracy": accuracy},
        )
        assert h.n_windows == 3

    def test_missing_store_and_path_raises(self, default_cfg, tiny_model):
        with pytest.raises(ValueError):
            WindowedHarness(default_cfg, tiny_model)

    def test_criterion_optimizer_required(self, default_cfg, tiny_model, tmp_path):
        store = _make_store(tmp_path / "s")
        h = WindowedHarness(default_cfg, tiny_model, store)
        with pytest.raises(NotImplementedError):
            h.get_criterion()
        with pytest.raises(NotImplementedError):
            h.get_optmizer()
