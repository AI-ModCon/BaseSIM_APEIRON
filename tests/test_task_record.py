"""Tests for apeiron.model.task_record (eval-set refs, persistence)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from apeiron.data.window_store import WindowStore
from apeiron.model.task_record import (
    InMemoryEvalSet,
    TaskRecord,
    TaskRecordStore,
    WindowEvalSetRef,
)


def _drain_y(loader):
    return torch.cat([y for _, y in loader])


class TestInMemoryEvalSet:
    def test_loader_returns_data(self):
        x = torch.randn(6, 3)
        y = torch.arange(6)
        ref = InMemoryEvalSet(x, y)
        gy = _drain_y(ref.loader(batch_size=2))
        assert torch.equal(gy, y)

    def test_spill_and_reload(self, tmp_path):
        x = torch.randn(5, 2)
        y = torch.arange(5)
        ref = InMemoryEvalSet(x, y)
        ref.spill(tmp_path / "e.pt")

        reloaded = InMemoryEvalSet(spill_path=tmp_path / "e.pt")
        gx, gy = next(iter(reloaded.loader(batch_size=5)))
        assert torch.equal(gy, y)
        assert torch.allclose(gx, x)


class TestWindowEvalSetRef:
    def test_pointer_loader_reads_committed_window(self, tmp_path):
        store = WindowStore(tmp_path, catalog=False)
        y = np.arange(10, dtype=np.int64)
        m = store.commit(np.zeros((10, 2), dtype=np.float32), y, val_fraction=0.4)
        ref = WindowEvalSetRef(store, m.window_id, "val")
        gy = _drain_y(ref.loader(batch_size=4))
        # val split is the trailing 40% -> samples 6..9.
        assert sorted(gy.tolist()) == [6, 7, 8, 9]

    def test_roundtrip_dict(self, tmp_path):
        store = WindowStore(tmp_path, catalog=False)
        m = store.commit(
            np.zeros((4, 2), dtype=np.float32),
            np.zeros(4, dtype=np.int64),
            val_fraction=0.5,
        )
        ref = WindowEvalSetRef(store, m.window_id, "val")
        d = ref.to_dict(None)
        assert d == {"kind": "window", "window_id": m.window_id, "split": "val"}
        back = WindowEvalSetRef.from_dict(d, store)
        assert back.window_id == m.window_id


class TestTaskRecordStore:
    def test_persist_and_reload_inmemory(self, tmp_path):
        store = TaskRecordStore(tmp_path / "recs")
        recs = [
            TaskRecord(0, [90.0], InMemoryEvalSet(torch.randn(4, 2), torch.arange(4))),
            TaskRecord(1, [80.0], InMemoryEvalSet(torch.randn(4, 2), torch.arange(4))),
        ]
        store.save(recs)

        loaded = store.load(None)
        assert [r.event_id for r in loaded] == [0, 1]
        assert [r.diagonal for r in loaded] == [[90.0], [80.0]]
        # The reloaded refs are usable.
        assert len(_drain_y(loaded[0].eval_ref.loader(batch_size=4))) == 4

    def test_persist_and_reload_window_pointer(self, tmp_path):
        wstore = WindowStore(tmp_path / "win", catalog=False)
        m = wstore.commit(
            np.zeros((6, 2), dtype=np.float32),
            np.arange(6, dtype=np.int64),
            val_fraction=0.5,
        )
        rstore = TaskRecordStore(tmp_path / "recs")
        rstore.save(
            [
                TaskRecord(
                    0,
                    [1.0],
                    WindowEvalSetRef(wstore, m.window_id, "val"),
                    window_id=m.window_id,
                )
            ]
        )

        loaded = rstore.load(wstore)
        assert isinstance(loaded[0].eval_ref, WindowEvalSetRef)
        assert loaded[0].window_id == m.window_id

    def test_window_reload_without_store_raises(self, tmp_path):
        wstore = WindowStore(tmp_path / "win", catalog=False)
        m = wstore.commit(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros(2, dtype=np.int64),
            val_fraction=0.5,
        )
        rstore = TaskRecordStore(tmp_path / "recs")
        rstore.save(
            [TaskRecord(0, [1.0], WindowEvalSetRef(wstore, m.window_id, "val"))]
        )
        with pytest.raises(ValueError):
            rstore.load(None)  # cannot rebuild a window pointer without the store

    def test_orphan_spills_pruned_on_evict(self, tmp_path):
        store = TaskRecordStore(tmp_path / "recs")
        store.save(
            [
                TaskRecord(
                    0, [1.0], InMemoryEvalSet(torch.randn(2, 2), torch.arange(2))
                ),
                TaskRecord(
                    1, [1.0], InMemoryEvalSet(torch.randn(2, 2), torch.arange(2))
                ),
            ]
        )
        assert (tmp_path / "recs" / "evalset_0.pt").exists()

        # Simulate FIFO eviction of task 0: only task 1 remains.
        store.save(
            [TaskRecord(1, [1.0], InMemoryEvalSet(torch.randn(2, 2), torch.arange(2)))]
        )
        assert not (tmp_path / "recs" / "evalset_0.pt").exists()
        assert (tmp_path / "recs" / "evalset_1.pt").exists()

    def test_load_missing_returns_empty(self, tmp_path):
        assert TaskRecordStore(tmp_path / "nope").load(None) == []
