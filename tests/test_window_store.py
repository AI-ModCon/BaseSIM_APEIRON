"""Tests for apeiron.data.window_store."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from apeiron.data.window_store import (
    MemmapWindowDataset,
    WindowSplit,
    WindowStore,
)


def _drain(loader):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


class TestCommitRoundTrip:
    def test_commit_and_read_all(self, tmp_path):
        store = WindowStore(tmp_path, catalog=False)
        x = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
        y = np.arange(20, dtype=np.int64)
        m = store.commit(x, y, val_fraction=0.25)

        assert m.window_id in store
        assert m.n_samples == 20
        assert m.x_shape == (3,)
        assert m.x_dtype == "float32"
        assert m.y_dtype == "int64"

        handle = store.window(m.window_id)
        gx, gy = _drain(handle.loader("all", batch_size=8))
        assert gx.shape == (20, 3)
        assert torch.equal(gy.sort().values, torch.arange(20))
        # Values survive the memmap round-trip exactly.
        assert torch.equal(gx, torch.from_numpy(x))

    def test_accepts_torch_tensors(self, tmp_path):
        store = WindowStore(tmp_path, catalog=False)
        x = torch.randn(8, 2)
        y = torch.randint(0, 4, (8,))
        m = store.commit(x, y)
        gx, gy = store.window(m.window_id).load_full("all")
        assert torch.allclose(gx, x)
        assert torch.equal(gy, y)

    def test_dtype_preserved_uint8(self, tmp_path):
        store = WindowStore(tmp_path, catalog=False)
        x = (np.random.rand(10, 4) * 255).astype(np.uint8)
        y = np.zeros(10, dtype=np.int64)
        m = store.commit(x, y)
        assert m.x_dtype == "uint8"
        gx, _ = store.window(m.window_id).load_full()
        assert gx.dtype == torch.uint8


class TestSplits:
    def test_val_fraction_range_split(self, tmp_path):
        store = WindowStore(tmp_path, catalog=False)
        x = np.zeros((10, 2), dtype=np.float32)
        y = np.arange(10, dtype=np.int64)
        m = store.commit(x, y, val_fraction=0.3)
        handle = store.window(m.window_id)
        assert list(handle.indices("train")) == list(range(0, 7))
        assert list(handle.indices("val")) == list(range(7, 10))

    def test_indexed_split_noncontiguous(self, tmp_path):
        store = WindowStore(tmp_path, catalog=False)
        x = np.zeros((10, 2), dtype=np.float32)
        y = np.arange(10, dtype=np.int64)
        train_idx = [0, 2, 4, 6, 8]
        val_idx = [1, 3, 5, 7, 9]
        m = store.commit_indexed(x, y, train_idx=train_idx, val_idx=val_idx)
        handle = store.window(m.window_id)
        _, ty = _drain(handle.loader("train", batch_size=32))
        _, vy = _drain(handle.loader("val", batch_size=32))
        assert sorted(ty.tolist()) == train_idx
        assert sorted(vy.tolist()) == val_idx

    def test_unknown_split_raises(self, tmp_path):
        store = WindowStore(tmp_path, catalog=False)
        m = store.commit(
            np.zeros((4, 2), dtype=np.float32), np.zeros(4, dtype=np.int64)
        )
        with pytest.raises(KeyError):
            store.window(m.window_id).indices("nope")

    def test_index_scheme_rejected_by_commit(self, tmp_path):
        store = WindowStore(tmp_path, catalog=False)
        with pytest.raises(ValueError, match="commit_indexed"):
            store.commit(
                np.zeros((4, 2), dtype=np.float32),
                np.zeros(4, dtype=np.int64),
                splits={"val": WindowSplit("index", idx_file="v.npy")},
            )


class TestSharding:
    def test_shards_partition_the_split(self, tmp_path):
        store = WindowStore(tmp_path, catalog=False)
        n = 23
        y = np.arange(n, dtype=np.int64)
        m = store.commit(np.zeros((n, 2), dtype=np.float32), y)
        handle = store.window(m.window_id)

        world = 4
        seen = []
        sizes = []
        for rank in range(world):
            idx = handle.indices("all", shard=(rank, world))
            sizes.append(len(idx))
            seen.extend(idx.tolist())
        # Contiguous, balanced, disjoint, covering.
        assert sorted(seen) == list(range(n))
        assert max(sizes) - min(sizes) <= 1
        assert sum(sizes) == n

    def test_shard_is_contiguous(self, tmp_path):
        store = WindowStore(tmp_path, catalog=False)
        m = store.commit(
            np.zeros((12, 2), dtype=np.float32), np.arange(12, dtype=np.int64)
        )
        idx = store.window(m.window_id).indices("all", shard=(1, 3))
        assert idx.tolist() == [4, 5, 6, 7]

    def test_bad_rank_raises(self, tmp_path):
        store = WindowStore(tmp_path, catalog=False)
        m = store.commit(
            np.zeros((4, 2), dtype=np.float32), np.zeros(4, dtype=np.int64)
        )
        with pytest.raises(ValueError):
            store.window(m.window_id).indices("all", shard=(5, 3))


class TestOrderingAndListing:
    def test_seq_is_monotonic_and_orders_windows(self, tmp_path):
        store = WindowStore(tmp_path, catalog=False)
        ids = [
            store.commit(
                np.zeros((3, 2), dtype=np.float32),
                np.zeros(3, dtype=np.int64),
                window_id=name,
            ).window_id
            for name in ("alpha", "beta", "gamma")
        ]
        assert store.window_ids() == ids  # commit order, not lexical
        assert [m.seq for m in store.list_manifests()] == [0, 1, 2]
        assert len(store) == 3

    def test_duplicate_window_id_rejected(self, tmp_path):
        store = WindowStore(tmp_path, catalog=False)
        store.commit(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros(2, dtype=np.int64),
            window_id="w",
        )
        with pytest.raises(FileExistsError):
            store.commit(
                np.zeros((2, 2), dtype=np.float32),
                np.zeros(2, dtype=np.int64),
                window_id="w",
            )

    def test_reopen_store_sees_committed_windows(self, tmp_path):
        WindowStore(tmp_path, catalog=False).commit(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros(2, dtype=np.int64),
            window_id="w0",
        )
        # A brand-new store object over the same root recovers the window and
        # continues the sequence.
        store2 = WindowStore(tmp_path, catalog=False)
        assert "w0" in store2
        m = store2.commit(
            np.zeros((2, 2), dtype=np.float32), np.zeros(2, dtype=np.int64)
        )
        assert m.seq == 1

    def test_mismatched_sample_counts_rejected(self, tmp_path):
        store = WindowStore(tmp_path, catalog=False)
        with pytest.raises(ValueError):
            store.commit(
                np.zeros((4, 2), dtype=np.float32), np.zeros(3, dtype=np.int64)
            )


class TestAtomicCommit:
    def test_failed_commit_leaves_nothing(self, tmp_path, monkeypatch):
        store = WindowStore(tmp_path, catalog=False)

        import apeiron.data.window_store as ws

        def boom(src, dst):
            raise OSError("simulated crash during rename")

        monkeypatch.setattr(ws.os, "replace", boom)
        with pytest.raises(OSError):
            store.commit(
                np.zeros((4, 2), dtype=np.float32), np.zeros(4, dtype=np.int64)
            )

        # No committed window, and no leftover staging directory.
        assert len(store) == 0
        assert not any(p.name.startswith(".tmp-") for p in tmp_path.iterdir())

    def test_tmp_dirs_ignored_by_listing(self, tmp_path):
        store = WindowStore(tmp_path, catalog=False)
        (tmp_path / ".tmp-orphan").mkdir()
        store.commit(np.zeros((2, 2), dtype=np.float32), np.zeros(2, dtype=np.int64))
        assert len(store) == 1  # the orphan tmp dir is not a committed window


class TestDelete:
    def test_delete_removes_window(self, tmp_path):
        store = WindowStore(tmp_path, catalog=False)
        m = store.commit(
            np.zeros((2, 2), dtype=np.float32), np.zeros(2, dtype=np.int64)
        )
        store.delete(m.window_id)
        assert m.window_id not in store
        assert len(store) == 0


class TestMemmapDataset:
    def test_lazy_open_after_pickle(self, tmp_path):
        # Simulates the DataLoader-worker path: the dataset must survive being
        # re-created from its constructor args without an open memmap.
        store = WindowStore(tmp_path, catalog=False)
        m = store.commit(
            np.arange(12, dtype=np.float32).reshape(6, 2), np.arange(6, dtype=np.int64)
        )
        handle = store.window(m.window_id)
        ds = handle.dataset("all")
        assert isinstance(ds, MemmapWindowDataset)
        import pickle

        ds2 = pickle.loads(pickle.dumps(ds))
        x0, y0 = ds2[0]
        assert x0.tolist() == [0.0, 1.0]
        assert int(y0) == 0
