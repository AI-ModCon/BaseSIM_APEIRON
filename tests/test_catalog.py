"""Tests for apeiron.data.catalog and its integration with WindowStore."""

from __future__ import annotations

import numpy as np

from apeiron.data.catalog import WindowCatalog
from apeiron.data.window_store import WindowStore


def _commit(store, window_id, *, t_start, t_end, detected=False):
    return store.commit(
        np.zeros((3, 2), dtype=np.float32),
        np.zeros(3, dtype=np.int64),
        window_id=window_id,
        t_start=t_start,
        t_end=t_end,
        detected=detected,
    )


class TestCatalogIntegration:
    def test_commit_populates_catalog(self, tmp_path):
        store = WindowStore(tmp_path, catalog=True)
        _commit(store, "w0", t_start="2026-01-01T00:00", t_end="2026-01-01T01:00")
        assert store.catalog is not None
        assert store.catalog.get("w0")["n_samples"] == 3
        assert len(store.catalog) == 1

    def test_delete_updates_catalog(self, tmp_path):
        store = WindowStore(tmp_path, catalog=True)
        _commit(store, "w0", t_start="a", t_end="b")
        store.delete("w0")
        assert store.catalog.get("w0") is None


class TestQueries:
    def _store(self, tmp_path):
        store = WindowStore(tmp_path, catalog=True)
        _commit(store, "w0", t_start="2026-03-01", t_end="2026-03-02")
        _commit(store, "w1", t_start="2026-03-02", t_end="2026-03-03", detected=True)
        _commit(store, "w2", t_start="2026-03-03", t_end="2026-03-04")
        _commit(store, "w3", t_start="2026-03-04", t_end="2026-03-05", detected=True)
        return store

    def test_time_range_query(self, tmp_path):
        cat = self._store(tmp_path).catalog
        # Historic-load creation: everything starting on/after 2026-03-02.
        assert cat.query(t_start_gte="2026-03-02") == ["w1", "w2", "w3"]

    def test_detected_query(self, tmp_path):
        cat = self._store(tmp_path).catalog
        assert cat.query(detected=True) == ["w1", "w3"]

    def test_seq_window_query(self, tmp_path):
        cat = self._store(tmp_path).catalog
        assert cat.query(seq_gte=1, seq_lt=3) == ["w1", "w2"]

    def test_ordering_by_seq(self, tmp_path):
        cat = self._store(tmp_path).catalog
        assert [r["window_id"] for r in cat.all()] == ["w0", "w1", "w2", "w3"]


class TestRebuild:
    def test_rebuild_from_store(self, tmp_path):
        store = WindowStore(tmp_path, catalog=True)
        _commit(store, "w0", t_start="a", t_end="b")
        _commit(store, "w1", t_start="c", t_end="d", detected=True)

        # Wipe the catalog rows, then rebuild purely from on-disk manifests.
        store.catalog.rebuild([])
        assert len(store.catalog) == 0
        store.rebuild_catalog()
        assert store.catalog.query(detected=True) == ["w1"]

    def test_standalone_catalog_upsert_and_remove(self, tmp_path):
        cat = WindowCatalog(tmp_path / "c.db")
        store = WindowStore(tmp_path / "s", catalog=False)
        m = store.commit(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros(2, dtype=np.int64),
            window_id="x",
        )
        cat.upsert(m)
        assert cat.get("x") is not None
        cat.remove("x")
        assert cat.get("x") is None
        cat.close()
