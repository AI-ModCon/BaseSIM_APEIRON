"""Tests for apeiron.model.checkpoint (retention + promotion rules)."""

from __future__ import annotations

import pytest
import torch

from apeiron.model.checkpoint import (
    CheckpointRecord,
    CheckpointStore,
    select_keep,
    select_one,
)


def _recs(specs):
    """specs: list of (event, metrics-dict)."""
    return [
        CheckpointRecord(event=e, file=f"drift_adaptation_{e}.pt", metrics=m)
        for e, m in specs
    ]


class TestSelectKeep:
    def test_fifo_keeps_newest_n(self):
        recs = _recs([(i, {}) for i in range(5)])
        assert select_keep(recs, "fifo", max_ckpts=2) == {3, 4}

    def test_max_metric(self):
        recs = _recs([(0, {"acc": 90}), (1, {"acc": 95}), (2, {"acc": 80})])
        assert select_keep(recs, "max:acc", max_ckpts=3) == {1}

    def test_min_metric(self):
        recs = _recs([(0, {"loss": 0.9}), (1, {"loss": 0.2}), (2, {"loss": 0.5})])
        assert select_keep(recs, "min:loss", max_ckpts=3) == {1}

    def test_composite_union(self):
        recs = _recs(
            [
                (0, {"hist": 70, "cur": 60}),
                (1, {"hist": 90, "cur": 65}),
                (2, {"hist": 80, "cur": 99}),
                (3, {"hist": 75, "cur": 70}),
            ]
        )
        # latest:1 -> {3}; best hist -> {1}; best cur -> {2}
        keep = select_keep(recs, "latest:1+max:hist+max:cur", max_ckpts=1)
        assert keep == {1, 2, 3}

    def test_count_suffix(self):
        recs = _recs([(i, {"acc": i}) for i in range(5)])
        assert select_keep(recs, "max:acc:2", max_ckpts=5) == {3, 4}

    def test_metric_absent_selects_nothing(self):
        recs = _recs([(0, {}), (1, {})])
        assert select_keep(recs, "max:acc", max_ckpts=5) == set()

    def test_unknown_clause_raises(self):
        with pytest.raises(ValueError):
            select_keep(_recs([(0, {})]), "banana", max_ckpts=1)


class TestSelectOne:
    def test_promotion_winner(self):
        recs = _recs([(0, {"h": 70}), (1, {"h": 95}), (2, {"h": 80})])
        assert select_one(recs, "max:h") == 1

    def test_empty_rule(self):
        assert select_one(_recs([(0, {})]), "") is None


class TestCheckpointStore:
    def _state(self, v):
        return {"w": torch.tensor([float(v)])}

    def test_save_writes_sidecar_and_latest(self, tmp_path):
        store = CheckpointStore(tmp_path, max_ckpts=5, retention="fifo")
        path = store.save(self._state(1), event=1, metrics={"test_curr_acc": 88.0})
        assert path.endswith("drift_adaptation_1.pt")
        assert store.latest() == "drift_adaptation_1.pt"
        recs = store.records()
        assert len(recs) == 1
        assert recs[0].metrics["test_curr_acc"] == 88.0

    def test_fifo_retention_prunes(self, tmp_path):
        store = CheckpointStore(tmp_path, max_ckpts=2, retention="fifo")
        for e in range(4):
            store.save(self._state(e), event=e, metrics={})
        events = {r.event for r in store.records()}
        assert events == {2, 3}
        assert not (tmp_path / "drift_adaptation_0.pt").exists()
        assert not (tmp_path / "drift_adaptation_0.json").exists()

    def test_metric_retention_keeps_best(self, tmp_path):
        store = CheckpointStore(
            tmp_path, max_ckpts=2, retention="latest:1+max:test_hist_acc"
        )
        store.save(self._state(0), event=0, metrics={"test_hist_acc": 99.0})
        store.save(self._state(1), event=1, metrics={"test_hist_acc": 50.0})
        store.save(self._state(2), event=2, metrics={"test_hist_acc": 60.0})
        # Keep newest (2) and best-hist (0); drop 1.
        assert {r.event for r in store.records()} == {0, 2}

    def test_deploy_promotion_pointer(self, tmp_path):
        store = CheckpointStore(
            tmp_path, max_ckpts=5, retention="fifo", deploy_rule="max:test_hist_acc"
        )
        store.save(self._state(0), event=0, metrics={"test_hist_acc": 70.0})
        store.save(self._state(1), event=1, metrics={"test_hist_acc": 95.0})
        store.save(self._state(2), event=2, metrics={"test_hist_acc": 80.0})
        assert store.deployed() == "drift_adaptation_1.pt"
        # The deployed checkpoint actually loads.
        state = torch.load(tmp_path / store.deployed(), weights_only=True)
        assert state["w"].item() == 1.0

    def test_records_ignore_missing_pt(self, tmp_path):
        store = CheckpointStore(tmp_path, max_ckpts=5)
        store.save(self._state(0), event=0, metrics={})
        (tmp_path / "drift_adaptation_0.pt").unlink()
        assert store.records() == []
