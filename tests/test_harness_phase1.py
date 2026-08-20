"""Integration tests for BaseModelHarness on the Phase-1 primitives.

Covers the pointer/in-memory task-record path, durability + resume, and
rule-based checkpointing -- while asserting the forgetting-measurement numerics
are unchanged from the copy-into-RAM design.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from apeiron.model.checkpoint import CheckpointStore
from apeiron.model.task_record import InMemoryEvalSet


def _cfg_with_ckpts(default_cfg, tmp_path, **model_kwargs):
    model = replace(
        default_cfg.model,
        ckpts_path=str(tmp_path),
        max_ckpts=model_kwargs.pop("max_ckpts", 3),
        **model_kwargs,
    )
    return replace(default_cfg, model=model)


class TestTaskRecordDefault:
    def test_default_evalset_is_in_memory(self, dummy_harness):
        dummy_harness.register_task(dummy_harness.eval())
        rec = dummy_harness._task_records[-1]
        assert isinstance(rec.eval_ref, InMemoryEvalSet)

    def test_eval_past_tasks_reproduces_diagonal(self, dummy_harness):
        # With the model frozen, re-scoring a registered task's frozen eval set
        # must return exactly its diagonal -- the correctness property the whole
        # forgetting measurement rests on.
        dummy_harness.model.eval()
        diag = dummy_harness.eval()
        dummy_harness.register_task(diag)
        past = dummy_harness.eval_past_tasks()
        assert past[0] == pytest.approx(dummy_harness.task_diagonals[0])
        assert dummy_harness.task_diagonals[0] == pytest.approx(diag)

    def test_fifo_eviction_of_task_records(self, dummy_harness):
        dummy_harness.max_task_records = 2
        for _ in range(4):
            dummy_harness.register_task(dummy_harness.eval())
        assert len(dummy_harness._task_records) == 2
        # Event ids keep climbing across eviction (no filename collisions).
        assert [r.event_id for r in dummy_harness._task_records] == [2, 3]


class TestDurabilityResume:
    def test_records_persist_and_reload(self, default_cfg, make_harness, tmp_path):
        cfg = _cfg_with_ckpts(default_cfg, tmp_path)
        h1 = make_harness(cfg)
        h1.register_task([90.0])
        h1.register_task([80.0])
        assert (tmp_path / "task_records" / "task_records.jsonl").exists()

        # A fresh harness (as after a crash/restart) recovers the history.
        h2 = make_harness(cfg)
        assert len(h2._task_records) == 0
        n = h2.load_task_records()
        assert n == 2
        assert [r.diagonal for r in h2._task_records] == [[90.0], [80.0]]
        # Reloaded eval sets are usable for BWT re-evaluation.
        assert len(h2.eval_past_tasks()) == 2

    def test_no_persistence_without_ckpts_path(self, dummy_harness):
        # default_cfg has no ckpts_path -> pure in-memory, nothing written.
        dummy_harness.register_task([1.0])
        assert dummy_harness._records_store is None
        assert dummy_harness.load_task_records() == 0


class TestRuleBasedCheckpointing:
    def test_save_ckpt_uses_store_and_metrics(
        self, default_cfg, make_harness, tmp_path
    ):
        cfg = _cfg_with_ckpts(default_cfg, tmp_path, max_ckpts=5)
        h = make_harness(cfg)
        assert isinstance(h._ckpt_store, CheckpointStore)

        h.last_metrics = {"test_hist_acc": 77.0}
        path = h.save_ckpt(event=1)
        assert path.endswith("drift_adaptation_1.pt")
        rec = h._ckpt_store.records()[0]
        assert rec.metrics["test_hist_acc"] == 77.0

    def test_retention_rule_from_config(self, default_cfg, make_harness, tmp_path):
        cfg = _cfg_with_ckpts(
            default_cfg,
            tmp_path,
            max_ckpts=2,
            ckpts_retention="latest:1+max:test_hist_acc",
            deploy_rule="max:test_hist_acc",
        )
        h = make_harness(cfg)
        for event, hist in [(0, 95.0), (1, 40.0), (2, 60.0)]:
            h.last_metrics = {"test_hist_acc": hist}
            h.save_ckpt(event=event)
        # Keep newest (2) and best-hist (0); deploy the best-hist checkpoint.
        assert {r.event for r in h._ckpt_store.records()} == {0, 2}
        assert h._ckpt_store.deployed() == "drift_adaptation_0.pt"

    def test_ckpts_enabled_flag(self, default_cfg, make_harness, tmp_path):
        assert make_harness(default_cfg).ckpts_enabled is False
        cfg = _cfg_with_ckpts(default_cfg, tmp_path, max_ckpts=1)
        assert make_harness(cfg).ckpts_enabled is True
