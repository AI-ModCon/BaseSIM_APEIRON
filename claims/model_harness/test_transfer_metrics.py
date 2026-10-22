"""Tests for the harness's per-task transfer metrics (the train-test matrix R).

register_task / eval_past_tasks / task_diagonals are documented in CLAUDE.md
and docs/tracking.md but had no criteria at all: a regrowth of the harness
passed the suite without them. These pin the semantics: frozen copies (not
references), oldest-first alignment of rows and diagonals, eviction under the
retention cap, and sample-weighted (not batch-weighted) metric averaging.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset


def _val_targets(harness) -> torch.Tensor:
    loader = harness.get_train_dataloaders()[1]
    return torch.cat([y for _, y in loader])


class TestRegisterTask:
    def test_rows_and_diagonals_start_empty(self, dummy_harness):
        assert dummy_harness.eval_past_tasks() == []
        assert dummy_harness.task_diagonals == []

    def test_diagonals_are_stored_oldest_first_and_copied(self, dummy_harness):
        d1 = [1.0, 2.0]
        dummy_harness.register_task(d1)
        dummy_harness.register_task([3.0, 4.0])
        d1.append(99.0)  # the caller's list must not alias the record
        assert dummy_harness.task_diagonals == [[1.0, 2.0], [3.0, 4.0]]

    def test_frozen_eval_set_is_a_copy_of_the_window(self, make_harness, default_cfg):
        n, in_f, nc = 12, 4, 3
        g = torch.Generator().manual_seed(0)
        val = TensorDataset(
            torch.randn(n, in_f, generator=g), torch.randint(0, nc, (n,), generator=g)
        )
        harness = make_harness(default_cfg, val_data=val)
        harness.register_task(harness.eval())
        frozen_before = harness.eval_past_tasks()

        # the stream moves on: the harness's val window is replaced entirely
        harness._val_ds = TensorDataset(
            torch.randn(n, in_f, generator=g) + 10.0,
            torch.randint(0, nc, (n,), generator=g),
        )
        assert harness.eval_past_tasks() == frozen_before, (
            "the registered task must evaluate on its frozen window, "
            "not on whatever the harness currently streams"
        )
        # and the frozen window carries the registered targets
        frozen_loader = harness._task_records[0][0]
        assert torch.equal(torch.cat([y for _, y in frozen_loader]), val.tensors[1])

    def test_past_task_scores_follow_the_current_model(self, dummy_harness):
        dummy_harness.register_task(dummy_harness.eval())
        before = dummy_harness.eval_past_tasks()
        with torch.no_grad():
            for p in dummy_harness.model.parameters():
                p.add_(torch.randn(p.shape, generator=torch.Generator().manual_seed(1)))
        after = dummy_harness.eval_past_tasks()
        assert before != after, (
            "eval_past_tasks scores the CURRENT model on the frozen window; "
            "a changed model must change the row"
        )

    def test_retention_cap_evicts_oldest_and_keeps_alignment(self, dummy_harness):
        dummy_harness.max_task_records = 2
        for i in range(3):
            dummy_harness.register_task([float(i)])
        assert dummy_harness.task_diagonals == [[1.0], [2.0]]
        assert len(dummy_harness.eval_past_tasks()) == 2


class TestEvalLoaderAveraging:
    def test_metrics_average_per_sample_not_per_batch(self, make_harness, default_cfg):
        # 12 samples at batch_size 8 yield batches of 8 and 4. A metric that
        # reports the batch size as its value exposes the weighting: the
        # sample-weighted mean is (8*8 + 4*4) / 12, a naive mean of batch
        # values would be 6.
        n, in_f, nc = 12, 4, 3
        g = torch.Generator().manual_seed(0)
        val = TensorDataset(
            torch.randn(n, in_f, generator=g), torch.randint(0, nc, (n,), generator=g)
        )
        harness = make_harness(default_cfg, val_data=val)
        harness.eval_metrics = {"batchsize": lambda y_hat, y: float(y.shape[0])}
        (value,) = harness.eval()
        assert value == pytest.approx((8 * 8 + 4 * 4) / 12)
