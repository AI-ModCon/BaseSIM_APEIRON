"""The harness surface other layers reach into, and the task registry's cap.

`ContinuousMonitor` calls two underscored methods on the harness it was
handed -- `_unpack` at `continuous_monitor.py:199` and `:219`, `_to_scalar`
at `:209` and `:228` -- and reads `ckpts_enabled` at `:338`.
`ContinuousTrainer` calls `register_task` at `:247` and reads
`eval_past_tasks()` and `task_diagonals` at `:124` and `:128`. All of that
is contract across a layer boundary, whatever the leading underscores
suggest, and none of it is asserted by this directory's tests.

Pinned here:

  the batch seam     `_unpack` returns the two tensors of a 2-tuple and
                     refuses any other arity. A subclassed dataloader is
                     the documented reason it exists, so its contract is
                     "two things out, or an error".
  the empty loader   `_eval_loader` raises rather than scoring. Both
                     rebuilds returned `[0.0, 0.0]` for a loader with no
                     batches, and that zero travels: into `eval_metrics`,
                     into the CSV, into the metric buffer the drift
                     detector averages. A model that was never evaluated
                     must not read as a model that scored zero.
  the registry cap   `max_task_records` defaults to 50, eviction keeps the
                     MOST RECENT records, and `task_diagonals` stays
                     aligned with `eval_past_tasks()` -- the alignment is
                     the point, per `register_task`'s own docstring
                     ("eviction can never misalign a task's eval set from
                     its diagonal"), and forgetting metrics are computed
                     from the pair. Rebuilds defaulted the cap to 20 and to
                     None.

NOT pinned, and filed instead (claims/LOG.md):

  * `_unpack` on a dict returns the dict's KEYS -- `('a', 'b')`, two
    strings where the monitor expects two tensors, which then fails at
    `.to(device)` with an error naming `str`. That is a defect, not a
    contract, and both rebuilds raise instead. Pinning the current answer
    would freeze it.
  * `_to_scalar` on an integer or boolean tensor raises out of torch's
    `mean()`, and on an empty tensor returns NaN. Both are plausible
    metric shapes and both answers are questionable; the rebuilds coerce.
  * `save_ckpt` writes a checkpoint even when `ckpts_enabled` is False.
    The monitor guards the call, so the harness arguably need not
    self-guard -- a question for whoever owns checkpointing.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


class TestUnpackSeam:
    """Two tensors out, or an error."""

    def test_a_two_tuple_gives_back_its_elements(self, dummy_harness):
        x = torch.zeros(2, 4)
        y = torch.zeros(2, dtype=torch.long)
        got_x, got_y = dummy_harness._unpack((x, y))
        assert got_x is x
        assert got_y is y

    def test_a_list_of_two_also_works(self, dummy_harness):
        """A dataloader's default collate yields a list, not a tuple."""
        x = torch.zeros(2, 4)
        y = torch.zeros(2, dtype=torch.long)
        got_x, got_y = dummy_harness._unpack([x, y])
        assert got_x is x
        assert got_y is y

    def test_a_three_tuple_raises(self, dummy_harness):
        """A loader yielding (x, y, weight) is a real shape, and silently
        dropping the third element would train on mislabelled pairs."""
        t = torch.zeros(2, 4)
        with pytest.raises(ValueError):
            dummy_harness._unpack((t, t, t))

    def test_a_one_tuple_raises(self, dummy_harness):
        with pytest.raises(ValueError):
            dummy_harness._unpack((torch.zeros(2, 4),))


class TestEmptyLoaderIsNotAZero:
    """An un-evaluated model must not score."""

    def test_eval_loader_refuses_an_empty_loader(self, dummy_harness):
        empty = DataLoader(
            TensorDataset(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)),
            batch_size=4,
        )
        with pytest.raises(RuntimeError):
            dummy_harness._eval_loader(empty)

    def test_a_populated_loader_scores(self, dummy_harness):
        """The control: the guard must not reject real work."""
        loader = DataLoader(
            TensorDataset(torch.randn(8, 4), torch.zeros(8, dtype=torch.long)),
            batch_size=4,
        )
        scores = dummy_harness._eval_loader(loader)
        assert len(scores) == len(dummy_harness.eval_metrics)


class TestTaskRegistryCap:
    """`register_task` bookkeeping, which the trainer reads."""

    def test_the_default_cap_is_fifty(self, dummy_harness):
        assert dummy_harness.max_task_records == 50

    def test_a_fresh_harness_has_no_past(self, dummy_harness):
        assert dummy_harness.task_diagonals == []
        assert len(dummy_harness.eval_past_tasks()) == 0

    def test_eviction_keeps_the_most_recent(self, dummy_harness):
        """Forgetting is measured against what came before, so dropping the
        NEWEST records would measure the wrong thing."""
        dummy_harness.max_task_records = 3
        for i in range(5):
            dummy_harness.register_task([float(i)])
        assert dummy_harness.task_diagonals == [[2.0], [3.0], [4.0]]

    def test_a_zero_cap_retains_nothing(self, dummy_harness):
        dummy_harness.max_task_records = 0
        for i in range(2):
            dummy_harness.register_task([float(i)])
        assert dummy_harness.task_diagonals == []
        assert len(dummy_harness.eval_past_tasks()) == 0

    @pytest.mark.parametrize(("cap", "registered"), [(3, 5), (50, 2), (1, 4)])
    def test_the_diagonals_stay_aligned_with_the_past_tasks(
        self, dummy_harness, cap, registered
    ):
        """`register_task`'s docstring: "eviction can never misalign a
        task's eval set from its diagonal". The transfer metrics index both
        by position, so a length mismatch silently scores task i against
        task j."""
        dummy_harness.max_task_records = cap
        for i in range(registered):
            dummy_harness.register_task([float(i)])
        assert len(dummy_harness.task_diagonals) == \
            len(dummy_harness.eval_past_tasks())
        assert len(dummy_harness.task_diagonals) == min(cap, registered)


class TestCkptsEnabled:
    """Read by the monitor before it asks for a checkpoint."""

    def test_it_is_a_bool(self, dummy_harness):
        assert isinstance(dummy_harness.ckpts_enabled, bool)
