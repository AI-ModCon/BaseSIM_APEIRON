"""How many rows a round logs, and which single one annotates the step.

The logger advances a per-stage step on every row unless the caller passes
`increment=False` (A68), so who passes it is a contract: a reader joins
rows by step, and a row that annotates where it should advance merges two
observations into one.

The trainer's rule, counted over one outer round against `src/`:

    cl    max_iter + 1 rows, every one advancing
    eval  2 rows: the validation annotation, then the advancing transfer row

confirmed at `max_iter` 1, 2, 3 and 5 -- 2, 3, 4 and 6 cl rows.

The single non-advancing call is `_log_validation`, which says so:

    # increment=False: annotate the CL round's step, do not advance it.

A rebuild passed `increment=False` at two more sites -- the CL profiler row
and the transfer-metrics row -- giving `cl` 1 non-advancing and `eval` 2.
Two flags. Over the pinned MNIST run that produced the same 465 steps, the
same 31 series and, checked value by value, **identical numbers
throughout**: 376 accuracies, 376 losses, 37 drift scores, the final test
accuracy, all byte-equal. The two rows shifted the global step of
everything after them, and 22 of 31 series failed their digest on the
label while agreeing on every value.

THE FIXTURE IS THE POINT OF THIS FILE. An earlier version of it passed
against the rebuild and was deleted for not discriminating, and the reason
was not scale -- it was that neither divergent site ran:

  * the CL profiler row is guarded by `if performance:`, and a profiler
    with no measured block reports nothing, so the row never appears.
    A block is measured below before the round starts.
  * the transfer row needs a computable BWT, which needs at least two
    registered tasks. Two are registered below.

With both reached, the counts above are exact and the criterion fails on
the rebuild. Without them it passes on everything, which is how two
`increment=` flags survived nine rebuilds and fifteen full-stack runs.
"""

from __future__ import annotations

import collections
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import TensorDataset

from apeiron.logger.logger import Logger
from apeiron.profilers.count_flops import FLOPSProfiler
from apeiron.training.continuous_trainer import ContinuousTrainer


def _fixed(seed: int, n: int = 32, in_f: int = 4, nc: int = 3) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    return TensorDataset(torch.randn(n, in_f, generator=generator),
                         torch.randint(0, nc, (n,), generator=generator))


def _call_sequence(cfg, harness) -> list[tuple[str, bool, tuple[str, ...]]]:
    """Every log call of one outer round, in order.

    Returns (stage, advances?, sorted metric names) per call, which is
    enough to assert the counts, the flags AND the order -- three
    independent properties of a produced sequence (R31).
    """
    profiler = FLOPSProfiler(warmup_iters=0)
    # A measured block, so `get_performance()` is non-empty and the CL
    # profiler row is actually logged. Without this the row is skipped and
    # the criterion cannot see its `increment=`.
    with profiler.measure_flops("warm"):
        harness.model(torch.zeros(2, 4))

    calls: list[tuple[str, bool, tuple[str, ...]]] = []
    stage = {"name": None}

    class Recording:
        step = 0

        def stage(self, name):
            stage["name"] = name

        def log(self, metrics, *args, increment=True, **kwargs):
            calls.append((stage["name"], bool(increment),
                          tuple(sorted(metrics))))

        def __getattr__(self, _name):
            return lambda *a, **kw: None

    recording = Recording()
    injected = Logger(backend="none",
                      csv_path=Path(tempfile.mkdtemp()) / "m.csv")
    injected.stage = recording.stage
    injected.log = recording.log
    with patch("apeiron.training.continuous_trainer.get_logger",
               return_value=recording):
        trainer = ContinuousTrainer(cfg, harness, injected, profiler)
        trainer.outer_cl_training_loop(drift_event_id=3)
    return calls


def _advance_counts(cfg, harness) -> collections.Counter:
    """(stage, advances?) -> how many log calls, over one outer round."""
    return collections.Counter((s, a) for s, a, _n in
                               _call_sequence(cfg, harness))


#: `max_iter` values the count rule is asserted at.
ITERATIONS = (1, 2, 3, 5)


@pytest.fixture()
def harness(default_cfg):
    """History, and TWO past tasks so the transfer row's BWT is computable."""
    from conftest import DummyHarness
    torch.manual_seed(5)
    built = DummyHarness(default_cfg, hist_train_data=_fixed(5),
                         hist_val_data=_fixed(9))
    built.register_task([1.0])
    built.register_task([2.0])
    return built


class TestWhoAnnotatesAndWhoAdvances:
    def test_no_cl_row_annotates(self, default_cfg, harness):
        """Every continual-learning row is its own observation, including
        the profiler row that closes the round."""
        counts = _advance_counts(default_cfg, harness)
        assert counts[("cl", False)] == 0, (
            f"{counts[('cl', False)]} cl row(s) passed increment=False; "
            f"each CL row is an observation of its own iteration"
        )

    def test_cl_rows_are_actually_logged(self, default_cfg, harness):
        """Guard: the assertion above is vacuous if the round logs no cl
        rows at all."""
        counts = _advance_counts(default_cfg, harness)
        assert counts[("cl", True)] > 0

    def test_exactly_one_eval_row_annotates(self, default_cfg, harness):
        """`_log_validation`, and only it."""
        counts = _advance_counts(default_cfg, harness)
        assert counts[("eval", False)] == 1, (
            f"{counts[('eval', False)]} eval row(s) passed increment=False; "
            f"only the post-CL validation annotates the step it shares"
        )

    def test_the_transfer_row_advances(self, default_cfg, harness):
        """The transfer metrics are an observation of the round that just
        finished, not an annotation of the validation before them. The
        source logs them `commit=False` and lets the step advance."""
        counts = _advance_counts(default_cfg, harness)
        assert counts[("eval", True)] >= 1, (
            "no eval row advanced the step; the transfer row must"
        )


class TestHowManyRowsARoundLogs:
    """The counts, because pinning only the FLAGS leaves the number free.

    An earlier version of this file asserted "no cl row annotates" and
    nothing about how many there are. A rebuild satisfied it by logging
    **two** cl rows per iteration -- 100 where the original logs 51 -- and
    the full-stack run went from 22 mismatched series to 31. Pinning a
    property of each element does not constrain the count of elements
    (R29, now twice over).
    """

    @pytest.mark.parametrize("max_iter", ITERATIONS)
    def test_the_cl_row_count_follows_max_iter(self, default_cfg, harness,
                                               max_iter):
        """One row per inner iteration, plus the profiler row that closes
        the round."""
        from dataclasses import replace
        cfg = replace(default_cfg,
                      train=replace(default_cfg.train, max_iter=max_iter))
        counts = _advance_counts(cfg, harness)
        total = counts[("cl", True)] + counts[("cl", False)]
        assert total == max_iter + 1, (
            f"max_iter={max_iter}: the round logged {total} cl rows, "
            f"expected {max_iter + 1} -- one per iteration plus the closing "
            f"profiler row"
        )

    @pytest.mark.parametrize("max_iter", ITERATIONS)
    def test_a_round_logs_exactly_two_eval_rows(self, default_cfg, harness,
                                                max_iter):
        """The validation annotation and the transfer row, whatever the
        iteration count."""
        from dataclasses import replace
        cfg = replace(default_cfg,
                      train=replace(default_cfg.train, max_iter=max_iter))
        counts = _advance_counts(cfg, harness)
        total = counts[("eval", True)] + counts[("eval", False)]
        assert total == 2, (
            f"max_iter={max_iter}: the round logged {total} eval rows, "
            f"expected 2"
        )


class TestTheFixtureReachesBothSites:
    """Without these the criterion passes on anything -- see the docstring."""

    def test_the_profiler_reports_something(self, default_cfg, harness):
        """The CL profiler row is guarded by `if performance:`."""
        profiler = FLOPSProfiler(warmup_iters=0)
        with profiler.measure_flops("warm"):
            harness.model(torch.zeros(2, 4))
        assert profiler.get_performance(), (
            "the profiler reported nothing, so the CL profiler row would be "
            "skipped and its increment= never observed"
        )

    def test_two_tasks_are_registered(self, harness):
        """The transfer row needs a computable BWT."""
        assert len(harness.task_diagonals) >= 2


class TestTheOrderOfTheClosingRows:
    """Which of the round's two closing rows comes first.

    The counts and the flags were both pinned and a rebuild still differed:
    it logged the CL profiler row BEFORE the transfer row where the source
    logs it after. Values identical, steps identical in shape -- and
    because each row takes the next global step, the pair swapped labels
    and 11 series failed their digest.

    Third independent property of the same sequence, after content and
    count (R31): the order.
    """

    def _closing(self, cfg, harness):
        calls = _call_sequence(cfg, harness)
        transfer = next(
            (i for i, (s, _a, names) in enumerate(calls)
             if s == "eval" and any(n.startswith("test_") for n in names)),
            None)
        profiler = next(
            (i for i, (s, _a, names) in enumerate(calls)
             if s == "cl" and all(n.startswith("cperf_") for n in names)),
            None)
        return calls, transfer, profiler

    def test_both_closing_rows_are_logged(self, default_cfg, harness):
        """Guard: the order assertion is vacuous if either is missing."""
        _calls, transfer, profiler = self._closing(default_cfg, harness)
        assert transfer is not None, "no transfer row was logged"
        assert profiler is not None, "no CL profiler row was logged"

    def test_the_transfer_row_precedes_the_profiler_row(self, default_cfg,
                                                        harness):
        """`src/` logs the transfer metrics, then the round's profiler
        summary. Each takes the next step, so swapping them swaps two
        labels and every series on those rows fails its digest."""
        _calls, transfer, profiler = self._closing(default_cfg, harness)
        assert transfer < profiler, (
            f"the CL profiler row was logged at position {profiler}, before "
            f"the transfer row at {transfer}; the transfer metrics close the "
            f"round first"
        )
