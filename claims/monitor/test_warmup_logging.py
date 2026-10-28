"""The warmup window is evaluated but NOT logged, and that is load-bearing.

`_evaluate_batch` has two branches (`continuous_monitor.py`):

    if self.batch_count > self.flops_profiler.warmup_iters:
        with self.flops_profiler.measure_flops(tag="infer"):
            ...  evaluate, then self.logger.stage("eval"); self.logger.log(...)
    else:
        ...  evaluate, and log nothing

so for the first `warmup_iters + 1` batches the model IS evaluated and the
result is NOT written down. With the default `warmup_iters = 1` that is
exactly two rows missing from the head of the metrics file.

This was found the long way. The assembled framework's full MNIST run
produced 378 `eval/accuracy` rows against the original's 376, and the
original's accuracy sequence was the rebuild's shifted by two -- so every
step label after the head was offset and the run claim read `drift fires
at [352]` where the pinned answer is `[350]`. Hashing the batch each
implementation unpacked settled what it was NOT: `_unpack` is called 126
times in both, with byte-identical batches in identical order. Both
evaluate the same data. The rebuild simply logs the warmup evaluations
that the original discards.

Pinned at the unit level rather than through a loop, because the fixture
harnesses do not reliably exercise it -- a seeded `DummyHarness` loop
agreed on the count in both implementations while the MNIST harness did
not, so a count assertion over a fixture would have passed and taught
nothing. What is asserted here is the gate itself: no row while
`batch_count <= warmup_iters`, a row after, at several warmup settings.

AN IMPROVEMENT WORTH RAISING, not pinned: the record's contents are
coupled to the PROFILER's warmup. `warmup_iters` exists so FLOP
measurements skip cold iterations, and it silently also decides how many
evaluations reach the metrics file -- so raising it to profile more
carefully deletes data points from the analysis, and the file carries no
indication that its first rows are missing. Logging the warmup
evaluations (with profiling still skipped) would decouple the two. That is
a behaviour change and belongs to whoever owns the metrics, so this file
pins the current gate and does not assume the fix.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from apeiron.config.configuration import DriftDetectionCfg
from apeiron.driver.continuous_monitor import ContinuousMonitor
from apeiron.profilers.count_flops import FLOPSProfiler


def _monitor(cfg, harness, warmup: int):
    monitor = ContinuousMonitor(cfg=cfg, modelHarness=harness)
    monitor.flops_profiler = FLOPSProfiler(warmup_iters=warmup)
    rows: list[dict] = []
    stage = {"name": None}
    real_stage = monitor.logger.stage

    def stage_hook(name):
        stage["name"] = name
        return real_stage(name)

    monitor.logger.stage = stage_hook
    monitor.logger.log = lambda metrics, *a, **kw: rows.append(
        (stage["name"], dict(metrics))
    )
    return monitor, rows


@pytest.fixture()
def cfg(default_cfg):
    return replace(default_cfg, drift_detection=DriftDetectionCfg())


def _one_batch(harness):
    return next(iter(harness.get_train_dataloaders()[0]))


class TestTheWarmupWindowIsSilent:
    @pytest.mark.parametrize("warmup", [0, 1, 2, 5])
    def test_no_eval_row_while_within_the_warmup_window(
        self, cfg, dummy_harness, warmup
    ):
        """`batch_count <= warmup_iters` evaluates and writes nothing."""
        monitor, rows = _monitor(cfg, dummy_harness, warmup)
        batch = _one_batch(dummy_harness)
        for count in range(warmup + 1):
            monitor.batch_count = count
            monitor._evaluate_batch(batch)
        eval_rows = [r for stage, r in rows if stage == "eval"]
        assert not eval_rows, (
            f"warmup_iters={warmup}: {len(eval_rows)} eval row(s) written "
            f"while batch_count was still inside the warmup window"
        )

    @pytest.mark.parametrize("warmup", [0, 1, 2, 5])
    def test_an_eval_row_appears_once_past_the_window(
        self, cfg, dummy_harness, warmup
    ):
        """The other half: the gate must open, or nothing would ever be
        recorded and the first assertion would pass vacuously."""
        monitor, rows = _monitor(cfg, dummy_harness, warmup)
        monitor.batch_count = warmup + 1
        monitor._evaluate_batch(_one_batch(dummy_harness))
        eval_rows = [r for stage, r in rows if stage == "eval"]
        assert len(eval_rows) == 1, (
            f"warmup_iters={warmup}: expected one eval row at "
            f"batch_count={warmup + 1}, got {len(eval_rows)}"
        )

    def test_the_boundary_is_strictly_greater(self, cfg, dummy_harness):
        """`>` and not `>=`: at `batch_count == warmup_iters` the window is
        still open. One off here shifts every row in the file by one."""
        monitor, rows = _monitor(cfg, dummy_harness, 3)
        batch = _one_batch(dummy_harness)
        monitor.batch_count = 3
        monitor._evaluate_batch(batch)
        assert not [r for stage, r in rows if stage == "eval"]
        monitor.batch_count = 4
        monitor._evaluate_batch(batch)
        assert len([r for stage, r in rows if stage == "eval"]) == 1

    @pytest.mark.parametrize("warmup", [0, 1, 2])
    def test_the_evaluation_still_happens_during_warmup(
        self, cfg, dummy_harness, warmup
    ):
        """Silent is not skipped. The metrics feed the drift buffer even
        while they are unlogged, so an implementation that skipped the
        forward pass entirely would change what the detector sees."""
        monitor, _rows = _monitor(cfg, dummy_harness, warmup)
        monitor.batch_count = 0
        metrics = monitor._evaluate_batch(_one_batch(dummy_harness))
        assert metrics is not None
        assert len(metrics) == len(dummy_harness.eval_metrics)


class TestBatchCountIsIncrementedAfterTheEvaluation:
    """Which batch sits at `batch_count == 0`, and so how wide the window is.

    The gate above is `batch_count > warmup_iters`, so the window's width
    depends entirely on the counter's value during the FIRST evaluation.
    The original evaluates and then increments (`_process_stream`:
    `metrics = self._evaluate_batch(batch)` followed by
    `self.batch_count += 1`), so the first batch is evaluated at
    `batch_count == 0` and the default `warmup_iters = 1` makes two batches
    silent.

    A rebuild incremented first. Its window was therefore one batch wide,
    it logged one extra row at the head of the file, and every row after
    was off by one -- `drift fires at [351]` against the pinned `[350]`.
    That was the residue left after the gate itself was pinned: the same
    comparison, one step instead of two, which is why the counter's
    ordering needs its own assertion rather than being implied.
    """

    def test_the_first_evaluation_sees_a_zero_counter(self, cfg, dummy_harness):
        monitor = ContinuousMonitor(cfg=cfg, modelHarness=dummy_harness)
        monitor.logger.log = lambda *a, **kw: None
        seen: list[int] = []
        real = type(monitor)._evaluate_batch

        def spy(self, batch):
            seen.append(self.batch_count)
            return real(self, batch)

        type(monitor)._evaluate_batch = spy
        try:
            monitor.run()
        finally:
            type(monitor)._evaluate_batch = real
        assert seen, "the loop evaluated nothing"
        assert seen[0] == 0, (
            f"the first batch was evaluated at batch_count={seen[0]}; the "
            f"counter must be incremented AFTER the evaluation, or the "
            f"warmup window is the wrong width"
        )

    def test_the_counter_advances_by_one_per_batch(self, cfg, dummy_harness):
        """No gaps and no repeats, so the window's width is exactly
        `warmup_iters + 1` batches."""
        monitor = ContinuousMonitor(cfg=cfg, modelHarness=dummy_harness)
        monitor.logger.log = lambda *a, **kw: None
        seen: list[int] = []
        real = type(monitor)._evaluate_batch

        def spy(self, batch):
            seen.append(self.batch_count)
            return real(self, batch)

        type(monitor)._evaluate_batch = spy
        try:
            monitor.run()
        finally:
            type(monitor)._evaluate_batch = real
        assert seen == list(range(len(seen))), (
            f"batch_count went {seen[:8]}...; expected 0, 1, 2, ..."
        )
