"""When the trainer measures, and the one batch it must not inspect.

Two unrelated branches of `inner_cl_training_loop`, both of which a
regrowth dropped and neither of which any other criterion here can see.

**1. Nothing is profiled during warmup.**

    if self.profiler and iter_count > self.profiler.warmup_iters and step == 0:
        with self.profiler.measure_flops(tag="update_fwd_bwd"):
            loss += self.cl_updater.fwd_bwd(...)
    else:
        loss += self.cl_updater.fwd_bwd(...)

Three conditions. `warmup_iters` exists because the first iterations of a
round are not representative -- lazy allocation, the first autograd graph,
caches not yet warm -- and `step == 0` because one measurement per
optimizer step is the unit `get_performance()` averages over.

A regrowth measured unconditionally. That is invisible to a check on the
reported FLOP number, because `get_performance()` returns the MEAN and
every block measures the same work: over the pinned MNIST run the
unconditional version took 100 measurements where `src/` takes 48, and
reported exactly the same `cperf_update_fwd_bwd_flop`. The mean of equal
numbers is insensitive to how many there are. What it does change is
`_time` and `_flops` -- which this repository deliberately leaves out of
the pinned digest, because timings are not reproducible -- so the
divergence sat in the one place a reproducible-run claim cannot look.
This is A73's sibling: both are about a measured window, and this one is
about *when the window opens* rather than what happens inside it.

Pinned as a count of measurements, not as the control flow: an
implementation may arrange the branch however it likes.

**2. A batch whose second element cannot be indexed is accepted.**

    try:
        shape = getattr(batch[1], "shape", None)
    except (IndexError, TypeError):
        # If we cannot inspect batch size, just accept the batch
        return current_iter, [b.to(self.cfg.device) for b in batch]

`_safe_next` enforces `min_batch` by reading `batch[1].shape`. A loader
whose batch is not a `(x, y)` pair -- a dict, a single tensor, a custom
collate -- raises on `batch[1]`, and the source treats "cannot inspect"
as "accept" rather than failing. The regrowth reads `batch[1]`
unguarded. The consequence is worse than an exception: `_safe_next`'s
other exit is to restart the loader and look again, so a harness with an
uninspectable batch either dies on the first step or, with a differently
shaped guard, spins.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from apeiron.logger.logger import Logger
from apeiron.profilers.count_flops import FLOPSProfiler
from apeiron.training.continuous_trainer import ContinuousTrainer

#: The profiler tag the inner loop opens around `fwd_bwd`.
TAG = "update_fwd_bwd"


def _loader(n: int = 32, in_f: int = 4, nc: int = 3, batch: int = 8) -> DataLoader:
    generator = torch.Generator().manual_seed(7)
    data = TensorDataset(torch.randn(n, in_f, generator=generator),
                         torch.randint(0, nc, (n,), generator=generator))
    return DataLoader(data, batch_size=batch)


def _trainer(cfg, harness, profiler):
    return ContinuousTrainer(cfg=cfg, modelHarness=harness,
                             logger=Logger(backend="none"), profiler=profiler)


def _measurements(cfg, harness, *, warmup: int, iterations: int) -> int:
    """How many `update_fwd_bwd` blocks `iterations` inner steps open."""
    profiler = FLOPSProfiler(warmup_iters=warmup)
    trainer = _trainer(cfg, harness, profiler)
    loader = _loader()
    train_iter = iter(loader)
    for iter_count in range(1, iterations + 1):
        _l, _r, train_iter, _h = trainer.inner_cl_training_loop(
            iter_count=iter_count, cur_train_loader=loader,
            train_iter=train_iter)
    return len(profiler.profiles.get(TAG, {}).get("flop", []))


class TestWarmupIsNotMeasured:
    def test_a_warmup_iteration_opens_no_block(self, default_cfg,
                                               dummy_harness):
        """The tag must not even exist: `measure_flops` creates the entry
        on first use, so an unconditional measurement is visible in the
        keys before any averaging happens."""
        profiler = FLOPSProfiler(warmup_iters=10)
        trainer = _trainer(default_cfg, dummy_harness, profiler)
        loader = _loader()
        trainer.inner_cl_training_loop(iter_count=1, cur_train_loader=loader,
                                       train_iter=iter(loader))
        assert TAG not in profiler.profiles, (
            f"iteration 1 of 10 warmup iterations was profiled; warmup "
            f"exists because the first steps are not representative, and "
            f"including them biases every reported figure"
        )

    @pytest.mark.parametrize("warmup", [0, 1, 3])
    def test_the_count_follows_warmup_iters(self, default_cfg, dummy_harness,
                                            warmup):
        """`iter_count > warmup_iters`, strictly: over five iterations
        numbered 1..5, five minus `warmup` are measured."""
        count = _measurements(default_cfg, dummy_harness,
                              warmup=warmup, iterations=5)
        assert count == 5 - warmup, (
            f"warmup_iters={warmup}: {count} block(s) opened over 5 "
            f"iterations, expected {5 - warmup}"
        )

    def test_one_block_per_iteration_not_per_accumulation_step(
        self, default_cfg, dummy_harness
    ):
        """`step == 0`. `get_performance()` averages over blocks, so the
        block has to be the unit it means to average: one optimizer step.
        Measuring every accumulation step leaves the mean unchanged and
        the throughput wrong."""
        cfg = replace(default_cfg,
                      train=replace(default_cfg.train,
                                    grad_accumulation_steps=4))
        count = _measurements(cfg, dummy_harness, warmup=0, iterations=3)
        assert count == 3, (
            f"{count} block(s) opened over 3 iterations with "
            f"grad_accumulation_steps=4; expected one per iteration"
        )

    def test_without_a_profiler_nothing_breaks(self, default_cfg,
                                               dummy_harness):
        """`if self.profiler and ...` -- the first condition."""
        trainer = _trainer(default_cfg, dummy_harness, None)
        loader = _loader()
        trainer.inner_cl_training_loop(iter_count=5, cur_train_loader=loader,
                                       train_iter=iter(loader))


class TestMeasurementStillHappens:
    """Guards. Every case above is satisfied by a trainer that never
    profiles at all."""

    def test_past_warmup_a_block_is_opened(self, default_cfg, dummy_harness):
        count = _measurements(default_cfg, dummy_harness,
                              warmup=1, iterations=3)
        assert count > 0, "nothing was ever profiled"

    def test_the_block_recorded_a_flop_count(self, default_cfg, dummy_harness):
        profiler = FLOPSProfiler(warmup_iters=0)
        trainer = _trainer(default_cfg, dummy_harness, profiler)
        loader = _loader()
        trainer.inner_cl_training_loop(iter_count=1, cur_train_loader=loader,
                                       train_iter=iter(loader))
        assert profiler.profiles[TAG]["flop"][0] > 0, (
            "the block opened but counted nothing, so the measurement is "
            "empty and the counts above say nothing about real work"
        )


class TestAnUninspectableBatchIsAccepted:
    def test_a_batch_that_refuses_indexing_does_not_raise(self, default_cfg,
                                                          dummy_harness):
        """`_safe_next` reads `batch[1].shape` to enforce `min_batch`. A
        batch that will not be indexed that way is accepted, because
        "cannot measure this" is not "this is too small"."""
        trainer = _trainer(default_cfg, dummy_harness, None)

        class Opaque(list):
            def __getitem__(self, index):
                if index == 1:
                    raise TypeError("not indexable the way you hoped")
                return super().__getitem__(index)

        batch = Opaque([torch.randn(8, 4), torch.randint(0, 3, (8,))])

        class OneBatch:
            def __iter__(self):
                return iter([batch])

        loader = OneBatch()
        _it, got = trainer._safe_next(iter(loader), loader, min_batch=64)
        assert got is not None, (
            "an uninspectable batch was rejected; `_safe_next`'s only "
            "other move is to restart the loader and look again, which "
            "for this loader never terminates"
        )

    def test_an_inspectable_undersized_batch_is_still_rejected(
        self, default_cfg, dummy_harness
    ):
        """Guard, and the half that must not be loosened: accepting what
        cannot be measured is not accepting everything. A loader whose
        batches ARE inspectable and too small must still be refused."""
        trainer = _trainer(default_cfg, dummy_harness, None)
        loader = _loader(n=8, batch=2)
        with pytest.raises(ValueError):
            trainer._safe_next(iter(loader), loader, min_batch=64)
