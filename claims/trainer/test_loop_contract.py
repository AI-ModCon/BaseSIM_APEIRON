"""The trainer's cross-layer surface, and its guard against a short window.

Reconstruction put this layer in an unusual position: both rebuilds produce
**bit-identical weights** on every configuration probed -- default,
`max_iter` 0/1/2/5, and all five update modes -- so the training
computation is already equivalent. What diverged was bookkeeping, and
almost all of it is internal to the class and therefore free. Two things
are not.

**`outer_cl_training_loop` is declared `-> int` and returns 0** (`:138`,
`:259`). Both rebuilds returned `None`. The value has no reader --
`continuous_monitor.py:336` calls it and discards the result -- but the
annotation is part of the declared API and this project runs `mypy .` as
a standing check (CLAUDE.md), so a reconstruction that returns `None`
under an `-> int` signature fails the project's own type gate. Pinned as
the signature it declares rather than as a value anybody consumes.

**`_safe_next` refuses a window smaller than one batch.** It takes
`min_batch` and raises when no batch in the loader can satisfy it:

    ValueError: no batch in this loader holds the 4 samples min_batch asks
    for -- the largest is 1. The window is smaller than one batch.

That is reachable from configuration, which is what makes it contract
rather than plumbing: `outer_cl_training_loop` passes
`min_batch=self.cfg.train.batch_size` (`:285`, `:291`), so any run whose
`[train] batch_size` exceeds the drift window's size arrives here. The
source stops with that sentence; one rebuild trained on the short batch
instead and reported losses. Silently training on a fraction of the
intended batch, after a drift event, is the kind of difference that moves
a published number and leaves no trace saying so.

Everything else this layer's probe flagged is internal and left free,
and the identical weights are the evidence that it can be: `_safe_next`'s
own signature, whether the trainer keeps a `criterion` attribute (the
updater builds its own from `modelHarness.get_criterion()`), and which
helper names exist. None has a reader outside `continuous_trainer.py`.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from apeiron.logger.logger import Logger
from apeiron.profilers.count_flops import FLOPSProfiler
from apeiron.training.continuous_trainer import ContinuousTrainer


@pytest.fixture()
def trainer(default_cfg, dummy_harness) -> ContinuousTrainer:
    return ContinuousTrainer(
        default_cfg, dummy_harness, Logger(backend="none"),
        FLOPSProfiler(warmup_iters=0),
    )


def _loader(rows: int, batch: int, features: int = 4) -> DataLoader:
    data = TensorDataset(torch.randn(rows, features),
                         torch.zeros(rows, dtype=torch.long))
    return DataLoader(data, batch_size=batch)


class TestOuterLoopSignature:
    """Declared `-> int`, and the project type-checks."""

    def test_it_returns_an_int(self, trainer):
        result = trainer.outer_cl_training_loop(0)
        assert isinstance(result, int)
        assert not isinstance(result, bool)

    def test_it_returns_an_int_for_a_named_event(self, trainer):
        assert isinstance(trainer.outer_cl_training_loop(drift_event_id=7), int)


class TestSafeNextShape:
    """`(iterator, batch)`, in that order."""

    def test_it_returns_the_iterator_first(self, trainer):
        loader = _loader(8, 4)
        result = trainer._safe_next(iter(loader), loader, min_batch=None)
        assert len(result) == 2
        iterator, batch = result
        assert hasattr(iterator, "__next__")
        assert len(batch) == 2

    def test_a_satisfiable_min_batch_returns_a_full_batch(self, trainer):
        loader = _loader(8, 4)
        _iterator, batch = trainer._safe_next(iter(loader), loader, min_batch=4)
        assert batch[0].shape[0] == 4


class TestShortWindowIsRefused:
    """The guard a misconfigured batch_size reaches."""

    def test_a_window_smaller_than_one_batch_raises(self, trainer):
        """`outer_cl_training_loop` passes `min_batch=cfg.train.batch_size`,
        so this is what a `batch_size` larger than the drift window does.
        Training on the short batch instead changes what the model learns
        after every drift event."""
        loader = _loader(2, 1)
        with pytest.raises(ValueError):
            trainer._safe_next(iter(loader), loader, min_batch=4)

    def test_the_error_names_the_numbers_involved(self, trainer):
        """So the operator can see which setting to change. Matched on the
        two quantities rather than on the sentence, so rewording is free."""
        loader = _loader(2, 1)
        with pytest.raises(ValueError) as caught:
            trainer._safe_next(iter(loader), loader, min_batch=4)
        message = str(caught.value)
        assert "4" in message
        assert "1" in message

    def test_it_does_not_loop_forever(self, trainer):
        """The guard exists because restarting the iterator would retry the
        same rejected data indefinitely. A reconstruction without it hangs
        rather than failing, which is worse than either answer."""
        loader = _loader(3, 1)
        with pytest.raises(ValueError):
            trainer._safe_next(iter(loader), loader, min_batch=99)

    def test_a_batch_element_without_a_shape_does_not_crash(self, trainer):
        """`_safe_next` inspects the batch size with
        `getattr(batch[1], "shape", None)` and falls through when there is
        none (`:68`), so a loader whose second element is not a tensor
        reaches the min_batch guard rather than an attribute error. A
        rebuild read `.shape` directly and raised
        `AttributeError: 'str' object has no attribute 'shape'` -- which
        names the wrong problem and points the operator at the batch
        instead of at the window size."""
        class Odd:
            def __iter__(self):
                return iter([(torch.zeros(1, 4), "not-a-tensor")])

        with pytest.raises(ValueError):
            trainer._safe_next(iter(Odd()), Odd(), min_batch=2)

    def test_an_exactly_sized_window_is_accepted(self, trainer):
        """The boundary: a batch equal to min_batch must pass, or the guard
        would reject correctly configured runs."""
        loader = _loader(4, 4)
        _iterator, batch = trainer._safe_next(iter(loader), loader, min_batch=4)
        assert batch[0].shape[0] == 4
