"""The post-CL validation row shares a step with the losses it describes.

`_log_validation` logs with `commit=False, increment=False`, and the
source says why in a comment:

    # increment=False: annotate the CL round's step, do not advance it.
    logger.log(payload, commit=False, increment=False)

So the validation figures land on the SAME global step as the CL round's
final losses, which is what makes them joinable: a reader selecting step
400 gets `cl/jvp_reg_total_loss` and `eval/val_post_cur_accuracy`
together, the loss and the accuracy it produced.

A rebuild advanced the step first. Its validation rows landed one step
late throughout -- 401 where the original writes 400, 402 where it writes
401 -- and because the run claim digests each series over its
`(step, value)` sequence, **16 of 31 series failed on the label while
their values were identical.** The two files had the same 465 steps, the
same 31 series and the same numbers; only the pairing was off by one.

Pinned as the co-location rather than as the keyword, because the keyword
is one way to achieve it and the pairing is the thing a reader depends on.
A rebuild that passed `increment=False` and still advanced the step
somewhere earlier would satisfy an assertion about the call and fail the
file.
"""

from __future__ import annotations

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


def _rows(cfg, harness) -> list[tuple[int, frozenset]]:
    """(step, metric names) for every row the outer loop records.

    A real `Logger` with a `csv_path`, because the step stamping and the
    `increment` handling are the logger's and a stub would not reproduce
    them. The trainer logs through a logger it fetches itself as well as
    the injected one (A67), so the same object is returned for both.
    """
    path = Path(tempfile.mkdtemp()) / "m.csv"
    logger = Logger(backend="none", csv_path=path)
    with patch("apeiron.training.continuous_trainer.get_logger",
               return_value=logger):
        trainer = ContinuousTrainer(cfg, harness, logger,
                                    FLOPSProfiler(warmup_iters=0))
        trainer.outer_cl_training_loop(drift_event_id=3)
    frame = logger.to_dataframe()
    assert frame is not None, "nothing was recorded"
    grouped: dict[int, set] = {}
    for step, metric in zip(frame["step"], frame["metric"]):
        grouped.setdefault(int(step), set()).add(str(metric))
    return [(s, frozenset(m)) for s, m in sorted(grouped.items())]


@pytest.fixture()
def harness_with_history(default_cfg):
    from conftest import DummyHarness
    harness = DummyHarness(default_cfg, hist_train_data=_fixed(5),
                           hist_val_data=_fixed(9))
    harness.register_task([1.0])
    return harness


def _step_of(rows, prefix: str) -> int | None:
    for step, names in rows:
        if any(n.split("/", 1)[-1].startswith(prefix) for n in names):
            return step
    return None


class TestTheValidationRowIsCoLocated:
    def test_validation_shares_a_step_with_the_cl_losses(
        self, default_cfg, harness_with_history
    ):
        """The pairing the comment promises: one step carries both."""
        rows = _rows(default_cfg, harness_with_history)
        loss_steps = {s for s, names in rows
                      if any("jvp_reg_total_loss" in n for n in names)}
        val_steps = {s for s, names in rows
                     if any("val_post_" in n for n in names)}
        assert loss_steps, "no CL loss row was recorded"
        assert val_steps, "no post-CL validation row was recorded"
        assert val_steps <= loss_steps, (
            f"validation landed on step(s) {sorted(val_steps - loss_steps)} "
            f"with no CL losses; the loss and the accuracy it produced must "
            f"share a step or a reader cannot join them"
        )

    def test_validation_lands_on_the_LAST_cl_step(self, default_cfg,
                                                   harness_with_history):
        """The exact failure mode: off by one, every value correct.

        It is the LAST loss row validation shares a step with, not the
        first -- the round logs one per inner iteration. The first version
        of this case compared against the first and failed against `src/`,
        where `max_iter=2` puts the losses on steps 1 and 2 and validation
        on 2.
        """
        rows = _rows(default_cfg, harness_with_history)
        loss_steps = [s for s, names in rows
                      if any("jvp_reg_total_loss" in n for n in names)]
        val_step = _step_of(rows, "val_post_")
        assert loss_steps and val_step is not None
        assert val_step == max(loss_steps), (
            f"validation is at step {val_step}; the round's last losses are "
            f"at {max(loss_steps)}. `increment=False` exists so the two "
            f"share a step and a reader can join them"
        )

    def test_the_validation_row_does_not_advance_the_step(
        self, default_cfg, harness_with_history
    ):
        """Whatever is logged after validation must not be pushed along by
        it either -- `increment=False` is about the rows that follow as
        much as the row itself."""
        rows = _rows(default_cfg, harness_with_history)
        val_step = _step_of(rows, "val_post_")
        after = [s for s, _n in rows if s > val_step]
        if after:
            assert min(after) == val_step + 1, (
                f"the row after validation is at step {min(after)}, "
                f"expected {val_step + 1}"
            )


class TestTheTransferRowIsAlsoAnnotation:
    """The same treatment for the transfer metrics."""

    def test_the_transfer_row_shares_a_step_with_something(
        self, default_cfg, harness_with_history
    ):
        """`fwt` and the test accuracies annotate a round rather than
        starting a new one, so their step must carry other metrics too."""
        rows = _rows(default_cfg, harness_with_history)
        transfer_steps = [(s, names) for s, names in rows
                          if any(n.endswith("/fwt") or "test_curr_acc" in n
                                 for n in names)]
        for step, names in transfer_steps:
            others = {n for n in names
                      if "fwt" not in n and "test_" not in n
                      and not n.endswith("step")}
            assert others, (
                f"step {step} carries only transfer metrics {sorted(names)}; "
                f"they annotate a round and should share its step"
            )
