"""Which side of `fwd_bwd` the second-order work happens on.

`fwd_bwd` is the only thing the trainer wraps in a FLOP counter:

    if self.profiler and iter_count > self.profiler.warmup_iters and step == 0:
        with self.profiler.measure_flops(tag="update_fwd_bwd"):
            loss += self.cl_updater.fwd_bwd(train_batch_tuple, hist_batch_tuple)

so whatever an updater chooses to do inside that call is reported as the
cost of a continual-learning update, and whatever it does in
`update_post_fwd_bwd` is not. The split is a measurement boundary, and an
updater sits on one side of it or the other.

K-FAC accumulates two curvature factors per layer, `aᵀa` and `gᵀg`, and
for a convolution those are the most expensive matmuls in the whole step
-- the unfolded patch matrix is (N·H·W, C·kH·kW). `src/` accumulates them
in `update_post_fwd_bwd`, outside the window. Measured over the shared
`TinyCNN` fixture, with the same counter the profiler uses:

| `update_mode` | inside `fwd_bwd` | inside `update_post_fwd_bwd` |
|---|---|---|
| `base` | 74304 | 0 |
| `none` | 0 | 0 |
| `jvp_reg` | 74304 | 0 |
| `ewc_online` | 74304 | 0 |
| `kfac_online` | 74304 | 100832 |

Every mode that trains pays exactly the model's own forward/backward
inside the window. K-FAC's extra work -- more than the pass itself --
lands outside it.

A rebuild accumulated the same two factors in the backward hook instead,
which is inside `fwd_bwd`: 174032 and 1104. The arithmetic is equivalent
and the run proved it -- over the pinned MNIST run the two
implementations agreed on **every** value of **every** series, 376
accuracies, 376 losses, 37 drift scores and the final test accuracy, with
one exception: `cperf_update_fwd_bwd_flop` read 40784777728 where the
original reads 4815519744. The reported cost of a CL update was 8.5x too
high, which is the number a reader would use to decide whether continual
learning is affordable.

Pinned relative to `base` rather than as absolute counts, so the
assertion holds for any model and batch while still refusing both
directions: work moved into the window, and K-FAC's curvature work not
being counted anywhere.

A NOTE ON WHAT THIS DOES NOT SAY. It does not say hooks are wrong. An
implementation may capture activations and grad-outputs in hooks -- `src/`
does -- as long as it only *stores* them there and contracts them after
the measured pass. The boundary is about where the arithmetic happens,
not where the tensors come from.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.flop_counter import FlopCounterMode

from apeiron.training.updater.create_updater import create_updater

#: Modes that run a forward/backward at all. `none` is handled separately.
TRAINING_MODES = ("base", "jvp_reg", "ewc_online", "kfac_online")


def _conv_data(seed: int, n: int = 16) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    return TensorDataset(torch.randn(n, 1, 8, 8, generator=generator),
                         torch.randint(0, 3, (n,), generator=generator))


def _split(default_cfg, make_harness, tiny_cnn_factory, mode: str) -> tuple[int, int]:
    """(FLOPs counted inside `fwd_bwd`, FLOPs counted inside the post hook).

    The same `FlopCounterMode` the profiler uses, so the numbers are the
    ones that reach the metrics file.
    """
    torch.manual_seed(0)
    cfg = replace(default_cfg,
                  continual_learning=replace(default_cfg.continual_learning,
                                             update_mode=mode))
    harness = make_harness(cfg, model=tiny_cnn_factory(),
                           train_data=_conv_data(1), val_data=_conv_data(2))
    updater = create_updater(cfg=cfg, modelHarness=harness)
    batch = next(iter(harness.get_train_dataloaders()[0]))

    updater.cl_preprocessing()
    updater.update_pre_fwd_bwd()

    inside = FlopCounterMode(display=False, depth=None)
    with inside:
        updater.fwd_bwd((batch[0], batch[1]), None)
    after = FlopCounterMode(display=False, depth=None)
    with after:
        updater.update_post_fwd_bwd()
    return inside.get_total_flops(), after.get_total_flops()


@pytest.fixture()
def cost(default_cfg, make_harness, tiny_cnn):
    """mode -> (flops inside fwd_bwd, flops inside update_post_fwd_bwd)."""
    def measure(mode: str) -> tuple[int, int]:
        return _split(default_cfg, make_harness, type(tiny_cnn), mode)
    return measure


class TestTheFixtureCanSeeADifference:
    """Guards. Without a convolution K-FAC's expensive factors never arise,
    and without a non-zero baseline every assertion below is vacuous."""

    def test_the_model_has_a_convolution(self, tiny_cnn):
        assert any(isinstance(m, nn.Conv2d) for m in tiny_cnn.modules()), (
            "no Conv2d in the fixture; K-FAC's unfolded patch matmuls are "
            "the expensive ones and a linear-only model would not show them"
        )

    def test_a_training_pass_costs_something(self, cost):
        inside, _after = cost("base")
        assert inside > 0, "the counter reported nothing for a vanilla pass"

    def test_the_counter_discriminates(self, cost):
        """The same measuring device must report different numbers for
        different work, or the comparisons below prove nothing."""
        assert cost("none")[0] == 0, (
            "the `none` updater trains nothing, so its measured pass must "
            "cost nothing; a counter that reports the same number for it "
            "and for `base` is not measuring"
        )

    def test_measuring_twice_gives_the_same_number(self, cost):
        """An op count, not a timing -- so it is comparable at all."""
        assert cost("kfac_online") == cost("kfac_online")


class TestTheMeasuredPassIsJustThePass:
    @pytest.mark.parametrize("mode", TRAINING_MODES)
    def test_every_mode_pays_the_same_inside_the_window(self, cost, mode):
        """The whole finding. `update_fwd_bwd` reports the cost of a CL
        update, and every mode's update is one forward/backward of the
        model -- whatever else the mode needs, it does outside."""
        baseline, _ = cost("base")
        inside, _ = cost(mode)
        assert inside == baseline, (
            f"`{mode}` counted {inside} FLOPs inside fwd_bwd against "
            f"{baseline} for a vanilla pass. Anything extra in there is "
            f"reported as the cost of an update: a rebuild that accumulated "
            f"K-FAC's factors in the backward hook reported 8.5x the real "
            f"figure over the MNIST run while every other value matched"
        )


class TestTheSecondOrderWorkIsStillCountedSomewhere:
    """The other direction: outside the window is not nowhere."""

    def test_kfac_does_its_curvature_work_after_the_pass(self, cost):
        """`aᵀa` and `gᵀg` per layer, and for a convolution they are
        larger than the pass that produced them. If this is zero the
        factors are not being accumulated at all -- which would satisfy
        the assertion above for the wrong reason."""
        _inside, after = cost("kfac_online")
        baseline, _ = cost("base")
        assert after > baseline, (
            f"K-FAC's post-pass hook counted {after} FLOPs against "
            f"{baseline} for the pass itself; the two curvature factors "
            f"are the most expensive matmuls in the step and must be "
            f"accumulated here, not inside the measured pass"
        )

    @pytest.mark.parametrize("mode", ("base", "jvp_reg", "ewc_online"))
    def test_the_other_modes_do_no_matmuls_after_the_pass(self, cost, mode):
        """Only K-FAC contracts anything afterwards. EWC's Fisher is
        element-wise on `.grad` and the counter does not count it, which
        is why this is zero rather than small."""
        _inside, after = cost(mode)
        assert after == 0, (
            f"`{mode}` counted {after} FLOPs after the pass; only "
            f"`kfac_online` contracts curvature factors there"
        )
