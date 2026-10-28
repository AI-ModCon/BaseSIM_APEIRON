"""The stages the trainer logs under, which decide its CSV namespace.

`Logger.stage(name)` sets the namespace every subsequent `log()` call is
written under, and it is sticky: whatever the last caller set stays in
force. The trainer sets it four times -- `"eval"` around validation,
`"cl"` around the continual-learning rows (`continuous_trainer.py:96`,
`:190`, `:233`, `:252`).

A rebuild called `stage()` **not once**. Its own tests never noticed,
because they read the metrics it passes to `log()` rather than the
namespace those metrics land in, and a trainer driven on its own has no
other caller to inherit a stage from.

In a real run it does. `ContinuousMonitor` sets `"drift"` before checking,
then dispatches the CL loop -- so every row the rebuilt trainer wrote went
into the drift namespace. Against the pinned MNIST claim the produced file
carried 27 metric series where the original carries 31, with the whole
`cl/*` namespace absent and its contents reappearing as `drift/gen_loss`,
`drift/reg_loss`, `drift/cl_iter`:

    original  cl/jvp_reg_total_loss, cl/jvp_reg_forgetting_loss,
              cl/jvp_reg_generation_loss, cl/drift_event_id, cl/step,
              cl/cperf_*, eval/val_post_*, eval/test_*, eval/fwt
    rebuild   drift/gen_loss, drift/reg_loss, drift/cl_iter,
              drift/val_post_*, drift/test_*, drift/fwt

An analysis selecting `cl/` finds nothing, and the drift namespace is
polluted with training rows. The run trained correctly and produced the
right weights.

Asserted by spying on `stage()` rather than on the rows, because the stage
is the contract and the row contents are already covered elsewhere. This
also makes the assertion independent of how far a fixture drives the loop:
the trainer either names its namespaces or it does not.
"""

from __future__ import annotations

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


def _trainer_and_stages(cfg, harness):
    """Run the outer loop, recording every stage the trainer sets.

    The stage calls do NOT go through the logger handed to the
    constructor: `outer_cl_training_loop` fetches its own with
    `get_logger(__name__)` (`continuous_trainer.py:140`). A spy on the
    injected logger therefore sees nothing, which is how the first version
    of this file came to fail against `src/`. Both are recorded here so
    the assertion holds whichever one an implementation uses.
    """
    from unittest.mock import patch

    injected = Logger(backend="none")
    stages: list[str] = []

    def record(name):
        stages.append(name)

    class Recording:
        """Stands in for the module-level logger."""
        step = 0

        def stage(self, name):
            record(name)

        def __getattr__(self, _name):
            return lambda *a, **kw: None

    injected.stage = record
    with patch("apeiron.training.continuous_trainer.get_logger",
               return_value=Recording()):
        trainer = ContinuousTrainer(cfg, harness, injected,
                                    FLOPSProfiler(warmup_iters=0))
        trainer.outer_cl_training_loop(drift_event_id=3)
    return stages


class TestTheTrainerNamesItsNamespaces:
    """`stage()` is sticky, so a caller that never sets it inherits one."""

    def test_it_sets_a_stage_at_all(self, default_cfg, dummy_harness):
        """The whole finding in one assertion: a trainer that never calls
        `stage()` writes its rows into whichever namespace the monitor left
        in force."""
        stages = _trainer_and_stages(default_cfg, dummy_harness)
        assert stages, (
            "the outer loop never called logger.stage(); every row it logs "
            "lands in the caller's namespace -- `drift/` in a real run"
        )

    def test_it_uses_the_cl_stage(self, default_cfg, dummy_harness):
        stages = _trainer_and_stages(default_cfg, dummy_harness)
        assert "cl" in stages, (
            f"the continual-learning rows need the `cl` namespace; "
            f"stages used: {sorted(set(stages))}"
        )

    def test_it_uses_the_eval_stage(self, default_cfg, dummy_harness):
        stages = _trainer_and_stages(default_cfg, dummy_harness)
        assert "eval" in stages, (
            f"validation rows need the `eval` namespace; "
            f"stages used: {sorted(set(stages))}"
        )

    def test_it_never_writes_into_the_drift_namespace(self, default_cfg,
                                                     dummy_harness):
        """The drift namespace belongs to the monitor's detection rows. A
        trainer that set it would make its training rows indistinguishable
        from drift measurements."""
        stages = _trainer_and_stages(default_cfg, dummy_harness)
        assert "drift" not in stages, (
            "the trainer set the `drift` stage; that namespace is the "
            "monitor's"
        )

    def test_it_sets_a_stage_with_history_present_too(self, default_cfg):
        """The historical-data path logs its own validation rows, and they
        belong in `eval` with the rest."""
        from conftest import DummyHarness
        harness = DummyHarness(default_cfg, hist_train_data=_fixed(5),
                               hist_val_data=_fixed(9))
        harness.register_task([1.0])
        stages = _trainer_and_stages(default_cfg, harness)
        assert "eval" in stages
        assert "cl" in stages


class TestTheStageNamesAreTheValidOnes:
    """`Logger.stage` rejects anything outside its own set."""

    @pytest.mark.parametrize("name", ["cl", "eval"])
    def test_the_logger_accepts_the_stages_the_trainer_uses(self, name):
        """A guard against pinning a stage the logger would refuse: the two
        names above have to be ones `Logger` actually permits."""
        Logger(backend="none").stage(name)
