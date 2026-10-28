"""What `update_post_fwd_bwd()` returns, which is a logged metric.

The hook's return value is not internal. `ContinuousTrainer` takes it as
`reg_loss` (`continuous_trainer.py:311`), hands it back from
`inner_cl_training_loop` as `forgetting_loss`, and logs it (`:191`-`:196`):

    "jvp_reg_total_loss": generation_loss + forgetting_loss,
    "jvp_reg_forgetting_loss": forgetting_loss,
    "jvp_reg_generation_loss": generation_loss,

So the number this hook returns becomes two columns in the metrics CSV --
the forgetting loss itself, and half of the total. Nothing in this
directory's tests asserts it.

**With no historical batch, every update mode returns exactly 0.0.** There
is nothing to forget yet: no past data entered the step, so no
regularisation term applies. A reconstruction returned the full training
loss instead, for every jvp_reg configuration, with weights
BIT-IDENTICAL to the original's -- the training was right and the metric
was wrong. That is the worst shape a divergence can take, because the run
looks healthy: the model trains correctly and the published forgetting
curve rises from zero to the generation loss, while `jvp_reg_total_loss`
double-counts the same number.

Measured across all five update modes and both `mix_historic_data`
settings, so the assertion is about the hook's contract rather than about
one mode's arithmetic.
"""

from __future__ import annotations

import pytest

from apeiron.training.updater.create_updater import create_updater

MODES = ("base", "jvp_reg", "ewc_online", "kfac_online", "none")


def _updater(default_cfg, dummy_harness, mode: str, mix: bool):
    from dataclasses import replace
    cfg = replace(
        default_cfg,
        continual_learning=replace(default_cfg.continual_learning,
                                   update_mode=mode, mix_historic_data=mix),
    )
    return cfg, create_updater(cfg, dummy_harness)


class TestNoHistoryMeansNoPenalty:
    """Nothing to forget yet, so the logged forgetting loss must be zero."""

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("mix", [False, True])
    def test_the_penalty_is_zero_without_a_historical_batch(
        self, default_cfg, dummy_harness, mode, mix
    ):
        _cfg, updater = _updater(default_cfg, dummy_harness, mode, mix)
        batch = next(iter(dummy_harness.get_train_dataloaders()[0]))
        updater.fwd_bwd(batch)                 # no historical batch supplied
        assert float(updater.update_post_fwd_bwd()) == pytest.approx(0.0)

    @pytest.mark.parametrize("mode", MODES)
    def test_it_is_not_merely_falsy(self, default_cfg, dummy_harness, mode):
        """`None` would also pass a truthiness check and would then crash
        the trainer's `generation_loss + forgetting_loss`."""
        _cfg, updater = _updater(default_cfg, dummy_harness, mode, False)
        batch = next(iter(dummy_harness.get_train_dataloaders()[0]))
        updater.fwd_bwd(batch)
        penalty = updater.update_post_fwd_bwd()
        assert penalty is not None
        assert float(penalty) + 1.0 == pytest.approx(1.0)

    @pytest.mark.parametrize("mode", MODES)
    def test_the_penalty_is_summable_with_a_loss(
        self, default_cfg, dummy_harness, mode
    ):
        """The trainer adds it to the generation loss, so whatever it is has
        to support that without a shape or type surprise."""
        _cfg, updater = _updater(default_cfg, dummy_harness, mode, False)
        batch = next(iter(dummy_harness.get_train_dataloaders()[0]))
        loss = updater.fwd_bwd(batch)
        total = float(loss) + float(updater.update_post_fwd_bwd())
        assert total == pytest.approx(float(loss))
