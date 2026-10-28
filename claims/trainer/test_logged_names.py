"""The exact column names the trainer writes, and how many times.

A67 pinned which STAGES the trainer logs under. This pins the names inside
them, which the stage namespace does not determine. Three differences
survived A67 in the assembled MNIST run -- 35 metric series against the
expected 31 -- and all three are names:

    original                             rebuild
    cl/jvp_reg_total_loss                cl/kfac_online_total_loss
    cl/jvp_reg_forgetting_loss           cl/kfac_online_forgetting_loss
    cl/jvp_reg_generation_loss           cl/kfac_online_generation_loss
    cperf_update_fwd_bwd_*               cperf_fwd_bwd_*
    cperf_optimizer_*                    cperf_opt_*
    (no val_pre_* anywhere)              eval/val_pre_cur_accuracy, ...

**The loss columns are hard-coded to one updater's name.** The trainer
writes `jvp_reg_total_loss` whatever `continual_learning.update_mode` is,
so a run using `kfac_online` files its losses under columns named after
`jvp_reg` (`continuous_trainer.py:193`-`:195`). The rebuild named them for
the configured mode, which is what a reader would expect. The current
names are pinned here anyway, because the claim's reproducibility depends
on them and a criterion records what the code does; the case for renaming
them is filed separately (claims/LOG.md, A69) and is the team's to make,
not a criterion's.

**The profiler tags decide the column names.** `tag="update_fwd_bwd"` and
`tag="optimizer"` (`:306`, `:316`) become `cperf_update_fwd_bwd_*` and
`cperf_optimizer_*`. Shorter tags are tempting and would rename columns in
every existing run; `docs/profiler.md` and the determinism filter match on
the current ones.

**Validation is logged ONCE, after the CL round.** The trainer calls
`_log_validation("post", ...)` at `:209` and nowhere else. The rebuild
also logged a `"pre"` set, which adds four columns the original never
writes. The pre-CL figure is not missing from the original -- it is
recorded as `test_pre_cl_acc` -- so the rebuild's extra set duplicates it
under a second naming scheme.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch.utils.data import TensorDataset

from apeiron.logger.logger import Logger
from apeiron.profilers.count_flops import FLOPSProfiler
from apeiron.training.continuous_trainer import ContinuousTrainer

#: The loss columns, written under the `cl` stage for every update mode.
LOSS_NAMES = {"jvp_reg_total_loss", "jvp_reg_forgetting_loss",
              "jvp_reg_generation_loss"}

#: Profiler tags, which become `cperf_{tag}_{flop,time,flops}` columns.
PROFILER_TAGS = {"update_fwd_bwd", "optimizer"}


def _fixed(seed: int, n: int = 32, in_f: int = 4, nc: int = 3) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    return TensorDataset(torch.randn(n, in_f, generator=generator),
                         torch.randint(0, nc, (n,), generator=generator))


def _run_and_collect(cfg, harness):
    """Every metric name the outer loop logs, and the tags it measures under.

    The trainer logs through a logger it fetches itself with
    `get_logger(__name__)`, not the injected one (see A67), so both are
    recorded.
    """
    names: set[str] = set()
    tags: set[str] = set()
    injected = Logger(backend="none")
    injected.log = lambda metrics, *a, **kw: names.update(metrics)

    class Recording:
        step = 0

        def log(self, metrics, *args, **kwargs):
            names.update(metrics)

        def stage(self, name):
            pass

        def __getattr__(self, _name):
            return lambda *a, **kw: None

    profiler = FLOPSProfiler(warmup_iters=0)
    real_measure = profiler.measure_flops
    real_optimizer = profiler.measure_flops_optimizer

    def measure(tag="default"):
        tags.add(tag)
        return real_measure(tag)

    def measure_optimizer(model, device, tag="optimizer"):
        tags.add(tag)
        return real_optimizer(model, device, tag)

    profiler.measure_flops = measure
    profiler.measure_flops_optimizer = measure_optimizer

    with patch("apeiron.training.continuous_trainer.get_logger",
               return_value=Recording()):
        trainer = ContinuousTrainer(cfg, harness, injected, profiler)
        trainer.outer_cl_training_loop(drift_event_id=3)
    return names, tags


@pytest.fixture()
def harness_with_history(default_cfg):
    from conftest import DummyHarness
    harness = DummyHarness(default_cfg, hist_train_data=_fixed(5),
                           hist_val_data=_fixed(9))
    harness.register_task([1.0])
    return harness


class TestTheLossColumnNames:
    def test_the_loss_columns_use_the_jvp_reg_prefix(self, default_cfg,
                                                    harness_with_history):
        """Hard-coded, and pinned because the record depends on it. See
        A69 for the case that it should follow `update_mode` instead."""
        names, _tags = _run_and_collect(default_cfg, harness_with_history)
        missing = sorted(LOSS_NAMES - names)
        assert not missing, f"the CL rows are missing {missing}"

    def test_the_prefix_does_not_follow_the_update_mode(self, default_cfg,
                                                       harness_with_history):
        """The surprising half, stated so a reconstruction cannot
        'improve' it without the gate noticing."""
        from dataclasses import replace
        cfg = replace(
            default_cfg,
            continual_learning=replace(default_cfg.continual_learning,
                                       update_mode="kfac_online"),
        )
        names, _tags = _run_and_collect(cfg, harness_with_history)
        assert "jvp_reg_total_loss" in names
        assert not [n for n in names if n.startswith("kfac_online_")]


class TestTheProfilerTags:
    def test_the_measured_tags_are_the_documented_ones(self, default_cfg,
                                                       harness_with_history):
        """These become `cperf_{tag}_*` columns, which `docs/profiler.md`
        and the determinism filter both match on."""
        _names, tags = _run_and_collect(default_cfg, harness_with_history)
        missing = sorted(PROFILER_TAGS - tags)
        assert not missing, f"the loop never measured under {missing}"

    def test_no_abbreviated_tag_is_used(self, default_cfg,
                                        harness_with_history):
        _names, tags = _run_and_collect(default_cfg, harness_with_history)
        assert "fwd_bwd" not in tags
        assert "opt" not in tags


class TestValidationIsLoggedOnce:
    def test_the_post_round_validation_is_recorded(self, default_cfg,
                                                   harness_with_history):
        names, _tags = _run_and_collect(default_cfg, harness_with_history)
        assert [n for n in names if n.startswith("val_post_")], (
            "the post-CL validation was not recorded"
        )

    def test_no_pre_round_validation_set_is_written(self, default_cfg,
                                                   harness_with_history):
        """`_log_validation` is called once, with `"post"`. The pre-CL
        figure is already recorded as `test_pre_cl_acc`, so a second
        `val_pre_*` set duplicates it under another scheme and adds four
        columns the original never writes."""
        names, _tags = _run_and_collect(default_cfg, harness_with_history)
        extra = sorted(n for n in names if n.startswith("val_pre_"))
        assert not extra, f"the loop wrote an extra validation set: {extra}"
