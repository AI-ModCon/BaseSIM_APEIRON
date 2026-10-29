"""Which rows advance the step, and which annotate the one before.

The logger stamps a per-stage step on every row and advances it unless the
caller passes `increment=False` (A68). That makes the step sequence a
contract in its own right: a reader joins rows by step, so a row that
should annotate and instead advances splits one observation into two, and
a row that should advance and instead annotates merges two into one.

Measured over the pinned MNIST run, the rule is simple:

| stage | rows | repeated steps |
|---|---|---|
| `cl` | 51 | 0 -- every row advances |
| `eval` | 378 | 1 -- the post-CL validation annotation |
| `drift` | 37 | 0 -- every row advances |

and `eval`'s single repeat is one per drift event: the validation figures
share the step of the evaluation they follow, which is the whole purpose
of `increment=False` in `_log_validation`.

A rebuild produced the same 465 steps, the same 31 series and -- verified
value by value -- **identical numbers throughout**: 376 accuracies, 37
drift scores, the final test accuracy, all byte-equal. It differed only in
this, with one extra non-advancing row in each of `cl` and `eval`:

    cl    src 49, 50, 51        rebuild 49, 50, 50
    eval  src 318, 318, 319     rebuild 318, 318, 318

Those two rows shifted the global step of everything after them, and 22 of
31 series failed their digest on the label while agreeing on every value.
This is the last structural difference between the two runs, and it is
worth stating how small it is: two `increment=` flags.

Pinned as the repeat COUNT per stage rather than the absolute step
numbers, so the assertion holds for any run length while still catching
both directions -- a row that annotates where it should advance, and one
that advances where it should annotate.
"""

from __future__ import annotations

import collections
from dataclasses import replace

import numpy as np
import torch
from torch.utils.data import TensorDataset

from apeiron.config.configuration import DriftDetectionCfg
from apeiron.driver.continuous_monitor import ContinuousMonitor

#: Page-Hinkley at a near-zero threshold, so the CL path is exercised.
FIRES = {"detector_name": "PageHinkleyDetector", "ph_threshold": 0.0001,
         "ph_min_instances": 1, "ph_delta": 0.0, "detection_interval": 4,
         "max_stream_updates": 1}


def _fixed(seed: int, n: int = 24, in_f: int = 4, nc: int = 3) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    return TensorDataset(torch.randn(n, in_f, generator=generator),
                         torch.randint(0, nc, (n,), generator=generator))


def _steps_by_stage(default_cfg):
    """stage -> the per-stage step recorded on each of its rows, in order."""
    from conftest import DummyHarness

    np.random.seed(5)
    torch.manual_seed(5)
    cfg = replace(default_cfg,
                  drift_detection=replace(DriftDetectionCfg(), **FIRES))
    harness = DummyHarness(cfg, train_data=_fixed(1), val_data=_fixed(2),
                           hist_train_data=_fixed(3), hist_val_data=_fixed(4))
    with torch.no_grad():
        for index, parameter in enumerate(harness.model.parameters()):
            parameter.copy_(torch.full_like(parameter, 0.01 * (index + 1)))

    monitor = ContinuousMonitor(cfg=cfg, modelHarness=harness)
    seen: dict[str, list[int]] = collections.defaultdict(list)
    stage = {"name": None}
    counters: dict[str, int] = collections.defaultdict(int)
    real_stage = monitor.logger.stage

    def stage_hook(name):
        stage["name"] = name
        return real_stage(name)

    def log(metrics, *args, increment=True, **kwargs):
        """Mirror the logger's own accounting: record the current step, then
        advance it only when the caller asked for that."""
        name = stage["name"]
        seen[name].append(counters[name])
        if increment:
            counters[name] += 1

    monitor.logger.stage = stage_hook
    monitor.logger.log = log
    monitor.run()
    return dict(seen), monitor.drift_event_count


def _repeats(steps: list[int]) -> int:
    return len(steps) - len(set(steps))


class TestWhichRowsAdvance:
    def test_every_cl_row_advances(self, default_cfg):
        """A continual-learning row is an observation of its own iteration.
        One that annotates instead merges two iterations onto one step."""
        steps, _events = _steps_by_stage(default_cfg)
        assert steps.get("cl"), "no cl row was recorded"
        assert _repeats(steps["cl"]) == 0, (
            f"{_repeats(steps['cl'])} cl row(s) shared a step with the row "
            f"before; every CL iteration is its own observation"
        )

    def test_every_drift_row_advances(self, default_cfg):
        steps, _events = _steps_by_stage(default_cfg)
        assert steps.get("drift"), "no drift row was recorded"
        assert _repeats(steps["drift"]) == 0

    def test_eval_repeats_once_per_drift_event(self, default_cfg):
        """The post-CL validation annotates the evaluation it follows, so
        it shares that step -- exactly once per adaptation, no more."""
        steps, events = _steps_by_stage(default_cfg)
        assert steps.get("eval"), "no eval row was recorded"
        assert events > 0, "the fixture never adapted; nothing to annotate"
        assert _repeats(steps["eval"]) == events, (
            f"{_repeats(steps['eval'])} eval row(s) annotate rather than "
            f"advance, with {events} drift event(s); expected one per event"
        )

    def test_the_steps_never_go_backwards(self, default_cfg):
        steps, _events = _steps_by_stage(default_cfg)
        for stage, values in steps.items():
            assert values == sorted(values), f"{stage} steps went backwards"


class TestTheFixtureExercisesBothKinds:
    """Guards: the counts above are only meaningful if both occur."""

    def test_an_annotating_row_happens_at_all(self, default_cfg):
        steps, _events = _steps_by_stage(default_cfg)
        assert _repeats(steps["eval"]) > 0, (
            "no row annotated, so the eval assertion would pass for an "
            "implementation that never uses increment=False"
        )

    def test_advancing_rows_happen_at_all(self, default_cfg):
        steps, _events = _steps_by_stage(default_cfg)
        assert len(set(steps["eval"])) > 1
        assert len(set(steps["cl"])) > 1
