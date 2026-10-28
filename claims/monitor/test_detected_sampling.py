"""`detected` is logged on real events and a sample of the rest.

`_log_metrics` decides whether to write the column at all:

    log_detected = drift_signal.drift_detected or (np.random.random() <= 0.1)
    ...
    **({"detected": int(drift_signal.drift_detected)} if log_detected else {})

so a real detection is always recorded -- the source's own comment is
"always included so no true drift events are dropped" -- and a non-event
is recorded about one time in ten.

A rebuild wrote the column on **every** check, unconditionally. By itself
that reads like more information, not less, and the monitor's own tests
never counted the rows. It broke the pinned MNIST run in a way that took
a while to see: the logger stamps a GLOBAL `step` on every row it records
(A68), and the run claim digests each series over its `(step, value)`
sequence. Thirty-five extra `detected` rows therefore shifted the step
label of every row logged after them, and **17 of 31 series failed their
digest while their values were identical**. The column that differed was
not among the ones that failed.

Pinned as a count under a seeded RNG, which is what makes it assertable
at all: the sampling draws from the global numpy RNG, and the run seeds it
from the config (A41), so for a given seed the number of sampled rows is a
fixed number rather than a distribution.

AN IMPROVEMENT, filed not pinned (claims/LOG.md): the sampling makes the
file's SCHEMA depend on the RNG -- whether a given non-event row carries
the column at all varies between runs unless numpy is seeded, and a run
that forgets to seed produces a differently-shaped file every time. A
column that is always present, carrying the flag, would cost one value per
row and remove that.
"""

from __future__ import annotations

import numpy as np
import pytest

from apeiron.drift_detection.detectors.base import DriftSignal, LearningRegime
from apeiron.driver.continuous_monitor import ContinuousMonitor


def _rows_with_detected(cfg, harness, *, checks: int, seed: int,
                        always_fire: bool = False) -> tuple[int, int]:
    """(rows carrying `detected`, total drift rows) over `checks` checks."""
    np.random.seed(seed)
    monitor = ContinuousMonitor(cfg=cfg, modelHarness=harness)

    if always_fire:
        class AlwaysFires:
            def update(self, value, **kwargs):
                return DriftSignal(regime=LearningRegime.CONTINUAL_LEARNING,
                                   drift_detected=True, drift_score=1.0,
                                   confidence=0.5, metadata={})

            def reset(self):
                pass

        monitor.__dict__["detector"] = AlwaysFires()

    counts = {"detected": 0, "total": 0}

    def log(metrics, *args, **kwargs):
        counts["total"] += 1
        if "detected" in metrics:
            counts["detected"] += 1

    monitor.logger.log = log
    monitor.logger.info = lambda *a, **kw: None
    for _ in range(checks):
        monitor.metric_buffer = [[1.0, 2.0]]
        try:
            monitor._check_drift()
        except Exception:                 # noqa: BLE001 — adaptation is not under test
            break
    return counts["detected"], counts["total"]


class TestANonEventIsSampled:
    """Not every check writes the column."""

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_far_fewer_than_every_check(self, default_cfg, dummy_harness, seed):
        """The whole finding: a rebuild wrote it on all of them, and the
        extra rows shifted every later step label."""
        detected, total = _rows_with_detected(
            default_cfg, dummy_harness, checks=40, seed=seed)
        assert total == 40, f"expected 40 drift rows, got {total}"
        assert detected < total, (
            f"`detected` was written on all {total} checks; it is sampled "
            f"for non-events, and the extra rows move the global step of "
            f"every row after them"
        )

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_the_sampled_share_is_about_a_tenth(self, default_cfg,
                                               dummy_harness, seed):
        """`<= 0.1` against a uniform draw. Asserted as a band rather than
        an exact count so the case does not depend on numpy's stream
        staying byte-identical across versions, while still refusing both
        'never' and 'always'."""
        detected, total = _rows_with_detected(
            default_cfg, dummy_harness, checks=200, seed=seed)
        assert 0 < detected <= total * 0.25, (
            f"{detected} of {total} non-events carried `detected`; expected "
            f"roughly a tenth"
        )

    def test_the_same_seed_gives_the_same_count(self, default_cfg,
                                                dummy_harness):
        """It draws from the global numpy RNG, so a seeded run is
        reproducible -- which is the only reason the MNIST claim can pin a
        digest over these rows at all."""
        first = _rows_with_detected(default_cfg, dummy_harness,
                                    checks=60, seed=7)
        second = _rows_with_detected(default_cfg, dummy_harness,
                                     checks=60, seed=7)
        assert first == second


class TestARealEventIsNeverDropped:
    """The half that must not be sampled."""

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_every_firing_check_carries_detected(self, default_cfg,
                                                 dummy_harness, seed):
        """"always included so no true drift events are dropped". With a
        detector that fires on every check, every row must have it --
        whatever the sampling draw would have said."""
        detected, total = _rows_with_detected(
            default_cfg, dummy_harness, checks=5, seed=seed, always_fire=True)
        assert total > 0, "no drift row was logged"
        assert detected == total, (
            f"only {detected} of {total} FIRING checks carried `detected`; "
            f"real events must never be sampled out"
        )
