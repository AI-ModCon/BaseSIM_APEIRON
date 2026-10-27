"""The surface `ContinuousMonitor`'s subclasses read off it.

`src/drift_only.py` and `src/cl_only.py` both subclass `ContinuousMonitor`
and reach into it for state and behaviour the class never promised in a
test:

  `label`              set by both subclasses (drift_only.py:56,
                       cl_only.py:257) and read back when the run is
                       announced, so the base class has to establish it.
  `metric_idx`         read to name the logged metric
                       (drift_only.py:112, drift_only.py:168,
                       cl_only.py:384) as `f"metric_{self.metric_idx}"`,
                       which puts it in the CSV column names every
                       downstream analysis reads.
  `last_agg_metric`    read after a check (drift_only.py:112) to log the
                       value the detector actually saw.
  `aggregation`        the reduction `_aggregate_buffer` applies.
  `_aggregate_buffer`  CALLED by cl_only.py:326. The leading underscore
                       says "internal to the class", and it is not: it is
                       internal to the class *hierarchy*, and a subclass in
                       another module is a caller like any other.
  `_adapt`             the documented seam for "a subclass that adapts on a
                       schedule rather than on a detection", per its own
                       docstring.

Nothing in this directory's other test files requires any of them. The
tests that do exercise some of this surface live with the entry points
(`tests/test_run_records.py`, `tests/test_entry_points.py`), one layer up
from the class that provides it -- so the monitor module's own tests can
pass in full while the contract its subclasses depend on is unasserted. An
implementation of the loop that keeps its bookkeeping under different names,
or aggregates inline instead of in a named method, satisfies every other
test here and breaks both side runners.

The aggregation table is pinned with it. `_aggregate_buffer` recognises
"mean", "median" and "last" and falls back to the mean for anything else
without warning (claims/LOG.md, A20), so a misspelled `aggregation` is
silently a mean -- which is worth asserting precisely because it is
surprising: the fallback is the current behaviour, and a reconstruction
that raised instead would be a change in behaviour, not a fix.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from apeiron.driver.continuous_monitor import ContinuousMonitor


@pytest.fixture(autouse=True)
def _patch_logger():
    mock_logger = MagicMock()
    mock_logger.step = 0
    with patch(
        "apeiron.driver.continuous_monitor.get_logger", return_value=mock_logger
    ):
        with patch(
            "apeiron.training.continuous_trainer.get_logger", return_value=mock_logger
        ):
            yield mock_logger


def _monitor(cfg, harness) -> ContinuousMonitor:
    return ContinuousMonitor(cfg=cfg, modelHarness=harness)


class TestSubclassState:
    """The attributes the side runners read."""

    def test_label_exists_and_is_a_string(self, default_cfg, dummy_harness):
        mon = _monitor(default_cfg, dummy_harness)
        assert isinstance(mon.label, str)
        assert mon.label

    def test_label_is_assignable(self, default_cfg, dummy_harness):
        """Both subclasses overwrite it in their own __init__."""
        mon = _monitor(default_cfg, dummy_harness)
        mon.label = "Drift-Only Monitoring"
        assert mon.label == "Drift-Only Monitoring"

    def test_metric_idx_comes_from_the_config(self, default_cfg, dummy_harness):
        """`f"metric_{self.metric_idx}"` names a CSV column, so the value
        has to follow `drift_detection.metric_index` rather than be
        re-derived or hard-coded."""
        for index in (0, 1, 3):
            cfg = replace(
                default_cfg,
                drift_detection=replace(
                    default_cfg.drift_detection, metric_index=index
                ),
            )
            assert _monitor(cfg, dummy_harness).metric_idx == index

    def test_last_agg_metric_starts_unset(self, default_cfg, dummy_harness):
        mon = _monitor(default_cfg, dummy_harness)
        assert mon.last_agg_metric is None

    def test_aggregation_comes_from_the_config(self, default_cfg, dummy_harness):
        for name in ("mean", "median", "last"):
            cfg = replace(
                default_cfg,
                drift_detection=replace(
                    default_cfg.drift_detection, aggregation=name
                ),
            )
            assert _monitor(cfg, dummy_harness).aggregation == name


class TestAggregateBufferSeam:
    """`cl_only.py` calls this by name, so the name is part of the contract."""

    def test_is_callable(self, default_cfg, dummy_harness):
        assert callable(getattr(_monitor(default_cfg, dummy_harness),
                                "_aggregate_buffer", None))

    @pytest.mark.parametrize(
        ("aggregation", "expected"),
        [
            ("mean", 2.75),
            ("median", 2.5),
            ("last", 2.0),
            # Not in the table, so the mean -- A20's silent fallback, and
            # the current behaviour rather than a desirable one.
            ("mien", 2.75),
            ("MEAN", 2.75),
            ("average", 2.75),
            ("max", 2.75),
            ("", 2.75),
        ],
    )
    def test_reduces_the_buffer_per_the_table(
        self, default_cfg, dummy_harness, aggregation, expected
    ):
        cfg = replace(
            default_cfg,
            drift_detection=replace(
                default_cfg.drift_detection, aggregation=aggregation
            ),
        )
        mon = _monitor(cfg, dummy_harness)
        mon.metric_buffer = [[1.0, 10.0], [5.0, 50.0], [3.0, 30.0], [2.0, 20.0]]
        assert mon._aggregate_buffer() == pytest.approx(expected)

    def test_reads_the_configured_metric_column(self, default_cfg, dummy_harness):
        """The second column, not the first, when metric_index says so."""
        cfg = replace(
            default_cfg,
            drift_detection=replace(default_cfg.drift_detection, metric_index=1),
        )
        mon = _monitor(cfg, dummy_harness)
        mon.metric_buffer = [[1.0, 10.0], [5.0, 50.0], [3.0, 30.0], [2.0, 20.0]]
        assert mon._aggregate_buffer() == pytest.approx(27.5)

    def test_clears_the_buffer(self, default_cfg, dummy_harness):
        """The next window has to start empty or every check re-counts the
        batches the previous one already saw."""
        mon = _monitor(default_cfg, dummy_harness)
        mon.metric_buffer = [[1.0], [2.0]]
        mon._aggregate_buffer()
        assert mon.metric_buffer == []

    def test_refuses_an_empty_buffer(self, default_cfg, dummy_harness):
        mon = _monitor(default_cfg, dummy_harness)
        mon.metric_buffer = []
        with pytest.raises(RuntimeError):
            mon._aggregate_buffer()

    def test_sets_last_agg_metric_on_a_check(self, default_cfg, dummy_harness):
        """`drift_only.py` logs `self.last_agg_metric` after a check, so the
        check has to have recorded it."""
        mon = _monitor(default_cfg, dummy_harness)
        mon.metric_buffer = [[1.0, 10.0], [5.0, 50.0], [3.0, 30.0], [2.0, 20.0]]
        mon._check_drift()
        assert mon.last_agg_metric == pytest.approx(2.75)


class TestAdaptSeam:
    """`_adapt(event_id)` is documented as the scheduled-adaptation seam."""

    def test_is_callable(self, default_cfg, dummy_harness):
        assert callable(getattr(_monitor(default_cfg, dummy_harness),
                                "_adapt", None))

    def test_dispatches_the_outer_cl_loop_with_the_event_id(
        self, default_cfg, dummy_harness
    ):
        mon = _monitor(default_cfg, dummy_harness)
        with patch.object(mon, "trainer") as trainer:
            mon._adapt(7)
        trainer.outer_cl_training_loop.assert_called_once_with(drift_event_id=7)
