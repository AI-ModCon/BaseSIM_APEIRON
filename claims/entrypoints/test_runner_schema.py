"""The columns the two side runners write, which make them comparable.

A drift-only run and a scheduled run exist to be compared against a detector
run, which requires the same column names. `decision_idx` and `trigger_count`
on a scheduled run's decision row were named by no test (claims/LOG.md, A32);
the `cperf_` prefix is pinned for the monitor itself in
tests/test_logged_schema.py and checked here for both runners.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from apeiron.config.configuration import DriftDetectionCfg

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cl_only  # noqa: E402


def _rows(obj) -> list[dict]:
    captured: list[dict] = []
    obj.logger.log = lambda metrics, *a, **kw: captured.append(dict(metrics))
    return captured


class TestSharedProfilerPrefix:
    @pytest.mark.parametrize("entry", ["drift_only", "cl_only"])
    def test_both_side_runners_use_the_same_prefix(self, entry):
        # the point of a drift-only or scheduled run is comparability with a
        # detector run, which requires the same column names
        module = importlib.import_module(entry)
        source = Path(module.__file__).read_text()
        assert 'f"cperf_{k}"' in source


class TestScheduledDecisionRow:
    """A scheduled run's decision row, which places it on the cost frontier."""

    def _runner(self, default_cfg, dummy_harness, schedule):
        cfg = replace(default_cfg, drift_detection=DriftDetectionCfg())
        with patch(
            "apeiron.driver.continuous_monitor.get_logger", return_value=MagicMock()
        ):
            return cl_only.ScheduledCLRunner(
                cfg=cfg, modelHarness=dummy_harness, schedule=schedule
            )

    def _decide(self, runner, fired, decision_idx=0):
        rows = _rows(runner)
        runner.decision_count = decision_idx
        runner.metric_buffer = [[1.0]]
        runner._log_decision(decision_idx, 1.0, fired)
        return rows[-1]

    def test_the_decision_index_is_recorded(self, default_cfg, dummy_harness):
        runner = self._runner(default_cfg, dummy_harness, cl_only.NeverSchedule())
        row = self._decide(runner, fired=False, decision_idx=4)
        assert row["decision_idx"] == 4

    def test_the_schedule_s_verdict_lands_in_the_detected_column(
        self, default_cfg, dummy_harness
    ):
        runner = self._runner(default_cfg, dummy_harness, cl_only.NeverSchedule())
        # the same column a detector run uses, carrying the schedule's choice
        assert self._decide(runner, fired=True)["detected"] == 1
        assert self._decide(runner, fired=False)["detected"] == 0

    def test_the_trigger_count_includes_this_decision(self, default_cfg, dummy_harness):
        runner = self._runner(default_cfg, dummy_harness, cl_only.NeverSchedule())
        runner.trigger_count = 2
        # subtle and worth pinning: the row reports the count AFTER this
        # decision, so a fired row reads 3 while the attribute is still 2
        assert self._decide(runner, fired=True)["trigger_count"] == 3
        assert self._decide(runner, fired=False)["trigger_count"] == 2

    def test_the_score_is_held_at_zero_not_faked(self, default_cfg, dummy_harness):
        runner = self._runner(default_cfg, dummy_harness, cl_only.NeverSchedule())
        row = self._decide(runner, fired=True)
        # there is no drift statistic in a scheduled run; a placeholder
        # severity would be indistinguishable from a real one in analysis
        assert row["score"] == 0.0
        assert row["regime"] == "scheduled"
        assert row["confidence"] == "N/A"
