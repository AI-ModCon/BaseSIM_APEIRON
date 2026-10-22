"""The column names a run writes, which downstream tooling reads.

Auditing every quantity the framework logs against the suite turned up three
nobody named: the `cperf_` prefix on profiler fields, and `decision_idx` and
`trigger_count` on a scheduled run's decision row (claims/LOG.md, A32).

The prefix is load-bearing rather than cosmetic. `docs/profiler.md` documents
`cperf_*` as the profiler namespace, and the determinism work filters it out
of a byte-for-byte CSV comparison — wall-clock measurements legitimately
differ between runs, so they are excluded by matching the prefix. Rename it
and that comparison silently starts comparing timings.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from apeiron.config.configuration import DriftDetectionCfg
from apeiron.driver.continuous_monitor import ContinuousMonitor
from apeiron.profilers import FLOPSProfiler


def _rows(obj) -> list[dict]:
    captured: list[dict] = []
    obj.logger.log = lambda metrics, *a, **kw: captured.append(dict(metrics))
    return captured


class TestProfilerFieldPrefix:
    """Profiler measurements reach the CSV under `cperf_`, from every runner."""

    def _measured_monitor(self, default_cfg, dummy_harness):
        cfg = replace(default_cfg, drift_detection=DriftDetectionCfg())
        mon = ContinuousMonitor(cfg=cfg, modelHarness=dummy_harness)
        # a profiler with something to report
        mon.flops_profiler = FLOPSProfiler(warmup_iters=0)
        mon.flops_profiler.profiles = {"detector": {"flop": [10.0], "time": [0.5]}}
        return mon

    def test_the_monitor_prefixes_its_profiler_fields(self, default_cfg, dummy_harness):
        mon = self._measured_monitor(default_cfg, dummy_harness)
        rows = _rows(mon)
        mon.metric_buffer = [[1.0]]
        mon._check_drift()

        row = rows[-1]
        profiler_fields = {k for k in row if k.startswith("cperf_")}
        assert profiler_fields, "no profiler field reached the row"
        # every reported performance key arrives, prefixed and nothing else
        reported = set(mon.flops_profiler.get_performance())
        assert profiler_fields == {f"cperf_{k}" for k in reported}

    def test_no_profiler_field_escapes_the_prefix(self, default_cfg, dummy_harness):
        mon = self._measured_monitor(default_cfg, dummy_harness)
        rows = _rows(mon)
        mon.metric_buffer = [[1.0]]
        mon._check_drift()

        # anything the profiler reports must be namespaced: a bare key would
        # be indistinguishable from a model metric during analysis, and would
        # survive a filter that drops wall-clock rows by prefix
        for key in mon.flops_profiler.get_performance():
            assert key not in rows[-1]

    def test_an_idle_profiler_contributes_no_columns(self, default_cfg, dummy_harness):
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        rows = _rows(mon)
        mon.metric_buffer = [[1.0]]
        mon._check_drift()
        assert not [k for k in rows[-1] if k.startswith("cperf_")]

    @pytest.mark.parametrize("entry", ["drift_only", "cl_only"])
    def test_both_side_runners_use_the_same_prefix(self, entry):
        # the point of a drift-only or scheduled run is comparability with a
        # detector run, which requires the same column names
        import importlib
        import sys
        from pathlib import Path

        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        module = importlib.import_module(entry)
        source = Path(module.__file__).read_text()
        assert 'f"cperf_{k}"' in source


class TestScheduledDecisionRow:
    """A scheduled run's decision row, which places it on the cost frontier."""

    def _runner(self, default_cfg, dummy_harness, schedule):
        import sys
        from pathlib import Path

        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        import cl_only

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
        import cl_only

        runner = self._runner(default_cfg, dummy_harness, cl_only.NeverSchedule())
        row = self._decide(runner, fired=False, decision_idx=4)
        assert row["decision_idx"] == 4

    def test_the_schedule_s_verdict_lands_in_the_detected_column(
        self, default_cfg, dummy_harness
    ):
        import cl_only

        runner = self._runner(default_cfg, dummy_harness, cl_only.NeverSchedule())
        # the same column a detector run uses, carrying the schedule's choice
        assert self._decide(runner, fired=True)["detected"] == 1
        assert self._decide(runner, fired=False)["detected"] == 0

    def test_the_trigger_count_includes_this_decision(self, default_cfg, dummy_harness):
        import cl_only

        runner = self._runner(default_cfg, dummy_harness, cl_only.NeverSchedule())
        runner.trigger_count = 2
        # subtle and worth pinning: the row reports the count AFTER this
        # decision, so a fired row reads 3 while the attribute is still 2
        assert self._decide(runner, fired=True)["trigger_count"] == 3
        assert self._decide(runner, fired=False)["trigger_count"] == 2

    def test_the_score_is_held_at_zero_not_faked(self, default_cfg, dummy_harness):
        import cl_only

        runner = self._runner(default_cfg, dummy_harness, cl_only.NeverSchedule())
        row = self._decide(runner, fired=True)
        # there is no drift statistic in a scheduled run; a placeholder
        # severity would be indistinguishable from a real one in analysis
        assert row["score"] == 0.0
        assert row["regime"] == "scheduled"
        assert row["confidence"] == "N/A"
