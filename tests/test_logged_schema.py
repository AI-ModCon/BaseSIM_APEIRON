"""The column names a run writes, which downstream tooling reads.

Auditing every quantity the framework logs against the suite turned up the
`cperf_` prefix on profiler fields, which nobody named (claims/LOG.md, A32).
The runners' own columns are pinned at their layer, in
tests/test_runner_schema.py.

The prefix is load-bearing rather than cosmetic. `docs/profiler.md` documents
`cperf_*` as the profiler namespace, and the determinism work filters it out
of a byte-for-byte CSV comparison — wall-clock measurements legitimately
differ between runs, so they are excluded by matching the prefix. Rename it
and that comparison silently starts comparing timings.
"""

from __future__ import annotations

from dataclasses import replace
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
