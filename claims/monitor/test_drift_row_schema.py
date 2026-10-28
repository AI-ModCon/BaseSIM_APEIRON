"""The metric names a run writes -- the metrics file's whole schema.

`test_logged_schema.py` pins the `cperf_` prefix on profiler fields.
Nothing pins the names themselves, or which stages a run logs under, and
those are the artefact: `[logging] metrics_output_path` is what every
downstream analysis opens.

What that cost. Nine blind rebuilds were assembled into a framework; it
imported, passed the pinned MNIST harness gate, and trained the entire run
without error. Then the run claim failed:

    AssertionError: drift fires at [], expected [350]

Drift had fired. The check selects `drift/score` from the CSV, exactly as
an analysis would, and the regrown monitor had named that column
`drift_score`. Measured as a unit, driving one monitor loop under each
implementation, the divergence is not one rename but the schema:

    only in the original   eval/accuracy, eval/mean_logit, drift/score,
                           drift/detected, drift/regime, drift/confidence,
                           and six drift/cperf_* fields
    only in the rebuild    drift/batch_count, drift/stream_update_count,
                           drift/drift_event_count, drift/drift_score,
                           drift/drift_detected

So the rebuild logs none of the per-batch evaluation metrics, none of the
profiler measurements, and no regime or confidence; it logs monitor
bookkeeping the original never writes, and it re-prefixes the two columns
it does share. A run against it completes, trains, and produces a file in
which no analysis can find the accuracy or the drift events.

This is the quietest failure in the campaign: every layer passed its gate,
the composition imported, training completed, the weights were computed --
and the artefact was unreadable by the tooling that consumes it. It is also
the reason a `grade 3` claim has to name the surface it covers: the probes
drove this layer to zero contract divergences on counters and weights, and
said nothing about what the run writes down.

Pinned here as the stage-qualified name set a loop logs, which is
implementation-agnostic -- it asserts the artefact, not the code that
writes it.

NOT pinned, filed instead: `detected` is also emitted on a 10% sample of
NON-events, chosen with the unseeded global numpy RNG
(`continuous_monitor.py`, `np.random.random() <= 0.1`). Whether that column
appears on a given non-event row therefore varies between runs of
identical code, so the schema itself is nondeterministic -- the A41 class,
and a decision for whoever owns the metrics. The assertions below require
`detected` only where drift actually fired.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from apeiron.config.configuration import DriftDetectionCfg
from apeiron.driver.continuous_monitor import ContinuousMonitor

#: Stage-qualified names a loop must write. The eval names are DERIVED
#: from the harness rather than listed: `eval_metrics` is the harness's to
#: define, and the first version of this file hard-coded the two names a
#: different fixture happened to have, so it failed against `src/`. A
#: criterion has to be measured with the fixture it will run under.
DRIFT_NAMES = {"drift/score", "drift/detected", "drift/regime",
               "drift/confidence"}

#: Names the original never writes. Monitor bookkeeping belongs to the
#: object, not to the measurement record.
FORBIDDEN = {"drift/drift_score", "drift/drift_detected",
             "drift/drift_regime", "drift/batch_count",
             "drift/stream_update_count", "drift/drift_event_count"}


def _logged_names(cfg, harness) -> set[str]:
    """Every stage-qualified metric name one monitor loop writes."""
    monitor = ContinuousMonitor(cfg=cfg, modelHarness=harness)
    seen: set[str] = set()
    stage = {"name": None}
    real_stage = monitor.logger.stage

    def stage_hook(name):
        stage["name"] = name
        return real_stage(name)

    monitor.logger.stage = stage_hook
    monitor.logger.log = lambda metrics, *a, **kw: seen.update(
        f"{stage['name']}/{key}" for key in metrics
    )
    monitor.run()
    return seen


@pytest.fixture()
def firing_cfg(default_cfg):
    """A loop short enough for a test and long enough to reach a check."""
    return replace(
        default_cfg,
        drift_detection=replace(DriftDetectionCfg(), max_stream_updates=3,
                                detection_interval=1),
    )


class TestTheRunWritesItsMeasurements:
    def test_the_evaluation_metrics_are_logged(self, firing_cfg, dummy_harness):
        """Per-batch evaluation is the reason the loop exists, and these are
        the columns the MNIST run claim reads for its final accuracy and
        its eval-point count. The rebuild logged none of them.

        Expected names come from the harness's own `eval_metrics`, so this
        holds for any harness rather than for one fixture's metric list."""
        expected = {f"eval/{name}" for name in dummy_harness.eval_metrics}
        assert expected, "the fixture harness declares no eval metrics"
        missing = sorted(expected - _logged_names(firing_cfg, dummy_harness))
        assert not missing, f"the run never logged {missing}"

    def test_the_drift_row_columns_are_logged(self, firing_cfg, dummy_harness):
        missing = sorted(DRIFT_NAMES - _logged_names(firing_cfg, dummy_harness))
        assert not missing, f"the drift row never logged {missing}"

    def test_the_profiler_fields_reach_the_record(self, firing_cfg, dummy_harness):
        """`cperf_`-namespaced, per `test_logged_schema.py`. The rebuild
        logged none of them, so the prefix test had nothing to check."""
        names = _logged_names(firing_cfg, dummy_harness)
        assert any(n.startswith("drift/cperf_") for n in names), (
            "no profiler measurement reached the record"
        )

    def test_the_metric_column_follows_the_configured_index(
        self, default_cfg, dummy_harness
    ):
        """The column name records WHICH metric was monitored, so an
        analysis joining runs can tell them apart."""
        for index in range(len(dummy_harness.eval_metrics)):
            cfg = replace(
                default_cfg,
                drift_detection=replace(DriftDetectionCfg(),
                                        max_stream_updates=3,
                                        detection_interval=1,
                                        metric_index=index),
            )
            assert f"drift/metric_{index}" in _logged_names(cfg, dummy_harness)


class TestItDoesNotRenameOrPadTheSchema:
    def test_no_column_is_re_prefixed_with_its_stage(self, firing_cfg,
                                                     dummy_harness):
        """The logger already supplies the `drift/` namespace from the
        stage, so `drift_score` becomes `drift/drift_score` -- which is
        what broke a full-stack run."""
        offenders = sorted(FORBIDDEN & _logged_names(firing_cfg, dummy_harness))
        assert not offenders, (
            f"the run writes {offenders}; the stage already names the "
            f"namespace, and an analysis selects on the original names"
        )

    def test_monitor_bookkeeping_is_not_a_measurement(self, firing_cfg,
                                                      dummy_harness):
        """Counters are state on the object. Writing them into the record
        makes them indistinguishable from model metrics during analysis."""
        names = _logged_names(firing_cfg, dummy_harness)
        for counter in ("batch_count", "stream_update_count",
                        "drift_event_count", "check_count"):
            assert f"drift/{counter}" not in names, (
                f"{counter} is monitor state, not a measurement"
            )
