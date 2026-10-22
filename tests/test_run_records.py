"""The two side runners' event records, summaries and module-level entries.

`drift_only` exists to produce a score trace a detector run can be compared
against, and `cl_only` to place a run on the accuracy-cost frontier. The
product of each is therefore its *record*, not its exit code — and none of it
was pinned, so a reconstruction satisfied the suite while dropping
`run_drift_only`, `run_manual_cl` and `drift_windows` wholesale
(claims/LOG.md, A28).

These tests pin that record: the per-event fields, the summary keys and their
semantics, the CSV field names the comparison depends on, and the two
module-level functions other code calls. They are also the prerequisite for
collapsing the duplicated monitoring loop into `ContinuousMonitor` — with the
record pinned, that refactor is checkable.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from apeiron.config.configuration import Config, build_config
from apeiron.evaluation.metrics import accuracy
from apeiron.logger import reset_logger
from apeiron.model.torch_model_harness import BaseModelHarness

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cl_only  # noqa: E402
import drift_only  # noqa: E402


class _Harness(BaseModelHarness):
    """A tiny stream that shifts every window, so drift fires repeatedly."""

    N, IN_F, NC = 24, 4, 3

    def __init__(self, cfg: Config):
        super().__init__(cfg, nn.Linear(self.IN_F, self.NC))
        self.eval_metrics = {"accuracy": accuracy, "loss": self.get_criterion()}
        self.higher_is_better = {"accuracy": True, "loss": False}
        self._window = 0
        self._train = self._make(0, 0.0)
        self._val = self._make(1, 0.0)

    def _make(self, seed: int, shift: float) -> TensorDataset:
        g = torch.Generator().manual_seed(seed)
        return TensorDataset(
            torch.randn(self.N, self.IN_F, generator=g) + shift,
            torch.randint(0, self.NC, (self.N,), generator=g),
        )

    def update_data_stream(self) -> None:
        self._val = self._make(100 + self._window, 3.0 * self._window)
        self._window += 1

    def get_stream_dataloader(self):
        return DataLoader(self._val, batch_size=self.cfg.data.batch_size)

    def get_train_dataloaders(self):
        bs = self.cfg.train.batch_size
        return DataLoader(self._train, batch_size=bs), DataLoader(
            self._val, batch_size=bs
        )

    def get_hist_dataloaders(self):
        return (None, None)

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self):
        return SGD(self.model.parameters(), lr=self.cfg.train.init_lr)


class _TimedHarness(_Harness):
    def update_data_stream(self) -> None:
        super().update_data_stream()
        self.current_window_timerange = (
            f"2026-01-{self._window:02d}",
            f"2026-01-{self._window + 1:02d}",
        )


@pytest.fixture(autouse=True)
def _fresh_logger():
    reset_logger()
    yield
    reset_logger()


@pytest.fixture()
def cfg(tmp_path) -> Config:
    toml = tmp_path / "cfg.toml"
    toml.write_text(
        textwrap.dedent(f"""\
        seed = 42
        device = "cpu"

        [model]
        name = "tiny"
        pretrained_path = ""

        [data]
        name = "test"
        path = "/tmp"
        batch_size = 4

        [train]
        batch_size = 8
        num_workers = 0
        init_lr = 0.01
        max_iter = 1

        [drift_detection]
        detector_name = "PageHinkleyDetector"
        detection_interval = 2
        metric_index = 0
        max_stream_updates = 3
        ph_min_instances = 2
        ph_delta = 0.0
        ph_threshold = 0.1

        [logging]
        backend = "none"
        metrics_output_path = "{tmp_path / "metrics.csv"}"
        """)
    )
    return build_config(["--config", str(toml)])


def _rows(logger) -> list[dict]:
    """Capture what the runner logs, in order.

    The returned list receives every row, eval rows included; `_drift_rows`
    narrows it to the drift-check trace.
    """
    captured: list[dict] = []
    logger.log = lambda metrics, *a, **kw: captured.append(dict(metrics))
    return captured


def _drift_rows(rows: list[dict]) -> list[dict]:
    """Only the drift-check rows — `check_idx` is unique to them."""
    return [row for row in rows if "check_idx" in row]


# ---------------------------------------------------------------------------
# drift_only: the event record
# ---------------------------------------------------------------------------
class TestDriftEventRecord:
    def test_every_event_carries_its_coordinates(self, cfg):
        summary = drift_only.run_drift_only(cfg=cfg, modelHarness=_Harness(cfg))
        assert summary["events"], "the stream must produce at least one event"

        for position, event in enumerate(summary["events"], start=1):
            # the event number is 1-based and dense: it IS the ordinal
            assert event["event"] == position
            # where in the run it happened, in three coordinates
            assert isinstance(event["batch"], int)
            assert isinstance(event["check"], int)
            assert isinstance(event["stream_idx"], int)
            assert 0 <= event["stream_idx"] < cfg.drift_detection.max_stream_updates
            # what the detector said
            assert isinstance(event["score"], float)
            assert isinstance(event["regime"], str)
            # the monitored metric, keyed by which metric was monitored
            assert f"metric_{cfg.drift_detection.metric_index}" in event

    def test_event_coordinates_never_go_backwards(self, cfg):
        summary = drift_only.run_drift_only(cfg=cfg, modelHarness=_Harness(cfg))
        for key in ("batch", "check", "stream_idx"):
            values = [e[key] for e in summary["events"]]
            assert values == sorted(values), f"{key} must be monotonic"

    def test_the_event_count_matches_the_record_length(self, cfg):
        summary = drift_only.run_drift_only(cfg=cfg, modelHarness=_Harness(cfg))
        assert summary["drift_events"] == len(summary["events"])

    def test_a_window_timerange_joins_the_event(self, cfg):
        summary = drift_only.run_drift_only(cfg=cfg, modelHarness=_TimedHarness(cfg))
        for event in summary["events"]:
            assert event["data_time_start"].startswith("2026-01-")
            assert event["data_time_end"].startswith("2026-01-")

    def test_no_timerange_leaves_the_fields_out(self, cfg):
        summary = drift_only.run_drift_only(cfg=cfg, modelHarness=_Harness(cfg))
        for event in summary["events"]:
            assert "data_time_start" not in event
            assert "data_time_end" not in event

    def test_drift_windows_deduplicates_and_sorts(self, cfg):
        monitor = drift_only.DriftOnlyMonitor(cfg=cfg, modelHarness=_Harness(cfg))
        monitor.events = [
            {"stream_idx": 2},
            {"stream_idx": 0},
            {"stream_idx": 2},
            {"stream_idx": 1},
        ]
        # coarser than the event list: repeated detections in one window
        # collapse, so this counts windows flagged, not how insistently
        assert monitor.drift_windows() == [0, 1, 2]

    def test_drift_windows_is_empty_without_events(self, cfg):
        monitor = drift_only.DriftOnlyMonitor(cfg=cfg, modelHarness=_Harness(cfg))
        assert monitor.drift_windows() == []

    def test_drift_windows_is_a_subset_of_the_event_windows(self, cfg):
        summary = drift_only.run_drift_only(cfg=cfg, modelHarness=_Harness(cfg))
        assert set(summary["drift_windows"]) == {
            e["stream_idx"] for e in summary["events"]
        }

    def test_the_returned_record_is_a_copy(self, cfg):
        monitor = drift_only.DriftOnlyMonitor(cfg=cfg, modelHarness=_Harness(cfg))
        summary = monitor.run()
        summary["events"].append({"event": 999})
        assert len(monitor.events) != len(summary["events"])


# ---------------------------------------------------------------------------
# drift_only: the summary contract
# ---------------------------------------------------------------------------
class TestDriftOnlySummary:
    EXPECTED = {
        "detector",
        "batches",
        "stream_updates",
        "drift_checks",
        "drift_events",
        "drift_windows",
        "events",
        "final_metrics",
    }

    def test_the_key_set_is_exact(self, cfg):
        summary = drift_only.run_drift_only(cfg=cfg, modelHarness=_Harness(cfg))
        assert set(summary) == self.EXPECTED

    def test_the_counters_describe_the_run(self, cfg):
        summary = drift_only.run_drift_only(cfg=cfg, modelHarness=_Harness(cfg))
        assert summary["detector"] == cfg.drift_detection.detector_name
        assert summary["stream_updates"] == cfg.drift_detection.max_stream_updates
        assert summary["batches"] > 0
        # one check per detection_interval batches, so checks trail batches
        assert 0 < summary["drift_checks"] <= summary["batches"]
        assert summary["drift_events"] <= summary["drift_checks"]

    def test_final_metrics_match_the_harness_metric_count(self, cfg):
        harness = _Harness(cfg)
        summary = drift_only.run_drift_only(cfg=cfg, modelHarness=harness)
        assert len(summary["final_metrics"]) == len(harness.eval_metrics)

    def test_the_model_is_never_adapted(self, cfg):
        harness = _Harness(cfg)
        before = [p.detach().clone() for p in harness.model.parameters()]
        drift_only.run_drift_only(cfg=cfg, modelHarness=harness)
        for p, b in zip(harness.model.parameters(), before):
            assert torch.equal(p.detach(), b)

    def test_run_drift_only_is_the_monitor_s_run(self, cfg):
        # the module-level entry other code calls; a reconstruction dropped it
        with patch.object(drift_only, "DriftOnlyMonitor") as built:
            built.return_value.run.return_value = {"sentinel": True}
            assert drift_only.run_drift_only(cfg=cfg, modelHarness=_Harness(cfg)) == {
                "sentinel": True
            }
        built.assert_called_once()


# ---------------------------------------------------------------------------
# drift_only: the logged trace, which is the point of the run
# ---------------------------------------------------------------------------
class TestDriftOnlyTrace:
    FIELDS = {"detected", "score", "regime", "confidence", "check_idx", "stream_idx"}

    def test_every_check_is_logged_not_sampled(self, cfg):
        monitor = drift_only.DriftOnlyMonitor(cfg=cfg, modelHarness=_Harness(cfg))
        rows = _rows(monitor.logger)
        summary = monitor.run()
        # a sampled trace cannot be used to re-tune a threshold offline, so
        # unlike ContinuousMonitor this logs detected=0 rows too
        checks = _drift_rows(rows)
        assert len(checks) == summary["drift_checks"]
        assert any(row["detected"] == 0 for row in checks)

    def test_the_schema_matches_the_detector_run(self, cfg):
        monitor = drift_only.DriftOnlyMonitor(cfg=cfg, modelHarness=_Harness(cfg))
        rows = _rows(monitor.logger)
        monitor.run()
        checks = _drift_rows(rows)
        assert checks
        for row in checks:
            assert self.FIELDS <= set(row)
            assert f"metric_{cfg.drift_detection.metric_index}" in row

    def test_check_and_stream_indices_advance(self, cfg):
        monitor = drift_only.DriftOnlyMonitor(cfg=cfg, modelHarness=_Harness(cfg))
        rows = _rows(monitor.logger)
        monitor.run()
        checks = _drift_rows(rows)
        assert [r["check_idx"] for r in checks] == sorted(
            r["check_idx"] for r in checks
        )
        assert checks[-1]["stream_idx"] >= checks[0]["stream_idx"]


# ---------------------------------------------------------------------------
# cl_only: the summary contract
# ---------------------------------------------------------------------------
class TestScheduledCLSummary:
    EXPECTED = {
        "schedule",
        "batches",
        "stream_updates",
        "decision_points",
        "triggers",
        "trigger_points",
        "final_metrics",
    }

    def _run(self, cfg, schedule):
        return cl_only.run_manual_cl(
            cfg=cfg, modelHarness=_Harness(cfg), schedule=schedule
        )

    def test_the_key_set_is_exact(self, cfg):
        assert set(self._run(cfg, cl_only.NeverSchedule())) == self.EXPECTED

    def test_a_never_schedule_triggers_nothing(self, cfg):
        summary = self._run(cfg, cl_only.NeverSchedule())
        assert summary["schedule"] == "never"
        assert summary["triggers"] == 0
        assert summary["trigger_points"] == []
        assert summary["decision_points"] == cfg.drift_detection.max_stream_updates

    def test_trigger_points_are_the_windows_that_fired(self, cfg):
        summary = self._run(cfg, cl_only.FixedSchedule([0, 2]))
        assert summary["trigger_points"] == [0, 2]
        assert summary["triggers"] == len(summary["trigger_points"])

    def test_a_periodic_schedule_fires_at_its_period(self, cfg):
        summary = self._run(cfg, cl_only.PeriodicSchedule(2))
        # decision points are 0..2 here, so period 2 fires once, at index 1
        assert summary["trigger_points"] == [1]
        assert summary["schedule"] == "periodic(every 2 window(s))"

    def test_one_decision_point_per_stream_window(self, cfg):
        summary = self._run(cfg, cl_only.NeverSchedule())
        assert summary["decision_points"] == summary["stream_updates"]

    def test_a_trigger_adapts_the_model(self, cfg):
        harness = _Harness(cfg)
        before = [p.detach().clone() for p in harness.model.parameters()]
        cl_only.run_manual_cl(
            cfg=cfg, modelHarness=harness, schedule=cl_only.FixedSchedule([0])
        )
        assert any(
            not torch.equal(p.detach(), b)
            for p, b in zip(harness.model.parameters(), before)
        ), "a fired trigger must run the CL loop"

    def test_run_manual_cl_is_the_runner_s_run(self, cfg):
        with patch.object(cl_only, "ScheduledCLRunner") as built:
            built.return_value.run.return_value = {"sentinel": True}
            assert self._run(cfg, cl_only.NeverSchedule()) == {"sentinel": True}
        built.assert_called_once()
