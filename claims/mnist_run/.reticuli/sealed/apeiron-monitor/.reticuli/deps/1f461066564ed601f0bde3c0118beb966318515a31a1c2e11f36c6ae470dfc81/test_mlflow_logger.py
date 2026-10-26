"""Tests for the MLflow metrics backend.

`MLFlowLogger` mirrors `WandBLogger`'s interface but differs in three ways
that matter and were unpinned: it prefixes with dots rather than slashes,
it logs against the PER-STAGE step so sparse global steps cannot downsample
away rare events, and it converts booleans to 0/1 instead of casting them
through float. The mlflow module itself is a recording fake — the logger
imports it lazily into `_mlflow`, so no server or tracking store is needed.
"""

from __future__ import annotations

from typing import Any

import pytest

from apeiron.logger.mlflow_logger import MLFlowLogger


class _FakeRunInfo:
    experiment_id = "exp-1"
    run_id = "run-9"


class _FakeRun:
    info = _FakeRunInfo()


class _FakeMlflow:
    """Records every call the logger makes instead of talking to a store."""

    def __init__(self, tracking_uri: str = "https://mlflow.example.org"):
        self.experiments: list[str] = []
        self.params: dict[str, Any] = {}
        self.metrics: list[tuple[dict[str, Any], int | None]] = []
        self.tags: dict[str, Any] = {}
        self.artifacts: list[str] = []
        self.started: list[dict[str, Any]] = []
        self.ended = 0
        self.system_metrics = 0
        self._tracking_uri = tracking_uri

    def set_experiment(self, name):
        self.experiments.append(name)

    def enable_system_metrics_logging(self):
        self.system_metrics += 1

    def start_run(self, run_name=None, tags=None):
        self.started.append({"run_name": run_name, "tags": tags})
        return _FakeRun()

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics, step=None):
        self.metrics.append((dict(metrics), step))

    def set_tag(self, key, value):
        self.tags[key] = value

    def log_artifact(self, path):
        self.artifacts.append(path)

    def end_run(self):
        self.ended += 1

    def get_tracking_uri(self):
        return self._tracking_uri


def _logger(csv_path=None, enabled=True, tracking_uri="https://mlflow.example.org"):
    lg = MLFlowLogger(enabled=enabled, csv_path=csv_path)
    fake = _FakeMlflow(tracking_uri)
    lg._mlflow = fake  # the lazy import is already satisfied
    return lg, fake


class TestLazyImport:
    def test_mlflow_is_not_imported_at_construction(self):
        assert MLFlowLogger()._mlflow is None

    def test_get_mlflow_caches(self):
        pytest.importorskip("mlflow")
        lg = MLFlowLogger()
        first = lg._get_mlflow()
        assert first is lg._get_mlflow()


class TestStages:
    def test_valid_stage_is_recorded(self):
        lg, _ = _logger()
        lg.stage("drift")
        assert lg.current_stage == "drift"

    def test_invalid_stage_refuses(self):
        lg, _ = _logger()
        with pytest.raises(ValueError, match="Invalid stage"):
            lg.stage("nonsense")  # type: ignore[arg-type]

    def test_stage_steps_start_at_zero(self):
        lg, _ = _logger()
        for stage in ("eval", "drift", "cl"):
            assert lg.get_stage_step(stage) == 0  # type: ignore[arg-type]

    def test_unknown_stage_step_is_zero(self):
        lg, _ = _logger()
        assert lg.get_stage_step("absent") == 0  # type: ignore[arg-type]


class TestInit:
    def test_disabled_logger_starts_no_run(self, default_cfg):
        lg, fake = _logger(enabled=False)
        assert lg.init(default_cfg) is None
        assert fake.started == []

    def test_experiment_run_and_params(self, default_cfg):
        lg, fake = _logger()
        run = lg.init(default_cfg, project="my-project", name="run-a")
        assert run is not None and lg.run is run
        assert fake.experiments == ["my-project"]
        assert fake.started[0]["run_name"] == "run-a"
        assert fake.system_metrics == 1
        # the frozen config arrives flattened with dot-separated keys
        assert fake.params["seed"] == default_cfg.seed
        assert fake.params["train.batch_size"] == default_cfg.train.batch_size
        assert fake.params["drift_detection.detector_name"] == (
            default_cfg.drift_detection.detector_name
        )

    def test_group_and_tags_become_run_tags(self, default_cfg):
        lg, fake = _logger()
        lg.init(default_cfg, tags=["alpha", "beta"], group="sweep-3")
        tags = fake.started[0]["tags"]
        assert tags == {"group": "sweep-3", "user_tag_0": "alpha", "user_tag_1": "beta"}

    def test_no_tags_passes_none(self, default_cfg):
        lg, fake = _logger()
        lg.init(default_cfg)
        assert fake.started[0]["tags"] is None

    def test_notes_become_the_run_description(self, default_cfg):
        lg, fake = _logger()
        lg.init(default_cfg, notes="a sentence")
        assert fake.tags["mlflow.note.content"] == "a sentence"

    def test_long_param_values_are_truncated(self, default_cfg):
        from dataclasses import replace

        from apeiron.config.configuration import ModelCfg

        long_path = "x" * 800
        cfg = replace(
            default_cfg, model=ModelCfg(name="tiny", pretrained_path=long_path)
        )
        lg, fake = _logger()
        lg.init(cfg)
        # mlflow caps params at 500 characters per value
        assert len(fake.params["model.pretrained_path"]) == 500


class TestFlattenConfig:
    def test_nested_keys_are_dotted(self):
        lg, _ = _logger()
        flat = lg._flatten_config({"a": 1, "b": {"c": 2, "d": {"e": 3}}})
        assert flat == {"a": 1, "b.c": 2, "b.d.e": 3}

    def test_empty_dict(self):
        lg, _ = _logger()
        assert lg._flatten_config({}) == {}


class TestLog:
    def test_step_and_stage_step_increment(self):
        lg, _ = _logger()
        lg.stage("eval")
        lg.log({"loss": 1.0})
        lg.log({"loss": 0.5})
        assert lg.step == 2
        assert lg.get_stage_step("eval") == 2

    def test_increment_false_holds_the_counters(self):
        lg, _ = _logger()
        lg.stage("eval")
        lg.log({"loss": 1.0}, increment=False)
        assert lg.step == 0
        assert lg.get_stage_step("eval") == 0

    def test_prefix_uses_dots_not_slashes(self, tmp_path):
        lg, _ = _logger(csv_path=tmp_path / "m.csv")
        lg.stage("cl")
        lg.log({"loss": 1.0})
        assert "cl.loss" in lg.metrics_history[0]

    def test_slashed_keys_are_converted_to_dots(self, tmp_path):
        lg, _ = _logger(csv_path=tmp_path / "m.csv")
        lg.stage("cl")
        lg.log({"drift/detected": 1})
        assert "drift.detected" in lg.metrics_history[0]

    def test_dotted_keys_pass_through_unprefixed(self, tmp_path):
        lg, _ = _logger(csv_path=tmp_path / "m.csv")
        lg.stage("cl")
        lg.log({"already.named": 1})
        assert "already.named" in lg.metrics_history[0]
        assert "cl.already.named" not in lg.metrics_history[0]

    def test_no_stage_means_no_prefix(self, tmp_path):
        lg, _ = _logger(csv_path=tmp_path / "m.csv")
        lg.log({"loss": 1.0})
        assert "loss" in lg.metrics_history[0]

    def test_explicit_step_overrides_the_counter(self, tmp_path):
        lg, _ = _logger(csv_path=tmp_path / "m.csv")
        lg.log({"loss": 1.0}, step=99)
        assert lg.metrics_history[0]["step"] == 99

    def test_history_is_only_kept_when_a_csv_is_wanted(self):
        lg, _ = _logger(csv_path=None)
        lg.log({"loss": 1.0})
        assert lg.metrics_history == []

    def test_nothing_reaches_mlflow_before_init(self):
        lg, fake = _logger()
        lg.log({"loss": 1.0})
        assert fake.metrics == []

    def test_metrics_log_against_the_per_stage_step(self, default_cfg):
        # deliberately unlike wandb: a sparse global step would let mlflow
        # downsample away rare detected=1 events
        lg, fake = _logger()
        lg.init(default_cfg)
        lg.stage("drift")
        lg.log({"loss": 1.0})
        lg.log({"loss": 2.0})
        assert [step for _, step in fake.metrics] == [1, 2]

    def test_without_a_stage_the_global_step_is_used(self, default_cfg):
        lg, fake = _logger()
        lg.init(default_cfg)
        lg.log({"loss": 1.0})
        assert fake.metrics[0][1] == 1

    def test_booleans_become_ints_not_floats(self, default_cfg):
        lg, fake = _logger()
        lg.init(default_cfg)
        lg.stage("drift")
        lg.log({"detected": True, "quiet": False})
        logged = fake.metrics[0][0]
        assert logged["drift.detected"] == 1
        assert isinstance(logged["drift.detected"], int)
        assert logged["drift.quiet"] == 0

    def test_non_numeric_values_are_dropped(self, default_cfg):
        lg, fake = _logger()
        lg.init(default_cfg)
        lg.stage("eval")
        lg.log({"loss": 1.0, "note": "a string", "where": None})
        logged = fake.metrics[0][0]
        assert "eval.loss" in logged
        assert "eval.note" not in logged
        assert "eval.where" not in logged

    def test_the_bare_step_key_is_not_logged_as_a_metric(self, default_cfg):
        lg, fake = _logger()
        lg.init(default_cfg)
        lg.stage("eval")
        lg.log({"loss": 1.0})
        assert "step" not in fake.metrics[0][0]

    def test_an_all_dropped_payload_logs_nothing(self, default_cfg):
        lg, fake = _logger()
        lg.init(default_cfg)
        lg.log({"note": "text"})
        assert fake.metrics == []


class TestFrameAndCsv:
    def test_dataframe_is_none_when_empty(self):
        lg, _ = _logger()
        assert lg.to_dataframe() is None

    def test_dataframe_is_long_format(self, tmp_path):
        lg, _ = _logger(csv_path=tmp_path / "m.csv")
        lg.stage("eval")
        lg.log({"loss": 1.0})
        frame = lg.to_dataframe()
        assert frame is not None
        assert list(frame.columns) == ["step", "metric", "value"]
        assert "eval.loss" in set(frame["metric"])
        assert "step" not in set(frame["metric"])

    def test_to_csv_writes_and_creates_parents(self, tmp_path):
        target = tmp_path / "nested" / "deeper" / "m.csv"
        lg, _ = _logger(csv_path=target)
        lg.log({"loss": 1.0})
        assert lg.to_csv() == target
        assert target.is_file()

    def test_to_csv_argument_overrides_the_configured_path(self, tmp_path):
        lg, _ = _logger(csv_path=tmp_path / "configured.csv")
        lg.log({"loss": 1.0})
        other = tmp_path / "explicit.csv"
        assert lg.to_csv(other) == other
        assert other.is_file()
        assert not (tmp_path / "configured.csv").exists()

    def test_to_csv_without_any_path_returns_none(self, tmp_path):
        lg, _ = _logger(csv_path=tmp_path / "m.csv")
        lg.log({"loss": 1.0})
        lg.csv_path = None
        assert lg.to_csv() is None

    def test_to_csv_returns_none_when_empty(self, tmp_path):
        lg, _ = _logger(csv_path=tmp_path / "m.csv")
        assert lg.to_csv() is None


class TestFinish:
    def test_ends_the_run_and_clears_it(self, default_cfg):
        lg, fake = _logger()
        lg.init(default_cfg)
        assert lg.finish() is None
        assert fake.ended == 1
        assert lg.run is None

    def test_writes_the_csv_and_returns_its_path(self, default_cfg, tmp_path):
        target = tmp_path / "m.csv"
        lg, _ = _logger(csv_path=target)
        lg.init(default_cfg)
        lg.log({"loss": 1.0})
        assert lg.finish() == target
        assert target.is_file()

    def test_save_csv_false_skips_the_file(self, default_cfg, tmp_path):
        target = tmp_path / "m.csv"
        lg, _ = _logger(csv_path=target)
        lg.init(default_cfg)
        lg.log({"loss": 1.0})
        assert lg.finish(save_csv=False) is None
        assert not target.exists()

    def test_nonzero_exit_code_marks_the_run_failed(self, default_cfg):
        lg, fake = _logger()
        lg.init(default_cfg)
        lg.finish(exit_code=1)
        assert fake.tags["mlflow.runStatus"] == "FAILED"

    def test_zero_exit_code_leaves_the_status_alone(self, default_cfg):
        lg, fake = _logger()
        lg.init(default_cfg)
        lg.finish(exit_code=0)
        assert "mlflow.runStatus" not in fake.tags

    def test_finish_without_a_run_is_harmless(self):
        lg, fake = _logger()
        assert lg.finish() is None
        assert fake.ended == 0


class TestSaveAndUrl:
    def test_save_logs_an_artifact(self, default_cfg, tmp_path):
        lg, fake = _logger()
        lg.init(default_cfg)
        f = tmp_path / "plot.png"
        f.write_bytes(b"")
        lg.save(f)
        assert fake.artifacts == [str(f)]

    def test_save_before_init_does_nothing(self, tmp_path):
        lg, fake = _logger()
        lg.save(tmp_path / "plot.png")
        assert fake.artifacts == []

    def test_url_is_none_without_a_run(self):
        lg, _ = _logger()
        assert lg.url is None

    def test_url_points_at_the_run(self, default_cfg):
        lg, _ = _logger(tracking_uri="https://mlflow.example.org")
        lg.init(default_cfg)
        assert lg.url == "https://mlflow.example.org/#/experiments/exp-1/runs/run-9"

    @pytest.mark.parametrize("uri", ["file:///tmp/mlruns", "/tmp/mlruns"])
    def test_local_tracking_stores_have_no_url(self, default_cfg, uri):
        lg, _ = _logger(tracking_uri=uri)
        lg.init(default_cfg)
        assert lg.url is None
