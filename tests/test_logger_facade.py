"""Tests for the Logger facade, its singleton, and backend configuration.

test_logger.py covers ConsoleLogger and WandBLogger directly; logger.py
itself — the facade every entry point builds, the `get_logger` singleton and
`configure_backend` — had no tests, so the backend selection and the
singleton's argument-ignoring behavior (a real trap: the entry points
comment on it) were unpinned.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from apeiron.config.configuration import Config, LoggingCfg
from apeiron.logger.logger import (
    Logger,
    configure_backend,
    get_logger,
    reset_logger,
)
from apeiron.logger.mlflow_logger import MLFlowLogger
from apeiron.logger.wandb_logger import WandBLogger


@pytest.fixture(autouse=True)
def _no_singleton_leak():
    reset_logger()
    yield
    reset_logger()


class TestBackendSelection:
    def test_none_backend_uses_a_disabled_wandb_logger(self):
        lg = Logger(backend="none")
        assert lg.backend == "none"
        assert isinstance(lg._metrics, WandBLogger)
        assert lg._metrics.enabled is False

    def test_wandb_backend_is_enabled(self):
        lg = Logger(backend="wandb")
        assert lg.backend == "wandb"
        assert isinstance(lg._metrics, WandBLogger)
        assert lg._metrics.enabled is True

    def test_mlflow_backend(self):
        lg = Logger(backend="mlflow")
        assert lg.backend == "mlflow"
        assert isinstance(lg._metrics, MLFlowLogger)

    def test_unknown_backend_falls_back_to_disabled(self):
        # anything that is not "wandb"/"mlflow" takes the "none" branch
        lg = Logger(backend="bogus")  # type: ignore[arg-type]
        assert lg._metrics.enabled is False


class TestFacadeDelegation:
    def test_metrics_calls_reach_the_backend(self, tmp_path):
        csv_path = tmp_path / "m.csv"
        lg = Logger(backend="none", csv_path=csv_path)
        lg.stage("eval")
        assert lg.current_stage == "eval"
        lg.log({"loss": 1.0})
        assert lg.step == 1
        assert lg.get_stage_step("eval") == 1
        frame = lg.to_dataframe()
        # long format: one row per (step, metric, value)
        assert frame is not None
        assert list(frame.columns) == ["step", "metric", "value"]
        assert "eval/loss" in set(frame["metric"])
        assert lg.finish() == csv_path
        assert csv_path.is_file()

    def test_console_verbosity_round_trips(self):
        lg = Logger(backend="none", verbosity="WARNING")
        assert lg.verbosity == "WARNING"
        lg.verbosity = "DEBUG"
        assert lg.verbosity == "DEBUG"

    def test_console_methods_do_not_raise(self):
        lg = Logger(backend="none", verbosity="DEBUG")
        for call in (lg.debug, lg.warning, lg.error, lg.critical):
            call("message")
        lg.info("tiered", 2)

    def test_url_is_none_without_a_run(self):
        assert Logger(backend="none").url is None


class TestSingleton:
    def test_get_logger_returns_one_instance(self):
        assert get_logger(backend="none") is get_logger(backend="none")

    def test_arguments_are_ignored_once_an_instance_exists(self):
        # the trap the entry points comment on: configure the singleton
        # before anything else logs, or the first caller's settings win
        first = get_logger(verbosity="WARNING", backend="none")
        second = get_logger(verbosity="DEBUG", backend="mlflow")
        assert second is first
        assert second.backend == "none"
        assert second.verbosity == "WARNING"

    def test_reset_allows_reconfiguration(self):
        first = get_logger(backend="none")
        reset_logger()
        second = get_logger(backend="mlflow")
        assert second is not first
        assert second.backend == "mlflow"


class TestConfigureBackend:
    def _cfg(self, logging_cfg: LoggingCfg | None, default_cfg: Config) -> Config:
        return replace(default_cfg, logging=logging_cfg)

    def test_no_config_defaults_to_wandb(self):
        assert configure_backend(None) == "wandb"

    def test_missing_logging_section_defaults_to_wandb(self, default_cfg):
        # worth knowing: a TOML with no [logging] section talks to wandb
        assert configure_backend(self._cfg(None, default_cfg)) == "wandb"

    @pytest.mark.parametrize("backend", ["wandb", "mlflow", "none"])
    def test_valid_backends_pass_through(self, backend, default_cfg):
        cfg = self._cfg(LoggingCfg(backend=backend), default_cfg)
        if backend == "mlflow":
            pytest.importorskip("mlflow")
        assert configure_backend(cfg) == backend

    def test_invalid_backend_raises(self, default_cfg):
        cfg = self._cfg(
            LoggingCfg(backend="bogus"),  # type: ignore[arg-type]
            default_cfg,
        )
        with pytest.raises(ValueError, match="Invalid logging backend"):
            configure_backend(cfg)

    def test_mlflow_tracking_uri_is_applied(self, default_cfg, tmp_path):
        mlflow = pytest.importorskip("mlflow")
        uri = (tmp_path / "mlruns").as_uri()
        cfg = self._cfg(
            LoggingCfg(backend="mlflow", mlflow_tracking_uri=uri), default_cfg
        )
        assert configure_backend(cfg) == "mlflow"
        assert mlflow.get_tracking_uri() == uri
