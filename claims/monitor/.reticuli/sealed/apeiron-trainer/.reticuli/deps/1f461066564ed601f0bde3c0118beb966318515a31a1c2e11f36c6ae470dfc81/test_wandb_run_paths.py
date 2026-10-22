"""The WandBLogger paths that only run once a run is attached.

test_logger.py exercises the logger without a run, which skips everything
guarded by `if self.enabled and self.run` — init, the log call, artifact
saving, finish, and the URL. The `wandb` module is patched in the logger's
own namespace, so nothing contacts Weights & Biases.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from apeiron.logger import wandb_logger as module
from apeiron.logger.wandb_logger import WandBLogger


class _FakeRun:
    url = "https://wandb.example.org/entity/project/runs/abc"

    def __init__(self):
        self.metric_definitions: list[tuple[str, str | None]] = []

    def define_metric(self, name, step_metric=None):
        self.metric_definitions.append((name, step_metric))


_DEFAULT_RUN = object()


class _FakeWandb:
    def __init__(self, run=_DEFAULT_RUN):
        # `run=None` must mean "wandb declined to create one", not "default"
        self.run_to_return = _FakeRun() if run is _DEFAULT_RUN else run
        self.init_kwargs: dict[str, Any] = {}
        self.logged: list[tuple[dict[str, Any], bool]] = []
        self.saved: list[dict[str, Any]] = []
        self.finished: list[int | None] = []

    # wandb.Settings(silent=True) is constructed inside init
    def Settings(self, **kwargs):  # noqa: N802 - mirrors the wandb API
        return kwargs

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run_to_return

    def log(self, metrics, commit=True):
        self.logged.append((dict(metrics), commit))

    def save(self, path, base_path=None, policy="now"):
        self.saved.append({"path": path, "base_path": base_path, "policy": policy})

    def finish(self, exit_code=None):
        self.finished.append(exit_code)


@pytest.fixture()
def fake_wandb():
    fake = _FakeWandb()
    with patch.object(module, "wandb", fake):
        yield fake


class TestInit:
    def test_disabled_logger_never_calls_wandb(self, fake_wandb, default_cfg):
        lg = WandBLogger(enabled=False)
        assert lg.init(default_cfg) is None
        assert fake_wandb.init_kwargs == {}

    def test_init_passes_the_config_and_metadata(self, fake_wandb, default_cfg):
        lg = WandBLogger()
        run = lg.init(
            default_cfg,
            project="proj",
            name="run-a",
            tags=["t1"],
            notes="a note",
            group="g",
        )
        assert run is lg.run
        kwargs = fake_wandb.init_kwargs
        assert kwargs["project"] == "proj"
        assert kwargs["name"] == "run-a"
        assert kwargs["tags"] == ["t1"]
        assert kwargs["notes"] == "a note"
        assert kwargs["group"] == "g"
        # the frozen dataclass arrives as a plain dict
        assert kwargs["config"]["seed"] == default_cfg.seed
        assert kwargs["config"]["train"]["batch_size"] == default_cfg.train.batch_size
        assert kwargs["settings"] == {"silent": True}

    def test_extra_kwargs_are_forwarded(self, fake_wandb, default_cfg):
        WandBLogger().init(default_cfg, entity="lbnl")
        assert fake_wandb.init_kwargs["entity"] == "lbnl"

    def test_step_becomes_the_x_axis_for_every_metric(self, fake_wandb, default_cfg):
        lg = WandBLogger()
        run = lg.init(default_cfg)
        assert ("step", None) in run.metric_definitions
        assert ("*", "step") in run.metric_definitions

    def test_a_run_wandb_declines_to_create_is_tolerated(self, default_cfg):
        fake = _FakeWandb(run=None)
        # wandb.init can return None; define_metric must not be called on it
        with patch.object(module, "wandb", fake):
            lg = WandBLogger()
            assert lg.init(default_cfg) is None


class TestLogWithARun:
    def test_metrics_reach_wandb_with_the_commit_flag(self, fake_wandb, default_cfg):
        lg = WandBLogger()
        lg.init(default_cfg)
        lg.stage("eval")
        lg.log({"loss": 1.0})
        metrics, commit = fake_wandb.logged[0]
        assert commit is True
        assert metrics["eval/loss"] == 1.0
        assert metrics["step"] == 1

    def test_commit_false_is_forwarded(self, fake_wandb, default_cfg):
        lg = WandBLogger()
        lg.init(default_cfg)
        lg.log({"loss": 1.0}, commit=False)
        assert fake_wandb.logged[0][1] is False

    def test_non_numeric_values_survive_unlike_mlflow(self, fake_wandb, default_cfg):
        # wandb accepts strings; the mlflow backend has to drop them
        lg = WandBLogger()
        lg.init(default_cfg)
        lg.stage("drift")
        lg.log({"regime": "stable"})
        assert fake_wandb.logged[0][0]["drift/regime"] == "stable"


class TestSaveWithARun:
    def test_save_forwards_the_path_and_policy(self, fake_wandb, default_cfg, tmp_path):
        lg = WandBLogger()
        lg.init(default_cfg)
        f = tmp_path / "plot.png"
        lg.save(f, base_path=tmp_path, policy="end")
        assert fake_wandb.saved == [
            {"path": str(f), "base_path": str(tmp_path), "policy": "end"}
        ]

    def test_absent_base_path_is_passed_as_none(
        self, fake_wandb, default_cfg, tmp_path
    ):
        lg = WandBLogger()
        lg.init(default_cfg)
        lg.save(tmp_path / "plot.png")
        assert fake_wandb.saved[0]["base_path"] is None

    def test_save_without_a_run_is_a_no_op(self, fake_wandb, tmp_path):
        WandBLogger().save(tmp_path / "plot.png")
        assert fake_wandb.saved == []


class TestFinishWithARun:
    def test_finish_ends_the_run_and_clears_it(self, fake_wandb, default_cfg):
        lg = WandBLogger()
        lg.init(default_cfg)
        assert lg.finish() is None
        assert fake_wandb.finished == [None]
        assert lg.run is None

    def test_exit_code_is_forwarded(self, fake_wandb, default_cfg):
        lg = WandBLogger()
        lg.init(default_cfg)
        lg.finish(exit_code=1)
        assert fake_wandb.finished == [1]

    def test_finish_writes_the_csv_and_returns_it(
        self, fake_wandb, default_cfg, tmp_path
    ):
        target = tmp_path / "nested" / "m.csv"
        lg = WandBLogger(csv_path=target)
        lg.init(default_cfg)
        lg.log({"loss": 1.0})
        assert lg.finish() == target
        assert target.is_file()

    def test_save_csv_false_skips_the_file(self, fake_wandb, default_cfg, tmp_path):
        target = tmp_path / "m.csv"
        lg = WandBLogger(csv_path=target)
        lg.init(default_cfg)
        lg.log({"loss": 1.0})
        assert lg.finish(save_csv=False) is None
        assert not target.exists()

    def test_to_csv_without_any_path_returns_none(self, fake_wandb, tmp_path):
        lg = WandBLogger(csv_path=tmp_path / "m.csv")
        lg.log({"loss": 1.0})
        lg.csv_path = None
        assert lg.to_csv() is None


class TestUrl:
    def test_url_comes_from_the_run_property(self, fake_wandb, default_cfg):
        # wandb removed Run.get_url() in 0.30; the property is the supported
        # accessor across the whole supported range
        lg = WandBLogger()
        lg.init(default_cfg)
        assert lg.url == _FakeRun.url
