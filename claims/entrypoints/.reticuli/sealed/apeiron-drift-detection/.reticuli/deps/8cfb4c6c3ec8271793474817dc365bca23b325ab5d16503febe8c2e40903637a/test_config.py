"""Tests for src/apeiron/config/configuration.py"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from apeiron.config.configuration import (
    Config,
    ContinualLearningCfg,
    DataCfg,
    DriftDetectionCfg,
    LoggingCfg,
    ModelCfg,
    TrainCfg,
    build_config,
    deep_update,
    env_overrides,
    get_available_device,
    kv_to_nested,
    load_toml,
    parse_args,
    _select_best_gpu,
)


# ---------------------------------------------------------------------------
# deep_update
# ---------------------------------------------------------------------------
class TestDeepUpdate:
    def test_flat_merge(self):
        x = {"a": 1, "b": 2}
        y = {"b": 3, "c": 4}
        result = deep_update(x, y)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        x = {"a": {"x": 1, "y": 2}, "b": 10}
        y = {"a": {"y": 99, "z": 100}}
        result = deep_update(x, y)
        assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 10}

    def test_overwrite_non_mapping_with_mapping(self):
        x = {"a": 1}
        y = {"a": {"nested": True}}
        result = deep_update(x, y)
        assert result == {"a": {"nested": True}}

    def test_overwrite_mapping_with_non_mapping(self):
        x = {"a": {"nested": True}}
        y = {"a": 42}
        result = deep_update(x, y)
        assert result == {"a": 42}

    def test_empty_y(self):
        x = {"a": 1}
        result = deep_update(x, {})
        assert result == {"a": 1}

    def test_empty_x(self):
        x: dict = {}
        result = deep_update(x, {"a": 1})
        assert result == {"a": 1}


# ---------------------------------------------------------------------------
# kv_to_nested
# ---------------------------------------------------------------------------
class TestKvToNested:
    def test_simple_key_value(self):
        result = kv_to_nested(["model.name=resnet"])
        assert result == {"model": {"name": "resnet"}}

    def test_numeric_value(self):
        result = kv_to_nested(["train.batch_size=32"])
        assert result == {"train": {"batch_size": 32}}

    def test_bool_value(self):
        result = kv_to_nested(["multi_gpu=true"])
        assert result == {"multi_gpu": True}

    def test_json_list_value(self):
        result = kv_to_nested(['tags=["a","b"]'])
        assert result == {"tags": ["a", "b"]}

    def test_string_value(self):
        result = kv_to_nested(["data.path=/some/path"])
        assert result == {"data": {"path": "/some/path"}}

    def test_multiple_items(self):
        items = ["train.batch_size=16", "train.init_lr=0.001"]
        result = kv_to_nested(items)
        assert result == {"train": {"batch_size": 16, "init_lr": 0.001}}

    def test_deeply_nested(self):
        result = kv_to_nested(["a.b.c.d=1"])
        assert result == {"a": {"b": {"c": {"d": 1}}}}

    def test_empty_list(self):
        assert kv_to_nested([]) == {}


# ---------------------------------------------------------------------------
# env_overrides
# ---------------------------------------------------------------------------
class TestEnvOverrides:
    def test_picks_up_app_prefix(self):
        with patch.dict(os.environ, {"APP_SEED": "123"}, clear=False):
            result = env_overrides("APP_")
            assert result == {"seed": 123}

    def test_nested_env_var(self):
        with patch.dict(os.environ, {"APP_TRAIN.BATCH_SIZE": "64"}, clear=False):
            result = env_overrides("APP_")
            assert "train" in result
            assert result["train"]["batch_size"] == 64

    def test_ignores_non_prefixed(self):
        with patch.dict(os.environ, {"HOME": "/home/user"}, clear=False):
            result = env_overrides("APP_")
            assert "home" not in result and "HOME" not in result


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------
class TestParseArgs:
    def test_config_required(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_config_parsed(self):
        args = parse_args(["--config", "test.toml"])
        assert args.config == Path("test.toml")

    def test_set_values(self):
        args = parse_args(["--config", "x.toml", "--set", "a=1", "--set", "b=2"])
        assert args.set == ["a=1", "b=2"]

    def test_device_override(self):
        args = parse_args(["--config", "x.toml", "--device", "cpu"])
        assert args.device == "cpu"

    def test_multi_gpu_flag(self):
        args = parse_args(["--config", "x.toml", "--multi-gpu"])
        assert args.multi_gpu is True


# ---------------------------------------------------------------------------
# load_toml
# ---------------------------------------------------------------------------
class TestLoadToml:
    def test_loads_valid_toml(self, tmp_path: Path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            textwrap.dedent("""\
            seed = 42
            device = "cpu"
            multi_gpu = false

            [model]
            name = "tiny"
            pretrained_path = ""

            [data]
            name = "test"
            path = "/tmp"

            [train]
            batch_size = 8
            num_workers = 0
            init_lr = 0.01

            [drift_detection]
            detector_name = "ADWINDetector"
            """)
        )
        result = load_toml(toml_file)
        assert result["model"]["name"] == "tiny"
        assert result["train"]["batch_size"] == 8
        assert result["seed"] == 42


# ---------------------------------------------------------------------------
# build_config (integration-ish)
# ---------------------------------------------------------------------------
class TestBuildConfig:
    def test_full_build(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        toml_file = tmp_path / "cfg.toml"
        toml_file.write_text(
            textwrap.dedent("""\
            seed = 42
            device = "cpu"
            multi_gpu = false

            [model]
            name = "tiny"
            pretrained_path = ""

            [data]
            name = "test"
            path = "/tmp"

            [train]
            batch_size = 8
            num_workers = 0
            init_lr = 0.01

            [drift_detection]
            detector_name = "ADWINDetector"
            """)
        )
        cfg = build_config(["--config", str(toml_file)])
        assert isinstance(cfg, Config)
        assert cfg.model.name == "tiny"
        assert cfg.device == "cpu"
        assert cfg.seed == 42
        assert isinstance(cfg.drift_detection, DriftDetectionCfg)

    def test_cli_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        toml_file = tmp_path / "cfg.toml"
        toml_file.write_text(
            textwrap.dedent("""\
            seed = 42
            device = "cpu"
            multi_gpu = false

            [model]
            name = "tiny"
            pretrained_path = ""

            [data]
            name = "test"
            path = "/tmp"

            [train]
            batch_size = 8
            num_workers = 0
            init_lr = 0.01

            [drift_detection]
            detector_name = "ADWINDetector"
            """)
        )
        cfg = build_config(["--config", str(toml_file), "--set", "seed=99"])
        assert cfg.seed == 99


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------
class TestFrozenDataclasses:
    def test_model_cfg_frozen(self):
        m = ModelCfg(name="x", pretrained_path="y")
        with pytest.raises(AttributeError):
            m.name = "z"  # type: ignore[misc]

    def test_train_cfg_defaults(self):
        t = TrainCfg(batch_size=8, num_workers=0, init_lr=0.01)
        assert t.grad_accumulation_steps == 1
        assert t.max_iter == 600

    def test_continual_learning_defaults(self):
        cl = ContinualLearningCfg()
        assert cl.update_mode == "base"
        assert cl.ewc_lambda == 1000.0

    def test_drift_detection_defaults(self):
        dd = DriftDetectionCfg()
        assert dd.detector_name == "ADWINDetector"
        assert dd.detection_interval == 10


# ---------------------------------------------------------------------------
# get_available_device / _select_best_gpu
# ---------------------------------------------------------------------------
class TestDeviceSelection:
    def test_select_best_gpu_no_nvidia_smi(self):
        with patch(
            "apeiron.config.configuration.subprocess.check_output",
            side_effect=FileNotFoundError,
        ):
            assert _select_best_gpu() is None

    def test_select_best_gpu_with_output(self):
        fake_output = b"1000\n2000\n500\n"
        with patch(
            "apeiron.config.configuration.subprocess.check_output",
            return_value=fake_output,
        ):
            assert _select_best_gpu() == 1  # index of 2000

    def test_get_available_device_cpu_fallback(self):
        with (
            patch("apeiron.config.configuration._select_best_gpu", return_value=None),
            patch("torch.cuda.is_available", return_value=False),
            patch.object(torch.backends, "mps", create=True) as mock_mps,
        ):
            mock_mps.is_available.return_value = False
            device = get_available_device()
            assert device == torch.device("cpu")


# ---------------------------------------------------------------------------
# The full configuration surface: every field, every default
# ---------------------------------------------------------------------------
class TestConfigurationSurface:
    """Pin the complete field set and default values of every section.

    These are the contract the rest of the framework reads: the detector
    loader dereferences the adwin/kswin/ph fields, the updaters the jvp/ewc/
    kfac fields, the entry points `verbosity` and `logging`. An
    implementation that parses the test TOMLs but drops fields passed the
    suite before this class existed.
    """

    def test_field_sets_are_complete(self):
        from dataclasses import fields

        expected = {
            ModelCfg: {"name", "pretrained_path", "max_ckpts", "ckpts_path"},
            TrainCfg: {
                "batch_size",
                "num_workers",
                "init_lr",
                "grad_accumulation_steps",
                "max_iter",
            },
            DataCfg: {"name", "path", "batch_size"},
            ContinualLearningCfg: {
                "update_mode",
                "mix_historic_data",
                "jvp_rho_theta",
                "jvp_rho_x",
                "jvp_data_sign",
                "ewc_lambda",
                "ewc_ema_decay",
                "kfac_lambda",
                "kfac_ema_decay",
            },
            DriftDetectionCfg: {
                "detector_name",
                "detection_interval",
                "aggregation",
                "metric_index",
                "reset_after_learning",
                "max_stream_updates",
                "adwin_delta",
                "adwin_minor_threshold",
                "adwin_moderate_threshold",
                "kswin_alpha",
                "kswin_window_size",
                "kswin_stat_size",
                "kswin_seed",
                "ph_min_instances",
                "ph_delta",
                "ph_threshold",
                "ph_alpha",
                "ensemble_detectors",
                "ensemble_voting",
            },
            Config: {
                "model",
                "data",
                "train",
                "continual_learning",
                "drift_detection",
                "seed",
                "device",
                "multi_gpu",
                "verbosity",
                "logging",
            },
        }
        for cls, names in expected.items():
            assert {f.name for f in fields(cls)} == names, cls.__name__

    def test_model_cfg_defaults(self):
        m = ModelCfg(name="x")
        assert m.pretrained_path == ""
        assert m.max_ckpts == 0
        assert m.ckpts_path == ""

    def test_train_cfg_all_defaults(self):
        t = TrainCfg(batch_size=8, num_workers=0, init_lr=0.01)
        assert t.grad_accumulation_steps == 1
        assert t.max_iter == 600

    def test_data_cfg_stream_batch_default(self):
        d = DataCfg(name="x", path="/tmp")
        assert d.batch_size == 1

    def test_continual_learning_all_defaults(self):
        cl = ContinualLearningCfg()
        assert cl.update_mode == "base"
        assert cl.mix_historic_data is False
        assert cl.jvp_rho_theta == 0.05
        assert cl.jvp_rho_x == 1.0
        assert cl.jvp_data_sign == 1.0
        assert cl.ewc_lambda == 1000.0
        assert cl.ewc_ema_decay == 0.95
        assert cl.kfac_lambda == 0.01
        assert cl.kfac_ema_decay == 0.95

    def test_drift_detection_all_defaults(self):
        dd = DriftDetectionCfg()
        assert dd.detector_name == "ADWINDetector"
        assert dd.detection_interval == 10
        assert dd.aggregation == "mean"
        assert dd.metric_index == 0
        assert dd.reset_after_learning is False
        assert dd.max_stream_updates == 20
        assert dd.adwin_delta == 0.002
        assert dd.adwin_minor_threshold == 0.3
        assert dd.adwin_moderate_threshold == 0.6
        assert dd.kswin_alpha == 0.005
        assert dd.kswin_window_size == 100
        assert dd.kswin_stat_size == 30
        assert dd.kswin_seed is None
        assert dd.ph_min_instances == 30
        assert dd.ph_delta == 0.005
        assert dd.ph_threshold == 50
        assert dd.ph_alpha == 0.9999
        assert dd.ensemble_detectors == ()
        assert dd.ensemble_voting == "majority"

    def test_logging_cfg_defaults(self):
        lg = LoggingCfg()
        assert lg.backend == "none"
        assert lg.experiment_name is None
        assert lg.mlflow_tracking_uri is None
        assert lg.metrics_output_path is None

    def test_config_top_level_defaults(self):
        cfg = Config(
            model=ModelCfg(name="x"),
            data=DataCfg(name="d", path="/tmp"),
            train=TrainCfg(batch_size=8, num_workers=0, init_lr=0.01),
            continual_learning=ContinualLearningCfg(),
            drift_detection=DriftDetectionCfg(),
            seed=1,
            device="cpu",
        )
        assert cfg.multi_gpu is False
        assert cfg.verbosity == "INFO"
        assert cfg.logging is None

    def test_ensemble_detectors_coerce_to_tuple(self):
        # TOML arrays arrive as lists; a bare string is ONE detector name,
        # never an iterable of characters
        dd = DriftDetectionCfg(ensemble_detectors=["ADWINDetector", "KSWINDetector"])
        assert dd.ensemble_detectors == ("ADWINDetector", "KSWINDetector")
        dd = DriftDetectionCfg(ensemble_detectors="ADWINDetector")
        assert dd.ensemble_detectors == ("ADWINDetector",)
