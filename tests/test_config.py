"""Tests for src/config/configuration.py"""

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
    DriftDetectionCfg,
    ModelCfg,
    TrainCfg,
    VisualizationCfg,
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

    def test_visualization_cfg(self):
        viz = VisualizationCfg()
        assert viz.input == "output/output.csv"


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
