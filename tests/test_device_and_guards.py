"""Device selection and the last few guard clauses.

`get_available_device` picks a device by probing hardware, so its branches
only run on the host that has that hardware; patching the probes covers all
four outcomes anywhere. The rest are guards reachable only by driving a
component out of order.
"""

from __future__ import annotations

import os
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from apeiron.config.configuration import (
    ContinualLearningCfg,
    DriftDetectionCfg,
    get_available_device,
)
from apeiron.drift_detection.load_drift_detector import load_drift_detector
from apeiron.training.updater.ewc import OnlineEWCUpdater
from apeiron.training.updater.kfac import OnlineKFACUpdater


class TestDeviceSelection:
    def test_multi_gpu_takes_the_whole_device(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert get_available_device(multi_gpu=True) == torch.device("cuda")

    def test_single_gpu_restricts_to_the_best_card(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        with (
            patch("apeiron.config.configuration._select_best_gpu", return_value=2),
            patch("torch.cuda.is_available", return_value=True),
        ):
            device = get_available_device()
        # the best card is pinned before CUDA initializes, so only cuda:0 is left
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
        assert device == torch.device("cuda:0")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def test_an_existing_visible_devices_setting_is_respected(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
        with (
            patch(
                "apeiron.config.configuration._select_best_gpu", return_value=2
            ) as chooser,
            patch("torch.cuda.is_available", return_value=True),
        ):
            get_available_device()
        chooser.assert_not_called()
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"

    def test_mps_is_used_when_cuda_is_absent(self):
        with (
            patch("apeiron.config.configuration._select_best_gpu", return_value=None),
            patch("torch.cuda.is_available", return_value=False),
            patch.object(torch.backends, "mps", create=True) as mps,
        ):
            mps.is_available.return_value = True
            assert get_available_device() == torch.device("mps")

    def test_a_raising_mps_probe_falls_back_to_cpu(self):
        with (
            patch("apeiron.config.configuration._select_best_gpu", return_value=None),
            patch("torch.cuda.is_available", return_value=False),
            patch.object(torch.backends, "mps", create=True) as mps,
        ):
            mps.is_available.side_effect = RuntimeError("no metal here")
            assert get_available_device() == torch.device("cpu")


class TestEnsembleValidation:
    def _cfg(self, default_cfg, **dd):
        return replace(default_cfg, drift_detection=DriftDetectionCfg(**dd))

    def test_an_ensemble_cannot_contain_itself(self, default_cfg):
        cfg = self._cfg(
            default_cfg,
            detector_name="EnsembleDetector",
            ensemble_detectors=["ADWINDetector", "EnsembleDetector"],
        )
        with pytest.raises(ValueError, match="cannot be nested inside itself"):
            load_drift_detector(cfg)


class TestOutOfOrderCommits:
    """Stepping an updater that was never prepared must not commit garbage."""

    def _cfg(self, default_cfg, mode):
        return replace(
            default_cfg, continual_learning=ContinualLearningCfg(update_mode=mode)
        )

    def test_ewc_steps_without_preparation_commit_nothing(
        self, make_harness, default_cfg
    ):
        cfg = self._cfg(default_cfg, "ewc_online")
        upd = OnlineEWCUpdater(cfg, make_harness(cfg))
        upd.update_post_optimizer_call()  # a step without cl_preprocessing
        assert upd._cl_steps == 1
        upd.cl_postprocessing()
        assert all(float(f.abs().sum()) == 0.0 for f in upd.fisher.values())

    def test_kfac_steps_without_preparation_commit_nothing(
        self, make_harness, default_cfg
    ):
        cfg = self._cfg(default_cfg, "kfac_online")
        upd = OnlineKFACUpdater(cfg, make_harness(cfg))
        upd.update_post_optimizer_call()
        assert upd._cl_steps == 1
        upd.cl_postprocessing()
        assert all(float(a.abs().sum()) == 0.0 for a in upd.A.values())
