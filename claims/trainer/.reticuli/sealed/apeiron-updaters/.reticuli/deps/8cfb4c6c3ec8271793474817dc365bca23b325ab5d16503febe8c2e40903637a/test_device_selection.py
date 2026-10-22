"""Device selection, with the hardware probes patched.\n\n`get_available_device` picks a device by probing for CUDA and MPS, so its\nbranches only run on a host that has that hardware. Patching the probes\ncovers all four outcomes anywhere."""

from __future__ import annotations

import os
from unittest.mock import patch

import torch

from apeiron.config.configuration import get_available_device


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
