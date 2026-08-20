"""Tests for apeiron.distributed.comm (single-process behavior + device bind).

The real multi-process path is exercised by a torchrun gloo smoke
(tests/dist_smoke.py); here we assert the single-process no-op semantics that
keep the rest of the framework unchanged, and the launch-detection / device
binding logic (which needs no process group).
"""

from __future__ import annotations

import torch

from apeiron.config.configuration import get_available_device
from apeiron.distributed.comm import DistContext, _detect_launch


class TestSingleProcessFacade:
    def test_defaults(self):
        c = DistContext()
        assert c.rank == 0
        assert c.world_size == 1
        assert c.is_main is True
        assert c.is_distributed is False
        assert c.initialized is False
        assert c.shard() is None

    def test_collectives_are_identity(self):
        c = DistContext()
        c.init_from_env()  # single process: no group created
        assert c.all_gather_object({"a": 1}) == [{"a": 1}]
        assert c.broadcast_object("payload") == "payload"
        t = torch.tensor([2.0, 6.0])
        c.all_reduce_mean_(t)
        assert t.tolist() == [2.0, 6.0]  # unchanged
        c.barrier()  # no-op
        c.shutdown()  # no-op

    def test_init_ignores_world_size_one(self, monkeypatch):
        # A launcher that reports WORLD_SIZE=1 must not create a process group.
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        c = DistContext()
        info = c.init_from_env()
        assert info.world_size == 1
        assert info.initialized is False


class TestLaunchDetection:
    def test_none_without_env(self, monkeypatch):
        for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_PROCID", "SLURM_NTASKS"):
            monkeypatch.delenv(var, raising=False)
        assert _detect_launch() is None

    def test_torchrun_env(self, monkeypatch):
        monkeypatch.setenv("RANK", "3")
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("LOCAL_RANK", "3")
        assert _detect_launch() == (3, 8, 3)

    def test_slurm_env(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.setenv("SLURM_PROCID", "5")
        monkeypatch.setenv("SLURM_NTASKS", "16")
        monkeypatch.setenv("SLURM_LOCALID", "1")
        assert _detect_launch() == (5, 16, 1)

    def test_torchrun_precedence_over_slurm(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("SLURM_PROCID", "9")
        monkeypatch.setenv("SLURM_NTASKS", "9")
        assert _detect_launch() == (0, 2, 0)


class TestDeviceBinding:
    def test_distributed_launch_binds_local_rank(self, monkeypatch):
        # Under a >1 world size, device resolution must not run the nvidia-smi
        # picker; it binds by LOCAL_RANK (cuda) or falls back (cpu/mps here).
        monkeypatch.setenv("LOCAL_RANK", "2")
        monkeypatch.setenv("WORLD_SIZE", "4")
        called = {"smi": False}

        import apeiron.config.configuration as configuration

        def _fail_smi():
            called["smi"] = True
            return 0

        monkeypatch.setattr(configuration, "_select_best_gpu", _fail_smi)
        dev = get_available_device()
        assert called["smi"] is False  # picker bypassed under distributed launch
        if torch.cuda.is_available():
            assert dev == torch.device("cuda:2")
        else:
            assert dev.type in ("cpu", "mps")

    def test_single_process_unaffected(self, monkeypatch):
        for var in ("LOCAL_RANK", "WORLD_SIZE", "SLURM_LOCALID", "SLURM_NTASKS"):
            monkeypatch.delenv(var, raising=False)
        # Should not raise and should return a valid device.
        dev = get_available_device()
        assert dev.type in ("cpu", "mps", "cuda")
