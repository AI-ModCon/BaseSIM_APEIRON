"""A thin facade over ``torch.distributed`` with a single-process fallback.

Every distributed operation in apeiron goes through the module-level ``comm``
object. Its defining property: **when the job was not launched distributed
(``world_size == 1``), every collective is an identity/no-op**. That is what lets
the monitoring loop, the harness, and the trainer call ``comm.all_gather_object``
/ ``comm.broadcast_object`` / ``comm.all_reduce_mean_`` unconditionally while the
single-process code path stays byte-for-byte what it was before Phase 3.

Launch detection supports the two ways these jobs actually start:

* **torchrun** -- ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` (and ``MASTER_ADDR``
  / ``MASTER_PORT``) are already in the environment.
* **SLURM srun** -- ``SLURM_PROCID`` / ``SLURM_NTASKS`` / ``SLURM_LOCALID``; the
  rendezvous address is derived from ``SLURM_NODELIST`` if not set (Frontier /
  Perlmutter).

Backend is ``nccl`` when CUDA/ROCm is available (ROCm's RCCL is exposed as
``nccl``), else ``gloo`` (CPU -- used for local multi-process testing).
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional, TypeVar

import torch

T = TypeVar("T")


@dataclass(frozen=True)
class DistInfo:
    """Immutable snapshot of the process's place in the job."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = "none"
    initialized: bool = False


def _detect_launch() -> Optional[tuple[int, int, int]]:
    """Return ``(rank, world_size, local_rank)`` from the environment, or None.

    None means "not launched distributed" -> run as a single process.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:  # torchrun
        return (
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
            int(os.environ.get("LOCAL_RANK", 0)),
        )
    if "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:  # srun
        return (
            int(os.environ["SLURM_PROCID"]),
            int(os.environ["SLURM_NTASKS"]),
            int(os.environ.get("SLURM_LOCALID", 0)),
        )
    return None


def _slurm_master_addr() -> str:
    """First hostname of the SLURM allocation (rendezvous host), else localhost."""
    nodelist = os.environ.get("SLURM_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        return "127.0.0.1"
    try:
        out = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist], stderr=subprocess.DEVNULL
        )
        hosts = out.decode().split()
        if hosts:
            return hosts[0]
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return "127.0.0.1"


class DistContext:
    """Process-group facade. Use the module-level singleton :data:`comm`."""

    def __init__(self) -> None:
        self._info = DistInfo()
        # Opt-in cumulative wall-clock spent in each collective, for the scaling
        # benchmark's Amdahl breakdown. Zero overhead unless time_collectives set.
        self.time_collectives = False
        self._timings: dict[str, float] = {}

    def timings(self) -> dict[str, float]:
        """Cumulative seconds spent in each collective since the last reset."""
        return dict(self._timings)

    def reset_timings(self) -> None:
        self._timings = {}

    def _tick(self, name: str, seconds: float) -> None:
        self._timings[name] = self._timings.get(name, 0.0) + seconds

    # -- introspection -----------------------------------------------------

    @property
    def info(self) -> DistInfo:
        return self._info

    @property
    def rank(self) -> int:
        return self._info.rank

    @property
    def world_size(self) -> int:
        return self._info.world_size

    @property
    def local_rank(self) -> int:
        return self._info.local_rank

    @property
    def is_main(self) -> bool:
        """True on rank 0 (and always true single-process). Owns all writes."""
        return self._info.rank == 0

    @property
    def is_distributed(self) -> bool:
        """True only when there is more than one process to coordinate."""
        return self._info.world_size > 1

    @property
    def initialized(self) -> bool:
        return self._info.initialized

    def shard(self) -> Optional[tuple[int, int]]:
        """``(rank, world_size)`` for :class:`WindowHandle` sharding, or None."""
        return (self.rank, self.world_size) if self.is_distributed else None

    # -- lifecycle ---------------------------------------------------------

    def init_from_env(self) -> DistInfo:
        """Detect the launch and, if distributed, create the process group.

        Idempotent and safe to call unconditionally in every entry point: a
        non-distributed launch just records single-process info and touches
        nothing.
        """
        launch = _detect_launch()
        if launch is None or launch[1] <= 1:
            local = launch[2] if launch else 0
            self._info = DistInfo(rank=0, world_size=1, local_rank=local)
            return self._info

        rank, world, local = launch
        import torch.distributed as dist

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        os.environ.setdefault("MASTER_ADDR", _slurm_master_addr())
        os.environ.setdefault("MASTER_PORT", "29500")
        if torch.cuda.is_available():
            torch.cuda.set_device(local)

        if not dist.is_initialized():
            dist.init_process_group(backend=backend, rank=rank, world_size=world)

        self._info = DistInfo(
            rank=rank,
            world_size=world,
            local_rank=local,
            backend=backend,
            initialized=True,
        )
        return self._info

    def shutdown(self) -> None:
        """Destroy the process group (safe to call unconditionally)."""
        if self._info.initialized:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        self._info = DistInfo()

    # -- collectives (identity/no-op when single-process) ------------------

    def barrier(self) -> None:
        if self._active():
            import torch.distributed as dist

            dist.barrier()

    def all_gather_object(self, obj: T) -> list[T]:
        """Gather ``obj`` from every rank into a rank-indexed list.

        Single-process: returns ``[obj]``. Because ranks own contiguous shards,
        concatenating the returned list in order reconstructs global order.
        """
        if not self._active():
            return [obj]
        import torch.distributed as dist

        t0 = time.perf_counter() if self.time_collectives else 0.0
        out: list[Any] = [None] * self.world_size
        dist.all_gather_object(out, obj)
        if self.time_collectives:
            self._tick("all_gather", time.perf_counter() - t0)
        return out

    def broadcast_object(self, obj: Optional[T], src: int = 0) -> T:
        """Broadcast ``obj`` from ``src`` to every rank (identity single-process)."""
        if not self._active():
            return obj  # type: ignore[return-value]
        import torch.distributed as dist

        t0 = time.perf_counter() if self.time_collectives else 0.0
        box: list[Any] = [obj if self.rank == src else None]
        dist.broadcast_object_list(box, src=src)
        if self.time_collectives:
            self._tick("broadcast", time.perf_counter() - t0)
        return box[0]

    def all_reduce_mean_(self, tensor: torch.Tensor) -> torch.Tensor:
        """In-place average of ``tensor`` across ranks (identity single-process)."""
        if not self._active():
            return tensor
        import torch.distributed as dist

        t0 = time.perf_counter() if self.time_collectives else 0.0
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(self.world_size)
        if self.time_collectives:
            self._tick("all_reduce", time.perf_counter() - t0)
        return tensor

    def broadcast_module_(self, module: torch.nn.Module, src: int = 0) -> None:
        """In-place broadcast of a module's params + buffers from ``src``.

        Manual data-parallel training requires every rank to start from the
        identical model; this enforces it. No-op single-process.
        """
        if not self._active():
            return
        import torch.distributed as dist

        for p in module.parameters():
            dist.broadcast(p.data, src=src)
        for b in module.buffers():
            dist.broadcast(b.data, src=src)

    def _active(self) -> bool:
        return self.is_distributed and self._info.initialized


#: Module-level singleton used throughout apeiron.
comm = DistContext()
