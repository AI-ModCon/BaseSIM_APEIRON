"""Distributed (multi-node / multi-GPU) support.

All coordination goes through the module-level :data:`comm` singleton, which is a
no-op in a single-process run. See :mod:`apeiron.distributed.comm`.
"""

from apeiron.distributed.comm import DistContext, DistInfo, comm

__all__ = ["comm", "DistContext", "DistInfo"]
