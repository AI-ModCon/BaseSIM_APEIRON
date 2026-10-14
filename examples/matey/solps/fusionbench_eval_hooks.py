"""FusionBench-compatible MATEY eval hooks (leadtime patch)."""

from __future__ import annotations

from typing import Any


def patch_leadtime(valid_dataset: Any, leadtime: int) -> None:
    """Force fixed leadtime on every dataset __getitem__ (FusionBench runtime)."""
    lt = int(leadtime)
    for sub in valid_dataset.sub_dsets:
        sub.leadtime_max = max(int(getattr(sub, "leadtime_max", 1)), lt)
        orig = sub.__getitem__

        def getitem(index, fixed_lt=lt, orig_fn=orig):
            if isinstance(index, (list, tuple)) and len(index) == 2:
                return orig_fn((index[0], fixed_lt))
            if isinstance(index, int):
                return orig_fn((index, fixed_lt))
            return orig_fn(index)

        sub.__getitem__ = getitem
