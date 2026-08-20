"""On-disk data management for committed data windows.

A *window* is a committed, immutable partition of the online stream: the data
collected over some ``delta_t`` and closed at a specific point. Once committed a
window never changes, so every consumer -- the monitoring stream, the
adaptation stream, the historical/replay stream, and offline analysis -- can map
the same file read-only, and many nodes can read it in parallel with no
coordination.

This package provides:

* :class:`~apeiron.data.window_store.WindowStore` -- atomic commit of windows
  to disk and memmap-backed read handles (the "write datasets that will not be
  kept in memory to disk" capability).
* :class:`~apeiron.data.catalog.WindowCatalog` -- a small queryable index over
  window manifests (the "query datasets needed for historic load creation"
  capability), rebuildable from the store at any time.
* :class:`~apeiron.data.windowed_harness.WindowedHarness` -- a
  :class:`~apeiron.model.torch_model_harness.BaseModelHarness` whose data
  methods are implemented in terms of a ``WindowStore``.
"""

from apeiron.data.window_store import (
    WindowHandle,
    WindowManifest,
    WindowSplit,
    WindowStore,
)
from apeiron.data.catalog import WindowCatalog
from apeiron.data.windowed_harness import WindowedHarness

__all__ = [
    "WindowStore",
    "WindowHandle",
    "WindowManifest",
    "WindowSplit",
    "WindowCatalog",
    "WindowedHarness",
]
