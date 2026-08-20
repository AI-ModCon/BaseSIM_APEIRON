"""A model harness whose data streams are committed windows on disk.

``WindowedHarness`` implements every data method of
:class:`~apeiron.model.torch_model_harness.BaseModelHarness` in terms of a
:class:`~apeiron.data.window_store.WindowStore`, so the three streams the
framework consumes all resolve to memory-mapped partitions rather than
in-process tensors:

* **monitoring stream** -- a loader over the current window (default: the whole
  window) that the driver runs inference on to feed the detector;
* **adaptation stream** -- the current window's ``train``/``val`` splits;
* **historical stream** -- the concatenation of all prior committed windows'
  ``train``/``val`` splits (the replay buffer / forgetting probe), or ``None``
  before any history exists.

Because the windows are immutable, a task's frozen validation set is stored as a
*pointer* (``WindowEvalSetRef``) into the committed window instead of being
copied into RAM -- the memory win described in the design notes.

The model, loss, and optimizer are not the store's concern: pass them to the
constructor (or subclass and override ``get_criterion``/``get_optmizer``). Each
committed window becomes one stream window, so drive a run with
``drift_detection.max_stream_updates`` no larger than ``len(store)``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from apeiron.config.configuration import Config
from apeiron.data.window_store import WindowHandle, WindowStore
from apeiron.distributed import comm
from apeiron.model.task_record import EvalSetRef, WindowEvalSetRef
from apeiron.model.torch_model_harness import BaseModelHarness, CriterionFn


class WindowedHarness(BaseModelHarness):
    """Harness backed by a :class:`WindowStore` of committed windows."""

    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        store: Optional[WindowStore] = None,
        *,
        criterion: Optional[CriterionFn] = None,
        optimizer_factory: Optional[Callable[[nn.Module], Optimizer]] = None,
        eval_metrics: Optional[Dict[str, Any]] = None,
        stream_split: str = "all",
        train_split: str = "train",
        val_split: str = "val",
    ) -> None:
        super().__init__(cfg=cfg, model=model)

        if store is None:
            if not cfg.data.window_store_path:
                raise ValueError(
                    "WindowedHarness needs a store or cfg.data.window_store_path"
                )
            store = WindowStore(cfg.data.window_store_path)
        self.store = store
        # Let restored task records rebuild their window pointers.
        self._window_store = store

        self._criterion = criterion
        self._optimizer_factory = optimizer_factory
        if eval_metrics is not None:
            self.eval_metrics = eval_metrics

        self.stream_split = stream_split
        self.train_split = train_split
        self.val_split = val_split

        # When True, freeze task eval sets by copying into RAM (the pre-Phase-1
        # behavior) instead of pointing into the committed window. Only for the
        # memory A/B benchmark; leave False in real runs.
        self._copy_task_evalsets = False

        # Snapshot of committed windows in commit order; refreshed on advance so
        # a long-running producer's newly committed windows become visible.
        self._windows = self.store.list_manifests()
        self._cursor = -1
        self.stream_exhausted = False
        self.current_window_timerange: Optional[Tuple[Any, Any]] = None

    # -- window navigation -------------------------------------------------

    @property
    def n_windows(self) -> int:
        return len(self._windows)

    def _handle(self) -> WindowHandle:
        if self._cursor < 0:
            raise RuntimeError("call update_data_stream() before reading the stream")
        manifest = self._windows[self._cursor]
        return WindowHandle(self.store.root / manifest.window_id, manifest)

    def update_data_stream(self) -> None:
        """Advance to the next committed window.

        Re-scans the store first so windows committed after construction are
        picked up. When the store is exhausted the cursor stays on the last
        window and :attr:`stream_exhausted` is set, so an over-long run re-streams
        the final window instead of crashing.
        """
        self._windows = self.store.list_manifests()
        if self._cursor + 1 < len(self._windows):
            self._cursor += 1
        else:
            self.stream_exhausted = True
            if self._cursor < 0 and self._windows:
                self._cursor = 0
        if not self._windows:
            raise RuntimeError(f"window store {self.store.root} is empty")

        manifest = self._windows[self._cursor]
        self.current_window_id = manifest.window_id
        self.current_window_timerange = (
            (manifest.t_start, manifest.t_end)
            if manifest.t_start is not None or manifest.t_end is not None
            else None
        )

    # -- streams -----------------------------------------------------------

    def _pin(self) -> bool:
        return torch.cuda.is_available()

    def get_stream_dataloader(self) -> DataLoader:
        # Shard the monitoring stream: each rank runs inference on a contiguous
        # slice of the window, and the engine gathers the per-batch metrics. In a
        # single-process run comm.shard() is None -> the whole window, unchanged.
        return self._handle().loader(
            self.stream_split,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            shard=comm.shard(),
            pin_memory=self._pin(),
        )

    def get_train_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        handle = self._handle()
        bs = self.cfg.train.batch_size
        nw = self.cfg.train.num_workers
        # Train split is sharded (data-parallel updates with gradient all-reduce);
        # the val split is kept whole so every rank computes the same eval metrics.
        train = handle.loader(
            self.train_split,
            batch_size=bs,
            shuffle=True,
            num_workers=nw,
            shard=comm.shard(),
            pin_memory=self._pin(),
        )
        val = handle.loader(
            self.val_split,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=self._pin(),
        )
        return train, val

    def get_hist_dataloaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """Replay/forgetting loaders over every window before the current one."""
        prior = self._windows[: self._cursor]
        if not prior:
            return None, None

        # Shard the replay (train) stream for data-parallel updates; keep the
        # historical val set whole so forgetting metrics agree across ranks.
        shard = comm.shard()
        train_sets: list[Dataset] = []
        val_sets: list[Dataset] = []
        for manifest in prior:
            handle = WindowHandle(self.store.root / manifest.window_id, manifest)
            train_sets.append(handle.dataset(self.train_split, shard=shard))
            val_sets.append(handle.dataset(self.val_split))

        bs = self.cfg.train.batch_size
        nw = self.cfg.train.num_workers
        hist_train: DataLoader = DataLoader(
            ConcatDataset(train_sets),
            batch_size=bs,
            shuffle=True,
            num_workers=nw,
            pin_memory=self._pin(),
        )
        hist_val: DataLoader = DataLoader(
            ConcatDataset(val_sets),
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=self._pin(),
        )
        return hist_train, hist_val

    # -- task eval set: pointer, not copy ---------------------------------

    def _freeze_task_evalset(self, window_id: Optional[str]) -> EvalSetRef:
        """Reference the committed window's val split instead of copying it."""
        if (
            not self._copy_task_evalsets
            and window_id is not None
            and window_id in self.store
        ):
            return WindowEvalSetRef(self.store, window_id, self.val_split)
        return super()._freeze_task_evalset(window_id)

    # -- model / loss / optimizer -----------------------------------------

    def get_criterion(self) -> CriterionFn:
        if self._criterion is None:
            raise NotImplementedError(
                "pass criterion= to WindowedHarness or override get_criterion()"
            )
        return self._criterion

    def get_optmizer(self) -> Optimizer:
        if self._optimizer_factory is None:
            raise NotImplementedError(
                "pass optimizer_factory= to WindowedHarness or override get_optmizer()"
            )
        return self._optimizer_factory(self.model)
