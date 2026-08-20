from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Callable, Tuple, List, Dict

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from apeiron.config.configuration import Config
from apeiron.distributed import comm
from apeiron.model.checkpoint import CheckpointStore
from apeiron.model.task_record import (
    EvalSetRef,
    InMemoryEvalSet,
    TaskRecord,
    TaskRecordStore,
)

if TYPE_CHECKING:
    from apeiron.data.window_store import WindowStore

MetricFn = Callable[[Tensor, Tensor], Any]
CriterionFn = Callable[[Tensor, Tensor], Tensor]


class BaseModelHarness(ABC):
    """
    Members
    -------
    self.model : nn.Module
    self.cfg   : Dict[str, Any]   # e.g., {"device": "cuda", ...}

    You must implement:
      - get_loader(self)    -> DataLoader | Iterable
      - get_criterion(self) -> CriterionFn
    """

    def __init__(self, cfg: Config, model: nn.Module):
        self.model = model
        self.cfg = cfg
        device = torch.device(self.cfg.device)
        self.model.to(device)

        self.eval_metrics: Dict[str, MetricFn] = {}

        # Provenance pointer for the window currently being streamed. Window-
        # backed harnesses set this; on-the-fly harnesses leave it None.
        self.current_window_id: Optional[str] = None
        # Metrics from the most recent CL round (post-CL accuracy, FWT, BWT, ...).
        # The trainer populates this; checkpoint retention/promotion rules read it.
        self.last_metrics: Dict[str, float] = {}

        # One TaskRecord per drift event, oldest first. Each holds R[i][i] and a
        # re-evaluable reference to that task's validation split (a pointer into
        # a committed window when available, else an in-memory copy).
        self._task_records: List[TaskRecord] = []
        self.max_task_records: int = 50
        # Monotonic id so spilled eval-set filenames never collide, even across
        # FIFO eviction.
        self._task_seq: int = 0

        # Set by window-backed subclasses; lets task records restore window refs.
        self._window_store: Optional["WindowStore"] = None

        # Durable stores, wired when a checkpoint path is configured. Task
        # records persist under <ckpts_path>/task_records so a crashed run's
        # forgetting history survives; checkpoints additionally need max_ckpts>0.
        self._records_store: Optional[TaskRecordStore] = None
        self._ckpt_store: Optional[CheckpointStore] = None
        if self.cfg.model.ckpts_path:
            base = Path(self.cfg.model.ckpts_path)
            self._records_store = TaskRecordStore(base / "task_records")
            if self.ckpts_enabled:
                self._ckpt_store = CheckpointStore(
                    base,
                    max_ckpts=self.cfg.model.max_ckpts,
                    retention=getattr(self.cfg.model, "ckpts_retention", "fifo"),
                    deploy_rule=getattr(self.cfg.model, "deploy_rule", ""),
                )

    @abstractmethod
    def get_optmizer(self) -> Optimizer:
        """
        Returns the optimizer object compatible with the trainable parameters
        supports parameter groups for, e.g., different learning rates
        """
        raise NotImplementedError

    # ----- subclass hooks -----

    @abstractmethod
    def update_data_stream(self) -> None:
        """
        Updates the data stream potentially leading to data drift
        """
        raise NotImplementedError

    @abstractmethod
    def get_stream_dataloader(self) -> DataLoader:
        """
        Returns a training and validation dataloader compatible with the model input
        that will be used for continual learning
        """
        raise NotImplementedError

    @abstractmethod
    def get_hist_dataloaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """
        Returns a training and validation dataloader with historical data (to measure drift) compatible with the model input
        If there is no historical data, return None
        """
        raise NotImplementedError

    @torch.no_grad()
    def get_train_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Returns a training and validation dataloader compatible with the model input
        that will be used to loop over for inference
        """
        raise NotImplementedError

    @abstractmethod
    def get_criterion(self) -> CriterionFn:
        """Return a loss function compatible with model output and dataloader labels"""
        raise NotImplementedError

    # ----- helpers -----
    def _unpack(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Returns the input and label from a batch
        Note: override for compatibility with subclassed dataloader

        :param batch: batch of data from dataloader
        :type batch: Tuple[Tensor, Tensor]

        :return: input and label of the batch
        :rtype: Tuple[Tensor, Tensor]
        """
        x, y = batch
        return x, y

    @staticmethod
    def _to_scalar(x: Tensor | float) -> float:
        if isinstance(x, torch.Tensor):
            return float(x.mean().item() if x.ndim > 0 else x.item())
        return float(x)

    @torch.no_grad()
    def _eval_loader(self, loader: DataLoader) -> List[float]:
        """Stream over batches; return mean(metric) over batches (order preserved)."""
        self.model.eval()
        sums = [0.0 for _ in self.eval_metrics]
        counts = [0 for _ in self.eval_metrics]

        for batch in loader:  # assumes iterable
            x, y = self._unpack(batch)
            x, y = x.to(self.cfg.device), y.to(self.cfg.device)

            # TODO: Add cuda amp support later. Needs config entry for amp
            # if self.cfg.amp:

            #     with torch.autocast(
            #         device_type=self.device.type,
            #         dtype=(
            #             torch.float16 if self.device.type == "cuda" else torch.bfloat16
            #         ),
            #     ):
            #         y_hat = self.model(x)
            # else:
            y_hat = self.model(x)

            batch_size = y.shape[0]
            for i, m in enumerate(self.eval_metrics.values()):
                metric_value = self._to_scalar(m(y_hat, y))
                # For metrics that return percentages (like accuracy), we need to
                # convert back to counts for proper averaging across variable batch sizes
                sums[i] += metric_value * batch_size
                counts[i] += batch_size

        if counts[0] == 0:
            raise RuntimeError("Empty loader: nothing to evaluate.")

        return [s / c for s, c in zip(sums, counts)]

    @torch.no_grad()
    def eval(self) -> List[float]:
        """Stream over batches; return mean(metric) over batches (order preserved)."""
        return self._eval_loader(self.get_train_dataloaders()[1])

    @torch.no_grad()
    def history_eval(self) -> Optional[List[float]]:
        """Stream over batches; return mean(metric) over batches (order preserved).

        Returns None if no historical data is available.
        """
        hist_loaders = self.get_hist_dataloaders()
        if hist_loaders is None or hist_loaders[1] is None:
            return None

        return self._eval_loader(hist_loaders[1])

    # ----- per-task evaluation (train-test matrix R) -----

    def _freeze_task_evalset(self, window_id: Optional[str]) -> EvalSetRef:
        """Build a re-evaluable reference to the current task's validation split.

        Default: copy the current validation split into memory -- what the
        harness has always done, because an on-the-fly harness drops its window
        tensors as the stream advances and a plain ``DataLoader`` reference would
        keep the whole window alive through a view.

        Window-backed harnesses override this to return a *pointer* into the
        committed window instead of copying (see
        :class:`~apeiron.model.task_record.WindowEvalSetRef`).
        """
        xs: List[Tensor] = []
        ys: List[Tensor] = []
        for batch in self.get_train_dataloaders()[1]:
            x, y = self._unpack(batch)
            xs.append(x.detach().cpu().clone())
            ys.append(y.detach().cpu().clone())
        return InMemoryEvalSet(torch.cat(xs), torch.cat(ys))

    def register_task(
        self, diagonal_metrics: List[float], window_id: Optional[str] = None
    ) -> None:
        """Record the task just finished so later events can measure forgetting.

        A *task* is one drift event: the window the detector fired on and the CL
        loop adapted to. This freezes a re-evaluable reference to that window's
        validation split and stores it alongside ``diagonal_metrics`` --
        ``R[i][i]``, the score on the window measured right after adapting to it.
        Both travel in the same record so eviction can never misalign a task's
        eval set from its diagonal.

        :param diagonal_metrics: ``eval()`` output for the current window, taken
            after the CL loop finished.
        :param window_id: Provenance of the task's data; defaults to the
            harness's current window (set by window-backed harnesses).
        """
        if window_id is None:
            window_id = self.current_window_id

        eval_ref = self._freeze_task_evalset(window_id)
        self._task_records.append(
            TaskRecord(
                event_id=self._task_seq,
                diagonal=list(diagonal_metrics),
                eval_ref=eval_ref,
                window_id=window_id,
            )
        )
        self._task_seq += 1

        # Cap retained tasks; BWT then averages over the surviving ones.
        while len(self._task_records) > self.max_task_records:
            self._task_records.pop(0)

        # Every rank keeps the in-memory record (so BWT agrees across ranks), but
        # only rank 0 writes the durable copy to avoid racing filesystem writes.
        if self._records_store is not None and comm.is_main:
            self._records_store.save(self._task_records)

    def load_task_records(self) -> int:
        """Restore persisted task records (for resuming a crashed run).

        Returns the number of records loaded. No-op (returns 0) when task-record
        persistence is not configured.
        """
        if self._records_store is None:
            return 0
        self._task_records = self._records_store.load(self._window_store)
        self._task_seq = 1 + max((r.event_id for r in self._task_records), default=-1)
        return len(self._task_records)

    @torch.no_grad()
    def eval_past_tasks(self) -> List[List[float]]:
        """Score the current model on every registered task's frozen eval set.

        Returns row ``T`` of the train-test matrix below the diagonal --
        ``[R[T][i] for i < T]``, oldest task first, index-aligned with
        :attr:`task_diagonals`. Empty until at least one task is registered.
        """
        bs = self.cfg.train.batch_size
        return [
            self._eval_loader(rec.eval_ref.loader(bs)) for rec in self._task_records
        ]

    @property
    def task_diagonals(self) -> List[List[float]]:
        """``R[i][i]`` per registered task, oldest first.

        Index-aligned with :meth:`eval_past_tasks`.
        """
        return [rec.diagonal for rec in self._task_records]

    @property
    def ckpts_enabled(self) -> bool:
        return self.cfg.model.max_ckpts > 0 and bool(self.cfg.model.ckpts_path)

    def save_ckpt(self, event: int) -> str:
        """Persist model state and apply the configured retention/promotion rules.

        The metrics gathered during the CL round (``self.last_metrics``: post-CL
        current/historical accuracy, FWT, BWT) ride along in a sidecar so
        retention (``[model] ckpts_retention``) and promotion (``deploy_rule``)
        can select by quality -- e.g. keep the best-historical-accuracy snapshot,
        not merely the newest. The default rule ``"fifo"`` keeps the newest
        ``max_ckpts`` by event, matching the prior behavior.
        """
        if self._ckpt_store is None:
            # Reachable only if a caller invokes save_ckpt without checkpointing
            # configured (ckpts_enabled is the guard everywhere in-tree).
            self._ckpt_store = CheckpointStore(
                self.cfg.model.ckpts_path,
                max_ckpts=self.cfg.model.max_ckpts,
                retention=getattr(self.cfg.model, "ckpts_retention", "fifo"),
                deploy_rule=getattr(self.cfg.model, "deploy_rule", ""),
            )
        return self._ckpt_store.save(
            self.model.state_dict(),
            event=event,
            metrics=dict(self.last_metrics),
            window_id=self.current_window_id,
        )
