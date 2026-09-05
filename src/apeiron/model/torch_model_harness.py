from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Callable, Tuple, List, Dict

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer

from apeiron.config.configuration import Config

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

        # One entry per drift event, oldest first: the frozen validation split of
        # the window that was adapted to, paired with R[i][i] (see register_task).
        self._task_records: List[Tuple[DataLoader, List[float]]] = []
        self.max_task_records: int = 50

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

    def register_task(self, diagonal_metrics: List[float]) -> None:
        """Record the task just finished so later events can measure forgetting.

        A *task* is one drift event: the window the detector fired on and the CL
        loop adapted to. This freezes that window's validation split into a
        standalone eval set and stores it alongside ``diagonal_metrics`` --
        ``R[i][i]``, the score on the window measured right after adapting to it.
        Both travel in the same record so eviction can never misalign a task's
        eval set from its diagonal.

        The split is copied into memory rather than referenced: the harness drops
        its window tensors as the stream advances, and a plain ``DataLoader``
        reference would keep the whole window alive through a view.

        :param diagonal_metrics: ``eval()`` output for the current window, taken
            after the CL loop finished.
        :type diagonal_metrics: List[float]
        """
        xs: List[Tensor] = []
        ys: List[Tensor] = []
        for batch in self.get_train_dataloaders()[1]:
            x, y = self._unpack(batch)
            xs.append(x.detach().cpu().clone())
            ys.append(y.detach().cpu().clone())

        frozen = DataLoader(
            TensorDataset(torch.cat(xs), torch.cat(ys)),
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
        )
        self._task_records.append((frozen, list(diagonal_metrics)))

        # Cap retained tasks; BWT then averages over the surviving ones.
        while len(self._task_records) > self.max_task_records:
            self._task_records.pop(0)

    @torch.no_grad()
    def eval_past_tasks(self) -> List[List[float]]:
        """Score the current model on every registered task's frozen eval set.

        Returns row ``T`` of the train-test matrix below the diagonal --
        ``[R[T][i] for i < T]``, oldest task first, index-aligned with
        :attr:`task_diagonals`. Empty until at least one task is registered.
        """
        return [self._eval_loader(loader) for loader, _ in self._task_records]

    @property
    def task_diagonals(self) -> List[List[float]]:
        """``R[i][i]`` per registered task, oldest first.

        Index-aligned with :meth:`eval_past_tasks`.
        """
        return [diagonal for _, diagonal in self._task_records]

    @property
    def ckpts_enabled(self) -> bool:
        return self.cfg.model.max_ckpts > 0 and bool(self.cfg.model.ckpts_path)

    def build_checkpoint_payload(self) -> Any:
        """Build the checkpoint object to save.

        Subclasses can override this to include additional metadata beyond weights
        (e.g., preprocessing scalers, feature names, architecture parameters)
        so that saved checkpoints match the format expected by the loader.

        Returns
        -------
        By default, returns ``model.state_dict()`` (weights only).
        """
        return self.model.state_dict()

    def save_ckpt(self, event: int) -> str:
        """Persist model state, evict oldest when over budget."""
        d = Path(self.cfg.model.ckpts_path)
        d.mkdir(parents=True, exist_ok=True)

        fname = f"drift_adaptation_{event}.pt"
        payload = self.build_checkpoint_payload()
        torch.save(payload, d / fname)
        (d / "latest").write_text(fname)

        # Guillotine the oldest survivors
        alive = sorted(d.glob("drift_adaptation_*.pt"), key=lambda p: p.stat().st_mtime)
        while len(alive) > self.cfg.model.max_ckpts:
            alive.pop(0).unlink()

        return str(d / fname)
