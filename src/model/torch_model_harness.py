from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, Callable, Tuple, List

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from config.configuration import Config

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

        self.eval_metrics: List[MetricFn] = []

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
    def get_cur_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Returns a training and validation dataloader compatible with the model input
        """
        raise NotImplementedError

    @abstractmethod
    def get_hist_data_loaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """
        Returns a training and validation dataloader with historical data (to measure drift) compatible with the model input
        If there is no historical data, return None
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
    def eval(self) -> List[float]:
        """Stream over batches; return mean(metric) over batches (order preserved)."""
        self.model.eval()
        sums = [0.0 for _ in self.eval_metrics]
        counts = [0 for _ in self.eval_metrics]

        for batch in self.get_cur_data_loaders()[1]:  # assumes iterable
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

            batch_size = y.size(0)
            for i, m in enumerate(self.eval_metrics):
                metric_value = self._to_scalar(m(y_hat, y))
                # For metrics that return percentages (like accuracy), we need to
                # convert back to counts for proper averaging across variable batch sizes
                sums[i] += metric_value * batch_size
                counts[i] += batch_size

        if counts[0] == 0:
            raise RuntimeError("Empty loader: nothing to evaluate.")

        return [s / c for s, c in zip(sums, counts)]
