"""Subsequence-level scoring strategies for active curriculum learning."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from torch import nn


class SubsequenceScorer(ABC):
    """Score dataset samples by informativeness. Higher = more informative."""

    @abstractmethod
    def score(self, model: nn.Module, dataset: Dataset, device: str) -> Tensor:
        """Return a score per sample in *dataset*."""
        raise NotImplementedError


class ResidualScorer(SubsequenceScorer):
    """Score = MSE(pred, target) per sample. One forward pass over the dataset."""

    def __init__(self, batch_size: int = 64) -> None:
        self.batch_size = batch_size

    @torch.no_grad()
    def score(self, model: nn.Module, dataset: Dataset, device: str) -> Tensor:
        model.eval()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        scores: list[Tensor] = []
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            pred = model(inp)
            # Per-sample MSE: mean over all dims except batch
            mse = (pred - tgt).pow(2).flatten(1).mean(dim=1)
            scores.append(mse.cpu())
        return torch.cat(scores)


class UncertaintyScorer(SubsequenceScorer):
    """Score = std of predictions across N stochastic forward passes (MC dropout).

    Requires the model to contain dropout layers. Calls ``model.train()``
    to enable dropout, then runs N forward passes per sample.
    """

    def __init__(self, mc_samples: int = 5, batch_size: int = 64) -> None:
        self.mc_samples = mc_samples
        self.batch_size = batch_size

    @torch.no_grad()
    def score(self, model: nn.Module, dataset: Dataset, device: str) -> Tensor:
        model.train()  # enable dropout
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_scores: list[Tensor] = []
        for inp, tgt in loader:
            inp = inp.to(device)
            preds: list[Tensor] = []
            for _ in range(self.mc_samples):
                pred = model(inp)
                preds.append(pred.cpu())
            # Stack -> (mc_samples, batch, *spatial), std over mc dim
            stacked = torch.stack(preds, dim=0)
            # Per-sample uncertainty: mean std across output dims
            uncertainty = stacked.std(dim=0).flatten(1).mean(dim=1)
            all_scores.append(uncertainty)

        model.eval()
        return torch.cat(all_scores)


def build_scorer(strategy: str, mc_samples: int = 5) -> SubsequenceScorer | None:
    """Factory: return a scorer instance or ``None`` for ``"none"``."""
    if strategy == "none":
        return None
    if strategy == "residual":
        return ResidualScorer()
    if strategy == "uncertainty":
        return UncertaintyScorer(mc_samples=mc_samples)
    raise ValueError(f"Unknown scoring strategy: {strategy!r}")
