from __future__ import annotations

import copy
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.utils.data import DataLoader

from config.configuration import Config
from model.torch_model_harness import BaseModelHarness


class BaseUpdater:
    """Base class for model updaters in continual learning.

    Defines the interface for gradient-based updates during training.
    Subclasses implement strategies like JVP or K-FAC regularization.

    Attributes:
        cfg: Configuration object.
        criterion: Loss function for training.
        model: Neural network model to update.
    """

    def __init__(self, cfg: Config, modelHarness: BaseModelHarness) -> None:
        """Initialize updater with config and model harness."""
        self.cfg = cfg
        self.criterion: Callable[..., torch.Tensor] = modelHarness.get_criterion()
        self.model: nn.Module = modelHarness.model

        # Anchor weights for importance weighting (shared across all updaters)
        self.theta_star: dict[str, torch.Tensor] = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        self.importance_weighting: bool = False
        self.importance_alpha: float = 1.0

    def _unreduced_criterion(
        self, outputs: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-sample loss (no reduction)."""
        crit = copy.copy(self.criterion)
        crit.reduction = "none"  # type: ignore[attr-defined]
        return crit(outputs, y)

    @torch.no_grad()
    def compute_sample_priorities(
        self, loader: DataLoader, device: str
    ) -> torch.Tensor:
        """Compute per-sample priority = (L_current - L_anchor)^alpha for prioritized sampling.

        Returns a 1-D tensor of priorities, one per sample in dataset order.
        """
        self.model.eval()
        anchor = {n: p.detach() for n, p in self.model.named_parameters()}
        anchor.update(self.theta_star)

        all_deltas: list[torch.Tensor] = []
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            cur_loss = self._unreduced_criterion(self.model(x), y)
            anchor_out = functional_call(self.model, anchor, (x,))
            anchor_loss = self._unreduced_criterion(anchor_out, y)
            delta = (cur_loss - anchor_loss).clamp(min=1e-8)
            all_deltas.append(delta.cpu())

        self.model.train()
        priorities = torch.cat(all_deltas)
        # Apply alpha exponent (alpha=1 → linear, <1 → flatter, >1 → sharper)
        return priorities.pow(self.importance_alpha)

    def fwd_bwd(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        hist_batch: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> float:
        """Perform forward and backward pass on current batch."""
        x, y = batch
        outputs = self.model(x)
        loss = self.criterion(outputs, y)

        loss = loss / self.cfg.train.grad_accumulation_steps
        loss.backward()

        return loss.item()

    @torch.no_grad()
    def cl_preprocessing(self) -> None:
        """Hook called before before the training loop starts"""
        pass

    @torch.no_grad()
    def cl_postprocessing(self) -> None:
        """Hook called after the training loop ends.

        Updates theta_star to current model parameters.
        """
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.theta_star:
                self.theta_star[n].copy_(p.detach())

    @torch.no_grad()
    def update_pre_fwd_bwd(self) -> None:
        """Hook called before gradient computation."""
        pass

    @torch.no_grad()
    def update_post_fwd_bwd(self) -> float:
        """Hook called after gradient computation, but before the optimizer. Returns regularization loss."""
        return 0.0

    @torch.no_grad()
    def update_post_optimizer_call(self) -> None:
        """Hook called after optimizer step to update internal state."""
        pass
