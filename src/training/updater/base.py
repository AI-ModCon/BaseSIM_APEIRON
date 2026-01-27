from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

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
        """Hook called after the training loop ends"""
        pass

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
