"""JVP (Jacobian-Vector Product) regularization updater for continual learning.

This module provides a JVP-regularized updater that extends the BaseUpdater
to mitigate catastrophic forgetting during continual learning.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch.func import functional_call, grad, jvp

from config.configuration import Config
from model.torch_model_harness import BaseModelHarness
from training.updater.base import BaseUpdater


class JVPRegUpdater(BaseUpdater):
    """JVP-regularized updater for continual learning.

    Combines gradients from current task, memory buffer, and JVP
    regularization term to prevent catastrophic forgetting.
    """

    def __init__(self, cfg: Config, modelHarness: BaseModelHarness) -> None:
        """Initialize JVP updater with config and model harness."""
        super().__init__(cfg, modelHarness)
        self.jvp_lambda: float = cfg.continual_learning.jvp_lambda
        self.jvp_deltax_norm: float = cfg.continual_learning.jvp_deltax_norm
        self._params: OrderedDict[str, torch.nn.Parameter] | None = None

        self.grad_dict: dict[str, torch.Tensor] | None = None
        self.loss_mem: float = 0.0

    def update_pre_fwd_bwd(self) -> None:
        """Cache parameters as dict for functional API."""
        if self._params is None:
            self._params = OrderedDict(self.model.named_parameters())

    def update_post_fwd_bwd(self) -> float:
        """Apply accumulated gradients and return memory loss."""
        if self.grad_dict is not None:
            for name, param in self.model.named_parameters():
                param.grad = self.grad_dict[name].detach()

        self.grad_dict = None
        loss_out = self.loss_mem
        self.loss_mem = 0.0
        return loss_out

    def fwd_bwd(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        hist_batch: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> float:
        """
        Args:
            current_batch: (input, target) for current task
            memory_batch: (input, target) for memory buffer

        Returns:
            grad_dict: Computed gradients for each parameter
            loss_curr: Loss on current task
            loss_mem: Loss on memory task
        """

        # Current gradients ###
        loss_cur = super().fwd_bwd(
            batch, hist_batch
        )  # fill the model gradients with the default grad info

        ### JVP gradients ###
        if self._params is not None and hist_batch is not None:
            self._compute_jvp_gradients(
                self._params, batch, hist_batch
            )  # adds jvp gradiens to existing gradients

        ### History gradients ####
        if hist_batch is None:
            return super().fwd_bwd(batch)
        x_mem, y_mem = hist_batch

        outputs_mem = self.model(x_mem)
        loss_mem = self.criterion(outputs_mem, y_mem)
        loss_mem = loss_mem / self.cfg.train.grad_accumulation_steps
        loss_mem.backward()  # adds the history gradients to the default gradients inplace.
        self.loss_mem += loss_mem.item()

        return loss_cur

    def _compute_jvp_gradients(
        self,
        params: OrderedDict[str, torch.nn.Parameter],
        batch: tuple[torch.Tensor, torch.Tensor],
        hist_batch: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Compute JVP-regularized gradients combining current, memory, and JVP terms."""

        # - Define loss function for functional API
        def f(p, x):
            pred = functional_call(self.model, (p,), (x,))
            return (
                self.criterion(pred, hist_batch[1])
                / self.cfg.train.grad_accumulation_steps
            )

        def jvp_func(p, tangents):
            return jvp(f, (p, hist_batch[0]), tangents)[1]

        # - Use current gradient as tangent direction
        tangents = OrderedDict(
            (name, param.grad.detach().clone())
            for name, param in params.items()
            if param.grad is not None
        )

        deltax = (
            self.jvp_deltax_norm
            * (hist_batch[0] - batch[0])
            / (torch.linalg.norm(hist_batch[0]) + torch.linalg.norm(batch[0]))
        )

        # - JVP computation
        grad_jvp = grad(jvp_func)(params, (tangents, deltax))

        # - Combine gradients
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                p.grad += self.jvp_lambda * grad_jvp[n]
            else:
                raise KeyError(
                    "param ", n, " has no grad, but JVP regularizer expected one."
                )
