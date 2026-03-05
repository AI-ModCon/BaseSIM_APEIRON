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
        if hist_batch is None:
            return super().fwd_bwd(batch, hist_batch)
        x_curr, y_curr = batch
        x_mem, y_mem = hist_batch

        # - Compute deltax direction
        deltax = (
            self.jvp_deltax_norm
            * (x_mem - x_curr)
            / (torch.linalg.norm(x_mem) + torch.linalg.norm(x_curr))
        )

        # - Compute combined gradients
        assert self._params is not None
        grad_dict, loss_curr, loss_mem = self._compute_jvp_gradients(
            self._params, x_curr, y_curr, x_mem, y_mem, deltax
        )

        self.grad_dict = (
            {k: self.grad_dict[k] + grad_dict[k] for k in grad_dict}
            if self.grad_dict is not None
            else grad_dict
        )

        self.loss_mem += loss_mem.item()
        return loss_curr.item()

    def _compute_jvp_gradients(
        self,
        params: OrderedDict[str, torch.nn.Parameter],
        x_curr: torch.Tensor,
        y_curr: torch.Tensor,
        x_mem: torch.Tensor,
        y_mem: torch.Tensor,
        deltax: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Compute JVP-regularized gradients combining current, memory, and JVP terms."""
        for p in params.values():
            p.requires_grad_(True)

        # - Define loss function for functional API
        def loss_fn(p, x, y):
            pred = functional_call(self.model, p, (x,))
            return self.criterion(pred, y) / self.cfg.train.grad_accumulation_steps

        # - Compute gradients
        grad_fn_curr = grad(loss_fn, argnums=0)
        grad_fn_mem = grad(loss_fn, argnums=0)

        # - Current task gradient
        grad_curr = grad_fn_curr(params, x_curr, y_curr)

        # - Memory task gradient
        grad_mem = grad_fn_mem(params, x_mem, y_mem)

        # - JVP computation
        def f(p, x):
            return loss_fn(p, x, y_mem)

        def jvp_func(p, tangents):
            return jvp(f, (p, x_mem), tangents)[1]

        # - Use current gradient as tangent direction
        tangents = OrderedDict((k, grad_curr[k]) for k in params)

        grad_jvp = grad(jvp_func)(params, (tangents, deltax))

        # - Combine gradients
        combined_grads = {
            k: grad_curr[k] + grad_mem[k] + self.jvp_lambda * grad_jvp[k]
            for k in params
        }

        # - Compute loss values
        with torch.no_grad():
            loss_curr = loss_fn(params, x_curr, y_curr)
            loss_mem = loss_fn(params, x_mem, y_mem)

        return combined_grads, loss_curr, loss_mem
