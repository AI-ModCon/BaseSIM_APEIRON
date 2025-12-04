"""JVP (Jacobian-Vector Product) regularization module for continual learning.

This module provides custom optimizer and loss implementations for training
neural networks with JVP regularization to mitigate catastrophic forgetting.

Refactored from src/training/updater/jvp_regularized.py for a cleaner
PyTorch-style implementation.

Comments:

- Optimizer now does per parameter step (Pytorch convention) instead of global step.
  - Some improvement to test accuracy was noticed.

"""

import torch


from src.config.configuration import Config
from src.profilers import FLOPSProfiler

import torch
import torch.nn as nn

from torch.func import grad, jvp, functional_call
from collections import OrderedDict


class JVPRegularizedLoss(nn.Module):
    """JVP-regularized loss for continual learning."""

    def __init__(self, model, criterion, jvp_reg=1.0, deltax_norm=1.0):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.jvp_reg = jvp_reg
        self.deltax_norm = deltax_norm

        # Cache params dict to avoid recreating every iteration
        self._params = None

    def forward(self, current_batch, memory_batch):
        """
        Args:
            current_batch: (input, target) for current task
            memory_batch: (input, target) for memory buffer

        Returns:
            grad_dict: Computed gradients for each parameter
            loss_curr: Loss on current task
            loss_mem: Loss on memory task
        """
        x_curr, y_curr = current_batch
        x_mem, y_mem = memory_batch

        # - Get parameters as dict for functional API (cached)
        if self._params is None:
            self._params = OrderedDict(self.model.named_parameters())

        # - Compute deltax direction
        deltax = (
            self.deltax_norm
            * (x_mem - x_curr)
            / (torch.linalg.norm(x_mem) + torch.linalg.norm(x_curr))
        )

        # - Compute combined gradients
        grad_dict, loss_curr, loss_mem = self._compute_jvp_gradients(
            self._params, x_curr, y_curr, x_mem, y_mem, deltax
        )

        return grad_dict, loss_curr, loss_mem

    def _compute_jvp_gradients(self, params, x_curr, y_curr, x_mem, y_mem, deltax):
        """Compute JVP-regularized gradients."""

        # - Ensure all params require grad
        for p in params.values():
            p.requires_grad_(True)

        # - Define loss function for functional API
        def loss_fn(p, x, y):
            pred = functional_call(self.model, p, (x,))
            return self.criterion(pred, y)

        # - Compute gradients
        grad_fn = grad(loss_fn, argnums=0)

        # - Current task gradient
        grad_curr = grad_fn(params, x_curr, y_curr)

        # - Memory task gradient
        grad_mem = grad_fn(params, x_mem, y_mem)

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
            k: grad_curr[k] + grad_mem[k] + self.jvp_reg * grad_jvp[k] for k in params
        }

        # - Compute loss values
        with torch.no_grad():
            loss_curr = loss_fn(params, x_curr, y_curr)
            loss_mem = loss_fn(params, x_mem, y_mem)

        return combined_grads, loss_curr, loss_mem


def step_method_jvp_reg(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    iter: int,
    train_batch: tuple,
    hist_batch: tuple,
    profiler: FLOPSProfiler,
    jvp_loss: JVPRegularizedLoss,
):
    if profiler and iter > profiler.warmup_iters:
        # Compute gradients
        with profiler.measure_flops(tag="jvp_hamil"):
            grads_dict, J_P, J_M = jvp_loss(train_batch, hist_batch)

        # Detach and assign gradients (outside profiling)
        # Memory operations add some latency.
        # Doing this externally to compare runtime w/ original implementation
        for name, param in model.named_parameters():
            param.grad = grads_dict[name].detach()

        # Optimizer step
        with profiler.measure_flops_optimizer(
            tag="jvp_optim", model=model, device=cfg.device
        ):
            optimizer.step()

    else:
        grads_dict, J_P, J_M = jvp_loss(train_batch, hist_batch)
        for name, param in model.named_parameters():
            param.grad = grads_dict[name].detach()
        optimizer.step()

    return J_P.item(), J_M.item(), (J_P + J_M).item()
