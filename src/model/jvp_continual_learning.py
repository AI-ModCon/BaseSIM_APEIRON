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
import torch.nn as nn
from torch.optim import Optimizer

from torch.func import grad, jvp, functional_call
from collections import OrderedDict

class JVPAdam(Optimizer):
    """Adam optimizer for JVP regularization."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(JVPAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                grad = p.grad

                # Update biased moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction - compute corrected moments WITHOUT modifying state
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Adam update
                denom = v_hat.sqrt().add_(group['eps'])
                p.add_(m_hat / denom, alpha=-group['lr'])

        return loss


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

        #- Get parameters as dict for functional API (cached)
        if self._params is None:
            self._params = OrderedDict(self.model.named_parameters())

        #- Compute deltax direction
        deltax = self.deltax_norm * (x_mem - x_curr) / (
            torch.linalg.norm(x_mem) + torch.linalg.norm(x_curr)
        )

        #- Compute combined gradients
        grad_dict, loss_curr, loss_mem = self._compute_jvp_gradients(
            self._params, x_curr, y_curr, x_mem, y_mem, deltax
        )

        return grad_dict, loss_curr, loss_mem

    def _compute_jvp_gradients(self, params, x_curr, y_curr, x_mem, y_mem, deltax):
        """Compute JVP-regularized gradients."""

        #- Ensure all params require grad
        for p in params.values():
            p.requires_grad_(True)

        #- Define loss function for functional API
        def loss_fn(p, x, y):
            pred = functional_call(self.model, p, (x,))
            return self.criterion(pred, y)

        #- Compute gradients
        grad_fn = grad(loss_fn, argnums=0)

        #- Current task gradient
        grad_curr = grad_fn(params, x_curr, y_curr)

        #- Memory task gradient
        grad_mem = grad_fn(params, x_mem, y_mem)

        #- JVP computation
        def f(p, x):
            return loss_fn(p, x, y_mem)

        def jvp_func(p, tangents):
            return jvp(f, (p, x_mem), tangents)[1]

        #- Use current gradient as tangent direction
        tangents = OrderedDict((k, grad_curr[k]) for k in params)

        grad_jvp = grad(jvp_func)(params, (tangents, deltax))

        #- Combine gradients
        combined_grads = {
            k: grad_curr[k] + grad_mem[k] + self.jvp_reg * grad_jvp[k]
            for k in params
        }

        #- Compute loss values
        with torch.no_grad():
            loss_curr = loss_fn(params, x_curr, y_curr)
            loss_mem = loss_fn(params, x_mem, y_mem)

        return combined_grads, loss_curr, loss_mem
