"""JVP (Jacobian-Vector Product) regularization using backward-mode AD only.

This module provides a transformer-compatible implementation of JVP regularization
for continual learning. Unlike the forward-mode AD implementation in jvp_reg.py,
this version uses only backward-mode AD, making it compatible with:
- Flash Attention
- Scaled Dot-Product Attention (SDPA)
- Any operation that doesn't support forward-mode AD

Mathematical Background:
========================
The original JVP regularization computes:
    grad_jvp = grad_θ [ JVP(L(θ, x_mem), (v_θ, v_x)) ]

Where JVP computes the directional derivative of L w.r.t. (θ, x) in direction (v_θ, v_x).

Using the chain rule: JVP(L, (v_θ, v_x)) = ∂L/∂θ · v_θ + ∂L/∂x · v_x

So we want: grad_θ [ (∂L/∂θ · v_θ) + (∂L/∂x · v_x) ]

This is computed with backward-mode AD by:
1. Computing ∂L/∂θ and ∂L/∂x (first backward pass, with create_graph=True)
2. Computing the scalar: s = sum(∂L/∂θ * v_θ) + sum(∂L/∂x * v_x)
3. Computing grad_θ(s) (second backward pass)

This is equivalent to computing the Hessian-vector product, using only backward passes.
"""

import torch
import torch.nn as nn

from config.configuration import Config
from profilers import FLOPSProfiler


class JVPRegularizedLossBackward(nn.Module):
    """JVP-regularized loss using only backward-mode AD.

    Compatible with flash attention and other ops that don't support forward-mode AD.
    """

    def __init__(self, model, criterion, jvp_reg=1.0, deltax_norm=1.0):
        """Initialize JVP regularized loss.

        Args:
            model: The neural network model
            criterion: Loss function (e.g., CrossEntropyLoss)
            jvp_reg: Regularization strength for JVP term
            deltax_norm: Scaling factor for input-space direction
        """
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.jvp_reg = jvp_reg
        self.deltax_norm = deltax_norm

    def _get_logits(self, output):
        """Extract logits from model output (handles HuggingFace models)."""
        if hasattr(output, "logits"):
            return output.logits
        return output

    def forward(self, current_batch, memory_batch):
        """
        Compute JVP-regularized gradients using only backward-mode AD.

        Args:
            current_batch: (input, target) for current task
            memory_batch: (input, target) for memory buffer

        Returns:
            grad_dict: Combined gradients for each parameter
            loss_curr: Loss on current task (detached)
            loss_mem: Loss on memory task (detached)
        """
        x_curr, y_curr = current_batch
        x_mem, y_mem = memory_batch

        # Compute deltax direction (normalized difference between memory and current inputs)
        deltax = (
            self.deltax_norm
            * (x_mem - x_curr)
            / (torch.linalg.norm(x_mem) + torch.linalg.norm(x_curr) + 1e-8)
        )

        # === Step 1: Compute grad_curr (gradient on current task) ===
        self.model.zero_grad()
        output_curr = self.model(x_curr)
        loss_curr = self.criterion(self._get_logits(output_curr), y_curr)

        grad_curr = torch.autograd.grad(
            loss_curr,
            self.model.parameters(),
            create_graph=False,
            retain_graph=False,
        )
        grad_curr_dict = {
            name: g for (name, _), g in zip(self.model.named_parameters(), grad_curr)
        }

        # === Step 2: Compute grad_mem (gradient on memory task) ===
        self.model.zero_grad()
        output_mem = self.model(x_mem)
        loss_mem = self.criterion(self._get_logits(output_mem), y_mem)

        grad_mem = torch.autograd.grad(
            loss_mem,
            self.model.parameters(),
            create_graph=False,
            retain_graph=False,
        )
        grad_mem_dict = {
            name: g for (name, _), g in zip(self.model.named_parameters(), grad_mem)
        }

        # === Step 3: Compute JVP regularization term using backward-mode AD ===
        # We want: grad_θ [ (∂L/∂θ · v_θ) + (∂L/∂x · v_x) ]
        # where v_θ = grad_curr and v_x = deltax

        self.model.zero_grad()

        # Enable gradient tracking on input for ∂L/∂x computation
        x_mem_grad = x_mem.detach().clone().requires_grad_(True)

        output_mem_2 = self.model(x_mem_grad)
        loss_mem_2 = self.criterion(self._get_logits(output_mem_2), y_mem)

        # Compute gradients w.r.t. both parameters AND input, with graph retained
        params_list = list(self.model.parameters())
        grads_and_input = torch.autograd.grad(
            loss_mem_2,
            params_list + [x_mem_grad],
            create_graph=True,  # Need graph for second derivative
            retain_graph=True,
        )

        grad_theta = grads_and_input[:-1]  # Gradients w.r.t. parameters
        grad_x = grads_and_input[-1]  # Gradient w.r.t. input

        # Compute the scalar: s = (∂L/∂θ · v_θ) + (∂L/∂x · v_x)
        # where v_θ = grad_curr (detached) and v_x = deltax
        dot_theta = sum(
            (g * v.detach()).sum() for g, v in zip(grad_theta, grad_curr)
        )
        dot_x = (grad_x * deltax).sum()

        s = dot_theta + dot_x

        # Compute grad_θ(s) - this is the JVP regularization gradient
        grad_jvp = torch.autograd.grad(
            s,
            params_list,
            create_graph=False,
            retain_graph=False,
        )
        grad_jvp_dict = {
            name: g for (name, _), g in zip(self.model.named_parameters(), grad_jvp)
        }

        # === Step 4: Combine gradients ===
        combined_grads = {}
        for name in grad_curr_dict:
            combined_grads[name] = (
                grad_curr_dict[name]
                + grad_mem_dict[name]
                + self.jvp_reg * grad_jvp_dict[name]
            )

        return combined_grads, loss_curr.detach(), loss_mem.detach()


def step_method_jvp_reg_backward(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    iter: int,
    train_batch: tuple,
    hist_batch: tuple,
    profiler: FLOPSProfiler,
    jvp_loss: JVPRegularizedLossBackward,
):
    """Perform one training step with JVP regularization (backward-mode AD).

    Args:
        model: The neural network model
        criterion: Loss function
        optimizer: Optimizer instance
        cfg: Configuration object
        iter: Current iteration number
        train_batch: (input, target) for current task
        hist_batch: (input, target) for historical/memory data
        profiler: FLOPS profiler for performance tracking
        jvp_loss: JVPRegularizedLossBackward instance

    Returns:
        Tuple of (loss_current, loss_memory, total_loss) as floats
    """
    if profiler and iter > profiler.warmup_iters:
        # Compute gradients with profiling
        with profiler.measure_flops(tag="jvp_backward"):
            grads_dict, J_P, J_M = jvp_loss(train_batch, hist_batch)

        # Assign gradients to parameters
        for name, param in model.named_parameters():
            param.grad = grads_dict[name].detach()

        # Optimizer step with profiling
        with profiler.measure_flops_optimizer(
            tag="jvp_optim", model=model, device=cfg.device
        ):
            optimizer.step()
    else:
        # Without profiling
        grads_dict, J_P, J_M = jvp_loss(train_batch, hist_batch)
        for name, param in model.named_parameters():
            param.grad = grads_dict[name].detach()
        optimizer.step()

    return J_P.item(), J_M.item(), (J_P + J_M).item()
