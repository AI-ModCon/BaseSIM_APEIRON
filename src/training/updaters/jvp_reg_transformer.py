"""JVP regularization implementations compatible with transformers (flash attention).

This module provides multiple approaches to compute JVP regularization gradients
without requiring second-order derivatives through flash attention.

All classes follow the same interface as JVPRegularizedLoss:
    - __init__(model, criterion, jvp_reg, deltax_norm, epsilon)
    - forward(current_batch, memory_batch, v_theta=None, v_x=None)
        -> (grad_dict, loss_curr, loss_mem)

Approaches:
1. MathBackend: Exact computation by disabling flash attention (slow but exact)
2. FiniteDiffHVP: Finite difference Hessian-vector product (fast, O(ε²) error)
3. GaussNewton: Gauss-Newton approximation (fast, drops model curvature term)
4. RichardsonHVP: Richardson extrapolation for improved finite diff accuracy (O(ε⁴) error)

Mathematical Background:
========================
The JVP regularization computes:
    combined_grads = grad_curr + grad_mem + jvp_reg * grad_jvp

Where:
    grad_jvp = ∇_θ [ ∂L/∂θ · v_θ + ∂L/∂x · v_x ]
             = H_θθ · v_θ + H_θx · v_x

Default directions (if not provided):
    v_theta = grad_curr (gradient on current task)
    v_x = deltax = normalized(x_mem - x_curr)
"""

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Dict, Tuple, Optional, Union, Type
from abc import ABC, abstractmethod


class BaseJVPRegularizedLoss(nn.Module, ABC):
    """Abstract base class for JVP regularization implementations."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        jvp_reg: float = 1.0,
        deltax_norm: float = 1.0,
        epsilon: float = 1e-4,
    ):
        """Initialize JVP regularized loss.

        Args:
            model: Neural network model
            criterion: Loss function (e.g., CrossEntropyLoss)
            jvp_reg: Regularization strength for JVP term
            deltax_norm: Scaling factor for input-space direction
            epsilon: Finite difference step size (used by subclasses)
        """
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.jvp_reg = jvp_reg
        self.deltax_norm = deltax_norm
        self.epsilon = epsilon

    def _get_logits(self, output):
        """Extract logits from model output (handles HuggingFace models)."""
        return output.logits if hasattr(output, "logits") else output

    def _compute_deltax(self, x_curr: torch.Tensor, x_mem: torch.Tensor) -> torch.Tensor:
        """Compute normalized direction from current to memory input."""
        return (
            self.deltax_norm
            * (x_mem - x_curr)
            / (torch.linalg.norm(x_mem) + torch.linalg.norm(x_curr) + 1e-8)
        )

    def _compute_grad_curr(
        self, x_curr: torch.Tensor, y_curr: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute gradient on current task."""
        self.model.zero_grad()
        logits = self._get_logits(self.model(x_curr))
        loss = self.criterion(logits, y_curr)
        loss.backward()
        return {
            n: p.grad.clone() for n, p in self.model.named_parameters() if p.grad is not None
        }

    def _compute_grad_mem(
        self, x_mem: torch.Tensor, y_mem: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Compute gradient on memory task and return loss."""
        self.model.zero_grad()
        logits = self._get_logits(self.model(x_mem))
        loss = self.criterion(logits, y_mem)
        loss.backward()
        return {
            n: p.grad.clone() for n, p in self.model.named_parameters() if p.grad is not None
        }, loss.detach()

    def _compute_loss_curr(self, x_curr: torch.Tensor, y_curr: torch.Tensor) -> torch.Tensor:
        """Compute loss on current task (no grad)."""
        with torch.no_grad():
            logits = self._get_logits(self.model(x_curr))
            return self.criterion(logits, y_curr)

    @abstractmethod
    def _compute_grad_jvp(
        self,
        x_mem: torch.Tensor,
        y_mem: torch.Tensor,
        v_theta: Dict[str, torch.Tensor],
        v_x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute ∇_θ(JVP) = H_θθ · v_θ + H_θx · v_x.

        Args:
            x_mem: Memory input tensor
            y_mem: Memory target tensor
            v_theta: Parameter-space direction (tangent vector for θ)
            v_x: Input-space direction (tangent vector for x)

        Returns:
            Dictionary mapping parameter names to gradient tensors
        """
        raise NotImplementedError

    def forward(
        self,
        current_batch: Tuple[torch.Tensor, torch.Tensor],
        memory_batch: Tuple[torch.Tensor, torch.Tensor],
        v_theta: Optional[Dict[str, torch.Tensor]] = None,
        v_x: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Compute JVP-regularized gradients.

        Args:
            current_batch: (input, target) for current task
            memory_batch: (input, target) for memory buffer
            v_theta: Optional parameter direction. If None, uses grad_curr.
            v_x: Optional input direction. If None, uses deltax = normalized(x_mem - x_curr).

        Returns:
            grad_dict: Combined gradients for each parameter
            loss_curr: Loss on current task (detached)
            loss_mem: Loss on memory task (detached)
        """
        x_curr, y_curr = current_batch
        x_mem, y_mem = memory_batch

        # Compute grad_curr (always needed for combined gradient)
        grad_curr = self._compute_grad_curr(x_curr, y_curr)

        # Use provided v_theta or default to grad_curr
        if v_theta is None:
            v_theta = grad_curr

        # Use provided v_x or compute deltax
        if v_x is None:
            v_x = self._compute_deltax(x_curr, x_mem)

        # Compute grad_mem
        grad_mem, loss_mem = self._compute_grad_mem(x_mem, y_mem)

        # Compute loss_curr
        loss_curr = self._compute_loss_curr(x_curr, y_curr)

        # Compute grad_jvp using the specific implementation
        grad_jvp = self._compute_grad_jvp(x_mem, y_mem, v_theta, v_x)

        # Combine gradients
        combined_grads = {
            n: grad_curr[n] + grad_mem[n] + self.jvp_reg * grad_jvp[n] for n in grad_curr
        }

        return combined_grads, loss_curr, loss_mem


# =============================================================================
# Approach 1: Math Backend (Exact, Slow)
# =============================================================================


@contextmanager
def sdpa_math_backend():
    """Context manager to force SDPA to use math backend (no flash attention)."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            yield
    except ImportError:
        # Fallback for older PyTorch versions
        old_flash = (
            torch.backends.cuda.flash_sdp_enabled()
            if hasattr(torch.backends.cuda, "flash_sdp_enabled")
            else None
        )
        old_mem = (
            torch.backends.cuda.mem_efficient_sdp_enabled()
            if hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled")
            else None
        )
        try:
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            yield
        finally:
            if old_flash is not None and hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(old_flash)
            if old_mem is not None and hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(old_mem)


class MathBackendJVPLoss(BaseJVPRegularizedLoss):
    """Exact JVP computation by disabling flash attention.

    This approach uses the math backend for attention, which supports
    second-order derivatives but is O(N²) in memory and slower.

    Use this as a ground truth reference for comparing other methods.
    """

    def _compute_grad_jvp(
        self,
        x_mem: torch.Tensor,
        y_mem: torch.Tensor,
        v_theta: Dict[str, torch.Tensor],
        v_x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute grad_jvp using exact second-order AD with math backend."""

        with sdpa_math_backend():
            self.model.zero_grad()

            # Enable grad on input for ∂L/∂x
            x_mem_grad = x_mem.detach().clone().requires_grad_(True)

            logits = self._get_logits(self.model(x_mem_grad))
            loss = self.criterion(logits, y_mem)

            # First backward with create_graph=True
            params_list = list(self.model.parameters())
            grads = torch.autograd.grad(
                loss,
                params_list + [x_mem_grad],
                create_graph=True,
            )

            grad_theta = grads[:-1]
            grad_x = grads[-1]

            # Compute scalar: s = ∂L/∂θ · v_θ + ∂L/∂x · v_x
            dot_theta = sum(
                (g * v.detach()).sum() for g, v in zip(grad_theta, v_theta.values())
            )
            dot_x = (grad_x * v_x).sum()
            s = dot_theta + dot_x

            # Second backward to get ∇_θ(s)
            grad_jvp_list = torch.autograd.grad(
                s,
                params_list,
                create_graph=False,
            )

        return {n: g for (n, _), g in zip(self.model.named_parameters(), grad_jvp_list)}


# =============================================================================
# Approach 2: Finite Difference HVP
# =============================================================================


class FiniteDiffHVPLoss(BaseJVPRegularizedLoss):
    """Finite difference Hessian-vector product.

    Computes:
        H_θθ · v_θ ≈ (∇L(θ+εv_θ) - ∇L(θ-εv_θ)) / 2ε
        H_θx · v_x ≈ (∇L(x+εv_x) - ∇L(x-εv_x)) / 2ε

    Accuracy: O(ε²) with central differences.
    Cost: 4 backward passes.
    """

    def _get_grad(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute gradient at given input."""
        self.model.zero_grad()
        logits = self._get_logits(self.model(x))
        loss = self.criterion(logits, y)
        loss.backward()
        return {
            n: p.grad.clone() for n, p in self.model.named_parameters() if p.grad is not None
        }

    def _compute_grad_jvp(
        self,
        x_mem: torch.Tensor,
        y_mem: torch.Tensor,
        v_theta: Dict[str, torch.Tensor],
        v_x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute grad_jvp via finite difference HVP."""

        # === H_θθ · v_θ ===
        # Perturb θ → θ + ε·v_θ
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                p.add_(self.epsilon * v_theta[n])
        grad_plus_theta = self._get_grad(x_mem, y_mem)

        # Perturb θ → θ - ε·v_θ (from θ + ε·v_θ)
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                p.sub_(2 * self.epsilon * v_theta[n])
        grad_minus_theta = self._get_grad(x_mem, y_mem)

        # Restore θ
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                p.add_(self.epsilon * v_theta[n])

        H_v_theta = {
            n: (grad_plus_theta[n] - grad_minus_theta[n]) / (2 * self.epsilon)
            for n in grad_plus_theta
        }

        # === H_θx · v_x ===
        grad_plus_x = self._get_grad(x_mem + self.epsilon * v_x, y_mem)
        grad_minus_x = self._get_grad(x_mem - self.epsilon * v_x, y_mem)

        H_v_x = {
            n: (grad_plus_x[n] - grad_minus_x[n]) / (2 * self.epsilon) for n in grad_plus_x
        }

        # Combine
        return {n: H_v_theta[n] + H_v_x[n] for n in H_v_theta}


# =============================================================================
# Approach 3: Gauss-Newton Approximation
# =============================================================================


class GaussNewtonJVPLoss(BaseJVPRegularizedLoss):
    """Gauss-Newton approximation for JVP regularization.

    Approximates:
        H ≈ H_GN = Jᵀ B J

    Where:
        J = ∂logits/∂θ (Jacobian)
        B = ∂²loss/∂logits² (Hessian of loss w.r.t. outputs)

    For cross-entropy: B = diag(p) - ppᵀ

    Computes:
        H_GN · v = Jᵀ B (J v)

    Steps:
        1. J·v via finite differences (2 forward passes)
        2. B·(J·v) analytically
        3. Jᵀ·(B·J·v) via backward (1 backward pass)

    Cost: 6 forward + 2 backward passes total.
    """

    def _apply_B_cross_entropy(self, p: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Apply B = diag(p) - ppᵀ to vector u.

        For cross-entropy loss with softmax, the Hessian w.r.t. logits is:
            B = diag(p) - p @ p.T

        B @ u = p * u - p * (p · u)
        """
        p_dot_u = (p * u).sum(dim=-1, keepdim=True)
        return p * u - p * p_dot_u

    def _compute_grad_jvp(
        self,
        x_mem: torch.Tensor,
        y_mem: torch.Tensor,
        v_theta: Dict[str, torch.Tensor],
        v_x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute grad_jvp via Gauss-Newton approximation."""

        # Get base logits and softmax
        with torch.no_grad():
            logits_base = self._get_logits(self.model(x_mem))
            p = torch.softmax(logits_base, dim=-1)

        # === H_GN_θθ · v_θ = Jᵀ B J v_θ ===

        # Step 1: J_θ · v_θ via finite diff
        with torch.no_grad():
            for n, param in self.model.named_parameters():
                param.add_(self.epsilon * v_theta[n])
            logits_plus = self._get_logits(self.model(x_mem))

            for n, param in self.model.named_parameters():
                param.sub_(2 * self.epsilon * v_theta[n])
            logits_minus = self._get_logits(self.model(x_mem))

            for n, param in self.model.named_parameters():
                param.add_(self.epsilon * v_theta[n])

        Jv_theta = (logits_plus - logits_minus) / (2 * self.epsilon)

        # Step 2: B · (J · v_θ)
        B_Jv_theta = self._apply_B_cross_entropy(p, Jv_theta)

        # Step 3: Jᵀ · (B · J · v_θ) via backward
        self.model.zero_grad()
        logits = self._get_logits(self.model(x_mem))
        pseudo_loss = (logits * B_Jv_theta.detach()).sum()
        pseudo_loss.backward()
        H_GN_v_theta = {
            n: p.grad.clone() for n, p in self.model.named_parameters() if p.grad is not None
        }

        # === H_GN_θx · v_x = Jᵀ B J_x v_x ===

        # Step 1: J_x · v_x via finite diff
        with torch.no_grad():
            logits_x_plus = self._get_logits(self.model(x_mem + self.epsilon * v_x))
            logits_x_minus = self._get_logits(self.model(x_mem - self.epsilon * v_x))

        Jv_x = (logits_x_plus - logits_x_minus) / (2 * self.epsilon)

        # Step 2: B · (J_x · v_x)
        B_Jv_x = self._apply_B_cross_entropy(p, Jv_x)

        # Step 3: Jᵀ · (B · J_x · v_x) via backward
        self.model.zero_grad()
        logits = self._get_logits(self.model(x_mem))
        pseudo_loss = (logits * B_Jv_x.detach()).sum()
        pseudo_loss.backward()
        H_GN_v_x = {
            n: p.grad.clone() for n, p in self.model.named_parameters() if p.grad is not None
        }

        # Combine
        return {n: H_GN_v_theta[n] + H_GN_v_x[n] for n in H_GN_v_theta}


# =============================================================================
# Approach 4: Richardson Extrapolation HVP
# =============================================================================


class RichardsonHVPLoss(BaseJVPRegularizedLoss):
    """Richardson extrapolation for improved finite difference accuracy.

    Uses central differences at two step sizes and extrapolates:
        HVP_improved = (4 * HVP(ε/2) - HVP(ε)) / 3

    This cancels the O(ε²) error term, giving O(ε⁴) accuracy.

    Cost: 8 backward passes (2x the basic finite diff).
    """

    def _get_grad(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute gradient at given input."""
        self.model.zero_grad()
        logits = self._get_logits(self.model(x))
        loss = self.criterion(logits, y)
        loss.backward()
        return {
            n: p.grad.clone() for n, p in self.model.named_parameters() if p.grad is not None
        }

    def _hvp_at_epsilon(
        self,
        x_mem: torch.Tensor,
        y_mem: torch.Tensor,
        v_theta: Dict[str, torch.Tensor],
        v_x: torch.Tensor,
        eps: float,
    ) -> Dict[str, torch.Tensor]:
        """Compute HVP using central differences at given epsilon."""

        # H_θθ · v_θ
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                p.add_(eps * v_theta[n])
        grad_plus_theta = self._get_grad(x_mem, y_mem)

        with torch.no_grad():
            for n, p in self.model.named_parameters():
                p.sub_(2 * eps * v_theta[n])
        grad_minus_theta = self._get_grad(x_mem, y_mem)

        with torch.no_grad():
            for n, p in self.model.named_parameters():
                p.add_(eps * v_theta[n])

        H_v_theta = {
            n: (grad_plus_theta[n] - grad_minus_theta[n]) / (2 * eps) for n in grad_plus_theta
        }

        # H_θx · v_x
        grad_plus_x = self._get_grad(x_mem + eps * v_x, y_mem)
        grad_minus_x = self._get_grad(x_mem - eps * v_x, y_mem)

        H_v_x = {
            n: (grad_plus_x[n] - grad_minus_x[n]) / (2 * eps) for n in grad_plus_x
        }

        return {n: H_v_theta[n] + H_v_x[n] for n in H_v_theta}

    def _compute_grad_jvp(
        self,
        x_mem: torch.Tensor,
        y_mem: torch.Tensor,
        v_theta: Dict[str, torch.Tensor],
        v_x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute grad_jvp via Richardson extrapolation."""

        # HVP at ε
        hvp_eps = self._hvp_at_epsilon(x_mem, y_mem, v_theta, v_x, self.epsilon)

        # HVP at ε/2
        hvp_half = self._hvp_at_epsilon(x_mem, y_mem, v_theta, v_x, self.epsilon / 2)

        # Richardson extrapolation: (4 * f(h/2) - f(h)) / 3
        return {n: (4 * hvp_half[n] - hvp_eps[n]) / 3 for n in hvp_eps}


# =============================================================================
# Factory function
# =============================================================================


def get_jvp_loss_for_transformer(
    model: nn.Module,
    criterion: nn.Module,
    method: str = "finite_diff",
    jvp_reg: float = 1.0,
    deltax_norm: float = 1.0,
    epsilon: float = 1e-4,
) -> BaseJVPRegularizedLoss:
    """Factory function to create JVP loss with specified method.

    Args:
        model: Neural network model
        criterion: Loss function
        method: One of "math_backend", "finite_diff", "gauss_newton", "richardson"
        jvp_reg: Regularization strength
        deltax_norm: Scaling for input direction
        epsilon: Finite difference step size

    Returns:
        JVP loss instance
    """
    methods: Dict[str, Type[BaseJVPRegularizedLoss]] = {
        "math_backend": MathBackendJVPLoss,
        "finite_diff": FiniteDiffHVPLoss,
        "gauss_newton": GaussNewtonJVPLoss,
        "richardson": RichardsonHVPLoss,
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")

    cls = methods[method]
    return cls(model, criterion, jvp_reg, deltax_norm, epsilon)
