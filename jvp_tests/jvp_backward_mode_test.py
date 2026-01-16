"""
Backward-mode only implementation of JVP regularization for transformers.

Mathematical Background:
========================
The original JVP regularization computes:
    grad_jvp = grad_θ [ JVP(L(θ, x_mem), (grad_θ, deltax)) ]

Where JVP computes the directional derivative of L w.r.t. (θ, x) in direction (grad_θ, deltax).

Using the chain rule, JVP(L, (v_θ, v_x)) = ∂L/∂θ · v_θ + ∂L/∂x · v_x

So we want: grad_θ [ (∂L/∂θ · v_θ) + (∂L/∂x · v_x) ]

where v_θ = grad_curr and v_x = deltax.

This can be computed with backward-mode AD by:
1. Computing ∂L/∂θ and ∂L/∂x (first backward pass, with create_graph=True)
2. Computing the dot products: s = sum(∂L/∂θ * v_θ) + sum(∂L/∂x * v_x)
3. Computing grad_θ(s) (second backward pass)

This is the "vector-Jacobian product of a vector-Jacobian product" approach,
which only uses backward-mode AD.
"""

import torch
import torch.nn as nn
from torch.func import functional_call
from collections import OrderedDict
from transformers import ViTForImageClassification


class JVPRegularizedLossBackward(nn.Module):
    """JVP-regularized loss using only backward-mode AD.

    Compatible with flash attention and other ops that don't support forward-mode AD.
    """

    def __init__(self, model, criterion, jvp_reg=1.0, deltax_norm=1.0):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.jvp_reg = jvp_reg
        self.deltax_norm = deltax_norm

    def _get_logits(self, output):
        """Extract logits from model output (handles HuggingFace models)."""
        if hasattr(output, 'logits'):
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
            loss_curr: Loss on current task
            loss_mem: Loss on memory task
        """
        x_curr, y_curr = current_batch
        x_mem, y_mem = memory_batch

        # Compute deltax direction
        deltax = (
            self.deltax_norm
            * (x_mem - x_curr)
            / (torch.linalg.norm(x_mem) + torch.linalg.norm(x_curr) + 1e-8)
        )

        # === Step 1: Compute grad_curr (gradient on current task) ===
        self.model.zero_grad()
        output_curr = self.model(x_curr)
        loss_curr = self.criterion(self._get_logits(output_curr), y_curr)

        # Get gradients for current task
        grad_curr = torch.autograd.grad(
            loss_curr,
            self.model.parameters(),
            create_graph=False,  # Don't need graph for this
            retain_graph=False
        )
        grad_curr_dict = {name: g for (name, _), g in zip(self.model.named_parameters(), grad_curr)}

        # === Step 2: Compute grad_mem (gradient on memory task) ===
        self.model.zero_grad()
        output_mem = self.model(x_mem)
        loss_mem = self.criterion(self._get_logits(output_mem), y_mem)

        grad_mem = torch.autograd.grad(
            loss_mem,
            self.model.parameters(),
            create_graph=False,
            retain_graph=False
        )
        grad_mem_dict = {name: g for (name, _), g in zip(self.model.named_parameters(), grad_mem)}

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
            retain_graph=True
        )

        grad_theta = grads_and_input[:-1]  # Gradients w.r.t. parameters
        grad_x = grads_and_input[-1]        # Gradient w.r.t. input

        # Compute the scalar: s = (∂L/∂θ · v_θ) + (∂L/∂x · v_x)
        # where v_θ = grad_curr (detached) and v_x = deltax
        dot_theta = sum(
            (g * v.detach()).sum()
            for g, v in zip(grad_theta, grad_curr)
        )
        dot_x = (grad_x * deltax).sum()

        s = dot_theta + dot_x

        # Now compute grad_θ(s) - this is the JVP regularization gradient
        grad_jvp = torch.autograd.grad(
            s,
            params_list,
            create_graph=False,
            retain_graph=False
        )
        grad_jvp_dict = {name: g for (name, _), g in zip(self.model.named_parameters(), grad_jvp)}

        # === Step 4: Combine gradients ===
        combined_grads = {}
        for name in grad_curr_dict:
            combined_grads[name] = (
                grad_curr_dict[name]
                + grad_mem_dict[name]
                + self.jvp_reg * grad_jvp_dict[name]
            )

        return combined_grads, loss_curr.detach(), loss_mem.detach()


def test_backward_mode_jvp():
    """Test the backward-mode JVP implementation with ViT."""

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load ViT model
    print("Loading ViT model...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=10,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)
    model.train()

    # Create dummy data
    batch_size = 2
    x_curr = torch.randn(batch_size, 3, 224, 224, device=device)
    y_curr = torch.randint(0, 10, (batch_size,), device=device)
    x_mem = torch.randn(batch_size, 3, 224, 224, device=device)
    y_mem = torch.randint(0, 10, (batch_size,), device=device)

    criterion = nn.CrossEntropyLoss()

    # Create the backward-mode JVP loss
    jvp_loss = JVPRegularizedLossBackward(
        model=model,
        criterion=criterion,
        jvp_reg=0.001,
        deltax_norm=1.0
    )

    print("\n=== Testing Backward-Mode JVP Regularization ===")
    try:
        grads, loss_curr, loss_mem = jvp_loss(
            (x_curr, y_curr),
            (x_mem, y_mem)
        )

        print(f"✓ Success!")
        print(f"  Loss (current): {loss_curr.item():.4f}")
        print(f"  Loss (memory):  {loss_mem.item():.4f}")
        print(f"  Gradients computed for {len(grads)} parameters")

        # Check gradient stats
        grad_norms = [g.norm().item() for g in grads.values()]
        print(f"  Gradient norm (min/mean/max): {min(grad_norms):.6f} / {sum(grad_norms)/len(grad_norms):.6f} / {max(grad_norms):.6f}")

        # Verify gradients are not all zeros
        total_norm = sum(g.norm().item() for g in grads.values())
        if total_norm > 0:
            print(f"  Total gradient norm: {total_norm:.6f} ✓")
        else:
            print(f"  WARNING: Total gradient norm is zero!")

    except Exception as e:
        print(f"✗ Failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # Test that we can actually do an optimizer step
    print("\n=== Testing Optimizer Step ===")
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        grads, loss_curr, loss_mem = jvp_loss(
            (x_curr, y_curr),
            (x_mem, y_mem)
        )

        # Assign gradients to parameters
        for name, param in model.named_parameters():
            param.grad = grads[name].detach()

        # Optimizer step
        optimizer.step()

        print("✓ Optimizer step completed successfully!")

    except Exception as e:
        print(f"✗ Optimizer step failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_backward_mode_jvp()
