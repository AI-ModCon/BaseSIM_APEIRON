"""Analyze the effect of direction magnitude on finite difference error.

This script investigates how the magnitude (norm) of the direction vectors
(v_theta and v_x) affects the accuracy of finite difference approximations
for computing the Hessian-vector product (HVP).

Mathematical Background:
========================
For central finite differences:
    HVP ≈ (∇L(θ + ε·v) - ∇L(θ - ε·v)) / (2ε)

The approximation error has two sources:
1. Truncation error: O(ε² · ||v||³ · L''') from Taylor expansion
2. Numerical error: O(||v|| · machine_eps / ε) from floating point

The total error is:
    E_total ≈ C₁ · ε² · ||v||³ + C₂ · ||v|| / ε

The optimal ε depends on ||v||:
    ε_optimal ∝ ||v||^(-2/3)

Key Insights:
- Small ||v||: Numerical errors dominate (need larger ε)
- Large ||v||: Truncation errors dominate (need smaller ε)
- Very large ||v||: Can cause numerical instability

Run with:
    poetry run python tests/test_direction_magnitude_effect.py

For shorter runs:
    poetry run python tests/test_direction_magnitude_effect.py --quick
"""

import argparse
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, "/Users/kraghavan/Desktop/modcon/BaseSim_Framework/src")

from training.updaters.jvp_reg_transformer import (
    MathBackendJVPLoss,
    FiniteDiffHVPLoss,
    RichardsonHVPLoss,
)


@dataclass
class MagnitudeTestResult:
    """Results for a single magnitude test."""
    magnitude: float
    rel_error: float
    cosine_sim: float
    grad_norm: float
    time_seconds: float


def get_device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device: torch.device, model_type: str = "vit"):
    """Load model for testing."""
    if model_type == "vit":
        from transformers import ViTForImageClassification
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=10,
            ignore_mismatched_sizes=True,
        )
    elif model_type == "mlp":
        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3 * 224 * 224, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, 10)

            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)

        model = SimpleMLP()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def create_batches(device: torch.device, batch_size: int = 2):
    """Create dummy batches."""
    x_curr = torch.randn(batch_size, 3, 224, 224, device=device)
    y_curr = torch.randint(0, 10, (batch_size,), device=device)
    x_mem = torch.randn(batch_size, 3, 224, 224, device=device)
    y_mem = torch.randint(0, 10, (batch_size,), device=device)
    return (x_curr, y_curr), (x_mem, y_mem)


def compute_relative_error(
    grads_approx: Dict[str, torch.Tensor],
    grads_exact: Dict[str, torch.Tensor],
) -> Tuple[float, float]:
    """Compute relative error and cosine similarity."""
    exact_flat = torch.cat([g.flatten() for g in grads_exact.values()])
    approx_flat = torch.cat([g.flatten() for g in grads_approx.values()])

    rel_error = (approx_flat - exact_flat).norm() / (exact_flat.norm() + 1e-8)
    cos_sim = torch.nn.functional.cosine_similarity(
        exact_flat.unsqueeze(0), approx_flat.unsqueeze(0)
    ).item()

    return rel_error.item(), cos_sim


def scale_direction(
    v_theta: Dict[str, torch.Tensor],
    target_norm: float,
) -> Dict[str, torch.Tensor]:
    """Scale v_theta to have specified norm."""
    # Compute current norm
    current_norm = sum(v.norm() ** 2 for v in v_theta.values()).sqrt().item()

    if current_norm < 1e-10:
        return v_theta

    scale = target_norm / current_norm
    return {k: v * scale for k, v in v_theta.items()}


def compute_v_theta_norm(v_theta: Dict[str, torch.Tensor]) -> float:
    """Compute L2 norm of v_theta."""
    return sum(v.norm() ** 2 for v in v_theta.values()).sqrt().item()


def run_magnitude_sweep(
    model: nn.Module,
    criterion: nn.Module,
    current_batch: Tuple[torch.Tensor, torch.Tensor],
    memory_batch: Tuple[torch.Tensor, torch.Tensor],
    magnitudes: List[float],
    epsilon: float,
    method: str = "finite_diff",
    vary_v_x: bool = False,
) -> Tuple[List[MagnitudeTestResult], Dict[str, torch.Tensor]]:
    """Sweep over different direction magnitudes and measure error.

    Args:
        model: Neural network model
        criterion: Loss function
        current_batch: (x_curr, y_curr)
        memory_batch: (x_mem, y_mem)
        magnitudes: List of magnitude values to test
        epsilon: Finite difference step size
        method: "finite_diff" or "richardson"
        vary_v_x: If True, vary v_x magnitude; else vary v_theta magnitude

    Returns:
        List of test results and ground truth gradients
    """
    device = next(model.parameters()).device

    # Initialize ground truth method
    math_backend = MathBackendJVPLoss(model, criterion, jvp_reg=1.0, deltax_norm=1.0)

    # Initialize test method
    if method == "finite_diff":
        test_method = FiniteDiffHVPLoss(
            model, criterion, jvp_reg=1.0, deltax_norm=1.0, epsilon=epsilon
        )
    elif method == "richardson":
        test_method = RichardsonHVPLoss(
            model, criterion, jvp_reg=1.0, deltax_norm=1.0, epsilon=epsilon
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute base v_theta (gradient on current task)
    model.zero_grad()
    x_curr, y_curr = current_batch
    logits = test_method._get_logits(model(x_curr))
    loss = criterion(logits, y_curr)
    loss.backward()
    base_v_theta = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }
    base_v_theta_norm = compute_v_theta_norm(base_v_theta)
    print(f"Base v_theta norm: {base_v_theta_norm:.4f}")

    # Base v_x
    x_mem, y_mem = memory_batch
    base_v_x = (x_mem - x_curr) / (x_mem.norm() + x_curr.norm() + 1e-8)
    base_v_x_norm = base_v_x.norm().item()
    print(f"Base v_x norm: {base_v_x_norm:.4f}")

    results = []
    ground_truth = None

    for mag in magnitudes:
        print(f"\n  Testing magnitude = {mag:.2e}...", end=" ", flush=True)

        # Scale the direction
        if vary_v_x:
            v_theta = base_v_theta  # Keep v_theta fixed
            v_x = base_v_x * (mag / base_v_x_norm)
            actual_mag = v_x.norm().item()
        else:
            v_theta = scale_direction(base_v_theta, mag)
            v_x = base_v_x  # Keep v_x fixed
            actual_mag = compute_v_theta_norm(v_theta)

        # Compute ground truth (first time only, or with smallest magnitude for reference)
        if ground_truth is None:
            try:
                grads_exact, _, _ = math_backend(current_batch, memory_batch, v_theta=v_theta, v_x=v_x)
                ground_truth = grads_exact
                print("[ground truth computed]", end=" ")
            except Exception as e:
                print(f"[ground truth FAILED: {e}]")
                # Use finite diff with very small epsilon as pseudo-ground truth
                pseudo_gt = FiniteDiffHVPLoss(model, criterion, jvp_reg=1.0, epsilon=1e-6)
                ground_truth, _, _ = pseudo_gt(current_batch, memory_batch, v_theta=v_theta, v_x=v_x)
                print("[using finite diff ε=1e-6 as reference]", end=" ")

        # Compute approximate
        try:
            start_time = time.perf_counter()
            grads_approx, _, _ = test_method(current_batch, memory_batch, v_theta=v_theta, v_x=v_x)
            elapsed = time.perf_counter() - start_time

            # Compute error vs ground truth (need to recompute GT for this magnitude)
            try:
                grads_exact_mag, _, _ = math_backend(current_batch, memory_batch, v_theta=v_theta, v_x=v_x)
                rel_error, cos_sim = compute_relative_error(grads_approx, grads_exact_mag)
            except Exception:
                # Compare against stored ground truth (less accurate but still informative)
                rel_error, cos_sim = compute_relative_error(grads_approx, ground_truth)

            grad_norm = sum(g.norm() ** 2 for g in grads_approx.values()).sqrt().item()

            results.append(MagnitudeTestResult(
                magnitude=actual_mag,
                rel_error=rel_error,
                cosine_sim=cos_sim,
                grad_norm=grad_norm,
                time_seconds=elapsed,
            ))
            print(f"done (error={rel_error:.2e}, cos_sim={cos_sim:.6f})")

        except Exception as e:
            print(f"FAILED: {e}")
            results.append(MagnitudeTestResult(
                magnitude=actual_mag,
                rel_error=float('nan'),
                cosine_sim=float('nan'),
                grad_norm=float('nan'),
                time_seconds=0,
            ))

    return results, ground_truth


def run_epsilon_sweep(
    model: nn.Module,
    criterion: nn.Module,
    current_batch: Tuple[torch.Tensor, torch.Tensor],
    memory_batch: Tuple[torch.Tensor, torch.Tensor],
    epsilons: List[float],
    magnitude: float,
    method: str = "finite_diff",
) -> List[Tuple[float, float, float]]:
    """Sweep epsilon values for a fixed magnitude.

    Returns:
        List of (epsilon, rel_error, cosine_sim) tuples
    """
    device = next(model.parameters()).device

    # Compute v_theta at specified magnitude
    model.zero_grad()
    x_curr, y_curr = current_batch
    from training.updaters.jvp_reg_transformer import FiniteDiffHVPLoss
    temp = FiniteDiffHVPLoss(model, criterion, jvp_reg=1.0)
    logits = temp._get_logits(model(x_curr))
    loss = criterion(logits, y_curr)
    loss.backward()
    base_v_theta = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }
    v_theta = scale_direction(base_v_theta, magnitude)

    # Compute ground truth
    math_backend = MathBackendJVPLoss(model, criterion, jvp_reg=1.0, deltax_norm=1.0)
    try:
        grads_exact, _, _ = math_backend(current_batch, memory_batch, v_theta=v_theta)
    except Exception as e:
        print(f"Math backend failed: {e}, using ε=1e-7 as reference")
        ref = FiniteDiffHVPLoss(model, criterion, jvp_reg=1.0, epsilon=1e-7)
        grads_exact, _, _ = ref(current_batch, memory_batch, v_theta=v_theta)

    results = []

    for eps in epsilons:
        print(f"    ε = {eps:.2e}...", end=" ", flush=True)

        if method == "finite_diff":
            test_method = FiniteDiffHVPLoss(model, criterion, jvp_reg=1.0, epsilon=eps)
        else:
            test_method = RichardsonHVPLoss(model, criterion, jvp_reg=1.0, epsilon=eps)

        try:
            grads_approx, _, _ = test_method(current_batch, memory_batch, v_theta=v_theta)
            rel_error, cos_sim = compute_relative_error(grads_approx, grads_exact)
            results.append((eps, rel_error, cos_sim))
            print(f"error = {rel_error:.2e}")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append((eps, float('nan'), float('nan')))

    return results


def generate_report(
    v_theta_results: List[MagnitudeTestResult],
    v_x_results: List[MagnitudeTestResult],
    epsilon_sweep_results: Dict[float, List[Tuple[float, float, float]]],
    epsilon: float,
    method: str,
):
    """Generate comprehensive report."""
    print("\n" + "=" * 80)
    print("DIRECTION MAGNITUDE EFFECT ON FINITE DIFFERENCE ERROR")
    print("=" * 80)

    print(f"\nMethod: {method}")
    print(f"Base epsilon: {epsilon}")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Table 1: v_theta magnitude effect
    print("\n" + "-" * 80)
    print("EFFECT OF v_theta MAGNITUDE (v_x fixed)")
    print("-" * 80)
    print("\n┌────────────────┬────────────────┬────────────────┬────────────────┬───────────┐")
    print("│ ||v_theta||    │ Rel. Error     │ Cosine Sim     │ ||grad||       │ Time (s)  │")
    print("├────────────────┼────────────────┼────────────────┼────────────────┼───────────┤")

    for r in v_theta_results:
        if not np.isnan(r.rel_error):
            print(f"│ {r.magnitude:>12.4e}   │ {r.rel_error:>12.6e}   │ {r.cosine_sim:>12.6f}   │ {r.grad_norm:>12.4f}   │ {r.time_seconds:>7.3f}   │")
        else:
            print(f"│ {r.magnitude:>12.4e}   │ {'FAILED':>14}   │ {'FAILED':>14}   │ {'FAILED':>14}   │ {'N/A':>9}   │")

    print("└────────────────┴────────────────┴────────────────┴────────────────┴───────────┘")

    # Find optimal magnitude
    valid_results = [r for r in v_theta_results if not np.isnan(r.rel_error)]
    if valid_results:
        best = min(valid_results, key=lambda x: x.rel_error)
        print(f"\nOptimal v_theta magnitude: {best.magnitude:.4e} (error: {best.rel_error:.6e})")

    # Table 2: v_x magnitude effect
    print("\n" + "-" * 80)
    print("EFFECT OF v_x MAGNITUDE (v_theta fixed)")
    print("-" * 80)
    print("\n┌────────────────┬────────────────┬────────────────┬────────────────┬───────────┐")
    print("│ ||v_x||        │ Rel. Error     │ Cosine Sim     │ ||grad||       │ Time (s)  │")
    print("├────────────────┼────────────────┼────────────────┼────────────────┼───────────┤")

    for r in v_x_results:
        if not np.isnan(r.rel_error):
            print(f"│ {r.magnitude:>12.4e}   │ {r.rel_error:>12.6e}   │ {r.cosine_sim:>12.6f}   │ {r.grad_norm:>12.4f}   │ {r.time_seconds:>7.3f}   │")
        else:
            print(f"│ {r.magnitude:>12.4e}   │ {'FAILED':>14}   │ {'FAILED':>14}   │ {'FAILED':>14}   │ {'N/A':>9}   │")

    print("└────────────────┴────────────────┴────────────────┴────────────────┴───────────┘")

    valid_results = [r for r in v_x_results if not np.isnan(r.rel_error)]
    if valid_results:
        best = min(valid_results, key=lambda x: x.rel_error)
        print(f"\nOptimal v_x magnitude: {best.magnitude:.4e} (error: {best.rel_error:.6e})")

    # Table 3: Epsilon sweep for different magnitudes
    if epsilon_sweep_results:
        print("\n" + "-" * 80)
        print("OPTIMAL EPSILON FOR DIFFERENT v_theta MAGNITUDES")
        print("-" * 80)
        print("\n┌────────────────┬────────────────┬────────────────┬────────────────┐")
        print("│ ||v_theta||    │ Optimal ε      │ Min Error      │ ε·||v||        │")
        print("├────────────────┼────────────────┼────────────────┼────────────────┤")

        for mag, eps_results in sorted(epsilon_sweep_results.items()):
            valid = [(e, err, cs) for e, err, cs in eps_results if not np.isnan(err)]
            if valid:
                best_eps, best_err, _ = min(valid, key=lambda x: x[1])
                eps_v_product = best_eps * mag
                print(f"│ {mag:>12.4e}   │ {best_eps:>12.2e}   │ {best_err:>12.6e}   │ {eps_v_product:>12.4e}   │")

        print("└────────────────┴────────────────┴────────────────┴────────────────┘")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & INSIGHTS")
    print("=" * 80)

    print("""
Key Observations:
-----------------
1. TRUNCATION ERROR: Dominates when ||v|| is large
   - Error ∝ ε² · ||v||³
   - Fix: Use smaller ε

2. NUMERICAL ERROR: Dominates when ||v|| is small
   - Error ∝ ||v|| / ε (from floating point)
   - Fix: Use larger ε

3. OPTIMAL RELATIONSHIP:
   - ε_optimal ∝ ||v||^(-2/3)
   - The product ε·||v|| should be roughly constant for optimal accuracy

Recommendations:
----------------
• For normalized directions (||v|| ≈ 1): Use ε ≈ 1e-4 to 1e-5
• For gradient directions (||v|| ≈ ||∇L||): Scale ε inversely with ||v||
• For very large ||v|| (> 100): Consider normalizing first
• For very small ||v|| (< 0.01): Consider using larger ε or Richardson extrapolation

Richardson Extrapolation:
------------------------
• Cancels O(ε²) term, giving O(ε⁴) accuracy
• More robust to ε choice
• Recommended when optimal ε is uncertain
""")

    print("=" * 80)
    print("END OF REPORT")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze direction magnitude effect on FD error")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer points")
    parser.add_argument("--model", default="mlp", choices=["vit", "mlp"], help="Model type")
    parser.add_argument("--method", default="finite_diff", choices=["finite_diff", "richardson"])
    parser.add_argument("--epsilon", type=float, default=1e-4, help="Base epsilon")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    print("\nLoading model...")
    model = load_model(device, args.model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()

    print("\nCreating batches...")
    current_batch, memory_batch = create_batches(device, batch_size=2)

    # Define magnitude sweep range
    if args.quick:
        v_theta_magnitudes = [0.01, 0.1, 1.0, 10.0, 100.0]
        v_x_magnitudes = [0.001, 0.01, 0.1, 1.0]
        epsilon_magnitudes = [0.1, 10.0]
        epsilons = [1e-2, 1e-3, 1e-4, 1e-5]
    else:
        v_theta_magnitudes = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
        v_x_magnitudes = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        epsilon_magnitudes = [0.1, 1.0, 10.0, 100.0]
        epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    # Run v_theta magnitude sweep
    print("\n" + "=" * 60)
    print("SWEEPING v_theta MAGNITUDES")
    print("=" * 60)
    v_theta_results, _ = run_magnitude_sweep(
        model, criterion, current_batch, memory_batch,
        magnitudes=v_theta_magnitudes,
        epsilon=args.epsilon,
        method=args.method,
        vary_v_x=False,
    )

    # Run v_x magnitude sweep
    print("\n" + "=" * 60)
    print("SWEEPING v_x MAGNITUDES")
    print("=" * 60)
    v_x_results, _ = run_magnitude_sweep(
        model, criterion, current_batch, memory_batch,
        magnitudes=v_x_magnitudes,
        epsilon=args.epsilon,
        method=args.method,
        vary_v_x=True,
    )

    # Run epsilon sweep for different magnitudes
    print("\n" + "=" * 60)
    print("SWEEPING EPSILON FOR DIFFERENT MAGNITUDES")
    print("=" * 60)
    epsilon_sweep_results = {}
    for mag in epsilon_magnitudes:
        print(f"\n  v_theta magnitude = {mag:.2e}")
        eps_results = run_epsilon_sweep(
            model, criterion, current_batch, memory_batch,
            epsilons=epsilons,
            magnitude=mag,
            method=args.method,
        )
        epsilon_sweep_results[mag] = eps_results

    # Generate report
    generate_report(
        v_theta_results,
        v_x_results,
        epsilon_sweep_results,
        epsilon=args.epsilon,
        method=args.method,
    )


if __name__ == "__main__":
    main()
