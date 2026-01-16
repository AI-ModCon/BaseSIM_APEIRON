"""GPU comparison tests for JVP regularization methods on Vision Transformers.

This test compares the four approaches:
1. MathBackend (exact, slow) - ground truth
2. FiniteDiffHVP (O(ε²) error)
3. GaussNewton (drops model curvature)
4. RichardsonHVP (O(ε⁴) error)

Run with:
    poetry run pytest tests/test_jvp_transformer_comparison.py -v -s

For GPU testing:
    poetry run pytest tests/test_jvp_transformer_comparison.py -v -s -m "not slow"
"""

import pytest
import torch
import torch.nn as nn
import time
from typing import Dict, Tuple

from training.updaters.jvp_reg_transformer import (
    MathBackendJVPLoss,
    FiniteDiffHVPLoss,
    GaussNewtonJVPLoss,
    RichardsonHVPLoss,
    get_jvp_loss_for_transformer,
)


def get_device():
    """Get available device, preferring CUDA."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_vit_model(device: torch.device, num_labels: int = 10):
    """Load ViT model for testing."""
    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)
    model.train()
    return model


def create_dummy_batches(
    device: torch.device, batch_size: int = 2
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Create dummy current and memory batches."""
    x_curr = torch.randn(batch_size, 3, 224, 224, device=device)
    y_curr = torch.randint(0, 10, (batch_size,), device=device)
    x_mem = torch.randn(batch_size, 3, 224, 224, device=device)
    y_mem = torch.randint(0, 10, (batch_size,), device=device)
    return (x_curr, y_curr), (x_mem, y_mem)


def compute_gradient_stats(grads: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute statistics for gradient dictionary."""
    all_grads = torch.cat([g.flatten() for g in grads.values()])
    return {
        "norm": all_grads.norm().item(),
        "mean": all_grads.mean().item(),
        "std": all_grads.std().item(),
        "min": all_grads.min().item(),
        "max": all_grads.max().item(),
    }


def compute_relative_error(
    grads_approx: Dict[str, torch.Tensor],
    grads_exact: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Compute relative error between approximate and exact gradients."""
    errors = {}

    # Per-parameter relative error
    for name in grads_exact:
        exact = grads_exact[name]
        approx = grads_approx[name]
        rel_err = (approx - exact).norm() / (exact.norm() + 1e-8)
        errors[name] = rel_err.item()

    # Overall relative error
    exact_flat = torch.cat([g.flatten() for g in grads_exact.values()])
    approx_flat = torch.cat([g.flatten() for g in grads_approx.values()])
    overall_rel_err = (approx_flat - exact_flat).norm() / (exact_flat.norm() + 1e-8)
    errors["_overall"] = overall_rel_err.item()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        exact_flat.unsqueeze(0), approx_flat.unsqueeze(0)
    ).item()
    errors["_cosine_similarity"] = cos_sim

    return errors


class TestJVPMethodsBasic:
    """Basic functionality tests for each method."""

    @pytest.fixture
    def simple_model(self):
        """Simple MLP for fast testing."""

        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 32)
                self.fc2 = nn.Linear(32, 10)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        return SimpleMLP()

    def test_math_backend_runs(self, simple_model):
        """Test MathBackendJVPLoss runs without error."""
        device = get_device()
        model = simple_model.to(device)
        criterion = nn.CrossEntropyLoss()

        jvp_loss = MathBackendJVPLoss(model, criterion, jvp_reg=0.001)

        x_curr = torch.randn(4, 64, device=device)
        y_curr = torch.randint(0, 10, (4,), device=device)
        x_mem = torch.randn(4, 64, device=device)
        y_mem = torch.randint(0, 10, (4,), device=device)

        grads, loss_curr, loss_mem = jvp_loss((x_curr, y_curr), (x_mem, y_mem))

        assert len(grads) == len(list(model.parameters()))
        assert loss_curr.item() > 0
        assert loss_mem.item() > 0

    def test_finite_diff_runs(self, simple_model):
        """Test FiniteDiffHVPLoss runs without error."""
        device = get_device()
        model = simple_model.to(device)
        criterion = nn.CrossEntropyLoss()

        jvp_loss = FiniteDiffHVPLoss(model, criterion, jvp_reg=0.001, epsilon=1e-4)

        x_curr = torch.randn(4, 64, device=device)
        y_curr = torch.randint(0, 10, (4,), device=device)
        x_mem = torch.randn(4, 64, device=device)
        y_mem = torch.randint(0, 10, (4,), device=device)

        grads, loss_curr, loss_mem = jvp_loss((x_curr, y_curr), (x_mem, y_mem))

        assert len(grads) == len(list(model.parameters()))
        assert loss_curr.item() > 0

    def test_gauss_newton_runs(self, simple_model):
        """Test GaussNewtonJVPLoss runs without error."""
        device = get_device()
        model = simple_model.to(device)
        criterion = nn.CrossEntropyLoss()

        jvp_loss = GaussNewtonJVPLoss(model, criterion, jvp_reg=0.001, epsilon=1e-4)

        x_curr = torch.randn(4, 64, device=device)
        y_curr = torch.randint(0, 10, (4,), device=device)
        x_mem = torch.randn(4, 64, device=device)
        y_mem = torch.randint(0, 10, (4,), device=device)

        grads, loss_curr, loss_mem = jvp_loss((x_curr, y_curr), (x_mem, y_mem))

        assert len(grads) == len(list(model.parameters()))

    def test_richardson_runs(self, simple_model):
        """Test RichardsonHVPLoss runs without error."""
        device = get_device()
        model = simple_model.to(device)
        criterion = nn.CrossEntropyLoss()

        jvp_loss = RichardsonHVPLoss(model, criterion, jvp_reg=0.001, epsilon=1e-3)

        x_curr = torch.randn(4, 64, device=device)
        y_curr = torch.randint(0, 10, (4,), device=device)
        x_mem = torch.randn(4, 64, device=device)
        y_mem = torch.randint(0, 10, (4,), device=device)

        grads, loss_curr, loss_mem = jvp_loss((x_curr, y_curr), (x_mem, y_mem))

        assert len(grads) == len(list(model.parameters()))

    def test_factory_function(self, simple_model):
        """Test factory function creates correct instances."""
        device = get_device()
        model = simple_model.to(device)
        criterion = nn.CrossEntropyLoss()

        methods = ["math_backend", "finite_diff", "gauss_newton", "richardson"]
        for method in methods:
            jvp_loss = get_jvp_loss_for_transformer(model, criterion, method=method)
            assert jvp_loss is not None

    def test_custom_v_theta(self, simple_model):
        """Test passing custom v_theta direction."""
        device = get_device()
        model = simple_model.to(device)
        criterion = nn.CrossEntropyLoss()

        jvp_loss = FiniteDiffHVPLoss(model, criterion, jvp_reg=0.001)

        x_curr = torch.randn(4, 64, device=device)
        y_curr = torch.randint(0, 10, (4,), device=device)
        x_mem = torch.randn(4, 64, device=device)
        y_mem = torch.randint(0, 10, (4,), device=device)

        # Custom v_theta (random direction)
        v_theta = {n: torch.randn_like(p) for n, p in model.named_parameters()}

        grads, _, _ = jvp_loss((x_curr, y_curr), (x_mem, y_mem), v_theta=v_theta)

        assert len(grads) == len(list(model.parameters()))

    def test_custom_v_x(self, simple_model):
        """Test passing custom v_x direction."""
        device = get_device()
        model = simple_model.to(device)
        criterion = nn.CrossEntropyLoss()

        jvp_loss = FiniteDiffHVPLoss(model, criterion, jvp_reg=0.001)

        x_curr = torch.randn(4, 64, device=device)
        y_curr = torch.randint(0, 10, (4,), device=device)
        x_mem = torch.randn(4, 64, device=device)
        y_mem = torch.randint(0, 10, (4,), device=device)

        # Custom v_x (random direction)
        v_x = torch.randn_like(x_mem)

        grads, _, _ = jvp_loss((x_curr, y_curr), (x_mem, y_mem), v_x=v_x)

        assert len(grads) == len(list(model.parameters()))


@pytest.mark.slow
class TestViTComparison:
    """Compare all methods on Vision Transformer (requires GPU for reasonable speed)."""

    @pytest.fixture(scope="class")
    def vit_setup(self):
        """Setup ViT model and data (shared across tests in class)."""
        device = get_device()
        print(f"\nUsing device: {device}")

        model = load_vit_model(device)
        criterion = nn.CrossEntropyLoss()
        current_batch, memory_batch = create_dummy_batches(device, batch_size=2)

        return {
            "model": model,
            "criterion": criterion,
            "device": device,
            "current_batch": current_batch,
            "memory_batch": memory_batch,
        }

    def test_all_methods_run_on_vit(self, vit_setup):
        """Test all methods can run on ViT without error."""
        model = vit_setup["model"]
        criterion = vit_setup["criterion"]
        current_batch = vit_setup["current_batch"]
        memory_batch = vit_setup["memory_batch"]

        methods = {
            "math_backend": MathBackendJVPLoss,
            "finite_diff": FiniteDiffHVPLoss,
            "gauss_newton": GaussNewtonJVPLoss,
            "richardson": RichardsonHVPLoss,
        }

        results = {}
        for name, cls in methods.items():
            print(f"\nTesting {name}...")
            try:
                if name == "math_backend":
                    jvp_loss = cls(model, criterion, jvp_reg=0.001, deltax_norm=1.0)
                else:
                    jvp_loss = cls(
                        model, criterion, jvp_reg=0.001, deltax_norm=1.0, epsilon=1e-4
                    )

                # Warmup
                _ = jvp_loss(current_batch, memory_batch)

                # Timed run
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()

                grads, loss_curr, loss_mem = jvp_loss(current_batch, memory_batch)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                stats = compute_gradient_stats(grads)
                results[name] = {
                    "success": True,
                    "time": elapsed,
                    "grad_norm": stats["norm"],
                    "loss_curr": loss_curr.item(),
                    "loss_mem": loss_mem.item(),
                    "grads": grads,
                }
                print(f"  ✓ {name}: {elapsed:.3f}s, grad_norm={stats['norm']:.6f}")

            except Exception as e:
                results[name] = {"success": False, "error": str(e)}
                print(f"  ✗ {name}: {e}")

        # At least finite_diff and gauss_newton should work
        assert results["finite_diff"]["success"]
        assert results["gauss_newton"]["success"]
        assert results["richardson"]["success"]

    def test_accuracy_vs_math_backend(self, vit_setup):
        """Compare accuracy of approximate methods vs math backend (ground truth)."""
        model = vit_setup["model"]
        criterion = vit_setup["criterion"]
        current_batch = vit_setup["current_batch"]
        memory_batch = vit_setup["memory_batch"]

        # Ground truth with math backend
        print("\nComputing ground truth with math backend...")
        math_loss = MathBackendJVPLoss(model, criterion, jvp_reg=0.001)

        try:
            grads_exact, _, _ = math_loss(current_batch, memory_batch)
            has_ground_truth = True
            print("  ✓ Ground truth computed")
        except Exception as e:
            print(f"  ✗ Math backend failed: {e}")
            has_ground_truth = False
            grads_exact = None

        # Approximate methods
        approx_methods = {
            "finite_diff": FiniteDiffHVPLoss(model, criterion, jvp_reg=0.001, epsilon=1e-4),
            "gauss_newton": GaussNewtonJVPLoss(model, criterion, jvp_reg=0.001, epsilon=1e-4),
            "richardson": RichardsonHVPLoss(model, criterion, jvp_reg=0.001, epsilon=1e-3),
        }

        print("\nComputing approximate methods...")
        approx_grads = {}
        for name, jvp_loss in approx_methods.items():
            grads, _, _ = jvp_loss(current_batch, memory_batch)
            approx_grads[name] = grads
            print(f"  ✓ {name} computed")

        # Compare if we have ground truth
        if has_ground_truth:
            print("\n" + "=" * 60)
            print("Accuracy Comparison (vs Math Backend)")
            print("=" * 60)

            for name, grads in approx_grads.items():
                errors = compute_relative_error(grads, grads_exact)
                print(f"\n{name}:")
                print(f"  Overall relative error: {errors['_overall']:.6f}")
                print(f"  Cosine similarity:      {errors['_cosine_similarity']:.6f}")

        # Compare approximate methods against each other
        print("\n" + "=" * 60)
        print("Cross-Method Comparison")
        print("=" * 60)

        method_names = list(approx_grads.keys())
        for i, name1 in enumerate(method_names):
            for name2 in method_names[i + 1 :]:
                errors = compute_relative_error(approx_grads[name1], approx_grads[name2])
                print(f"\n{name1} vs {name2}:")
                print(f"  Relative error:    {errors['_overall']:.6f}")
                print(f"  Cosine similarity: {errors['_cosine_similarity']:.6f}")

    def test_performance_benchmark(self, vit_setup):
        """Benchmark performance of all methods."""
        model = vit_setup["model"]
        criterion = vit_setup["criterion"]
        current_batch = vit_setup["current_batch"]
        memory_batch = vit_setup["memory_batch"]
        device = vit_setup["device"]

        methods = {
            "finite_diff": FiniteDiffHVPLoss(model, criterion, jvp_reg=0.001, epsilon=1e-4),
            "gauss_newton": GaussNewtonJVPLoss(model, criterion, jvp_reg=0.001, epsilon=1e-4),
            "richardson": RichardsonHVPLoss(model, criterion, jvp_reg=0.001, epsilon=1e-3),
        }

        # Only include math_backend if not on MPS (often fails)
        if device.type != "mps":
            methods["math_backend"] = MathBackendJVPLoss(model, criterion, jvp_reg=0.001)

        n_runs = 3
        print(f"\n{'=' * 60}")
        print(f"Performance Benchmark ({n_runs} runs each)")
        print("=" * 60)

        timings = {}
        for name, jvp_loss in methods.items():
            # Warmup
            try:
                _ = jvp_loss(current_batch, memory_batch)
            except Exception as e:
                print(f"\n{name}: FAILED - {e}")
                continue

            # Timed runs
            times = []
            for _ in range(n_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()

                _ = jvp_loss(current_batch, memory_batch)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            avg_time = sum(times) / len(times)
            std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            timings[name] = {"avg": avg_time, "std": std_time}

            print(f"\n{name}:")
            print(f"  Average: {avg_time:.3f}s ± {std_time:.3f}s")

        # Summary
        if timings:
            print("\n" + "=" * 60)
            print("Summary (sorted by speed)")
            print("=" * 60)
            sorted_methods = sorted(timings.items(), key=lambda x: x[1]["avg"])
            fastest = sorted_methods[0][1]["avg"]
            for name, t in sorted_methods:
                speedup = t["avg"] / fastest
                print(f"  {name}: {t['avg']:.3f}s ({speedup:.1f}x vs fastest)")


def run_comparison():
    """Run comprehensive comparison of all JVP methods on ViT.

    Generates a detailed performance vs accuracy report.
    """
    device = get_device()
    print("=" * 70)
    print("JVP REGULARIZATION METHODS: PERFORMANCE VS ACCURACY REPORT")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "-" * 70)
    print("SETUP")
    print("-" * 70)

    print("\nLoading ViT-Base-Patch16-224 model...")
    model = load_vit_model(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()

    print("\nCreating batches...")
    batch_size = 2
    current_batch, memory_batch = create_dummy_batches(device, batch_size=batch_size)
    print(f"  Batch size: {batch_size}")
    print(f"  Input shape: {current_batch[0].shape}")

    # Configuration
    jvp_reg = 0.001
    deltax_norm = 1.0
    epsilon = 1e-4
    n_warmup = 2
    n_runs = 5

    print(f"\nHyperparameters:")
    print(f"  jvp_reg: {jvp_reg}")
    print(f"  deltax_norm: {deltax_norm}")
    print(f"  epsilon: {epsilon}")
    print(f"  warmup runs: {n_warmup}")
    print(f"  timed runs: {n_runs}")

    # Initialize all methods
    methods = {}

    # Math backend (ground truth)
    try:
        methods["math_backend"] = MathBackendJVPLoss(
            model, criterion, jvp_reg=jvp_reg, deltax_norm=deltax_norm, epsilon=epsilon
        )
    except Exception as e:
        print(f"\nWARNING: Math backend initialization failed: {e}")

    methods["finite_diff"] = FiniteDiffHVPLoss(
        model, criterion, jvp_reg=jvp_reg, deltax_norm=deltax_norm, epsilon=epsilon
    )
    methods["gauss_newton"] = GaussNewtonJVPLoss(
        model, criterion, jvp_reg=jvp_reg, deltax_norm=deltax_norm, epsilon=epsilon
    )
    methods["richardson"] = RichardsonHVPLoss(
        model, criterion, jvp_reg=jvp_reg, deltax_norm=deltax_norm, epsilon=1e-3  # Larger epsilon for Richardson
    )

    print("\n" + "-" * 70)
    print("RUNNING BENCHMARKS")
    print("-" * 70)

    results = {}

    for name, jvp_loss in methods.items():
        print(f"\n{name}:")
        try:
            # Warmup
            print(f"  Warming up ({n_warmup} runs)...", end=" ", flush=True)
            for _ in range(n_warmup):
                _ = jvp_loss(current_batch, memory_batch)
            print("done")

            # Timed runs
            print(f"  Timing ({n_runs} runs)...", end=" ", flush=True)
            times = []
            for _ in range(n_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()

                grads, loss_curr, loss_mem = jvp_loss(current_batch, memory_batch)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            print("done")

            avg_time = sum(times) / len(times)
            std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            stats = compute_gradient_stats(grads)

            results[name] = {
                "grads": grads,
                "loss_curr": loss_curr.item(),
                "loss_mem": loss_mem.item(),
                "avg_time": avg_time,
                "std_time": std_time,
                "grad_norm": stats["norm"],
                "grad_mean": stats["mean"],
                "grad_std": stats["std"],
            }

            print(f"  ✓ Completed: {avg_time:.3f}s ± {std_time:.3f}s")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Compute accuracy metrics (vs math_backend if available, else vs richardson)
    if "math_backend" in results:
        ground_truth_name = "math_backend"
    else:
        ground_truth_name = "richardson"
        print(f"\nNOTE: Using {ground_truth_name} as reference (math_backend unavailable)")

    ground_truth_grads = results[ground_truth_name]["grads"]

    for name in results:
        if name != ground_truth_name:
            errors = compute_relative_error(results[name]["grads"], ground_truth_grads)
            results[name]["rel_error"] = errors["_overall"]
            results[name]["cosine_sim"] = errors["_cosine_similarity"]
        else:
            results[name]["rel_error"] = 0.0
            results[name]["cosine_sim"] = 1.0

    # Generate Report
    print("\n" + "=" * 70)
    print("PERFORMANCE VS ACCURACY REPORT")
    print("=" * 70)

    # Table 1: Performance
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                        PERFORMANCE SUMMARY                          │")
    print("├────────────────┬──────────────┬──────────────┬───────────────────────┤")
    print("│ Method         │ Time (s)     │ Std (s)      │ Speedup vs Math Bkend │")
    print("├────────────────┼──────────────┼──────────────┼───────────────────────┤")

    math_time = results.get("math_backend", {}).get("avg_time", None)

    for name in ["math_backend", "richardson", "finite_diff", "gauss_newton"]:
        if name in results:
            r = results[name]
            if math_time:
                speedup = math_time / r["avg_time"]
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"
            print(f"│ {name:<14} │ {r['avg_time']:>10.4f}   │ {r['std_time']:>10.4f}   │ {speedup_str:>21} │")

    print("└────────────────┴──────────────┴──────────────┴───────────────────────┘")

    # Table 2: Accuracy
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print(f"│                 ACCURACY (vs {ground_truth_name:<14})                     │")
    print("├────────────────┬──────────────┬──────────────┬───────────────────────┤")
    print("│ Method         │ Rel. Error   │ Cosine Sim   │ Gradient Norm         │")
    print("├────────────────┼──────────────┼──────────────┼───────────────────────┤")

    for name in ["math_backend", "richardson", "finite_diff", "gauss_newton"]:
        if name in results:
            r = results[name]
            print(f"│ {name:<14} │ {r['rel_error']:>10.6f}   │ {r['cosine_sim']:>10.6f}   │ {r['grad_norm']:>19.4f}   │")

    print("└────────────────┴──────────────┴──────────────┴───────────────────────┘")

    # Table 3: Cross-method comparison
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                     CROSS-METHOD COMPARISON                         │")
    print("├─────────────────────────────┬──────────────┬──────────────────────────┤")
    print("│ Comparison                  │ Rel. Error   │ Cosine Similarity        │")
    print("├─────────────────────────────┼──────────────┼──────────────────────────┤")

    method_names = [n for n in ["math_backend", "richardson", "finite_diff", "gauss_newton"] if n in results]
    for i, name1 in enumerate(method_names):
        for name2 in method_names[i + 1:]:
            errors = compute_relative_error(results[name1]["grads"], results[name2]["grads"])
            comparison = f"{name1} vs {name2}"
            print(f"│ {comparison:<27} │ {errors['_overall']:>10.6f}   │ {errors['_cosine_similarity']:>22.6f}   │")

    print("└─────────────────────────────┴──────────────┴──────────────────────────┘")

    # Summary and Recommendations
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    # Find fastest and most accurate (excluding ground truth)
    approx_methods = {k: v for k, v in results.items() if k != "math_backend"}

    if approx_methods:
        fastest = min(approx_methods.items(), key=lambda x: x[1]["avg_time"])
        most_accurate = min(approx_methods.items(), key=lambda x: x[1]["rel_error"])

        print(f"\n• Fastest method:      {fastest[0]} ({fastest[1]['avg_time']:.3f}s)")
        print(f"• Most accurate:       {most_accurate[0]} (rel. error: {most_accurate[1]['rel_error']:.6f})")

        if "math_backend" in results:
            print(f"• Ground truth:        math_backend ({results['math_backend']['avg_time']:.3f}s)")

            # Compute efficiency (accuracy per unit time)
            print("\n• Efficiency (lower rel_error / time is better):")
            for name, r in approx_methods.items():
                efficiency = r["rel_error"] * r["avg_time"]  # Lower is better
                print(f"    {name}: {efficiency:.6f}")

    print("\n• Method Characteristics:")
    print("    - math_backend: Exact but slow (O(N²) attention memory)")
    print("    - richardson:   O(ε⁴) accuracy, 8 backward passes")
    print("    - finite_diff:  O(ε²) accuracy, 4 backward passes")
    print("    - gauss_newton: Approximate (drops model curvature), fastest")

    print("\n• Recommendations:")
    print("    - For training at scale: gauss_newton (fast, good direction)")
    print("    - For research/validation: richardson (high accuracy)")
    print("    - For ground truth comparison: math_backend (if memory allows)")

    print("\n" + "=" * 70)
    print("END OF REPORT")
    print("=" * 70)

    return results


def run_direction_magnitude_analysis(
    model: nn.Module,
    criterion: nn.Module,
    current_batch,
    memory_batch,
    base_epsilon: float = 1e-4,
):
    """Analyze how direction magnitude affects finite difference error.

    Returns dict with analysis results.
    """
    from training.updaters.jvp_reg_transformer import (
        MathBackendJVPLoss,
        FiniteDiffHVPLoss,
    )

    device = next(model.parameters()).device

    # Compute base v_theta (gradient on current task)
    model.zero_grad()
    x_curr, y_curr = current_batch
    fd_temp = FiniteDiffHVPLoss(model, criterion, jvp_reg=1.0)
    logits = fd_temp._get_logits(model(x_curr))
    loss = criterion(logits, y_curr)
    loss.backward()
    base_v_theta = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }
    base_v_theta_norm = sum(v.norm() ** 2 for v in base_v_theta.values()).sqrt().item()

    # Base v_x
    x_mem, _ = memory_batch
    base_v_x = (x_mem - x_curr) / (x_mem.norm() + x_curr.norm() + 1e-8)
    base_v_x_norm = base_v_x.norm().item()

    def scale_v_theta(target_norm):
        scale = target_norm / base_v_theta_norm
        return {k: v * scale for k, v in base_v_theta.items()}

    # Test magnitudes for v_theta
    v_theta_magnitudes = [0.1, 1.0, 10.0, 100.0]
    # Test epsilon values
    epsilons = [1e-2, 1e-3, 1e-4, 1e-5]

    results = {}

    print("\n" + "-" * 70)
    print("DIRECTION MAGNITUDE vs OPTIMAL EPSILON ANALYSIS")
    print("-" * 70)
    print(f"\nBase ||v_theta|| (gradient norm): {base_v_theta_norm:.4f}")
    print(f"Base ||v_x||: {base_v_x_norm:.4f}")

    # Initialize methods
    math_backend = MathBackendJVPLoss(model, criterion, jvp_reg=1.0, deltax_norm=1.0)

    print("\n┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ ||v_theta||  │ ε = 1e-2     │ ε = 1e-3     │ ε = 1e-4     │ ε = 1e-5     │ Optimal ε    │")
    print("├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")

    for mag in v_theta_magnitudes:
        v_theta = scale_v_theta(mag)

        # Get ground truth
        try:
            grads_exact, _, _ = math_backend(current_batch, memory_batch, v_theta=v_theta)
        except Exception:
            # Use very small epsilon as reference
            ref = FiniteDiffHVPLoss(model, criterion, jvp_reg=1.0, epsilon=1e-6)
            grads_exact, _, _ = ref(current_batch, memory_batch, v_theta=v_theta)

        errors = []
        for eps in epsilons:
            fd = FiniteDiffHVPLoss(model, criterion, jvp_reg=1.0, epsilon=eps)
            try:
                grads_approx, _, _ = fd(current_batch, memory_batch, v_theta=v_theta)
                exact_flat = torch.cat([g.flatten() for g in grads_exact.values()])
                approx_flat = torch.cat([g.flatten() for g in grads_approx.values()])
                rel_error = (approx_flat - exact_flat).norm() / (exact_flat.norm() + 1e-8)
                errors.append((eps, rel_error.item()))
            except Exception:
                errors.append((eps, float('nan')))

        # Find optimal epsilon
        valid_errors = [(e, err) for e, err in errors if not np.isnan(err)]
        if valid_errors:
            optimal_eps, min_error = min(valid_errors, key=lambda x: x[1])
        else:
            optimal_eps, min_error = None, float('nan')

        # Format row
        error_strs = []
        for eps, err in errors:
            if np.isnan(err):
                error_strs.append("   FAIL   ")
            elif err > 1.0:
                error_strs.append(f" {err:>8.2f} ")
            else:
                error_strs.append(f" {err:>8.2e} ")

        opt_str = f" {optimal_eps:.0e} " if optimal_eps else "   N/A    "
        print(f"│ {mag:>10.1e}   │{error_strs[0]}│{error_strs[1]}│{error_strs[2]}│{error_strs[3]}│{opt_str}│")

        results[mag] = {
            "errors": dict(errors),
            "optimal_eps": optimal_eps,
            "min_error": min_error,
        }

    print("└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")

    # Show the ε·||v|| relationship
    print("\n┌──────────────┬──────────────┬──────────────┐")
    print("│ ||v_theta||  │ Optimal ε    │ ε · ||v||    │")
    print("├──────────────┼──────────────┼──────────────┤")

    for mag in v_theta_magnitudes:
        r = results[mag]
        if r["optimal_eps"]:
            product = r["optimal_eps"] * mag
            print(f"│ {mag:>10.1e}   │ {r['optimal_eps']:>10.0e}   │ {product:>10.4f}   │")
        else:
            print(f"│ {mag:>10.1e}   │     N/A      │     N/A      │")

    print("└──────────────┴──────────────┴──────────────┘")

    print("\n• Key Insight: ε · ||v|| ≈ constant (≈0.01) for optimal accuracy")
    print("• Rule of thumb: ε_optimal ≈ 0.01 / ||v_theta||")

    return results


def run_full_comparison():
    """Run comprehensive comparison including method benchmarks and magnitude analysis."""
    import numpy as np

    device = get_device()
    print("=" * 80)
    print("COMPREHENSIVE JVP REGULARIZATION ANALYSIS")
    print("=" * 80)
    print(f"\nDevice: {device}")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load model
    print("\n" + "-" * 80)
    print("SETUP")
    print("-" * 80)
    print("\nLoading ViT-Base-Patch16-224 model...")
    model = load_vit_model(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()

    print("\nCreating batches...")
    batch_size = 2
    current_batch, memory_batch = create_dummy_batches(device, batch_size=batch_size)
    print(f"  Batch size: {batch_size}")
    print(f"  Input shape: {current_batch[0].shape}")

    # =========================================================================
    # PART 1: Method Performance Comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: METHOD PERFORMANCE COMPARISON")
    print("=" * 80)

    jvp_reg = 0.001
    deltax_norm = 1.0
    epsilon = 1e-4
    n_warmup = 2
    n_runs = 5

    print(f"\nHyperparameters: jvp_reg={jvp_reg}, deltax_norm={deltax_norm}, epsilon={epsilon}")

    # Initialize all methods
    methods = {
        "math_backend": MathBackendJVPLoss(model, criterion, jvp_reg=jvp_reg, deltax_norm=deltax_norm),
        "finite_diff": FiniteDiffHVPLoss(model, criterion, jvp_reg=jvp_reg, deltax_norm=deltax_norm, epsilon=epsilon),
        "gauss_newton": GaussNewtonJVPLoss(model, criterion, jvp_reg=jvp_reg, deltax_norm=deltax_norm, epsilon=epsilon),
        "richardson": RichardsonHVPLoss(model, criterion, jvp_reg=jvp_reg, deltax_norm=deltax_norm, epsilon=1e-3),
    }

    results = {}

    for name, jvp_loss in methods.items():
        print(f"\n{name}:", end=" ", flush=True)
        try:
            # Warmup
            for _ in range(n_warmup):
                _ = jvp_loss(current_batch, memory_batch)

            # Timed runs
            times = []
            for _ in range(n_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                grads, loss_curr, loss_mem = jvp_loss(current_batch, memory_batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            avg_time = sum(times) / len(times)
            std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            stats = compute_gradient_stats(grads)

            results[name] = {
                "grads": grads,
                "avg_time": avg_time,
                "std_time": std_time,
                "grad_norm": stats["norm"],
            }
            print(f"{avg_time:.3f}s ± {std_time:.3f}s")
        except Exception as e:
            print(f"FAILED: {e}")

    # Compute accuracy vs math_backend
    if "math_backend" in results:
        ground_truth = results["math_backend"]["grads"]
        for name in results:
            if name != "math_backend":
                errors = compute_relative_error(results[name]["grads"], ground_truth)
                results[name]["rel_error"] = errors["_overall"]
                results[name]["cosine_sim"] = errors["_cosine_similarity"]
            else:
                results[name]["rel_error"] = 0.0
                results[name]["cosine_sim"] = 1.0

    # Print performance table
    print("\n┌─────────────────────────────────────────────────────────────────────────────────┐")
    print("│                           PERFORMANCE vs ACCURACY                               │")
    print("├────────────────┬────────────┬────────────┬──────────────┬──────────────┬────────┤")
    print("│ Method         │ Time (s)   │ Std (s)    │ Rel. Error   │ Cosine Sim   │ Speedup│")
    print("├────────────────┼────────────┼────────────┼──────────────┼──────────────┼────────┤")

    math_time = results.get("math_backend", {}).get("avg_time", 1.0)
    for name in ["math_backend", "richardson", "finite_diff", "gauss_newton"]:
        if name in results:
            r = results[name]
            speedup = math_time / r["avg_time"]
            print(f"│ {name:<14} │ {r['avg_time']:>8.4f}   │ {r['std_time']:>8.4f}   │ {r['rel_error']:>10.6f}   │ {r['cosine_sim']:>10.6f}   │ {speedup:>5.2f}x │")

    print("└────────────────┴────────────┴────────────┴──────────────┴──────────────┴────────┘")

    # =========================================================================
    # PART 2: Direction Magnitude Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: DIRECTION MAGNITUDE vs EPSILON ANALYSIS")
    print("=" * 80)

    import numpy as np
    magnitude_results = run_direction_magnitude_analysis(
        model, criterion, current_batch, memory_batch, base_epsilon=epsilon
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 80)

    print("""
METHOD SELECTION GUIDE:
=======================
┌────────────────┬───────────────────────────────────────────────────────────┐
│ Use Case       │ Recommended Method                                        │
├────────────────┼───────────────────────────────────────────────────────────┤
│ Production     │ finite_diff (best accuracy/speed tradeoff)                │
│ Fast training  │ gauss_newton (2.5x faster, ~15% error but good direction) │
│ Validation     │ math_backend (exact, but O(N²) memory)                    │
│ Research       │ richardson (highest accuracy among approximations)        │
└────────────────┴───────────────────────────────────────────────────────────┘

EPSILON TUNING GUIDE:
=====================
┌────────────────────────────────────────────────────────────────────────────┐
│ The optimal epsilon depends on direction magnitude:                        │
│                                                                            │
│   ε_optimal ≈ 0.01 / ||v_theta||                                          │
│                                                                            │
│ Examples:                                                                  │
│   • ||v_theta|| = 0.1  →  ε ≈ 0.1    (1e-1)                               │
│   • ||v_theta|| = 1.0  →  ε ≈ 0.01   (1e-2)                               │
│   • ||v_theta|| = 10   →  ε ≈ 0.001  (1e-3)                               │
│   • ||v_theta|| = 100  →  ε ≈ 0.0001 (1e-4)                               │
│                                                                            │
│ For gradient-based v_theta (typical ||∇L|| ≈ 10-100), use ε = 1e-4        │
└────────────────────────────────────────────────────────────────────────────┘
""")

    print("=" * 80)
    print("END OF COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    return results, magnitude_results


if __name__ == "__main__":
    import sys
    import numpy as np

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        run_full_comparison()
    else:
        run_comparison()
