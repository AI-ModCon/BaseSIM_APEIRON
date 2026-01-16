"""
Minimal working example to reproduce JVP + Flash Attention incompatibility
with Vision Transformers from HuggingFace.
"""

import torch
from torch.func import grad, jvp, functional_call
from collections import OrderedDict
from transformers import ViTForImageClassification

def main():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load a small ViT model
    print("Loading ViT model...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=10,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)
    model.eval()

    # Create dummy data (batch_size=2, 3 channels, 224x224)
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    # Perturbation direction for JVP
    deltax = torch.randn_like(x) * 0.01

    # Get parameters as OrderedDict for functional API
    params = OrderedDict(model.named_parameters())

    # Ensure all params require grad
    for p in params.values():
        p.requires_grad_(True)

    criterion = torch.nn.CrossEntropyLoss()

    # Define loss function for functional API
    def loss_fn(p, x, y):
        # HuggingFace models return an object with .logits
        output = functional_call(model, p, (x,))
        logits = output.logits if hasattr(output, 'logits') else output
        return criterion(logits, y)

    # Test 1: Basic forward pass
    print("\n=== Test 1: Basic forward pass ===")
    try:
        with torch.no_grad():
            output = model(x)
            print(f"Forward pass OK. Output logits shape: {output.logits.shape}")
    except Exception as e:
        print(f"Forward pass FAILED: {type(e).__name__}: {e}")

    # Test 2: Gradient computation (backward mode AD)
    print("\n=== Test 2: Backward mode AD (grad) ===")
    try:
        grad_fn = grad(loss_fn, argnums=0)
        grads = grad_fn(params, x, y)
        print(f"Backward mode AD OK. Got gradients for {len(grads)} parameters")
    except Exception as e:
        print(f"Backward mode AD FAILED: {type(e).__name__}: {e}")

    # Test 3: JVP computation (forward mode AD) - This is expected to fail with flash attention
    print("\n=== Test 3: Forward mode AD (jvp) on input ===")
    try:
        def f_input(x):
            output = model(x)
            logits = output.logits if hasattr(output, 'logits') else output
            return criterion(logits, y)

        # JVP w.r.t. input
        primal_out, tangent_out = jvp(f_input, (x,), (deltax,))
        print(f"JVP on input OK. Primal: {primal_out.item():.4f}, Tangent: {tangent_out.item():.4f}")
    except Exception as e:
        print(f"JVP on input FAILED: {type(e).__name__}: {e}")

    # Test 4: JVP on parameters (forward mode AD)
    print("\n=== Test 4: Forward mode AD (jvp) on parameters ===")
    try:
        def f_params(p):
            output = functional_call(model, p, (x,))
            logits = output.logits if hasattr(output, 'logits') else output
            return criterion(logits, y)

        # Create tangent vectors for params
        tangents_params = OrderedDict((k, torch.randn_like(v) * 0.01) for k, v in params.items())

        primal_out, tangent_out = jvp(f_params, (params,), (tangents_params,))
        print(f"JVP on params OK. Primal: {primal_out.item():.4f}, Tangent: {tangent_out.item():.4f}")
    except Exception as e:
        print(f"JVP on params FAILED: {type(e).__name__}: {e}")

    # Test 5: The actual JVP computation from jvp_reg.py
    print("\n=== Test 5: Full JVP regularization computation ===")
    try:
        # This mimics what jvp_reg.py does
        def f(p, x_input):
            output = functional_call(model, p, (x_input,))
            logits = output.logits if hasattr(output, 'logits') else output
            return criterion(logits, y)

        # First compute grad_curr (backward mode - should work)
        grad_fn = grad(f, argnums=0)
        grad_curr = grad_fn(params, x)
        print("  Step 1 (grad_curr): OK")

        # Now the problematic part: JVP with grad_curr as tangent
        def jvp_func(p, tangents):
            # tangents is a tuple of (param_tangents, input_tangent)
            return jvp(f, (p, x), tangents)[1]

        tangents = (OrderedDict((k, grad_curr[k]) for k in params), deltax)

        # This takes gradient of the JVP output
        grad_jvp = grad(jvp_func)(params, tangents)
        print("  Step 2 (grad of jvp): OK")
        print(f"Full JVP regularization OK. Got gradients for {len(grad_jvp)} parameters")

    except Exception as e:
        print(f"Full JVP regularization FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
