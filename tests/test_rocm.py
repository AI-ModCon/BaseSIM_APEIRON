"""Tests to verify ROCm/CUDA installation and PyTorch GPU support."""

import pytest
import torch

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No CUDA/ROCm GPU available"
)
requires_rocm = pytest.mark.skipif(
    getattr(torch.version, "hip", None) is None, reason="ROCm/HIP not available"
)


@pytest.fixture
def gpu_device():
    """Return the GPU device string."""
    return "cuda"


def test_torch_import():
    """Test that PyTorch can be imported."""
    assert torch is not None, "PyTorch import failed"


def test_torchvision_import():
    """Test that torchvision can be imported."""
    import torchvision

    assert torchvision is not None, "torchvision import failed"


@requires_gpu
def test_rocm_available():
    """Test that CUDA/ROCm is available through PyTorch."""
    assert torch.cuda.is_available(), "CUDA/ROCm is not available"


@requires_gpu
def test_gpu_count():
    """Test that at least one GPU is detected."""
    gpu_count = torch.cuda.device_count()
    assert gpu_count > 0, f"No GPUs detected, found {gpu_count}"


@requires_gpu
def test_gpu_properties():
    """Test that GPU properties can be queried."""
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        assert props.name is not None
        assert props.total_memory > 0


@requires_gpu
def test_tensor_on_gpu():
    """Test that tensors can be created and moved to GPU."""
    x = torch.randn(100, 100).cuda()
    assert x.is_cuda, "Tensor not on GPU"
    y = x @ x.T
    assert y.is_cuda, "Result tensor not on GPU"


@requires_gpu
def test_tensor_to_device(gpu_device):
    """Test that tensor.to(device) works (used by harness and JVP tests)."""
    x = torch.randn(32, 10)
    x_gpu = x.to(gpu_device)
    assert x_gpu.is_cuda, "tensor.to(device) failed"
    assert x_gpu.shape == x.shape, "Shape changed after .to()"


@requires_gpu
def test_autograd_on_gpu(gpu_device):
    """Test that backward pass and gradient computation work on GPU."""
    x = torch.randn(16, 4, device=gpu_device, requires_grad=True)
    w = torch.randn(4, 2, device=gpu_device, requires_grad=True)
    loss = (x @ w).sum()
    loss.backward()
    assert w.grad is not None, "Gradients not computed"
    assert w.grad.is_cuda, "Gradients not on GPU"
    assert w.grad.shape == w.shape, "Gradient shape mismatch"


@requires_gpu
def test_optimizer_step_on_gpu(gpu_device):
    """Test that optimizer zero_grad/step work on GPU parameters."""
    param = torch.nn.Parameter(torch.randn(4, 4, device=gpu_device))
    optimizer = torch.optim.SGD([param], lr=0.1)

    initial = param.clone().detach()
    loss = param.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.equal(param, initial), "Weights not updated after step"


@requires_gpu
def test_no_grad_context(gpu_device):
    """Test that torch.no_grad() inference mode works on GPU."""
    w = torch.randn(4, 4, device=gpu_device, requires_grad=True)
    with torch.no_grad():
        y = w @ w.T
    assert y.is_cuda, "Output not on GPU"
    assert not y.requires_grad, "Output should not require grad inside no_grad"


@requires_gpu
def test_tensor_clone_detach(gpu_device):
    """Test that clone/detach work on GPU tensors (used for weight snapshots)."""
    x = torch.randn(4, 4, device=gpu_device, requires_grad=True)
    y = x.clone().detach()
    assert y.is_cuda, "Cloned tensor not on GPU"
    assert not y.requires_grad, "Detached tensor should not require grad"
    assert torch.equal(x, y), "Cloned tensor values differ"


@requires_gpu
def test_torch_comparison_ops(gpu_device):
    """Test torch.equal, torch.allclose, and torch.isnan on GPU tensors."""
    a = torch.tensor([1.0, 2.0, 3.0], device=gpu_device)
    b = a.clone()

    assert torch.equal(a, b), "torch.equal failed on identical GPU tensors"
    assert torch.allclose(a, b, atol=1e-6), "torch.allclose failed"

    c = torch.tensor([1.0, float("nan"), 3.0], device=gpu_device)
    nan_mask = torch.isnan(c)
    assert nan_mask[1].item(), "torch.isnan failed to detect NaN on GPU"
    assert not nan_mask[0].item(), "torch.isnan false positive on GPU"


@requires_rocm
def test_torch_rocm_build():
    """Test that PyTorch was built with ROCm support."""
    hip_version = getattr(torch.version, "hip", None)
    assert hip_version is not None, "PyTorch not built with ROCm/HIP support"
