"""Tests to verify ROCm installation and PyTorch GPU support."""


def test_torch_import():
    """Test that PyTorch can be imported."""
    import torch

    assert torch is not None, "PyTorch import failed"


def test_torchvision_import():
    """Test that torchvision can be imported."""
    import torchvision

    assert torchvision is not None, "torchvision import failed"


def test_rocm_available():
    """Test that ROCm/HIP is available through PyTorch."""
    import torch

    assert torch.cuda.is_available(), "CUDA/ROCm is not available"


def test_gpu_count():
    """Test that at least one GPU is detected."""
    import torch

    gpu_count = torch.cuda.device_count()
    assert gpu_count > 0, f"No GPUs detected, found {gpu_count}"


def test_gpu_properties():
    """Test that GPU properties can be queried."""
    import torch

    assert torch.cuda.is_available(), "CUDA/ROCm not available"
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        assert props.name is not None
        assert props.total_memory > 0


def test_tensor_on_gpu():
    """Test that tensors can be created and moved to GPU."""
    import torch

    assert torch.cuda.is_available(), "CUDA/ROCm not available"
    x = torch.randn(100, 100).cuda()
    assert x.is_cuda, "Tensor not on GPU"
    y = x @ x.T
    assert y.is_cuda, "Result tensor not on GPU"


def test_torch_rocm_build():
    """Test that PyTorch was built with ROCm support."""
    import torch

    hip_version = getattr(torch.version, "hip", None)
    assert hip_version is not None, "PyTorch not built with ROCm/HIP support"
