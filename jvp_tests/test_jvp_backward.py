"""Tests for backward-mode JVP regularization with transformers."""

import pytest
import torch
import torch.nn as nn

from training.updaters.jvp_reg_backward import (
    JVPRegularizedLossBackward,
    step_method_jvp_reg_backward,
)
from training.continual_learning import _is_transformer_model
from profilers import FLOPSProfiler


class SimpleCNN(nn.Module):
    """Simple CNN for testing non-transformer detection."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = nn.functional.adaptive_avg_pool2d(x, (8, 8))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SimpleTransformer(nn.Module):
    """Simple transformer-like model for testing detection."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(64, 32)
        self.attention = nn.MultiheadAttention(32, 4)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        # x: (batch, seq, features)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq, batch, features)
        x, _ = self.attention(x, x, x)
        x = x.mean(dim=0)  # (batch, features)
        return self.fc(x)


def test_is_transformer_model_cnn():
    """Test that CNN is not detected as transformer."""
    model = SimpleCNN()
    assert not _is_transformer_model(model)


def test_is_transformer_model_transformer():
    """Test that transformer is detected."""
    model = SimpleTransformer()
    assert _is_transformer_model(model)


def test_jvp_backward_basic():
    """Test basic JVP backward-mode computation."""
    device = "cpu"
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    jvp_loss = JVPRegularizedLossBackward(
        model=model,
        criterion=criterion,
        jvp_reg=0.001,
        deltax_norm=1.0,
    )

    # Create dummy batches
    x_curr = torch.randn(4, 3, 32, 32, device=device)
    y_curr = torch.randint(0, 10, (4,), device=device)
    x_mem = torch.randn(4, 3, 32, 32, device=device)
    y_mem = torch.randint(0, 10, (4,), device=device)

    grads, loss_curr, loss_mem = jvp_loss((x_curr, y_curr), (x_mem, y_mem))

    # Check we got gradients for all parameters
    assert len(grads) == len(list(model.named_parameters()))

    # Check gradients are not all zeros
    total_norm = sum(g.norm().item() for g in grads.values())
    assert total_norm > 0

    # Check losses are reasonable
    assert loss_curr.item() > 0
    assert loss_mem.item() > 0


def test_jvp_backward_with_transformer():
    """Test JVP backward-mode with transformer model."""
    device = "cpu"
    model = SimpleTransformer().to(device)
    criterion = nn.CrossEntropyLoss()

    jvp_loss = JVPRegularizedLossBackward(
        model=model,
        criterion=criterion,
        jvp_reg=0.001,
        deltax_norm=1.0,
    )

    # Create dummy batches (batch, seq, features)
    x_curr = torch.randn(4, 10, 64, device=device)
    y_curr = torch.randint(0, 10, (4,), device=device)
    x_mem = torch.randn(4, 10, 64, device=device)
    y_mem = torch.randint(0, 10, (4,), device=device)

    grads, loss_curr, loss_mem = jvp_loss((x_curr, y_curr), (x_mem, y_mem))

    # Check we got gradients for all parameters
    assert len(grads) == len(list(model.named_parameters()))

    # Check gradients are not all zeros
    total_norm = sum(g.norm().item() for g in grads.values())
    assert total_norm > 0


@pytest.mark.slow
def test_jvp_backward_with_huggingface_vit():
    """Test JVP backward-mode with HuggingFace ViT model."""
    pytest.importorskip("transformers")
    from transformers import ViTForImageClassification

    device = "cpu"
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=10,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)

    # Should be detected as transformer
    assert _is_transformer_model(model)

    criterion = nn.CrossEntropyLoss()

    jvp_loss = JVPRegularizedLossBackward(
        model=model,
        criterion=criterion,
        jvp_reg=0.001,
        deltax_norm=1.0,
    )

    # Create dummy batches
    x_curr = torch.randn(2, 3, 224, 224, device=device)
    y_curr = torch.randint(0, 10, (2,), device=device)
    x_mem = torch.randn(2, 3, 224, 224, device=device)
    y_mem = torch.randint(0, 10, (2,), device=device)

    grads, loss_curr, loss_mem = jvp_loss((x_curr, y_curr), (x_mem, y_mem))

    # Check we got gradients for all parameters
    assert len(grads) == len(list(model.named_parameters()))

    # Check gradients are not all zeros
    total_norm = sum(g.norm().item() for g in grads.values())
    assert total_norm > 0
