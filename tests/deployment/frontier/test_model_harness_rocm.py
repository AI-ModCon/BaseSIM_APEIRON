"""Tests to verify MNIST model harness works correctly with ROCm."""

import pytest
import torch

from config.configuration import (
    Config,
    ModelCfg,
    DataCfg,
    TrainCfg,
    ContinualLearningCfg,
    DriftDetectionCfg,
)
from examples.mnist.model import MNIST_CNN


@pytest.fixture
def rocm_config():
    """Create a config for ROCm/GPU testing."""
    return Config(
        model=ModelCfg(name="mnist_cnn", pretrained_path=""),
        data=DataCfg(name="mnist", path="./data"),
        train=TrainCfg(batch_size=32, num_workers=0, init_lr=0.001),
        continual_learning=ContinualLearningCfg(),
        drift_detection=DriftDetectionCfg(),
        seed=42,
        device="cuda",
        multi_gpu=False,
    )


@pytest.fixture
def harness(rocm_config):
    """Create MNIST harness and initialize data stream."""
    harness = MNIST_CNN(rocm_config)
    harness.update_data_stream()
    return harness


class TestModelLoading:
    """Tests for model loading and GPU placement."""

    def test_model_on_gpu(self, harness):
        """Test that model is moved to GPU."""
        device = next(harness.model.parameters()).device
        assert device.type == "cuda", f"Model not on GPU, found {device}"

    def test_model_device_matches_config(self, harness):
        """Test that model device matches config device."""
        device = next(harness.model.parameters()).device
        assert str(device).startswith("cuda")


class TestDataLoader:
    """Tests for data loader functionality."""

    def test_data_loaders_created(self, harness):
        """Test that data loaders are created after update_data_stream."""
        train_loader, val_loader = harness.get_cur_data_loaders()
        assert train_loader is not None, "Train loader is None"
        assert val_loader is not None, "Val loader is None"

    def test_data_loader_batch_shape(self, harness):
        """Test that data loader produces correct batch shapes."""
        train_loader, _ = harness.get_cur_data_loaders()
        batch = next(iter(train_loader))
        x, y = batch
        assert x.dim() == 3, f"Expected 3D input (B, H, W), got {x.dim()}D"
        assert y.dim() == 1, f"Expected 1D labels, got {y.dim()}D"
        assert x.shape[0] == y.shape[0], "Batch size mismatch between x and y"

    def test_data_moves_to_gpu(self, harness):
        """Test that data can be moved to GPU."""
        train_loader, _ = harness.get_cur_data_loaders()
        batch = next(iter(train_loader))
        x, y = batch
        x_gpu = x.to(harness.cfg.device)
        y_gpu = y.to(harness.cfg.device)
        assert x_gpu.is_cuda, "Input tensor not on GPU"
        assert y_gpu.is_cuda, "Label tensor not on GPU"


class TestForwardPass:
    """Tests for model forward pass."""

    def test_forward_pass_runs(self, harness):
        """Test that forward pass executes without error."""
        train_loader, _ = harness.get_cur_data_loaders()
        batch = next(iter(train_loader))
        x, y = batch
        x = x.to(harness.cfg.device)

        harness.model.eval()
        with torch.no_grad():
            output = harness.model(x)

        assert output is not None, "Forward pass returned None"

    def test_forward_pass_output_shape(self, harness):
        """Test that forward pass produces correct output shape."""
        train_loader, _ = harness.get_cur_data_loaders()
        batch = next(iter(train_loader))
        x, y = batch
        x = x.to(harness.cfg.device)

        harness.model.eval()
        with torch.no_grad():
            output = harness.model(x)

        assert output.shape[0] == x.shape[0], "Batch size mismatch"
        assert output.shape[1] == 10, f"Expected 10 classes, got {output.shape[1]}"

    def test_forward_pass_output_on_gpu(self, harness):
        """Test that forward pass output is on GPU."""
        train_loader, _ = harness.get_cur_data_loaders()
        batch = next(iter(train_loader))
        x, _ = batch
        x = x.to(harness.cfg.device)

        harness.model.eval()
        with torch.no_grad():
            output = harness.model(x)

        assert output.is_cuda, "Output tensor not on GPU"


class TestEval:
    """Tests for harness eval method."""

    def test_eval_runs(self, harness):
        """Test that eval method executes without error."""
        metrics = harness.eval()
        assert metrics is not None, "Eval returned None"

    def test_eval_returns_metrics(self, harness):
        """Test that eval returns expected number of metrics."""
        metrics = harness.eval()
        assert len(metrics) == len(harness.eval_metrics), (
            f"Expected {len(harness.eval_metrics)} metrics, got {len(metrics)}"
        )

    def test_eval_metrics_are_valid(self, harness):
        """Test that eval metrics are valid floats."""
        metrics = harness.eval()
        for i, metric in enumerate(metrics):
            assert isinstance(metric, float), f"Metric {i} is not a float"
            assert not torch.isnan(torch.tensor(metric)), f"Metric {i} is NaN"


class TestTrainingStep:
    """Tests for a single training step."""

    def test_training_step(self, harness):
        """Test that a single training step executes without error."""
        train_loader, _ = harness.get_cur_data_loaders()
        batch = next(iter(train_loader))
        x, y = batch
        x = x.to(harness.cfg.device)
        y = y.to(harness.cfg.device)

        harness.model.train()
        optimizer = harness.get_optmizer()
        criterion = harness.get_criterion()

        optimizer.zero_grad()
        output = harness.model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0, "Loss should be positive"

    def test_gradients_computed(self, harness):
        """Test that gradients are computed during backward pass."""
        train_loader, _ = harness.get_cur_data_loaders()
        batch = next(iter(train_loader))
        x, y = batch
        x = x.to(harness.cfg.device)
        y = y.to(harness.cfg.device)

        harness.model.train()
        optimizer = harness.get_optmizer()
        criterion = harness.get_criterion()

        optimizer.zero_grad()
        output = harness.model(x)
        loss = criterion(output, y)
        loss.backward()

        has_grad = False
        for param in harness.model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients computed"

    def test_weights_updated(self, harness):
        """Test that weights are updated after optimizer step."""
        train_loader, _ = harness.get_cur_data_loaders()
        batch = next(iter(train_loader))
        x, y = batch
        x = x.to(harness.cfg.device)
        y = y.to(harness.cfg.device)

        harness.model.train()
        optimizer = harness.get_optmizer()
        criterion = harness.get_criterion()

        # Get initial weights
        initial_weights = {
            name: param.clone() for name, param in harness.model.named_parameters()
        }

        optimizer.zero_grad()
        output = harness.model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Check weights changed
        weights_changed = False
        for name, param in harness.model.named_parameters():
            if not torch.equal(param, initial_weights[name]):
                weights_changed = True
                break

        assert weights_changed, "Weights not updated after optimizer step"
