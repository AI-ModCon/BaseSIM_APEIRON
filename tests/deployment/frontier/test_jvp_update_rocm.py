"""Tests to verify JVP regularized update works correctly with ROCm."""

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

from training.updater.jvp_reg import JVPRegUpdater
from profilers import FLOPSProfiler


@pytest.fixture
def rocm_config():
    """Create a config for ROCm/GPU testing."""
    return Config(
        model=ModelCfg(name="mnist_cnn", pretrained_path=""),
        data=DataCfg(name="mnist", path="./data"),
        train=TrainCfg(batch_size=32, num_workers=0, init_lr=0.001),
        continual_learning=ContinualLearningCfg(
            jvp_reg=0.001, deltax_norm=1.0, max_iter=5
        ),
        drift_detection=DriftDetectionCfg(),
        seed=42,
        device="cuda",
        multi_gpu=False,
    )


@pytest.fixture
def harness_with_history(rocm_config):
    """Create MNIST harness with historical data."""
    harness = MNIST_CNN(rocm_config)
    harness.update_data_stream()  # First stream
    harness.update_data_stream()  # Second stream (creates history)
    return harness


class TestJVPRegularizedLoss:
    """Tests for JVPRegularizedLoss module."""

    def test_jvp_loss_creation(self, harness_with_history):
        """Test that JVPRegularizedLoss can be created."""
        criterion = harness_with_history.get_criterion()
        jvp_loss = JVPRegUpdater(
            model=harness_with_history.model,
            criterion=criterion,
            jvp_reg=0.001,
            deltax_norm=1.0,
        )
        assert jvp_loss is not None

    def test_jvp_loss_forward(self, rocm_config, harness_with_history):
        """Test that JVPRegularizedLoss forward pass works on GPU."""
        criterion = harness_with_history.get_criterion()
        jvp_loss = JVPRegularizedLoss(
            model=harness_with_history.model,
            criterion=criterion,
            jvp_reg=0.001,
            deltax_norm=1.0,
        )

        # Get batches
        train_loader, _ = harness_with_history.get_cur_data_loaders()
        hist_train_loader, _ = harness_with_history.get_hist_data_loaders()

        train_batch = next(iter(train_loader))
        hist_batch = next(iter(hist_train_loader))

        # Move to device
        train_batch = [b.to(rocm_config.device) for b in train_batch]
        hist_batch = [b.to(rocm_config.device) for b in hist_batch]

        # Forward pass
        grad_dict, loss_curr, loss_mem = jvp_loss(train_batch, hist_batch)

        assert grad_dict is not None, "Gradient dict is None"
        assert loss_curr is not None, "Current loss is None"
        assert loss_mem is not None, "Memory loss is None"

    def test_jvp_loss_gradients_on_gpu(self, rocm_config, harness_with_history):
        """Test that JVP gradients are computed on GPU."""
        criterion = harness_with_history.get_criterion()
        jvp_loss = JVPRegularizedLoss(
            model=harness_with_history.model,
            criterion=criterion,
            jvp_reg=0.001,
            deltax_norm=1.0,
        )

        # Get batches
        train_loader, _ = harness_with_history.get_cur_data_loaders()
        hist_train_loader, _ = harness_with_history.get_hist_data_loaders()

        train_batch = next(iter(train_loader))
        hist_batch = next(iter(hist_train_loader))

        train_batch = [b.to(rocm_config.device) for b in train_batch]
        hist_batch = [b.to(rocm_config.device) for b in hist_batch]

        # Compute gradients
        grad_dict, _, _ = jvp_loss(train_batch, hist_batch)

        # Check gradients exist for all parameters
        for name, param in harness_with_history.model.named_parameters():
            assert name in grad_dict, f"No gradient for {name}"
            assert grad_dict[name].is_cuda, f"Gradient for {name} not on GPU"
            assert not torch.isnan(grad_dict[name]).any(), f"NaN in gradient for {name}"


class TestJVPUpdateStep:
    """Tests for step_method_jvp_reg function."""

    def test_jvp_step_runs(self, rocm_config, harness_with_history):
        """Test that JVP update step executes without error."""
        criterion = harness_with_history.get_criterion()
        optimizer = harness_with_history.get_optmizer()
        model = harness_with_history.model
        profiler = FLOPSProfiler()

        jvp_loss = JVPRegularizedLoss(
            model=model,
            criterion=criterion,
            jvp_reg=rocm_config.continuous_learning.jvp_reg,
            deltax_norm=rocm_config.continuous_learning.deltax_norm,
        )

        # Get batches
        train_loader, _ = harness_with_history.get_cur_data_loaders()
        hist_train_loader, _ = harness_with_history.get_hist_data_loaders()

        train_batch = next(iter(train_loader))
        hist_batch = next(iter(hist_train_loader))

        train_batch = [b.to(rocm_config.device) for b in train_batch]
        hist_batch = [b.to(rocm_config.device) for b in hist_batch]

        # Run update step
        loss_curr, loss_mem, loss_total = step_method_jvp_reg(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            cfg=rocm_config,
            iter=0,
            train_batch=train_batch,
            hist_batch=hist_batch,
            profiler=profiler,
            jvp_loss=jvp_loss,
        )

        assert loss_curr > 0, "Current loss should be positive"
        assert loss_mem > 0, "Memory loss should be positive"
        assert loss_total > 0, "Total loss should be positive"

    def test_jvp_step_updates_weights(self, rocm_config, harness_with_history):
        """Test that JVP update step modifies model weights."""
        criterion = harness_with_history.get_criterion()
        optimizer = harness_with_history.get_optmizer()
        model = harness_with_history.model
        profiler = FLOPSProfiler()

        # Get initial weights
        initial_weights = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }

        jvp_loss = JVPRegularizedLoss(
            model=model,
            criterion=criterion,
            jvp_reg=rocm_config.continuous_learning.jvp_reg,
            deltax_norm=rocm_config.continuous_learning.deltax_norm,
        )

        # Get batches
        train_loader, _ = harness_with_history.get_cur_data_loaders()
        hist_train_loader, _ = harness_with_history.get_hist_data_loaders()

        train_batch = next(iter(train_loader))
        hist_batch = next(iter(hist_train_loader))

        train_batch = [b.to(rocm_config.device) for b in train_batch]
        hist_batch = [b.to(rocm_config.device) for b in hist_batch]

        # Run update step
        step_method_jvp_reg(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            cfg=rocm_config,
            iter=0,
            train_batch=train_batch,
            hist_batch=hist_batch,
            profiler=profiler,
            jvp_loss=jvp_loss,
        )

        # Check weights changed
        weights_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_weights[name], atol=1e-6):
                weights_changed = True
                break

        assert weights_changed, "No weights updated after JVP step"

    def test_jvp_step_multiple_iterations(self, rocm_config, harness_with_history):
        """Test that multiple JVP update steps work correctly."""
        criterion = harness_with_history.get_criterion()
        optimizer = harness_with_history.get_optmizer()
        model = harness_with_history.model
        profiler = FLOPSProfiler()

        jvp_loss = JVPRegularizedLoss(
            model=model,
            criterion=criterion,
            jvp_reg=rocm_config.continuous_learning.jvp_reg,
            deltax_norm=rocm_config.continuous_learning.deltax_norm,
        )

        # Get loaders
        train_loader, _ = harness_with_history.get_cur_data_loaders()
        hist_train_loader, _ = harness_with_history.get_hist_data_loaders()

        train_iter = iter(train_loader)
        hist_iter = iter(hist_train_loader)

        losses = []
        for i in range(5):
            train_batch = next(train_iter)
            hist_batch = next(hist_iter)

            train_batch = [b.to(rocm_config.device) for b in train_batch]
            hist_batch = [b.to(rocm_config.device) for b in hist_batch]

            loss_curr, loss_mem, loss_total = step_method_jvp_reg(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                cfg=rocm_config,
                iter=i,
                train_batch=train_batch,
                hist_batch=hist_batch,
                profiler=profiler,
                jvp_loss=jvp_loss,
            )

            losses.append(loss_total)

        # All losses should be positive
        assert all(loss > 0 for loss in losses), "Some losses are not positive"
