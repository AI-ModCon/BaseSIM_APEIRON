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


@pytest.fixture
def rocm_config():
    """Create a config for ROCm/GPU testing."""
    return Config(
        model=ModelCfg(name="mnist_cnn", pretrained_path=""),
        data=DataCfg(name="mnist", path="./data"),
        train=TrainCfg(batch_size=32, num_workers=0, init_lr=0.001),
        continual_learning=ContinualLearningCfg(
            jvp_lambda=0.001, jvp_deltax_norm=1.0, max_iter=5
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


class TestJVPRegUpdater:
    """Tests for JVPRegUpdater class."""

    def test_jvp_updater_creation(self, rocm_config, harness_with_history):
        """Test that JVPRegUpdater can be created."""
        jvp_updater = JVPRegUpdater(
            cfg=rocm_config,
            modelHarness=harness_with_history,
        )
        assert jvp_updater is not None

    def test_jvp_updater_forward_backward(self, rocm_config, harness_with_history):
        """Test that JVPRegUpdater forward-backward pass works on GPU."""
        jvp_updater = JVPRegUpdater(
            cfg=rocm_config,
            modelHarness=harness_with_history,
        )

        # Get batches
        train_loader, _ = harness_with_history.get_cur_data_loaders()
        hist_train_loader, _ = harness_with_history.get_hist_data_loaders()

        train_batch = next(iter(train_loader))
        hist_batch = next(iter(hist_train_loader))

        # Move to device
        train_batch = tuple(b.to(rocm_config.device) for b in train_batch)
        hist_batch = tuple(b.to(rocm_config.device) for b in hist_batch)

        # Run forward-backward pass
        jvp_updater.update_pre_fwd_bwd()
        loss_curr = jvp_updater.fwd_bwd(train_batch, hist_batch)
        loss_mem = jvp_updater.update_post_fwd_bwd()

        assert loss_curr is not None, "Current loss is None"
        assert loss_mem is not None, "Memory loss is None"
        assert loss_curr > 0, "Current loss should be positive"

    def test_jvp_gradients_on_gpu(self, rocm_config, harness_with_history):
        """Test that JVP gradients are computed on GPU."""
        jvp_updater = JVPRegUpdater(
            cfg=rocm_config,
            modelHarness=harness_with_history,
        )

        # Get batches
        train_loader, _ = harness_with_history.get_cur_data_loaders()
        hist_train_loader, _ = harness_with_history.get_hist_data_loaders()

        train_batch = next(iter(train_loader))
        hist_batch = next(iter(hist_train_loader))

        train_batch = tuple(b.to(rocm_config.device) for b in train_batch)
        hist_batch = tuple(b.to(rocm_config.device) for b in hist_batch)

        # Compute gradients
        jvp_updater.update_pre_fwd_bwd()
        jvp_updater.fwd_bwd(train_batch, hist_batch)
        jvp_updater.update_post_fwd_bwd()

        # Check gradients exist for all parameters
        for name, param in harness_with_history.model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.is_cuda, f"Gradient for {name} not on GPU"
            assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"


class TestJVPUpdateStep:
    """Tests for JVP update step with optimizer."""

    def test_jvp_step_runs(self, rocm_config, harness_with_history):
        """Test that JVP update step executes without error."""
        optimizer = harness_with_history.get_optmizer()

        jvp_updater = JVPRegUpdater(
            cfg=rocm_config,
            modelHarness=harness_with_history,
        )

        # Get batches
        train_loader, _ = harness_with_history.get_cur_data_loaders()
        hist_train_loader, _ = harness_with_history.get_hist_data_loaders()

        train_batch = next(iter(train_loader))
        hist_batch = next(iter(hist_train_loader))

        train_batch = tuple(b.to(rocm_config.device) for b in train_batch)
        hist_batch = tuple(b.to(rocm_config.device) for b in hist_batch)

        # Run update step
        optimizer.zero_grad()
        jvp_updater.update_pre_fwd_bwd()
        loss_curr = jvp_updater.fwd_bwd(train_batch, hist_batch)
        loss_mem = jvp_updater.update_post_fwd_bwd()
        optimizer.step()

        assert loss_curr > 0, "Current loss should be positive"
        assert loss_mem >= 0, "Memory loss should be non-negative"

    def test_jvp_step_updates_weights(self, rocm_config, harness_with_history):
        """Test that JVP update step modifies model weights."""
        optimizer = harness_with_history.get_optmizer()
        model = harness_with_history.model

        # Get initial weights
        initial_weights = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }

        jvp_updater = JVPRegUpdater(
            cfg=rocm_config,
            modelHarness=harness_with_history,
        )

        # Get batches
        train_loader, _ = harness_with_history.get_cur_data_loaders()
        hist_train_loader, _ = harness_with_history.get_hist_data_loaders()

        train_batch = next(iter(train_loader))
        hist_batch = next(iter(hist_train_loader))

        train_batch = tuple(b.to(rocm_config.device) for b in train_batch)
        hist_batch = tuple(b.to(rocm_config.device) for b in hist_batch)

        # Run update step
        optimizer.zero_grad()
        jvp_updater.update_pre_fwd_bwd()
        jvp_updater.fwd_bwd(train_batch, hist_batch)
        jvp_updater.update_post_fwd_bwd()
        optimizer.step()

        # Check weights changed
        weights_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_weights[name], atol=1e-6):
                weights_changed = True
                break

        assert weights_changed, "No weights updated after JVP step"

    def test_jvp_step_multiple_iterations(self, rocm_config, harness_with_history):
        """Test that multiple JVP update steps work correctly."""
        optimizer = harness_with_history.get_optmizer()

        jvp_updater = JVPRegUpdater(
            cfg=rocm_config,
            modelHarness=harness_with_history,
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

            train_batch = tuple(b.to(rocm_config.device) for b in train_batch)
            hist_batch = tuple(b.to(rocm_config.device) for b in hist_batch)

            optimizer.zero_grad()
            jvp_updater.update_pre_fwd_bwd()
            loss_curr = jvp_updater.fwd_bwd(train_batch, hist_batch)
            loss_mem = jvp_updater.update_post_fwd_bwd()
            optimizer.step()

            loss_total = loss_curr + loss_mem
            losses.append(loss_total)

        # All losses should be positive
        assert all(loss > 0 for loss in losses), "Some losses are not positive"
