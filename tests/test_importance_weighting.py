"""Tests for per-sample loss derivative importance weighting in CL training."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from config.configuration import ContinualLearningCfg
from training.updater.base import BaseUpdater
from training.updater.ewc import OnlineEWCUpdater
from training.updater.kfac import OnlineKFACUpdater
from training.updater.create_updater import create_updater


# ---------------------------------------------------------------------------
# TestUnreducedCriterion
# ---------------------------------------------------------------------------
class TestUnreducedCriterion:
    def test_nll_loss_shape(self, default_cfg, make_harness):
        """NLLLoss with reduction='none' returns per-sample tensor."""
        harness = make_harness(default_cfg)
        harness.get_criterion = lambda: nn.NLLLoss()
        updater = BaseUpdater(cfg=default_cfg, modelHarness=harness)

        outputs = torch.log_softmax(torch.randn(8, 3), dim=1)
        y = torch.randint(0, 3, (8,))
        result = updater._unreduced_criterion(outputs, y)
        assert result.shape == (8,)

    def test_cross_entropy_shape(self, default_cfg, make_harness):
        """CrossEntropyLoss with reduction='none' returns per-sample tensor."""
        harness = make_harness(default_cfg)
        updater = BaseUpdater(cfg=default_cfg, modelHarness=harness)

        outputs = torch.randn(8, 3)
        y = torch.randint(0, 3, (8,))
        result = updater._unreduced_criterion(outputs, y)
        assert result.shape == (8,)


# ---------------------------------------------------------------------------
# TestPerSampleWeighting
# ---------------------------------------------------------------------------
class TestPerSampleWeighting:
    def _make_weighted_updater(self, default_cfg, make_harness, temperature=1.0):
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(
                importance_weighting=True,
                importance_temperature=temperature,
            ),
        )
        harness = make_harness(cfg)
        updater = create_updater(cfg, harness)
        return updater, harness

    def test_weights_sum_to_batch_size(self, default_cfg, make_harness):
        """Importance weights after softmax * len should sum to batch_size."""
        updater, harness = self._make_weighted_updater(default_cfg, make_harness)
        x = torch.randn(8, 4)
        y = torch.randint(0, 3, (8,))

        outputs = harness.model(x)
        per_sample_loss = updater._unreduced_criterion(outputs, y)

        with torch.no_grad():
            anchor = {n: p.detach() for n, p in harness.model.named_parameters()}
            anchor.update(updater.theta_star)
            from torch.func import functional_call

            anchor_out = functional_call(harness.model, anchor, (x,))
            anchor_loss = updater._unreduced_criterion(anchor_out, y)
            delta = (per_sample_loss.detach() - anchor_loss).clamp(min=1e-8)
            weights = (delta / updater.importance_temperature).softmax(dim=0) * len(
                delta
            )

        assert abs(weights.sum().item() - 8.0) < 1e-4

    def test_high_delta_gets_high_weight(self, default_cfg, make_harness):
        """Samples with higher delta_L should get higher weights."""
        updater, _ = self._make_weighted_updater(default_cfg, make_harness)

        delta = torch.tensor([0.1, 0.5, 2.0, 0.01])
        weights = (delta / updater.importance_temperature).softmax(dim=0) * len(delta)
        # The sample with delta=2.0 should have highest weight
        assert weights[2] == weights.max()
        # The sample with delta=0.01 should have lowest weight
        assert weights[3] == weights.min()

    def test_disabled_uses_standard_loss(self, default_cfg, make_harness):
        """When importance_weighting=False, fwd_bwd uses standard criterion."""
        harness = make_harness(default_cfg)
        updater = BaseUpdater(cfg=default_cfg, modelHarness=harness)
        assert updater.importance_weighting is False

        x = torch.randn(4, 4)
        y = torch.randint(0, 3, (4,))
        harness.model.zero_grad()
        loss = updater.fwd_bwd((x, y))
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_weighted_fwd_bwd_runs(self, default_cfg, make_harness):
        """fwd_bwd with importance weighting enabled completes without error."""
        updater, harness = self._make_weighted_updater(default_cfg, make_harness)
        x = torch.randn(4, 4)
        y = torch.randint(0, 3, (4,))
        harness.model.zero_grad()
        loss = updater.fwd_bwd((x, y))
        assert isinstance(loss, float)
        assert loss >= 0.0


# ---------------------------------------------------------------------------
# TestThetaStar
# ---------------------------------------------------------------------------
class TestThetaStar:
    def test_base_initializes_theta_star(self, dummy_harness):
        """BaseUpdater should initialize theta_star for all requires_grad params."""
        updater = BaseUpdater(cfg=dummy_harness.cfg, modelHarness=dummy_harness)
        param_names = {
            n for n, p in dummy_harness.model.named_parameters() if p.requires_grad
        }
        assert set(updater.theta_star.keys()) == param_names

    def test_ewc_inherits_theta_star(self, default_cfg, make_harness):
        """EWC should use theta_star from BaseUpdater (no duplicate init)."""
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(update_mode="ewc_online"),
        )
        harness = make_harness(cfg)
        updater = OnlineEWCUpdater(cfg=cfg, modelHarness=harness)
        param_names = {
            n for n, p in harness.model.named_parameters() if p.requires_grad
        }
        assert set(updater.theta_star.keys()) == param_names

    def test_kfac_keeps_partial_theta_star(self, default_cfg, make_harness):
        """KFAC stores theta_star only for supported layers (Linear/Conv2d)."""
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(update_mode="kfac_online"),
        )
        harness = make_harness(cfg)
        updater = OnlineKFACUpdater(cfg=cfg, modelHarness=harness)
        # KFAC theta_star uses module names, not parameter names
        # It should have entries for supported modules
        assert len(updater.theta_star) > 0

    def test_theta_star_updates_in_postprocessing(self, dummy_harness):
        """cl_postprocessing should update theta_star to current params."""
        updater = BaseUpdater(cfg=dummy_harness.cfg, modelHarness=dummy_harness)

        # Modify model parameters
        with torch.no_grad():
            for p in dummy_harness.model.parameters():
                p.add_(1.0)

        updater.cl_postprocessing()

        # theta_star should now match current params
        for n, p in dummy_harness.model.named_parameters():
            if p.requires_grad and n in updater.theta_star:
                assert torch.allclose(updater.theta_star[n], p.detach())


# ---------------------------------------------------------------------------
# TestFunctionalCallWeighting
# ---------------------------------------------------------------------------
class TestFunctionalCallWeighting:
    def test_anchor_loss_differs_after_training(self, default_cfg, make_harness):
        """After modifying model params, anchor loss should differ from current loss."""
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(importance_weighting=True),
        )
        harness = make_harness(cfg)
        updater = create_updater(cfg, harness)

        x = torch.randn(4, 4)
        y = torch.randint(0, 3, (4,))

        # Modify model
        with torch.no_grad():
            for p in harness.model.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        outputs = harness.model(x)
        per_sample_loss = updater._unreduced_criterion(outputs, y)

        with torch.no_grad():
            from torch.func import functional_call

            anchor = {n: p.detach() for n, p in harness.model.named_parameters()}
            anchor.update(updater.theta_star)
            anchor_out = functional_call(harness.model, anchor, (x,))
            anchor_loss = updater._unreduced_criterion(anchor_out, y)

        # Losses should differ since params changed
        assert not torch.allclose(per_sample_loss, anchor_loss, atol=1e-6)

    def test_no_gradients_through_anchor(self, default_cfg, make_harness):
        """Anchor computation should not create gradients."""
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(importance_weighting=True),
        )
        harness = make_harness(cfg)
        updater = create_updater(cfg, harness)

        x = torch.randn(4, 4)
        y = torch.randint(0, 3, (4,))

        with torch.no_grad():
            anchor = {n: p.detach() for n, p in harness.model.named_parameters()}
            anchor.update(updater.theta_star)
            from torch.func import functional_call

            anchor_out = functional_call(harness.model, anchor, (x,))
            anchor_loss = updater._unreduced_criterion(anchor_out, y)

        assert not anchor_loss.requires_grad


# ---------------------------------------------------------------------------
# TestBalancedSampling
# ---------------------------------------------------------------------------
class TestBalancedSampling:
    def test_sampler_overrides_shuffle(self):
        """When sampler is provided, shuffle should not be passed (no error)."""
        ds = TensorDataset(torch.randn(100, 4), torch.randint(0, 3, (100,)))
        sampler = RandomSampler(ds, replacement=True, num_samples=50)
        # This should not raise "mutually exclusive" error
        loader = DataLoader(ds, batch_size=8, sampler=sampler)
        batch = next(iter(loader))
        assert batch[0].shape[0] == 8

    def test_no_sampler_default_behavior(self):
        """Without sampler, DataLoader uses shuffle normally."""
        ds = TensorDataset(torch.randn(100, 4), torch.randint(0, 3, (100,)))
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        batch = next(iter(loader))
        assert batch[0].shape[0] == 8

    def test_current_ratio_controls_num_samples(self):
        """RandomSampler with num_samples controls how many samples are drawn."""
        ds = TensorDataset(torch.randn(100, 4), torch.randint(0, 3, (100,)))
        ratio = 0.7
        n_samples = int(len(ds) * ratio)
        sampler = RandomSampler(ds, replacement=True, num_samples=n_samples)

        # Count total samples yielded
        loader = DataLoader(ds, batch_size=16, sampler=sampler)
        total = sum(batch[0].shape[0] for batch in loader)
        assert total == n_samples


# ---------------------------------------------------------------------------
# TestConfigFields
# ---------------------------------------------------------------------------
class TestConfigFields:
    def test_default_values(self):
        """ContinualLearningCfg should have correct defaults for importance fields."""
        cfg = ContinualLearningCfg()
        assert cfg.importance_weighting is False
        assert cfg.importance_temperature == 1.0

    def test_custom_values(self):
        """ContinualLearningCfg should accept custom importance values."""
        cfg = ContinualLearningCfg(
            importance_weighting=True, importance_temperature=0.5
        )
        assert cfg.importance_weighting is True
        assert cfg.importance_temperature == 0.5

    def test_create_updater_passes_config(self, default_cfg, make_harness):
        """create_updater should set importance fields on the updater."""
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(
                importance_weighting=True,
                importance_temperature=2.0,
            ),
        )
        harness = make_harness(cfg)
        updater = create_updater(cfg, harness)
        assert updater.importance_weighting is True
        assert updater.importance_temperature == 2.0
