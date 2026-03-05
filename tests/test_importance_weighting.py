"""Tests for prioritized sampling importance weighting in CL training."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    TensorDataset,
    WeightedRandomSampler,
)

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
# TestComputeSamplePriorities
# ---------------------------------------------------------------------------
class TestComputeSamplePriorities:
    def test_priorities_shape(self, default_cfg, make_harness):
        """compute_sample_priorities returns one priority per sample."""
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(importance_weighting=True),
        )
        harness = make_harness(cfg)
        updater = create_updater(cfg, harness)

        ds = TensorDataset(torch.randn(20, 4), torch.randint(0, 3, (20,)))
        loader = DataLoader(ds, batch_size=8)
        priorities = updater.compute_sample_priorities(loader, "cpu")
        assert priorities.shape == (20,)

    def test_priorities_positive(self, default_cfg, make_harness):
        """All priorities should be positive (clamped at 1e-8)."""
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(importance_weighting=True),
        )
        harness = make_harness(cfg)
        updater = create_updater(cfg, harness)

        ds = TensorDataset(torch.randn(16, 4), torch.randint(0, 3, (16,)))
        loader = DataLoader(ds, batch_size=8)
        priorities = updater.compute_sample_priorities(loader, "cpu")
        assert (priorities > 0).all()

    def test_priorities_differ_after_param_change(self, default_cfg, make_harness):
        """After modifying params, priorities should differ from uniform."""
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(importance_weighting=True),
        )
        harness = make_harness(cfg)
        updater = create_updater(cfg, harness)

        # Modify model away from anchor
        with torch.no_grad():
            for p in harness.model.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        ds = TensorDataset(torch.randn(16, 4), torch.randint(0, 3, (16,)))
        loader = DataLoader(ds, batch_size=8)
        priorities = updater.compute_sample_priorities(loader, "cpu")
        # Not all priorities should be identical
        assert priorities.std() > 0

    def test_alpha_controls_sharpness(self, default_cfg, make_harness):
        """Higher alpha should produce more varied priorities."""
        harness = make_harness(default_cfg)

        # Create updaters FIRST (anchors theta_star at current params)
        updater_low = BaseUpdater(cfg=default_cfg, modelHarness=harness)
        updater_low.importance_weighting = True
        updater_low.importance_alpha = 0.5

        updater_high = BaseUpdater(cfg=default_cfg, modelHarness=harness)
        updater_high.importance_weighting = True
        updater_high.importance_alpha = 2.0

        # THEN modify model away from anchor
        with torch.no_grad():
            for p in harness.model.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        ds = TensorDataset(torch.randn(32, 4), torch.randint(0, 3, (32,)))
        loader = DataLoader(ds, batch_size=16)

        p_low = updater_low.compute_sample_priorities(loader, "cpu")
        p_high = updater_high.compute_sample_priorities(loader, "cpu")

        # Higher alpha → more variance in priorities
        assert p_high.std() > p_low.std()

    def test_weighted_sampler_from_priorities(self, default_cfg, make_harness):
        """Priorities can be used with WeightedRandomSampler."""
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(importance_weighting=True),
        )
        harness = make_harness(cfg)
        updater = create_updater(cfg, harness)

        ds = TensorDataset(torch.randn(20, 4), torch.randint(0, 3, (20,)))
        loader = DataLoader(ds, batch_size=8)
        priorities = updater.compute_sample_priorities(loader, "cpu")

        sampler = WeightedRandomSampler(
            priorities, num_samples=len(priorities), replacement=True
        )
        new_loader = DataLoader(ds, batch_size=8, sampler=sampler)
        batch = next(iter(new_loader))
        assert batch[0].shape[0] == 8


# ---------------------------------------------------------------------------
# TestStandardFwdBwd
# ---------------------------------------------------------------------------
class TestStandardFwdBwd:
    def test_fwd_bwd_uses_standard_loss(self, default_cfg, make_harness):
        """fwd_bwd always uses standard criterion (sampling handles importance)."""
        harness = make_harness(default_cfg)
        updater = BaseUpdater(cfg=default_cfg, modelHarness=harness)

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
        assert cfg.importance_alpha == 1.0

    def test_custom_values(self):
        """ContinualLearningCfg should accept custom importance values."""
        cfg = ContinualLearningCfg(importance_weighting=True, importance_alpha=0.5)
        assert cfg.importance_weighting is True
        assert cfg.importance_alpha == 0.5

    def test_create_updater_passes_config(self, default_cfg, make_harness):
        """create_updater should set importance fields on the updater."""
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(
                importance_weighting=True,
                importance_alpha=2.0,
            ),
        )
        harness = make_harness(cfg)
        updater = create_updater(cfg, harness)
        assert updater.importance_weighting is True
        assert updater.importance_alpha == 2.0
