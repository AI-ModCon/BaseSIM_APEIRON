"""Tests for training updaters: base, ewc, kfac, jvp_reg, no_updater, and factory."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from apeiron.config.configuration import ContinualLearningCfg
from apeiron.training.updater.base import BaseUpdater
from apeiron.training.updater.no_updater import NoUpdater
from apeiron.training.updater.ewc import OnlineEWCUpdater
from apeiron.training.updater.kfac import OnlineKFACUpdater
from apeiron.training.updater.jvp_reg import JVPRegUpdater
from apeiron.training.updater.create_updater import create_updater


# ---------------------------------------------------------------------------
# BaseUpdater
# ---------------------------------------------------------------------------
class TestBaseUpdater:
    def test_fwd_bwd_returns_loss(self, dummy_harness):
        updater = BaseUpdater(cfg=dummy_harness.cfg, modelHarness=dummy_harness)
        x = torch.randn(4, 4)
        y = torch.randint(0, 3, (4,))
        loss = updater.fwd_bwd((x, y))
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_fwd_bwd_populates_gradients(self, dummy_harness):
        updater = BaseUpdater(cfg=dummy_harness.cfg, modelHarness=dummy_harness)
        x = torch.randn(4, 4)
        y = torch.randint(0, 3, (4,))
        dummy_harness.model.zero_grad()
        updater.fwd_bwd((x, y))
        for p in dummy_harness.model.parameters():
            assert p.grad is not None

    def test_hooks_are_noop(self, dummy_harness):
        updater = BaseUpdater(cfg=dummy_harness.cfg, modelHarness=dummy_harness)
        updater.cl_preprocessing()
        updater.update_pre_fwd_bwd()
        reg = updater.update_post_fwd_bwd()
        assert reg == 0.0
        updater.update_post_optimizer_call()
        updater.cl_postprocessing()

    @staticmethod
    def _grads(model):
        return [p.grad.clone() for p in model.parameters() if p.grad is not None]

    def _mixing_updater(self, default_cfg, make_harness, enabled: bool):
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(mix_historic_data=enabled),
        )
        harness = make_harness(cfg)
        return BaseUpdater(cfg=cfg, modelHarness=harness), harness

    def test_hist_batch_ignored_by_default(self, dummy_harness):
        """mix_historic_data defaults to False, so hist_batch must not change grads."""
        updater = BaseUpdater(cfg=dummy_harness.cfg, modelHarness=dummy_harness)
        assert updater.mix_historic_data is False
        batch = (torch.randn(4, 4), torch.randint(0, 3, (4,)))
        hist = (torch.randn(4, 4), torch.randint(0, 3, (4,)))

        dummy_harness.model.zero_grad()
        updater.fwd_bwd(batch)
        grad_cur_only = self._grads(dummy_harness.model)

        dummy_harness.model.zero_grad()
        updater.fwd_bwd(batch, hist)
        grad_with_hist = self._grads(dummy_harness.model)

        for a, b in zip(grad_cur_only, grad_with_hist):
            assert torch.allclose(a, b, atol=1e-6)

    def test_hist_batch_is_mixed_when_enabled(self, default_cfg, make_harness):
        updater, harness = self._mixing_updater(default_cfg, make_harness, enabled=True)
        batch = (torch.randn(4, 4), torch.randint(0, 3, (4,)))
        hist = (torch.randn(4, 4), torch.randint(0, 3, (4,)))

        harness.model.zero_grad()
        updater.fwd_bwd(batch)
        grad_cur_only = self._grads(harness.model)

        harness.model.zero_grad()
        updater.fwd_bwd(batch, hist)
        grad_mixed = self._grads(harness.model)

        assert any(
            not torch.allclose(a, b) for a, b in zip(grad_cur_only, grad_mixed)
        ), "hist_batch did not influence the gradients"

    def test_hist_batch_matches_half_and_half_pass(self, default_cfg, make_harness):
        """Mixing must equal one fwd/bwd over half the current plus half the hist batch."""
        updater, harness = self._mixing_updater(default_cfg, make_harness, enabled=True)
        batch = (torch.randn(8, 4), torch.randint(0, 3, (8,)))
        hist = (torch.randn(8, 4), torch.randint(0, 3, (8,)))

        harness.model.zero_grad()
        updater.fwd_bwd(batch, hist)
        grad_mixed = self._grads(harness.model)

        harness.model.zero_grad()
        x = torch.cat([batch[0][:4], hist[0][:4]], dim=0)
        y = torch.cat([batch[1][:4], hist[1][:4]], dim=0)
        loss = harness.get_criterion()(harness.model(x), y)
        (loss / harness.cfg.train.grad_accumulation_steps).backward()
        grad_ref = self._grads(harness.model)

        for a, b in zip(grad_mixed, grad_ref):
            assert torch.allclose(a, b, atol=1e-6)

    def test_grad_accumulation_scaling(self, default_cfg, make_harness):
        cfg2 = replace(
            default_cfg,
            train=replace(default_cfg.train, grad_accumulation_steps=4),
        )
        harness2 = make_harness(cfg2)
        updater = BaseUpdater(cfg=cfg2, modelHarness=harness2)
        x = torch.randn(4, 4)
        y = torch.randint(0, 3, (4,))
        loss = updater.fwd_bwd((x, y))
        # With grad_accumulation_steps=4, loss is divided by 4
        assert loss >= 0.0


# ---------------------------------------------------------------------------
# NoUpdater
# ---------------------------------------------------------------------------
class TestNoUpdater:
    def test_returns_sentinel(self, dummy_harness):
        updater = NoUpdater(cfg=dummy_harness.cfg, modelHarness=dummy_harness)
        loss = updater.fwd_bwd((torch.randn(4, 4), torch.randint(0, 3, (4,))))
        assert loss == -1.0


# ---------------------------------------------------------------------------
# OnlineEWCUpdater
# ---------------------------------------------------------------------------
class TestOnlineEWCUpdater:
    def test_init_creates_anchor_and_fisher(self, default_cfg, make_harness):
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(
                update_mode="ewc_online", ewc_lambda=10.0, ewc_ema_decay=0.9
            ),
        )
        harness = make_harness(cfg)
        updater = OnlineEWCUpdater(cfg=cfg, modelHarness=harness)
        assert len(updater.theta_star) > 0
        assert len(updater.fisher) > 0

    def test_cl_lifecycle(self, default_cfg, make_harness):
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(update_mode="ewc_online"),
        )
        harness = make_harness(cfg)
        updater = OnlineEWCUpdater(cfg=cfg, modelHarness=harness)

        updater.cl_preprocessing()
        assert updater._cl_fisher_accum is not None
        assert updater._cl_steps == 0

        # Simulate one training step
        updater.update_pre_fwd_bwd()
        x = torch.randn(4, 4)
        y = torch.randint(0, 3, (4,))
        loss = updater.fwd_bwd((x, y))
        assert loss >= 0.0
        reg_loss = updater.update_post_fwd_bwd()
        assert isinstance(reg_loss, float)
        updater.update_post_optimizer_call()
        assert updater._cl_steps == 1

        updater.cl_postprocessing()
        assert updater._cl_fisher_accum is None
        assert updater._cl_steps == 0

    def test_postprocessing_noop_with_zero_steps(self, default_cfg, make_harness):
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(update_mode="ewc_online"),
        )
        harness = make_harness(cfg)
        updater = OnlineEWCUpdater(cfg=cfg, modelHarness=harness)
        updater.cl_preprocessing()
        # No steps taken
        updater.cl_postprocessing()  # should not error


# ---------------------------------------------------------------------------
# OnlineKFACUpdater
# ---------------------------------------------------------------------------
class TestOnlineKFACUpdater:
    def _make_updater(self, default_cfg, make_harness):
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(update_mode="kfac_online"),
        )
        harness = make_harness(cfg)
        updater = OnlineKFACUpdater(cfg=cfg, modelHarness=harness)
        return updater, harness

    def test_init(self, default_cfg, make_harness):
        updater, _ = self._make_updater(default_cfg, make_harness)
        # Linear layer should be tracked
        assert len(updater.theta_star) > 0
        assert len(updater.A) > 0
        assert len(updater.G) > 0

    def test_cl_lifecycle(self, default_cfg, make_harness):
        updater, harness = self._make_updater(default_cfg, make_harness)
        updater.cl_preprocessing()
        assert updater._A_accum is not None

        # One training step
        x = torch.randn(4, 4)
        y = torch.randint(0, 3, (4,))
        harness.model.zero_grad()
        updater.fwd_bwd((x, y))
        reg = updater.update_post_fwd_bwd()
        assert isinstance(reg, float)
        updater.update_post_optimizer_call()
        assert updater._cl_steps == 1

        updater.cl_postprocessing()
        assert updater._A_accum is None
        assert updater._cl_steps == 0

    def test_supported_layers(self, default_cfg, make_harness):
        updater, _ = self._make_updater(default_cfg, make_harness)
        assert updater._supported(nn.Linear(4, 4)) is True
        assert updater._supported(nn.Conv2d(1, 1, 3)) is True
        assert updater._supported(nn.ReLU()) is False


# ---------------------------------------------------------------------------
# JVPRegUpdater
# ---------------------------------------------------------------------------
class TestJVPRegUpdater:
    def _make_updater(self, default_cfg, make_harness):
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(update_mode="jvp_reg"),
        )
        harness = make_harness(cfg)
        updater = JVPRegUpdater(cfg=cfg, modelHarness=harness)
        return updater, harness

    def test_fwd_bwd_no_hist_falls_back_to_base(self, default_cfg, make_harness):
        updater, _ = self._make_updater(default_cfg, make_harness)
        updater.update_pre_fwd_bwd()
        x = torch.randn(4, 4)
        y = torch.randint(0, 3, (4,))
        loss = updater.fwd_bwd((x, y), hist_batch=None)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_fwd_bwd_with_hist(self, default_cfg, make_harness):
        updater, _ = self._make_updater(default_cfg, make_harness)
        updater.update_pre_fwd_bwd()
        batch = (torch.randn(4, 4), torch.randint(0, 3, (4,)))
        hist = (torch.randn(4, 4), torch.randint(0, 3, (4,)))
        loss = updater.fwd_bwd(batch, hist_batch=hist)
        assert isinstance(loss, float)

    def test_update_post_fwd_bwd_clears_state(self, default_cfg, make_harness):
        updater, _ = self._make_updater(default_cfg, make_harness)
        updater.update_pre_fwd_bwd()
        batch = (torch.randn(4, 4), torch.randint(0, 3, (4,)))
        hist = (torch.randn(4, 4), torch.randint(0, 3, (4,)))
        updater.fwd_bwd(batch, hist_batch=hist)
        reg = updater.update_post_fwd_bwd()
        assert isinstance(reg, float)
        assert updater.grad_dict is None
        assert updater.loss_mem == 0.0


# ---------------------------------------------------------------------------
# create_updater factory
# ---------------------------------------------------------------------------
class TestCreateUpdater:
    def test_base(self, default_cfg, make_harness):
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(update_mode="base"),
        )
        u = create_updater(cfg, make_harness(cfg))
        assert isinstance(u, BaseUpdater)
        assert not isinstance(
            u, (OnlineEWCUpdater, OnlineKFACUpdater, JVPRegUpdater, NoUpdater)
        )

    def test_ewc(self, default_cfg, make_harness):
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(update_mode="ewc_online"),
        )
        u = create_updater(cfg, make_harness(cfg))
        assert isinstance(u, OnlineEWCUpdater)

    def test_kfac(self, default_cfg, make_harness):
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(update_mode="kfac_online"),
        )
        u = create_updater(cfg, make_harness(cfg))
        assert isinstance(u, OnlineKFACUpdater)

    def test_jvp(self, default_cfg, make_harness):
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(update_mode="jvp_reg"),
        )
        u = create_updater(cfg, make_harness(cfg))
        assert isinstance(u, JVPRegUpdater)

    def test_none(self, default_cfg, make_harness):
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(update_mode="none"),
        )
        u = create_updater(cfg, make_harness(cfg))
        assert isinstance(u, NoUpdater)

    def test_unknown_raises(self, default_cfg, make_harness):
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(update_mode="unknown_xyz"),
        )
        with pytest.raises(NotImplementedError, match="Unknown update_mode"):
            create_updater(cfg, make_harness(cfg))
