"""Numerical tests for the continual-learning updaters.

The behavioral tests in test_updaters.py check that the updaters run; these
check that they compute the documented quantities. Every expectation is
re-derived independently in the test on a hook-free copy of the model, so a
different-but-plausible algorithm (wrong perturbation direction, wrong EMA,
wrong penalty shape) fails even though it would train without error.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.testing import assert_close

from apeiron.config.configuration import ContinualLearningCfg
from apeiron.training.updater.ewc import OnlineEWCUpdater
from apeiron.training.updater.jvp_reg import JVPRegUpdater
from apeiron.training.updater.kfac import OnlineKFACUpdater

A_STEPS = 1  # default grad_accumulation_steps in the test config


def _fresh_copy(model: torch.nn.Module) -> torch.nn.Module:
    """An identical model with no hooks and no shared storage."""
    ref = type(model)()
    ref.load_state_dict(model.state_dict())
    return ref


def _params(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


def _grads_at(model, criterion, x, y):
    """Reference gradient of criterion(model(x), y) / A_STEPS, hook-free."""
    loss = criterion(model(x), y) / A_STEPS
    return torch.autograd.grad(loss, _params(model))


def _batch(seed: int, n: int = 8, scale: float = 1.0, shift: float = 0.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g) * scale + shift
    y = torch.randint(0, 3, (n,), generator=g)
    return x, y


# ---------------------------------------------------------------------------
# Online EWC
# ---------------------------------------------------------------------------
class TestEWCNumerics:
    def _updater(self, make_harness, default_cfg, lam=2.0, decay=0.5):
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(
                update_mode="ewc_online", ewc_lambda=lam, ewc_ema_decay=decay
            ),
        )
        torch.manual_seed(0)
        harness = make_harness(cfg)
        return OnlineEWCUpdater(cfg, harness), harness

    def _one_cl_step(self, upd, harness, batch):
        harness.model.zero_grad()
        upd.update_pre_fwd_bwd()
        upd.fwd_bwd(batch)
        penalty = upd.update_post_fwd_bwd()
        grads = {
            n: p.grad.detach().clone()
            for n, p in harness.model.named_parameters()
            if p.requires_grad
        }
        upd.update_post_optimizer_call()
        return penalty, grads

    def test_first_commit_sets_fisher_to_mean_squared_gradient(
        self, make_harness, default_cfg
    ):
        upd, harness = self._updater(make_harness, default_cfg)
        upd.cl_preprocessing()
        penalty, grads = self._one_cl_step(upd, harness, _batch(1))
        assert penalty == 0.0, "no prior yet: the first penalty must be zero"
        upd.cl_postprocessing()
        for name, g in grads.items():
            # F <- decay * 0 + mean over 1 step of grad^2
            assert_close(upd.fisher[name], g.pow(2))
            assert_close(
                upd.theta_star[name],
                dict(harness.model.named_parameters())[name].detach(),
            )

    def test_penalty_value_and_gradient_term(self, make_harness, default_cfg):
        lam = 2.0
        upd, harness = self._updater(make_harness, default_cfg, lam=lam)
        upd.cl_preprocessing()
        self._one_cl_step(upd, harness, _batch(1))
        upd.cl_postprocessing()

        delta = 0.1
        with torch.no_grad():
            for p in _params(harness.model):
                p.add_(delta)

        harness.model.zero_grad()
        upd.update_pre_fwd_bwd()
        upd.fwd_bwd(_batch(2))
        base = {
            n: p.grad.detach().clone()
            for n, p in harness.model.named_parameters()
            if p.requires_grad
        }
        penalty = upd.update_post_fwd_bwd()

        # 0.5 * lambda * sum(F * (theta - theta*)^2), with theta - theta* = delta
        expected = (
            0.5 * lam * sum((f * delta**2).sum().item() for f in upd.fisher.values())
        )
        assert penalty == pytest.approx(expected, rel=1e-5)
        # the gradient gains lambda * F * (theta - theta*)
        for name, p in harness.model.named_parameters():
            if p.requires_grad:
                assert_close(p.grad, base[name] + lam * upd.fisher[name] * delta)

    def test_fisher_commit_uses_the_ema_decay(self, make_harness, default_cfg):
        decay = 0.5
        upd, harness = self._updater(make_harness, default_cfg, decay=decay)
        upd.cl_preprocessing()
        self._one_cl_step(upd, harness, _batch(1))
        upd.cl_postprocessing()
        fisher_1 = {n: f.clone() for n, f in upd.fisher.items()}

        upd.cl_preprocessing()
        _, grads_2 = self._one_cl_step(upd, harness, _batch(2))
        upd.cl_postprocessing()
        for name in fisher_1:
            # grads_2 is captured after fwd_bwd and before update_post_fwd_bwd
            # adds the penalty term, which is exactly the gradient the Fisher
            # estimate squares.
            assert_close(
                upd.fisher[name], decay * fisher_1[name] + grads_2[name].pow(2)
            )


# ---------------------------------------------------------------------------
# Online KFAC
# ---------------------------------------------------------------------------
class TestKFACNumerics:
    def _updater(self, make_harness, default_cfg, lam=2.0, decay=0.5):
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(
                update_mode="kfac_online", kfac_lambda=lam, kfac_ema_decay=decay
            ),
        )
        torch.manual_seed(0)
        harness = make_harness(cfg)
        return OnlineKFACUpdater(cfg, harness), harness

    def test_committed_factors_are_the_kronecker_statistics(
        self, make_harness, default_cfg
    ):
        upd, harness = self._updater(make_harness, default_cfg)
        model = harness.model
        x, y = _batch(1)

        # reference activation/grad-output statistics on a hook-free copy
        ref = _fresh_copy(model)
        out = ref(x)
        loss = harness.get_criterion()(out, y) / A_STEPS
        out.retain_grad()
        loss.backward()
        expected_A = x.T @ x / x.shape[0]
        expected_G = out.grad.T @ out.grad / out.grad.shape[0]

        upd.cl_preprocessing()
        model.zero_grad()
        upd.fwd_bwd((x, y))
        penalty = upd.update_post_fwd_bwd()
        assert penalty == 0.0, "no prior yet: the first penalty must be zero"
        upd.update_post_optimizer_call()
        upd.cl_postprocessing()

        assert_close(upd.A["fc"], expected_A)
        assert_close(upd.G["fc"], expected_G)
        assert_close(upd.theta_star["fc"], model.fc.weight.detach())

    def test_penalty_is_kfac_structured(self, make_harness, default_cfg):
        lam = 2.0
        upd, harness = self._updater(make_harness, default_cfg, lam=lam)
        model = harness.model
        upd.cl_preprocessing()
        model.zero_grad()
        upd.fwd_bwd(_batch(1))
        upd.update_post_fwd_bwd()
        upd.update_post_optimizer_call()
        upd.cl_postprocessing()

        # a non-constant perturbation: cross-entropy grad-outputs sum to zero
        # across classes, so G annihilates constant columns and a uniform
        # delta would make the penalty vanish identically
        diff = torch.randn(
            model.fc.weight.shape, generator=torch.Generator().manual_seed(3)
        )
        with torch.no_grad():
            model.fc.weight.add_(diff)

        model.zero_grad()
        upd.fwd_bwd(_batch(2))
        base = model.fc.weight.grad.detach().clone()
        penalty = upd.update_post_fwd_bwd()

        grad_penalty = upd.G["fc"] @ diff @ upd.A["fc"]
        assert penalty == pytest.approx(
            0.5 * lam * (diff * grad_penalty).sum().item(), rel=1e-5
        )
        assert_close(model.fc.weight.grad, base + lam * grad_penalty)


# ---------------------------------------------------------------------------
# JVP / first-order SAM robust updater
# ---------------------------------------------------------------------------
class TestJVPNumerics:
    def _updater(
        self, make_harness, default_cfg, rho_theta=0.5, rho_x=0.0, data_sign=1.0
    ):
        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(
                update_mode="jvp_reg",
                jvp_rho_theta=rho_theta,
                jvp_rho_x=rho_x,
                jvp_data_sign=data_sign,
            ),
        )
        torch.manual_seed(0)
        harness = make_harness(cfg)
        return JVPRegUpdater(cfg, harness), harness

    def _reference_grads(
        self, model, criterion, x, y, xh, yh, *, rho_theta, rho_x=0.0, data_sign=1.0
    ):
        """The documented algorithm, re-derived from scratch.

        The perturbation direction is the NEGATIVE, CURRENT-ONLY gradient —
        not the combined one and not ascent — and the backward pass happens
        at the perturbed point over the half/half mixed batch. A regrowth
        that perturbs along the combined gradient passed the behavioral
        suite once (claims/LOG.md, A13); this pins the real thing.
        """
        ref = _fresh_copy(model)
        g_cur = _grads_at(ref, criterion, x, y)
        gnorm = torch.sqrt(torch.stack([(g**2).sum() for g in g_cur]).sum()) + 1e-12
        with torch.no_grad():
            for p, g in zip(_params(ref), g_cur):
                p.add_(rho_theta * (-g / gnorm))

        n_total = x.shape[0]
        n_hist = min(n_total // 2, xh.shape[0])
        n_cur = n_total - n_hist
        x_h = xh[:n_hist]
        if rho_x != 0.0 and n_hist > 0:
            d = data_sign * (
                x.reshape(x.shape[0], -1).mean(0) - xh.reshape(xh.shape[0], -1).mean(0)
            )
            d = d / (d.norm() + 1e-12)
            x_h = x_h + rho_x * d.reshape(xh.shape[1:])
        x_mix = torch.cat([x[:n_cur], x_h], dim=0)
        y_mix = torch.cat([y[:n_cur], yh[:n_hist]], dim=0)
        loss_mix = criterion(ref(x_mix), y_mix) / A_STEPS
        return torch.autograd.grad(loss_mix, _params(ref)), float(loss_mix.item())

    def test_perturbs_along_negative_current_only_gradient(
        self, make_harness, default_cfg
    ):
        upd, harness = self._updater(make_harness, default_cfg, rho_theta=0.5)
        model = harness.model
        # a historical batch far from the current one, so the current-only
        # and combined gradients disagree and a wrong direction is visible
        x, y = _batch(1)
        xh, yh = _batch(7, scale=3.0, shift=2.0)
        before = [p.detach().clone() for p in _params(model)]

        model.zero_grad()
        loss = upd.fwd_bwd((x, y), (xh, yh))

        expected, expected_loss = self._reference_grads(
            model, harness.get_criterion(), x, y, xh, yh, rho_theta=0.5
        )
        for p, e in zip(_params(model), expected):
            assert_close(p.grad, e)
        assert loss == pytest.approx(expected_loss, rel=1e-6)
        # parameters are always restored
        for p, b in zip(_params(model), before):
            assert_close(p.detach(), b)

    @pytest.mark.parametrize("data_sign", [1.0, -1.0])
    def test_rho_x_shifts_history_along_signed_drift_direction(
        self, make_harness, default_cfg, data_sign
    ):
        upd, harness = self._updater(
            make_harness, default_cfg, rho_theta=0.5, rho_x=2.0, data_sign=data_sign
        )
        model = harness.model
        x, y = _batch(1)
        xh, yh = _batch(7, scale=3.0, shift=2.0)

        model.zero_grad()
        upd.fwd_bwd((x, y), (xh, yh))
        expected, _ = self._reference_grads(
            model,
            harness.get_criterion(),
            x,
            y,
            xh,
            yh,
            rho_theta=0.5,
            rho_x=2.0,
            data_sign=data_sign,
        )
        for p, e in zip(_params(model), expected):
            assert_close(p.grad, e)

    def test_rho_x_zero_and_nonzero_differ(self, make_harness, default_cfg):
        # the input perturbation must actually reach the loss: a regrowth
        # that drops rho_x entirely passed the behavioral suite once
        x, y = _batch(1)
        xh, yh = _batch(7, scale=3.0, shift=2.0)
        grads = []
        for rho_x in (0.0, 2.0):
            upd, harness = self._updater(
                make_harness, default_cfg, rho_theta=0.5, rho_x=rho_x
            )
            harness.model.zero_grad()
            upd.fwd_bwd((x, y), (xh, yh))
            grads.append([p.grad.detach().clone() for p in _params(harness.model)])
        assert any(not torch.allclose(a, b) for a, b in zip(grads[0], grads[1])), (
            "rho_x changed nothing: the input perturbation never reached the loss"
        )

    def test_without_history_is_a_plain_current_step(self, make_harness, default_cfg):
        upd, harness = self._updater(make_harness, default_cfg, rho_theta=0.5)
        model = harness.model
        x, y = _batch(1)
        model.zero_grad()
        loss = upd.fwd_bwd((x, y))
        expected = _grads_at(_fresh_copy(model), harness.get_criterion(), x, y)
        for p, e in zip(_params(model), expected):
            assert_close(p.grad, e)
        assert loss == pytest.approx(
            (harness.get_criterion()(_fresh_copy(model)(x), y) / A_STEPS).item(),
            rel=1e-6,
        )

    def test_post_fwd_bwd_reports_and_resets_the_accumulated_loss(
        self, make_harness, default_cfg
    ):
        upd, harness = self._updater(make_harness, default_cfg)
        x, y = _batch(1)
        xh, yh = _batch(7)
        harness.model.zero_grad()
        loss = upd.fwd_bwd((x, y), (xh, yh))
        assert upd.update_post_fwd_bwd() == pytest.approx(loss)
        assert upd.update_post_fwd_bwd() == 0.0


# ---------------------------------------------------------------------------
# Online KFAC over Conv2d layers
# ---------------------------------------------------------------------------
class TestKFACConv2d:
    """The Conv2d branches of the KFAC factors.

    conftest ships a TinyCNN "for updater tests that need Conv2d layers" and
    nothing used it, so `_a_dim`/`_g_dim`/`_extract_a`/`_extract_g` had never
    run on a convolution and neither had the penalty's `view_as` reshape.
    Assertions are structural — Kronecker factors are Gram matrices, so their
    dimensions, symmetry and positive-semidefiniteness hold whatever the
    implementation — plus the documented gradient relation.
    """

    IMG, CH, NC = 8, 1, 3

    def _setup(self, make_harness, default_cfg, tiny_cnn, lam=2.0, decay=0.5):
        from torch.utils.data import TensorDataset

        cfg = replace(
            default_cfg,
            continual_learning=ContinualLearningCfg(
                update_mode="kfac_online", kfac_lambda=lam, kfac_ema_decay=decay
            ),
        )
        g = torch.Generator().manual_seed(0)
        data = TensorDataset(
            torch.randn(8, self.CH, self.IMG, self.IMG, generator=g),
            torch.randint(0, self.NC, (8,), generator=g),
        )
        torch.manual_seed(0)
        harness = make_harness(cfg, model=tiny_cnn, train_data=data, val_data=data)
        return OnlineKFACUpdater(cfg, harness), harness, data

    def _batch(self, data):
        return data.tensors[0], data.tensors[1]

    def test_factor_dimensions_follow_the_layer_shapes(
        self, make_harness, default_cfg, tiny_cnn
    ):
        upd, harness, _ = self._setup(make_harness, default_cfg, tiny_cnn)
        conv = harness.model.conv
        # A: in_channels * kH * kW; G: out_channels
        a_dim = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
        assert upd.A["conv"].shape == (a_dim, a_dim)
        assert upd.G["conv"].shape == (conv.out_channels, conv.out_channels)
        # the Linear head keeps its own feature dimensions
        assert upd.A["fc"].shape == (harness.model.fc.in_features,) * 2
        assert upd.G["fc"].shape == (harness.model.fc.out_features,) * 2

    def test_committed_conv_factors_are_symmetric_psd(
        self, make_harness, default_cfg, tiny_cnn
    ):
        upd, harness, data = self._setup(make_harness, default_cfg, tiny_cnn)
        upd.cl_preprocessing()
        harness.model.zero_grad()
        upd.fwd_bwd(self._batch(data))
        upd.update_post_fwd_bwd()
        upd.update_post_optimizer_call()
        upd.cl_postprocessing()

        for name in ("conv", "fc"):
            for factor in (upd.A[name], upd.G[name]):
                assert factor.abs().sum() > 0, f"{name}: factor never accumulated"
                assert_close(factor, factor.T)  # Gram matrices are symmetric
                eigenvalues = torch.linalg.eigvalsh(factor.double())
                assert eigenvalues.min() > -1e-6, f"{name}: not PSD"

    def test_conv_penalty_is_applied_with_the_weight_shape(
        self, make_harness, default_cfg, tiny_cnn
    ):
        lam = 2.0
        upd, harness, data = self._setup(make_harness, default_cfg, tiny_cnn, lam=lam)
        model = harness.model
        upd.cl_preprocessing()
        model.zero_grad()
        upd.fwd_bwd(self._batch(data))
        upd.update_post_fwd_bwd()
        upd.update_post_optimizer_call()
        upd.cl_postprocessing()

        weight = model.conv.weight
        diff = torch.randn(weight.shape, generator=torch.Generator().manual_seed(5))
        with torch.no_grad():
            weight.add_(diff)

        model.zero_grad()
        upd.fwd_bwd(self._batch(data))
        base = weight.grad.detach().clone()
        penalty = upd.update_post_fwd_bwd()

        flat = diff.view(weight.shape[0], -1)
        structured = upd.G["conv"] @ flat @ upd.A["conv"]
        # the gradient term is reshaped back to the 4-D weight
        assert weight.grad.shape == weight.shape
        assert_close(
            weight.grad - base, lam * structured.view_as(weight), atol=1e-5, rtol=1e-4
        )
        assert penalty > 0.0

    def test_unsupported_layers_are_skipped(self, make_harness, default_cfg, tiny_cnn):
        # AdaptiveAvgPool2d holds no weight: it must not appear among the factors
        upd, _, _ = self._setup(make_harness, default_cfg, tiny_cnn)
        assert set(upd.A) == {"conv", "fc"}
        assert "pool" not in upd.theta_star
