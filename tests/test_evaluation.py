"""Tests for src/apeiron/evaluation/metrics.py and src/apeiron/evaluation/evaluation.py"""

from __future__ import annotations

import torch
import pytest

from apeiron.evaluation.metrics import accuracy, mae, mse, vrmse


class TestAccuracy:
    def test_perfect_predictions(self):
        output = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        target = torch.tensor([0, 1, 2])
        acc = accuracy(output, target)
        assert acc.item() == pytest.approx(100.0)

    def test_all_wrong(self):
        output = torch.tensor([[0.0, 10.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        target = torch.tensor([0, 1, 2])
        acc = accuracy(output, target)
        assert acc.item() == pytest.approx(0.0)

    def test_partial_correct(self):
        output = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
        target = torch.tensor([0, 1, 1, 0])
        acc = accuracy(output, target)
        assert acc.item() == pytest.approx(50.0)

    def test_single_sample(self):
        output = torch.tensor([[5.0, 1.0, 0.0]])
        target = torch.tensor([0])
        acc = accuracy(output, target)
        assert acc.item() == pytest.approx(100.0)


class TestRegressionMetrics:
    def test_mae_and_mse(self):
        out = torch.tensor([1.0, 2.0, 3.0])
        tgt = torch.tensor([1.0, 4.0, 3.0])  # errors: 0, 2, 0
        assert mae(out, tgt).item() == pytest.approx(2.0 / 3.0)
        assert mse(out, tgt).item() == pytest.approx(4.0 / 3.0)

    def test_zero_error(self):
        x = torch.randn(2, 3, 8, 8)
        assert mae(x, x).item() == pytest.approx(0.0)
        assert mse(x, x).item() == pytest.approx(0.0)
        assert vrmse(x, x).item() == pytest.approx(0.0)

    def test_vrmse_predicting_the_mean_is_one(self):
        # Predicting each field's spatial mean gives VRMSE == 1 (by definition).
        target = torch.randn(4, 2, 16, 16)
        pred = target.mean(dim=(2, 3), keepdim=True).expand_as(target)
        assert vrmse(pred, target).item() == pytest.approx(1.0, abs=1e-3)

    def test_vrmse_scale_invariant(self):
        # Scaling a channel scales its error and its std equally -> VRMSE stable.
        target = torch.randn(3, 2, 12, 12)
        pred = target + 0.1 * torch.randn_like(target)
        base = vrmse(pred, target).item()
        scaled = vrmse(pred * 1000, target * 1000).item()
        assert scaled == pytest.approx(base, rel=1e-4)
