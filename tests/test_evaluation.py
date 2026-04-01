"""Tests for src/evaluation/metrics.py and src/evaluation/evaluation.py"""

from __future__ import annotations

import torch
import pytest

from apeiron.evaluation.metrics import accuracy


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
