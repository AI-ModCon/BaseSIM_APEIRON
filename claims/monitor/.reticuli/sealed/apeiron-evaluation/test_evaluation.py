"""Tests for src/apeiron/evaluation/metrics.py and src/apeiron/evaluation/evaluation.py"""

from __future__ import annotations

import torch
import pytest

from apeiron.evaluation.metrics import accuracy, accuracy_topk


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


class TestAccuracyTopK:
    """The general form behind `accuracy`.

    `accuracy_topk` is documented in CLAUDE.md as part of the evaluation
    layer, is the function `accuracy` delegates to, and had no criteria of
    its own — so every blind reconstruction of this layer dropped it and
    passed (claims/LOG.md, A3).
    """

    def _logits(self):
        # row 0: class order 2, 1, 0 by score; row 1: 0, 2, 1; row 2: 1, 0, 2
        return torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [3.0, 1.0, 2.0],
                [2.0, 3.0, 1.0],
            ]
        )

    def test_returns_one_value_per_k_in_order(self):
        res = accuracy_topk(self._logits(), torch.tensor([2, 0, 1]), topk=(1, 2, 3))
        assert len(res) == 3
        assert [r.item() for r in res] == [pytest.approx(100.0)] * 3

    def test_k_of_one_matches_accuracy(self):
        output, target = self._logits(), torch.tensor([0, 1, 2])
        assert accuracy_topk(output, target, topk=(1,))[0].item() == pytest.approx(
            accuracy(output, target).item()
        )

    def test_a_larger_k_is_never_stricter(self):
        # every target sits second-best, so top-1 misses all and top-2 catches all
        target = torch.tensor([1, 2, 0])
        top1, top2 = accuracy_topk(self._logits(), target, topk=(1, 2))
        assert top1.item() == pytest.approx(0.0)
        assert top2.item() == pytest.approx(100.0)

    def test_partial_credit_at_k(self):
        # rows 0 and 1 have their target in the top 2, row 2 does not
        target = torch.tensor([1, 2, 2])
        (top2,) = accuracy_topk(self._logits(), target, topk=(2,))
        assert top2.item() == pytest.approx(200.0 / 3)

    def test_default_k_is_one(self):
        output, target = self._logits(), torch.tensor([2, 0, 1])
        assert accuracy_topk(output, target)[0].item() == pytest.approx(
            accuracy_topk(output, target, topk=(1,))[0].item()
        )

    def test_k_equal_to_the_class_count_is_always_perfect(self):
        for target in (torch.tensor([0, 0, 0]), torch.tensor([2, 1, 0])):
            (top3,) = accuracy_topk(self._logits(), target, topk=(3,))
            assert top3.item() == pytest.approx(100.0)

    def test_runs_without_building_a_graph(self):
        output = self._logits().requires_grad_(True)
        (res,) = accuracy_topk(output, torch.tensor([2, 0, 1]), topk=(1,))
        assert not res.requires_grad
