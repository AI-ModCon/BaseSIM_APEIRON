"""Tests for apeiron.evaluation.evaluation.

This module is exported through `docs/api/evaluation.md` but nothing in the
framework calls it, and bringing it under test turned up two defects. They
are recorded here as expected failures rather than pinned as behavior: a
passing test asserting a 6x loss error would make the error official. See
claims/LOG.md (A19) for the recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from apeiron.evaluation.evaluation import test as evaluate


@dataclass
class _Cfg:
    device: str = "cpu"


def _data(n: int = 12, seed: int = 0) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    return TensorDataset(
        torch.randn(n, 4, generator=g), torch.randint(0, 3, (n,), generator=g)
    )


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(4, 3)


class TestAccuracy:
    def test_accuracy_is_a_percentage_of_correct_predictions(self):
        data = _data()
        model = _model()
        accuracy, _ = evaluate(
            model, DataLoader(data, batch_size=8), nn.CrossEntropyLoss(), _Cfg()
        )

        with torch.no_grad():
            predicted = model(data.tensors[0]).argmax(dim=1)
        expected = 100.0 * (predicted == data.tensors[1]).sum().item() / len(data)
        assert accuracy == pytest.approx(expected)

    def test_perfect_and_zero_accuracy(self):
        # a model that always answers class 0, against all-0 then all-1 labels
        model = nn.Linear(4, 3)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.copy_(torch.tensor([10.0, 0.0, 0.0]))
        x = torch.randn(6, 4, generator=torch.Generator().manual_seed(1))

        right = TensorDataset(x, torch.zeros(6, dtype=torch.long))
        wrong = TensorDataset(x, torch.ones(6, dtype=torch.long))
        crit = nn.CrossEntropyLoss()
        assert (
            evaluate(model, DataLoader(right, batch_size=4), crit, _Cfg())[0] == 100.0
        )
        assert evaluate(model, DataLoader(wrong, batch_size=4), crit, _Cfg())[0] == 0.0

    def test_model_is_left_in_eval_mode(self):
        model = _model()
        model.train()
        evaluate(
            model, DataLoader(_data(), batch_size=8), nn.CrossEntropyLoss(), _Cfg()
        )
        assert not model.training

    def test_batch_size_does_not_change_accuracy(self):
        data, model, crit = _data(), _model(), nn.CrossEntropyLoss()
        scores = [
            evaluate(model, DataLoader(data, batch_size=bs), crit, _Cfg())[0]
            for bs in (1, 5, 12)
        ]
        assert scores[0] == pytest.approx(scores[1]) == pytest.approx(scores[2])


class TestEmptyLoader:
    @pytest.mark.xfail(
        reason="A19: returns a bare 0.0 instead of a (accuracy, loss) pair, "
        "so every caller unpacking the result raises TypeError",
        strict=True,
    )
    def test_empty_loader_returns_a_pair(self):
        empty = TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long))
        accuracy, loss = evaluate(
            _model(), DataLoader(empty, batch_size=4), nn.CrossEntropyLoss(), _Cfg()
        )
        assert (accuracy, loss) == (0.0, 0.0)

    def test_empty_loader_currently_returns_a_scalar(self):
        # the present behavior, recorded so the xfail above has a counterpart
        empty = TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long))
        assert (
            evaluate(
                _model(), DataLoader(empty, batch_size=4), nn.CrossEntropyLoss(), _Cfg()
            )
            == 0.0
        )


class TestLoss:
    @pytest.mark.xfail(
        reason="A19: batch losses from a mean-reduction criterion are summed and "
        "then divided by the SAMPLE count, under-reporting by samples/batches "
        "(6x for 12 samples in 2 batches). The code comment claims a sum "
        "reduction it never requests, and no harness supplies one.",
        strict=True,
    )
    def test_loss_is_the_mean_over_samples(self):
        data, model = _data(), _model()
        criterion = nn.CrossEntropyLoss()  # what every harness returns
        _, reported = evaluate(model, DataLoader(data, batch_size=8), criterion, _Cfg())
        with torch.no_grad():
            expected = criterion(model(data.tensors[0]), data.tensors[1]).item()
        assert reported == pytest.approx(expected, rel=1e-5)

    def test_loss_is_correct_only_for_a_sum_reduction_criterion(self):
        # the one caller shape that makes the arithmetic come out right
        data, model = _data(), _model()
        _, reported = evaluate(
            model,
            DataLoader(data, batch_size=8),
            nn.CrossEntropyLoss(reduction="sum"),
            _Cfg(),
        )
        with torch.no_grad():
            expected = nn.CrossEntropyLoss()(
                model(data.tensors[0]), data.tensors[1]
            ).item()
        assert reported == pytest.approx(expected, rel=1e-5)

    def test_reported_loss_scales_with_the_batch_count(self):
        # the defect's signature: a quantity that should be batch-size
        # invariant moves by the number of batches
        data, model, crit = _data(n=12), _model(), nn.CrossEntropyLoss()
        one_batch = evaluate(model, DataLoader(data, batch_size=12), crit, _Cfg())[1]
        six_batches = evaluate(model, DataLoader(data, batch_size=2), crit, _Cfg())[1]
        assert six_batches == pytest.approx(6 * one_batch, rel=1e-5)
