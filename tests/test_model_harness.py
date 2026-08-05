"""Tests for src/model/torch_model_harness.py (BaseModelHarness via DummyHarness)."""

from __future__ import annotations

import pytest
import torch

from apeiron.model.torch_model_harness import BaseModelHarness


class TestUnpack:
    def test_unpack_tuple(self, dummy_harness):
        batch = (torch.randn(4, 4), torch.randint(0, 3, (4,)))
        x, y = dummy_harness._unpack(batch)
        assert x.shape == (4, 4)
        assert y.shape == (4,)


class TestToScalar:
    def test_scalar_tensor(self):
        assert BaseModelHarness._to_scalar(torch.tensor(3.14)) == pytest.approx(3.14)

    def test_1d_tensor_takes_mean(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        assert BaseModelHarness._to_scalar(t) == pytest.approx(2.0)

    def test_float_passthrough(self):
        assert BaseModelHarness._to_scalar(4.5) == 4.5

    def test_int_passthrough(self):
        assert BaseModelHarness._to_scalar(7) == 7.0


class TestEval:
    # TODO: should probably write a test that actually verifies the results here
    def test_returns_list_of_metrics(self, dummy_harness):
        result = dummy_harness.eval()
        assert isinstance(result, list)
        assert len(result) == 1  # just accuracy
        assert 0.0 <= result[0] <= 100.0

    def test_eval_puts_model_in_eval_mode(self, dummy_harness):
        dummy_harness.model.train()
        dummy_harness.eval()
        assert not dummy_harness.model.training


class TestHistoryEval:
    def test_returns_none_without_history(self, dummy_harness):
        result = dummy_harness.history_eval()
        assert result is None

    def test_returns_metrics_with_history(self, dummy_harness_with_history):
        result = dummy_harness_with_history.history_eval()
        assert isinstance(result, list)
        assert len(result) == 1
        assert 0.0 <= result[0] <= 100.0


class TestHarnessAbstract:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            BaseModelHarness(cfg=None, model=None)  # type: ignore[arg-type]


class _StructuredTarget:
    """A target that exposes ``.shape`` but not ``.size()``."""

    def __init__(self, tensor: torch.Tensor, leadtime: int = 1):
        self.tensor = tensor
        self.leadtime = leadtime

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    def to(self, device) -> "_StructuredTarget":
        return _StructuredTarget(self.tensor.to(device), self.leadtime)


class _Scalarish:
    shape = ()


class TestBatchSize:
    @pytest.mark.parametrize(
        "target, expected",
        [
            (torch.randn(7, 3, 3), 7),
            (torch.tensor(1.0), 1),
            (_StructuredTarget(torch.randn(5, 2)), 5),
            (object(), 1),
            (_Scalarish(), 1),
        ],
    )
    def test_leading_dimension(self, target, expected):
        assert BaseModelHarness._batch_size(target) == expected


class TestEvalWithStructuredTargets:
    """Regression: ``eval`` used ``y.size(0)``, which crashed on non-Tensor targets."""

    @staticmethod
    def _run(harness) -> float:
        batches = [
            (torch.randn(5, 4), _StructuredTarget(torch.full((5,), 2.0))),
            (torch.randn(3, 4), _StructuredTarget(torch.full((3,), 10.0))),
        ]
        harness.get_train_dataloaders = lambda: (None, batches)
        harness.eval_metrics = {"mean_target": lambda y_hat, y: y.tensor.mean()}
        (value,) = harness.eval()
        return value

    def test_does_not_crash_and_weights_by_batch_size(self, dummy_harness):
        # (2.0 * 5 + 10.0 * 3) / 8 == 5.0; an unweighted mean would give 6.0.
        assert self._run(dummy_harness) == pytest.approx(5.0)
