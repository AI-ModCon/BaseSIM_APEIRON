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
    """Stand-in for a harness that yields a structured target rather than a Tensor.

    Mirrors the duck type of MATEY's ``MateyTargetBatch``: it carries the target
    alongside per-batch metadata, exposes ``.shape`` and ``.to()``, and
    deliberately does *not* implement ``.size()``.
    """

    def __init__(self, tensor: torch.Tensor, leadtime: int = 1):
        self.tensor = tensor
        self.leadtime = leadtime

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    def to(self, device) -> "_StructuredTarget":
        return _StructuredTarget(self.tensor.to(device), self.leadtime)


class TestBatchSize:
    def test_tensor_uses_leading_dimension(self):
        assert BaseModelHarness._batch_size(torch.randn(7, 3, 3)) == 7

    def test_zero_dim_tensor_counts_as_one(self):
        assert BaseModelHarness._batch_size(torch.tensor(1.0)) == 1

    def test_structured_target_uses_shape(self):
        y = _StructuredTarget(torch.randn(5, 2))
        assert not hasattr(y, "size")
        assert BaseModelHarness._batch_size(y) == 5

    def test_shapeless_target_counts_as_one(self):
        assert BaseModelHarness._batch_size(object()) == 1

    def test_empty_shape_counts_as_one(self):
        class _Scalarish:
            shape = ()

        assert BaseModelHarness._batch_size(_Scalarish()) == 1


class TestEvalWithStructuredTargets:
    """Regression: ``eval`` used ``y.size(0)``, which crashed on non-Tensor targets.

    The crash surfaced only once a detector fired and continual learning began,
    so it is worth pinning both that the call survives and that the per-batch
    weighting still uses the true batch size -- a wrong batch size would
    silently mis-average across variable-sized batches instead of raising.
    """

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
