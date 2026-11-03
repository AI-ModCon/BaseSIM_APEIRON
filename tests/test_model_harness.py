"""Tests for src/model/torch_model_harness.py (BaseModelHarness via DummyHarness)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from model.torch_model_harness import BaseModelHarness


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


class TestPrometheusLoaders:
    @pytest.fixture()
    def prometheus_harness(self, monkeypatch):
        monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
        from examples.prometheus_torch.model import PrometheusHarness

        return object.__new__(PrometheusHarness)

    @pytest.mark.parametrize(
        ("has_train", "has_val"), [(False, False), (True, False), (False, True)]
    )
    def test_requires_initialized_loaders(
        self, dummy_harness, prometheus_harness, has_train, has_val
    ):
        train, val = dummy_harness.get_cur_data_loaders()
        harness = prometheus_harness
        harness._cur_train_loader = train if has_train else None
        harness._cur_val_loader = val if has_val else None

        with pytest.raises(RuntimeError, match=r"Call update_data_stream\(\)"):
            harness.get_cur_data_loaders()

    def test_returns_initialized_loaders(self, dummy_harness, prometheus_harness):
        train, val = dummy_harness.get_cur_data_loaders()
        harness = prometheus_harness
        harness._cur_train_loader = train
        harness._cur_val_loader = val

        assert harness.get_cur_data_loaders() == (train, val)
