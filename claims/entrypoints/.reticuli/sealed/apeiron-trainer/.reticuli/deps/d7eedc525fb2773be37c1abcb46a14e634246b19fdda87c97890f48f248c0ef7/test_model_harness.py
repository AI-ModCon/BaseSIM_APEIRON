"""Tests for src/apeiron/model/torch_model_harness.py (BaseModelHarness via DummyHarness)."""

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


class TestGetOptimizerDeprecation:
    """The misspelled `get_optmizer` hook is deprecated but must keep working."""

    def test_legacy_subclass_is_bridged_and_warns(self, default_cfg, tiny_model):
        """A harness implementing only `get_optmizer` still satisfies the ABC."""
        with pytest.warns(DeprecationWarning, match="get_optmizer"):

            class LegacyHarness(BaseModelHarness):
                def get_optmizer(self):
                    return "sentinel-optimizer"

                def update_data_stream(self):
                    pass

                def get_stream_dataloader(self):
                    raise NotImplementedError

                def get_hist_dataloaders(self):
                    raise NotImplementedError

                def get_criterion(self):
                    raise NotImplementedError

        harness = LegacyHarness(default_cfg, tiny_model)
        # The framework only calls get_optimizer(); it must reach the legacy impl.
        assert harness.get_optimizer() == "sentinel-optimizer"
        assert harness.get_optmizer() == "sentinel-optimizer"

    def test_new_subclass_does_not_warn(self, recwarn, dummy_harness):
        """The in-repo harnesses use the new spelling and must stay silent."""
        assert dummy_harness.get_optimizer() is not None
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]

    def test_old_name_still_callable_on_new_subclass(self, dummy_harness):
        """External callers using the old name get a warning but a working result."""
        with pytest.warns(DeprecationWarning, match="get_optmizer"):
            legacy_call = dummy_harness.get_optmizer()
        assert type(legacy_call) is type(dummy_harness.get_optimizer())
