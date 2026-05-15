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


class TestCheckpoints:
    """Tests for save_ckpt with metadata, retain-all, and FIFO eviction."""

    def _make_harness(self, tmp_path, make_harness, max_ckpts):
        from apeiron.config.configuration import (
            Config, ModelCfg, DataCfg, TrainCfg,
            ContinualLearningCfg, DriftDetectionCfg,
        )
        cfg = Config(
            model=ModelCfg(
                name="tiny",
                pretrained_path="",
                max_ckpts=max_ckpts,
                ckpts_path=str(tmp_path / "ckpts"),
            ),
            data=DataCfg(name="test", path="/tmp"),
            train=TrainCfg(batch_size=8, num_workers=0, init_lr=0.01, max_iter=2),
            continual_learning=ContinualLearningCfg(),
            drift_detection=DriftDetectionCfg(),
            seed=42,
            device="cpu",
            multi_gpu=False,
        )
        return make_harness(cfg)

    def test_metadata_saved_and_recoverable(self, tmp_path, make_harness):
        h = self._make_harness(tmp_path, make_harness, max_ckpts=-1)
        meta = {"updater": "ewc_online", "drift_event_id": 1, "phase": "post"}
        path = h.save_ckpt(event=1, metadata=meta, tag="post")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert "state_dict" in payload
        assert payload["metadata"] == meta

    def test_retain_all_with_negative_max_ckpts(self, tmp_path, make_harness):
        h = self._make_harness(tmp_path, make_harness, max_ckpts=-1)
        assert h.ckpts_enabled
        for i in range(5):
            h.save_ckpt(event=i, tag="post")
        from pathlib import Path
        ckpt_dir = Path(h.cfg.model.ckpts_path)
        assert len(list(ckpt_dir.glob("drift_adaptation_*.pt"))) == 5

    def test_fifo_eviction_with_positive_max_ckpts(self, tmp_path, make_harness):
        h = self._make_harness(tmp_path, make_harness, max_ckpts=2)
        assert h.ckpts_enabled
        for i in range(4):
            h.save_ckpt(event=i, tag="post")
        from pathlib import Path
        ckpt_dir = Path(h.cfg.model.ckpts_path)
        surviving = list(ckpt_dir.glob("drift_adaptation_*.pt"))
        assert len(surviving) == 2

    def test_ckpts_disabled_when_zero(self, default_cfg, make_harness):
        """max_ckpts=0 (default) disables checkpointing."""
        h = make_harness(default_cfg)
        assert not h.ckpts_enabled

    def test_tag_in_filename(self, tmp_path, make_harness):
        h = self._make_harness(tmp_path, make_harness, max_ckpts=-1)
        path = h.save_ckpt(event=3, tag="pre")
        assert "drift_adaptation_3_pre.pt" in path


class TestHarnessAbstract:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            BaseModelHarness(cfg=None, model=None)  # type: ignore[arg-type]
