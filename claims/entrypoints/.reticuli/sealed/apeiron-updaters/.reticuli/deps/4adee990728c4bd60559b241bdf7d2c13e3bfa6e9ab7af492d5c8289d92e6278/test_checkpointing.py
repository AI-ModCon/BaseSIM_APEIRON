"""Tests for the harness's FIFO post-adaptation checkpointing.

`ckpts_enabled`, `build_checkpoint_payload`, and `save_ckpt` are consulted by
both the monitor and the scheduled-CL runner after every adaptation, and
shipped without tests. The retention budget is the part worth pinning: it
decides which states a later restart can still reach.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from apeiron.config.configuration import ModelCfg


def _cfg(default_cfg, tmp_path, max_ckpts):
    return replace(
        default_cfg,
        model=ModelCfg(
            name="tiny",
            pretrained_path="",
            max_ckpts=max_ckpts,
            ckpts_path=str(tmp_path / "ckpts"),
        ),
    )


def _saved(tmp_path):
    return sorted(p.name for p in (tmp_path / "ckpts").glob("drift_adaptation_*.pt"))


class TestCkptsEnabled:
    def test_disabled_by_default(self, dummy_harness):
        assert dummy_harness.ckpts_enabled is False

    def test_needs_both_a_budget_and_a_path(self, make_harness, default_cfg, tmp_path):
        assert make_harness(_cfg(default_cfg, tmp_path, 0)).ckpts_enabled is False
        no_path = replace(
            default_cfg,
            model=ModelCfg(name="tiny", pretrained_path="", max_ckpts=3, ckpts_path=""),
        )
        assert make_harness(no_path).ckpts_enabled is False
        assert make_harness(_cfg(default_cfg, tmp_path, 3)).ckpts_enabled is True


class TestSaveCkpt:
    def test_writes_the_payload_and_the_latest_pointer(
        self, make_harness, default_cfg, tmp_path
    ):
        harness = make_harness(_cfg(default_cfg, tmp_path, 3))
        path = harness.save_ckpt(event=4)

        assert path.endswith("drift_adaptation_4.pt")
        assert _saved(tmp_path) == ["drift_adaptation_4.pt"]
        assert (tmp_path / "ckpts" / "latest").read_text() == "drift_adaptation_4.pt"

        loaded = torch.load(path, weights_only=True)
        expected = harness.build_checkpoint_payload()
        assert loaded.keys() == expected.keys()
        for key, value in expected.items():
            assert torch.equal(loaded[key], value)

    def test_creates_the_directory(self, make_harness, default_cfg, tmp_path):
        harness = make_harness(_cfg(default_cfg, tmp_path, 1))
        assert not (tmp_path / "ckpts").exists()
        harness.save_ckpt(event=0)
        assert (tmp_path / "ckpts").is_dir()

    def test_budget_evicts_the_oldest(self, make_harness, default_cfg, tmp_path):
        # eviction orders by mtime, so this pins the documented intent on any
        # filesystem whose timestamps resolve finer than a save takes
        harness = make_harness(_cfg(default_cfg, tmp_path, 2))
        for event in range(4):
            harness.save_ckpt(event=event)
        assert len(_saved(tmp_path)) == 2, _saved(tmp_path)
        assert "drift_adaptation_3.pt" in _saved(tmp_path)
        assert "drift_adaptation_0.pt" not in _saved(tmp_path)

    def test_latest_always_names_the_newest(self, make_harness, default_cfg, tmp_path):
        harness = make_harness(_cfg(default_cfg, tmp_path, 2))
        for event in range(3):
            harness.save_ckpt(event=event)
        assert (tmp_path / "ckpts" / "latest").read_text() == "drift_adaptation_2.pt"

    def test_payload_tracks_the_trained_weights(
        self, make_harness, default_cfg, tmp_path
    ):
        harness = make_harness(_cfg(default_cfg, tmp_path, 2))
        first = harness.save_ckpt(event=0)
        with torch.no_grad():
            for p in harness.model.parameters():
                p.add_(1.0)
        second = harness.save_ckpt(event=1)

        before = torch.load(first, weights_only=True)
        after = torch.load(second, weights_only=True)
        assert any(not torch.equal(before[k], after[k]) for k in before), (
            "a checkpoint must capture the weights at its own event"
        )

    def test_subclasses_can_extend_the_payload(
        self, make_harness, default_cfg, tmp_path
    ):
        harness = make_harness(_cfg(default_cfg, tmp_path, 1))
        harness.build_checkpoint_payload = lambda: {  # type: ignore[method-assign]
            "state": harness.model.state_dict(),
            "scaler": 2.5,
        }
        loaded = torch.load(harness.save_ckpt(event=0), weights_only=False)
        assert loaded["scaler"] == 2.5
        assert "state" in loaded
