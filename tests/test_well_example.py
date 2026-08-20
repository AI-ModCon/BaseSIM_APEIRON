"""Tests for the Well example: converter, surrogate/harness, and benchmark.

Gated on h5py (the only extra dep the Well path needs). Everything runs against
the schema-identical fixture -- no download.
"""

from __future__ import annotations

import glob

import pytest

pytest.importorskip("h5py")

import torch  # noqa: E402

from apeiron.config.configuration import (  # noqa: E402
    Config,
    ContinualLearningCfg,
    DataCfg,
    DriftDetectionCfg,
    ModelCfg,
    TrainCfg,
)
from apeiron.data.window_store import WindowStore  # noqa: E402
from apeiron.model.task_record import InMemoryEvalSet, WindowEvalSetRef  # noqa: E402
from examples.utils import get_example  # noqa: E402
from examples.well.convert import convert_files  # noqa: E402
from examples.well.fixture import generate_fixture  # noqa: E402
from examples.well.model import PDESurrogate, WellHarness  # noqa: E402
from examples.well.wellio import read_well_file  # noqa: E402


@pytest.fixture()
def well_store(tmp_path):
    """A small converted Well store (3 tcool regimes) + its metadata."""
    generate_fixture(
        tmp_path / "src", tcools=(0.03, 0.3, 1.0), n_time=16, height=8, width=12
    )
    meta = convert_files(
        sorted(glob.glob(str(tmp_path / "src" / "*.hdf5"))),
        tmp_path / "store",
        window_steps=8,
        val_fraction=0.25,
    )
    return tmp_path / "store", meta


def _cfg(store, **over):
    base = dict(
        model=ModelCfg(name="well_surrogate", width=8, depth=2),
        data=DataCfg(
            name="well", path=str(store), window_store_path=str(store), batch_size=6
        ),
        train=TrainCfg(batch_size=6, num_workers=0, init_lr=1e-3, max_iter=5),
        continual_learning=ContinualLearningCfg(update_mode="base"),
        drift_detection=DriftDetectionCfg(
            detector_name="PageHinkleyDetector",
            metric_index=0,
            aggregation="mean",
            max_stream_updates=6,
        ),
        seed=0,
        device="cpu",
    )
    base.update(over)
    return Config(**base)


class TestWellIO:
    def test_channel_layout_and_shape(self, tmp_path):
        paths = generate_fixture(tmp_path, tcools=(0.1,), n_time=10, height=8, width=12)
        traj = read_well_file(paths[0])
        assert traj.channels == ("density", "pressure", "velocity_0", "velocity_1")
        assert traj.fields.shape == (10, 4, 8, 12)  # [time, C, H, W]
        assert traj.params["tcool"] == pytest.approx(0.1)


class TestConverter:
    def test_windows_and_drift_order(self, well_store):
        store_path, meta = well_store
        assert meta["channels"] == ["density", "pressure", "velocity_0", "velocity_1"]
        assert meta["grid"] == [8, 12]
        # regimes ordered ascending by tcool -> monotonic drift stream.
        assert meta["regime_order"] == sorted(meta["regime_order"])
        assert len(meta["norm_mean"]) == 4 and len(meta["norm_std"]) == 4

        store = WindowStore(store_path, catalog=False)
        assert len(store) == meta["n_windows"] > 0
        x, y = store.window(store.window_ids()[0]).load_full("all")
        assert x.shape[1:] == (4, 8, 12)
        assert x.shape == y.shape

    def test_skips_windows_too_small_to_split(self, tmp_path):
        # window_steps that leaves a 1-pair tail -> that window is dropped.
        generate_fixture(tmp_path / "src", tcools=(0.1,), n_time=14, height=8, width=8)
        meta = convert_files(
            sorted(glob.glob(str(tmp_path / "src" / "*.hdf5"))),
            tmp_path / "store",
            window_steps=12,
            val_fraction=0.25,
        )
        assert meta["skipped_small_windows"] >= 1


class TestSurrogate:
    def test_residual_shape_preserved(self):
        m = PDESurrogate(4, mean=[0] * 4, std=[1] * 4, width=8, depth=2)
        x = torch.randn(2, 4, 8, 12)
        assert m(x).shape == x.shape

    def test_zero_body_is_identity(self):
        m = PDESurrogate(4, mean=[0] * 4, std=[1] * 4, width=8, depth=2)
        torch.nn.init.zeros_(m.head.weight)
        torch.nn.init.zeros_(m.head.bias)
        x = torch.randn(2, 4, 8, 12)
        # residual with a zero head returns the input unchanged.
        assert torch.allclose(m(x), x, atol=1e-6)


class TestHarness:
    def test_get_example_builds_well_harness(self, well_store):
        h = get_example(_cfg(well_store[0]))
        assert isinstance(h, WellHarness)
        assert list(h.eval_metrics) == ["vrmse", "mae"]
        assert h.higher_is_better == {"vrmse": False, "mae": False}

    def test_task_evalset_is_a_window_pointer(self, well_store):
        h = get_example(_cfg(well_store[0]))
        h.update_data_stream()
        h.register_task([0.0], window_id=h.current_window_id)
        assert isinstance(h._task_records[-1].eval_ref, WindowEvalSetRef)

    def test_copy_mode_uses_in_memory_evalset(self, well_store):
        h = get_example(_cfg(well_store[0]))
        h._copy_task_evalsets = True
        h.update_data_stream()
        h.register_task([0.0], window_id=h.current_window_id)
        assert isinstance(h._task_records[-1].eval_ref, InMemoryEvalSet)

    def test_cl_reduces_error_on_current_window(self, well_store):
        from apeiron.logger.logger import Logger
        from apeiron.training.continuous_trainer import ContinuousTrainer

        cfg = _cfg(well_store[0], train=TrainCfg(6, 0, 1e-3, max_iter=25))
        h = get_example(cfg)
        h.update_data_stream()
        pre = h.eval()[0]
        ContinuousTrainer(
            cfg, h, Logger(verbosity="ERROR", backend="none"), profiler=None
        ).outer_cl_training_loop(drift_event_id=1)
        post = h.eval()[0]
        assert post < pre  # VRMSE went down
