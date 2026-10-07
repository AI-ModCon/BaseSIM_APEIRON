"""Tests for the Well example (examples/well).

Everything here runs against a miniature dataset with the real one's structure,
so nothing downloads and nothing depends on the ~0.6 GB regime files.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch
import yaml

from examples.well import datasets as well_datasets
from examples.well.datasets import (
    CONVERSION_RECIPE,
    RECIPE_SUFFIX,
    RegimeCache,
    WellDataset,
    convert_to_array,
    get_dataset,
    read,
    trailing_number,
    writing,
)
from examples.well.unet import UNetClassic

# Small, but a multiple of 16 on both axes so the U-Net's four poolings work.
N_TRAJ, N_TIME, HEIGHT, WIDTH = 2, 9, 32, 48

MINI = WellDataset(
    name="mini",
    fields=(
        ("t0_fields", "density", None),
        ("t0_fields", "pressure", None),
        ("t1_fields", "velocity", 0),
        ("t1_fields", "velocity", 1),
    ),
    spatial_resolution=(HEIGHT, WIDTH),
)


def write_regime(path: Path, seed: int) -> None:
    """A miniature file shaped like a Well 2D regime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as handle:
        for group, field, _ in MINI.fields:
            key = f"{group}/{field}"
            if key in handle:
                continue
            shape: tuple[int, ...] = (N_TRAJ, N_TIME, HEIGHT, WIDTH)
            if field == "velocity":
                shape = shape + (2,)
            handle.create_dataset(key, data=rng.normal(size=shape).astype("float32"))


def write_dataset(root: Path, regimes: dict[str, int]) -> Path:
    """A whole miniature Well dataset: regimes plus the stats file."""
    base = root / MINI.name
    for regime, seed in regimes.items():
        for split in ("train", "valid"):
            write_regime(base / "data" / split / f"{regime}.hdf5", seed)
    stats = {
        "mean": {"density": 0.0, "pressure": 0.0, "velocity": [0.0, 0.0]},
        "std": {"density": 1.0, "pressure": 1.0, "velocity": [1.0, 1.0]},
    }
    (base / "stats.yaml").write_text(yaml.safe_dump(stats))
    return base


@pytest.fixture()
def mini_dataset(tmp_path) -> Path:
    write_dataset(tmp_path / "well", {"mini_tcool_0.30": 1, "mini_tcool_1.00": 2})
    return tmp_path / "well"


@pytest.fixture()
def cache(tmp_path, mini_dataset) -> RegimeCache:
    return RegimeCache(MINI, str(mini_dataset), root=tmp_path / "cache")


# ---------------------------------------------------------------------------
# the registry: what makes this extensible to the other Well datasets
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_the_shipped_dataset_is_registered(self):
        spec = get_dataset("turbulent_radiative_layer_2D")
        assert spec.n_channels == 4
        assert spec.spatial_resolution == (128, 384)
        assert spec.n_spatial_dims == 2

    def test_channel_names_label_vector_components(self):
        assert get_dataset("turbulent_radiative_layer_2D").channel_names == [
            "density",
            "pressure",
            "velocity[0]",
            "velocity[1]",
        ]

    def test_an_unknown_dataset_says_what_is_on_offer(self):
        with pytest.raises(NotImplementedError) as excinfo:
            get_dataset("rayleigh_benard")
        message = str(excinfo.value)
        assert "turbulent_radiative_layer_2D" in message
        assert "REGISTRY" in message


class TestOrdering:
    def test_sorts_by_the_swept_parameter(self):
        names = [
            "turbulent_radiative_layer_tcool_0.18",
            "turbulent_radiative_layer_tcool_0.03",
            "turbulent_radiative_layer_tcool_3.16",
        ]
        assert [trailing_number(n) for n in sorted(names, key=trailing_number)] == [
            0.03,
            0.18,
            3.16,
        ]

    def test_beats_a_string_sort_on_unpadded_names(self):
        names = ["x_0.1", "x_0.03", "x_10.0", "x_2.0"]
        assert sorted(names, key=trailing_number) == [
            "x_0.03",
            "x_0.1",
            "x_2.0",
            "x_10.0",
        ]
        assert sorted(names) != sorted(names, key=trailing_number)

    def test_names_without_a_number_sort_last_instead_of_raising(self):
        # The signal that a dataset needs its own `order`. Two-parameter sweeps
        # parse but sort by the wrong parameter, which is why `order` is a row.
        assert trailing_number("viscoelastic_instability_AH") == float("inf")
        assert trailing_number("rayleigh_benard_Rayleigh_1e10_Prandtl_1") == 1.0

    def test_a_row_can_override_the_key(self):
        spec = WellDataset(
            name="two_params",
            fields=MINI.fields,
            spatial_resolution=(16, 16),
            order=lambda n: float(n.split("_")[1]),
        )
        names = ["r_3_p_1", "r_1_p_9"]
        assert sorted(names, key=spec.order) == ["r_1_p_9", "r_3_p_1"]


# ---------------------------------------------------------------------------
# mapped arrays
# ---------------------------------------------------------------------------


class TestWriting:
    def test_lands_the_file_only_on_success(self, tmp_path):
        dest = tmp_path / "a.npy"
        with writing(dest, (2, 3), "float32") as out:
            out[:] = 1.0
        assert dest.exists()
        assert read(dest).shape == (2, 3)

    def test_a_failed_write_leaves_nothing_behind(self, tmp_path):
        dest = tmp_path / "b.npy"
        with pytest.raises(RuntimeError):
            with writing(dest, (2, 3), "float32") as out:
                out[0] = 1.0
                raise RuntimeError("interrupted")
        assert not dest.exists()
        assert list(tmp_path.glob("*.part*")) == []

    def test_reads_are_a_view_not_a_copy(self, tmp_path):
        dest = tmp_path / "c.npy"
        with writing(dest, (4, 4), "float32") as out:
            out[:] = 2.0
        assert isinstance(read(dest), np.memmap)


class TestConversion:
    def test_shape_and_channel_order(self, tmp_path):
        source = tmp_path / "r.hdf5"
        write_regime(source, seed=7)
        dest = tmp_path / "r.npy"
        convert_to_array(MINI, source, dest)

        mapped = read(dest)
        assert mapped.shape == (N_TRAJ, N_TIME, 4, HEIGHT, WIDTH)
        assert mapped.dtype == np.dtype("float32")

        with h5py.File(source) as handle:
            assert np.allclose(mapped[:, :, 0], handle["t0_fields/density"][:])
            assert np.allclose(mapped[:, :, 1], handle["t0_fields/pressure"][:])
            assert np.allclose(mapped[:, :, 2], handle["t1_fields/velocity"][..., 0])
            assert np.allclose(mapped[:, :, 3], handle["t1_fields/velocity"][..., 1])

    def test_stamps_the_recipe(self, tmp_path):
        source = tmp_path / "r.hdf5"
        write_regime(source, seed=7)
        dest = tmp_path / "r.npy"
        convert_to_array(MINI, source, dest)
        assert dest.with_name(dest.name + RECIPE_SUFFIX).read_text() == (
            CONVERSION_RECIPE
        )

    def test_a_wrong_grid_in_the_row_is_reported_with_what_was_found(self, tmp_path):
        source = tmp_path / "r.hdf5"
        write_regime(source, seed=7)
        wrong = WellDataset(
            name="wrong", fields=MINI.fields, spatial_resolution=(64, 64)
        )
        with pytest.raises(ValueError) as excinfo:
            convert_to_array(wrong, source, tmp_path / "r.npy")
        assert f"({HEIGHT}, {WIDTH})" in str(excinfo.value)
        assert "(64, 64)" in str(excinfo.value)


# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------


class TestRegimeCache:
    def test_lists_regimes_in_order_without_reading_them(self, cache):
        assert cache.regimes() == ["mini_tcool_0.30", "mini_tcool_1.00"]

    def test_an_empty_source_is_reported(self, tmp_path):
        empty = RegimeCache(MINI, str(tmp_path / "nothing"), root=tmp_path / "cache")
        with pytest.raises(FileNotFoundError):
            empty.regimes()

    def test_converts_once_and_reuses(self, cache):
        first = cache.array("train", "mini_tcool_0.30")
        stamp = first.stat().st_mtime_ns
        again = cache.array("train", "mini_tcool_0.30")
        assert again == first
        assert again.stat().st_mtime_ns == stamp

    def test_rebuilds_when_the_recipe_moves(self, cache):
        path = cache.array("train", "mini_tcool_0.30")
        recipe = path.with_name(path.name + RECIPE_SUFFIX)
        recipe.write_text("well-hdf5->something-else:v0")
        stamp = path.stat().st_mtime_ns

        rebuilt = cache.array("train", "mini_tcool_0.30")
        assert rebuilt.stat().st_mtime_ns != stamp
        assert recipe.read_text() == CONVERSION_RECIPE

    def test_rebuilds_when_the_stamp_is_missing(self, cache):
        path = cache.array("train", "mini_tcool_0.30")
        path.with_name(path.name + RECIPE_SUFFIX).unlink()
        assert cache.array("train", "mini_tcool_0.30").exists()

    def test_a_local_source_is_read_in_place_and_never_deleted(
        self, cache, mini_dataset
    ):
        cache.array("train", "mini_tcool_0.30")
        assert (
            mini_dataset / MINI.name / "data" / "train" / "mini_tcool_0.30.hdf5"
        ).exists()

    def test_statistics_follow_the_channel_order(self, cache):
        mean, std = cache.statistics()
        assert mean.shape == (4,)
        assert std.shape == (4,)
        assert np.allclose(std, 1.0)

    def test_a_missing_local_file_is_reported(self, cache):
        with pytest.raises(FileNotFoundError):
            cache.array("train", "no_such_regime")


class TestRemoteCache:
    def _cache(self, tmp_path) -> RegimeCache:
        return RegimeCache(MINI, "hf://datasets/polymathic-ai", root=tmp_path / "cache")

    def test_builds_a_download_url(self, tmp_path):
        cache = self._cache(tmp_path)
        assert cache.remote
        assert cache._url("data/train/x.hdf5") == (
            "https://huggingface.co/datasets/polymathic-ai/mini/resolve/main/"
            "data/train/x.hdf5"
        )

    def test_fetches_then_drops_the_hdf5_once_converted(self, tmp_path, mini_dataset):
        cache = self._cache(tmp_path)
        source = mini_dataset / MINI.name / "data" / "train" / "mini_tcool_0.30.hdf5"

        def land(url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read_bytes())

        with patch.object(well_datasets, "fetch", side_effect=land) as fetched:
            path = cache.array("train", "mini_tcool_0.30")

        assert fetched.call_count == 1
        assert read(path).shape == (N_TRAJ, N_TIME, 4, HEIGHT, WIDTH)
        # Re-fetchable, and nothing reads it again: half of what a regime costs.
        assert list((cache.root / "download").rglob("*.hdf5")) == []


# ---------------------------------------------------------------------------
# the dataset
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class TestUNet:
    def test_shapes_match_the_data(self):
        model = UNetClassic(
            dim_in=16, dim_out=4, spatial_resolution=(HEIGHT, WIDTH), init_features=4
        )
        out = model(torch.randn(2, 16, HEIGHT, WIDTH))
        assert out.shape == (2, 4, HEIGHT, WIDTH)

    def test_published_layer_names_are_kept(self):
        # The point of vendoring this: a published checkpoint has to load into
        # it by name. Renaming any of these silently breaks that.
        keys = UNetClassic(dim_in=4, dim_out=4, init_features=4).state_dict().keys()
        for expected in (
            "encoder1.enc1conv1.weight",
            "encoder1.enc1norm1.weight",
            "bottleneck.bottleneckconv2.weight",
            "decoder4.dec4conv1.weight",
            "upconv4.weight",
            "conv.weight",
        ):
            assert expected in keys

    def test_a_resolution_it_cannot_pool_is_refused(self):
        with pytest.raises(ValueError, match="multiple of 16"):
            UNetClassic(dim_in=4, dim_out=4, spatial_resolution=(30, 48))
