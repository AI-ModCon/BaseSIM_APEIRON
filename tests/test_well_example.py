"""Tests for the Well example (examples/well).

Everything here runs against a miniature dataset with the real one's structure,
so nothing downloads and nothing depends on the ~0.6 GB regime files.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch
import yaml

from apeiron.config.configuration import (
    Config,
    ContinualLearningCfg,
    DataCfg,
    DriftDetectionCfg,
    ModelCfg,
    TrainCfg,
)
from examples.well import datasets as well_datasets
from examples.well import evaluate
from examples.well import plot
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
from examples.well.model import (
    FrozenBatchNormUNet,
    N_STEPS_INPUT,
    N_STEPS_OUTPUT,
    UNET_WIDTHS,
    WELL_UNET,
    RegimeDataset,
    RegimeFrames,
    vrmse,
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


def well_cfg(data_root: Path, name: str = "unet_small") -> Config:
    return Config(
        model=ModelCfg(name=name),
        data=DataCfg(name="well:mini", path=str(data_root), batch_size=2),
        train=TrainCfg(batch_size=2, num_workers=0, init_lr=1e-3, max_iter=1),
        continual_learning=ContinualLearningCfg(update_mode="base"),
        drift_detection=DriftDetectionCfg(detection_interval=2, max_stream_updates=1),
        seed=3,
        device="cpu",
    )


@pytest.fixture()
def registered():
    """Make the miniature dataset visible to get_dataset()."""
    with patch.dict(well_datasets.REGISTRY, {MINI.name: MINI}):
        yield MINI


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


# ---------------------------------------------------------------------------
# the dataset
# ---------------------------------------------------------------------------


class TestOfflineUse:
    def test_a_remote_listing_is_cached_and_reused_without_network(self, tmp_path):
        """After one listing (or a prefetch), regimes() reads regimes.json and
        never touches the network -- which is what lets a run start on a
        compute node that has no internet access."""
        cache = RegimeCache(MINI, "hf://datasets/polymathic-ai", root=tmp_path / "c")
        listing = [
            {"path": "data/train/mini_tcool_1.00.hdf5"},
            {"path": "data/train/mini_tcool_0.30.hdf5"},
            {"path": "data/train/README.md"},
        ]
        response = io.BytesIO(json.dumps(listing).encode())
        with patch.object(
            well_datasets.urllib.request, "urlopen", return_value=response
        ) as opened:
            first = cache.regimes()
        assert opened.call_count == 1
        assert first == ["mini_tcool_0.30", "mini_tcool_1.00"]
        assert (tmp_path / "c" / "regimes.json").is_file()

        def no_network(*args, **kwargs):
            raise AssertionError("regimes() touched the network")

        with patch.object(well_datasets.urllib.request, "urlopen", no_network):
            assert cache.regimes() == first

    def test_prefetch_converts_every_requested_split(
        self, registered, mini_dataset, tmp_path, capsys
    ):
        config = tmp_path / "mini.toml"
        config.write_text(
            "seed = 3\n"
            'device = "cpu"\n'
            "[model]\n"
            'name = "unet_small"\n'
            "[data]\n"
            'name = "well:mini"\n'
            f'path = "{mini_dataset}"\n'
            "batch_size = 2\n"
            "[train]\n"
            "batch_size = 2\n"
            "num_workers = 0\n"
            "init_lr = 1e-3\n"
            "[drift_detection]\n"
            'detector_name = "PageHinkleyDetector"\n'
        )
        with patch.object(well_datasets, "CACHE_ROOT", tmp_path / "root"):
            assert (
                well_datasets.main(["--config", str(config), "--splits", "train,valid"])
                == 0
            )
        for split in ("train", "valid"):
            for regime in ("mini_tcool_0.30", "mini_tcool_1.00"):
                assert (tmp_path / "root" / "mini" / split / f"{regime}.npy").is_file()
        assert "2 regimes" in capsys.readouterr().out


class TestRegimeDataset:
    def _dataset(self, cache) -> RegimeDataset:
        mean, std = cache.statistics()
        frames = RegimeFrames(cache.array("train", "mini_tcool_0.30"))
        return RegimeDataset(frames, mean, std)

    def test_sample_shapes_fold_time_into_channels(self, cache):
        x, y = self._dataset(cache)[0]
        assert x.shape == (N_STEPS_INPUT * 4, HEIGHT, WIDTH)
        assert y.shape == (N_STEPS_OUTPUT * 4, HEIGHT, WIDTH)

    def test_length_counts_every_sliding_window(self, cache):
        per_trajectory = N_TIME - N_STEPS_INPUT - N_STEPS_OUTPUT + 1
        assert len(self._dataset(cache)) == N_TRAJ * per_trajectory

    def test_the_target_follows_the_inputs_in_time(self, cache):
        dataset = self._dataset(cache)
        x_first, _ = dataset[0]
        _, y_last_of_first = dataset[N_STEPS_INPUT]
        # Sample N_STEPS_INPUT predicts the frame that sample 0's input ended at
        # plus N_STEPS_INPUT, so the two overlap in a checkable way.
        assert x_first.shape[0] == N_STEPS_INPUT * 4
        assert y_last_of_first.shape[0] == N_STEPS_OUTPUT * 4

    def test_normalisation_is_applied(self, cache):
        mean, std = cache.statistics()
        frames = RegimeFrames(cache.array("train", "mini_tcool_0.30"))
        raw, _ = RegimeDataset(frames, mean, std)[0]
        shifted, _ = RegimeDataset(frames, mean + 1.0, std)[0]
        assert torch.allclose(raw - 1.0, shifted, atol=1e-5)

    def test_too_short_a_regime_is_reported(self, tmp_path):
        dest = tmp_path / "short.npy"
        with writing(dest, (1, 2, 4, HEIGHT, WIDTH), "float32") as out:
            out[:] = 0.0
        mean = np.zeros(4, dtype="float32")
        with pytest.raises(ValueError, match="too few time steps"):
            RegimeDataset(RegimeFrames(dest), mean, mean + 1.0)


# ---------------------------------------------------------------------------
# the metric
# ---------------------------------------------------------------------------


class TestVrmse:
    def test_zero_for_a_perfect_prediction(self):
        y = torch.randn(2, 4, 8, 8)
        assert vrmse(y.clone(), y).item() == pytest.approx(0.0, abs=1e-6)

    def test_scale_invariant_per_channel(self):
        y = torch.randn(2, 4, 8, 8)
        y_hat = y + 0.1 * torch.randn_like(y)
        scaled = torch.tensor([1.0, 10.0, 100.0, 1000.0]).view(1, 4, 1, 1)
        assert vrmse(y_hat, y).item() == pytest.approx(
            vrmse(y_hat * scaled, y * scaled).item(), rel=1e-4
        )

    def test_grows_with_error(self):
        y = torch.randn(2, 4, 8, 8)
        near = vrmse(y + 0.01 * torch.randn_like(y), y)
        far = vrmse(y + 1.00 * torch.randn_like(y), y)
        assert far > near

    def test_square_root_is_taken_before_averaging(self):
        """Pins the reduction order against The Well's own definition.

        Their NRMSE applies sqrt to the normalised MSE tensor and leaves the
        reduction to the caller, so VRMSE is mean(sqrt(ratio)). Averaging the
        ratios and taking one sqrt gives a larger number -- it agrees to three
        decimals on homoscedastic noise, which is why this needs a case where
        the per-sample ratios differ by orders of magnitude, and is worth a test
        because the wrong form is not comparable with their published results.
        """
        # Two samples, one channel, constructed so the ratios are 0.01 and 4.0.
        y = torch.tensor([[[0.0, 0.0, 2.0, 2.0]], [[0.0, 0.0, 0.0, 4.0]]])
        # unbiased spatial variance: 4/3 and 4.0
        assert torch.allclose(y.var(dim=2), torch.tensor([[4 / 3], [4.0]]))
        y_hat = y + torch.tensor([[[0.11547]], [[4.0]]])

        # ratios 0.01 and 4.0 -> sqrt 0.1 and 2.0 -> mean 1.05
        assert vrmse(y_hat, y).item() == pytest.approx(1.05, abs=1e-3)
        # the wrong form would give sqrt((0.01 + 4.0) / 2) = 1.4160
        assert vrmse(y_hat, y).item() != pytest.approx(1.4160, abs=1e-2)

    def test_matches_a_reference_value_from_the_well(self):
        """A value computed with the_well's own VRMSE on a fixed input.

        the_well is not a dependency of this repository, so the number is
        pinned here rather than recomputed. Reproduce with:

            from the_well.benchmark.metrics import VRMSE
            VRMSE.eval(y_hat_channels_last, y_channels_last, meta).mean()
        """
        torch.manual_seed(0)
        y = torch.randn(3, 4, 32, 48)
        y_hat = y + 0.3 * torch.randn(3, 4, 32, 48)
        assert vrmse(y_hat, y).item() == pytest.approx(0.297809, abs=1e-5)


# ---------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------


class TestHarness:
    def test_discovers_regimes_in_order(self, registered, mini_dataset):
        harness = WELL_UNET(cfg=well_cfg(mini_dataset))
        assert harness.regimes == ["mini_tcool_0.30", "mini_tcool_1.00"]

    def test_a_name_without_a_dataset_is_refused(self, registered, mini_dataset):
        cfg = well_cfg(mini_dataset)
        bad = Config(
            **{**cfg.__dict__, "data": DataCfg(name="well", path="", batch_size=1)}
        )
        with pytest.raises(ValueError, match="well:<dataset>"):
            WELL_UNET(cfg=bad)

    def test_an_unknown_model_name_is_refused(self, registered, mini_dataset):
        with pytest.raises(ValueError, match="unet"):
            WELL_UNET(cfg=well_cfg(mini_dataset, name="resnet"))

    @pytest.mark.parametrize("name", sorted(UNET_WIDTHS))
    def test_builds_at_either_width(self, registered, mini_dataset, name):
        harness = WELL_UNET(cfg=well_cfg(mini_dataset, name=name))
        widths = {p.shape[0] for p in harness.model.encoder1.parameters()}
        assert widths == {UNET_WIDTHS[name]}

    def test_a_window_end_to_end(self, registered, mini_dataset):
        harness = WELL_UNET(cfg=well_cfg(mini_dataset))
        harness.update_data_stream()

        assert harness.window == 0
        x, y = next(iter(harness.get_stream_dataloader()))
        assert x.shape[1:] == (N_STEPS_INPUT * 4, HEIGHT, WIDTH)

        with torch.no_grad():
            assert harness.model(x).shape == y.shape

    def test_history_appears_only_after_the_first_window(
        self, registered, mini_dataset
    ):
        harness = WELL_UNET(cfg=well_cfg(mini_dataset))
        harness.update_data_stream()
        assert harness.get_hist_dataloaders() == (None, None)

        harness.update_data_stream()
        hist_train, hist_valid = harness.get_hist_dataloaders()
        assert hist_train is not None and hist_valid is not None
        # One prior regime, and it is the one the first window showed.
        assert len(hist_train.dataset) == len(
            harness.get_train_dataloaders()[0].dataset
        )

    def test_the_last_regime_holds_if_the_stream_runs_on(
        self, registered, mini_dataset
    ):
        harness = WELL_UNET(cfg=well_cfg(mini_dataset))
        for _ in range(4):
            harness.update_data_stream()
        assert harness.window == 3
        assert harness._regime(harness.window) == "mini_tcool_1.00"

    def test_the_model_init_is_seeded(self, registered, mini_dataset):
        """Same seed, same weights.

        Nothing in the framework seeds a torch RNG, so without the harness
        doing it the model is a fresh random draw per run -- and with an
        untrained model the per-batch error *is* the drift signal, so the whole
        trace moves and detector settings stop meaning anything.
        """
        cfg = well_cfg(mini_dataset)
        first = WELL_UNET(cfg=cfg).model.state_dict()["encoder1.enc1conv1.weight"]
        again = WELL_UNET(cfg=cfg).model.state_dict()["encoder1.enc1conv1.weight"]
        assert torch.equal(first, again)

        other = Config(**{**cfg.__dict__, "seed": cfg.seed + 1})
        differs = WELL_UNET(cfg=other).model.state_dict()["encoder1.enc1conv1.weight"]
        assert not torch.equal(first, differs)

    def test_shuffled_loaders_are_repeatable(self, registered, mini_dataset):
        def first_batch_indices():
            harness = WELL_UNET(cfg=well_cfg(mini_dataset))
            harness.update_data_stream()
            x, _ = next(iter(harness.get_train_dataloaders()[0]))
            return x.clone()

        # The framework seeds nothing globally, so the harness seeds its own
        # loaders; without that the drift trace would differ run to run.
        assert torch.equal(first_batch_indices(), first_batch_indices())

    def test_evaluates_both_metrics(self, registered, mini_dataset):
        harness = WELL_UNET(cfg=well_cfg(mini_dataset))
        harness.update_data_stream()
        assert list(harness.eval_metrics) == ["vrmse", "mse"]
        scores = harness.eval()
        assert len(scores) == 2
        assert all(np.isfinite(s) for s in scores)


# ---------------------------------------------------------------------------
# frozen batch statistics
# ---------------------------------------------------------------------------


class TestFrozenBatchNorm:
    def test_batchnorm_stays_in_eval_while_the_model_trains(self):
        model = FrozenBatchNormUNet(
            dim_in=16, dim_out=4, spatial_resolution=(HEIGHT, WIDTH), init_features=4
        )
        model.train()
        assert model.training
        layers = [
            m
            for m in model.modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
        ]
        assert layers and all(not m.training for m in layers)

    def test_running_statistics_survive_training_mode_passes(self):
        model = FrozenBatchNormUNet(
            dim_in=16, dim_out=4, spatial_resolution=(HEIGHT, WIDTH), init_features=4
        )
        model.train()
        layer = next(
            m
            for m in model.modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
        )
        before = layer.running_mean.clone()
        with torch.no_grad():
            for _ in range(5):
                model(torch.randn(2, 16, HEIGHT, WIDTH))
        assert torch.equal(before, layer.running_mean)

    def test_an_ordinary_unet_does_move_its_statistics(self):
        """The control: this is the behaviour the frozen variant exists to stop."""
        model = UNetClassic(
            dim_in=16, dim_out=4, spatial_resolution=(HEIGHT, WIDTH), init_features=4
        )
        model.train()
        layer = next(
            m
            for m in model.modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
        )
        before = layer.running_mean.clone()
        with torch.no_grad():
            for _ in range(5):
                model(torch.randn(2, 16, HEIGHT, WIDTH))
        assert not torch.equal(before, layer.running_mean)

    def test_the_published_checkpoint_shape_still_loads(self):
        """Freezing must not rename or add anything."""
        plain = UNetClassic(dim_in=16, dim_out=4, init_features=4)
        frozen = FrozenBatchNormUNet(dim_in=16, dim_out=4, init_features=4)
        assert list(plain.state_dict()) == list(frozen.state_dict())
        frozen.load_state_dict(plain.state_dict(), strict=True)

    def test_the_harness_always_builds_it(self, registered, mini_dataset):
        """Not a variant to opt into: every window the harness serves is a
        different regime, so live batch statistics are never what is wanted."""
        for name in sorted(UNET_WIDTHS):
            harness = WELL_UNET(cfg=well_cfg(mini_dataset, name=name))
            assert isinstance(harness.model, FrozenBatchNormUNet)


# ---------------------------------------------------------------------------
# the test-split evaluation pass
# ---------------------------------------------------------------------------


class TestEvaluate:
    def _checkpoint_dir(self, tmp_path, events=(1, 2, 10)) -> Path:
        """A directory shaped like what save_ckpt leaves behind."""
        directory = tmp_path / "ckpts"
        directory.mkdir()
        for event in events:
            model = evaluate.build_model(MINI, 4)
            torch.save(model.state_dict(), directory / f"drift_adaptation_{event}.pt")
        (directory / "latest").write_text(f"drift_adaptation_{max(events)}.pt")
        return directory

    def test_a_reference_is_built_at_the_width_in_its_own_file(self, tmp_path):
        """The Well's published weights are 48 whatever width a run used.

        Reading the width from the file is what lets them be scored beside
        checkpoints from a narrower run, in one table.
        """
        wide = evaluate.build_model(MINI, 8)
        path = tmp_path / "wide.pt"
        torch.save(wide.state_dict(), path)

        weights = evaluate.load_weights(path)
        assert evaluate.width_of(weights) == 8

        rebuilt = evaluate.build_model(MINI, evaluate.width_of(weights))
        rebuilt.load_state_dict(weights, strict=True)

    def test_checkpoints_are_found_in_event_order(self, tmp_path):
        directory = self._checkpoint_dir(tmp_path, events=(10, 2, 1))
        found = evaluate.checkpoints(directory)
        assert [label for label, _ in found] == ["event_001", "event_002", "event_010"]

    def test_the_latest_marker_is_not_mistaken_for_a_checkpoint(self, tmp_path):
        directory = self._checkpoint_dir(tmp_path, events=(1,))
        assert len(evaluate.checkpoints(directory)) == 1

    def test_weights_round_trip_through_a_checkpoint(self, tmp_path):
        model = evaluate.build_model(MINI, 4)
        path = tmp_path / "drift_adaptation_1.pt"
        torch.save(model.state_dict(), path)
        loaded = evaluate.load_weights(path)
        assert set(loaded) == set(model.state_dict())

    def test_a_payload_wrapping_the_state_dict_is_accepted(self, tmp_path):
        model = evaluate.build_model(MINI, 4)
        path = tmp_path / "wrapped.pt"
        torch.save({"state_dict": model.state_dict(), "window": 3}, path)
        assert set(evaluate.load_weights(path)) == set(model.state_dict())

    def test_persistence_repeats_the_last_input_frame(self):
        channels = MINI.n_channels
        x = torch.arange(float(N_STEPS_INPUT * channels)).reshape(1, -1, 1, 1)
        last = evaluate.persistence(x, channels)
        assert last.shape[1] == N_STEPS_OUTPUT * channels
        assert torch.equal(last, x[:, -N_STEPS_OUTPUT * channels :])

    def test_scores_checkpoints_and_baselines_against_the_test_split(
        self, tmp_path, mini_dataset, capsys
    ):
        write_regime(
            mini_dataset / MINI.name / "data" / "test" / "mini_tcool_0.30.hdf5", 5
        )
        write_regime(
            mini_dataset / MINI.name / "data" / "test" / "mini_tcool_1.00.hdf5", 6
        )
        config = tmp_path / "mini.toml"
        config.write_text(
            "seed = 3\n"
            'device = "cpu"\n'
            "[model]\n"
            'name = "unet_small"\n'
            f'ckpts_path = "{self._checkpoint_dir(tmp_path, events=(1, 2))}"\n'
            "max_ckpts = 5\n"
            "[data]\n"
            'name = "well:mini"\n'
            f'path = "{mini_dataset}"\n'
            "batch_size = 2\n"
            "[train]\n"
            "batch_size = 2\n"
            "num_workers = 0\n"
            "init_lr = 1e-3\n"
            "[drift_detection]\n"
            'detector_name = "PageHinkleyDetector"\n'
        )
        narrow = WellDataset(
            name="mini",
            fields=MINI.fields,
            spatial_resolution=MINI.spatial_resolution,
        )
        with (
            patch.dict(well_datasets.REGISTRY, {"mini": narrow}),
            patch.dict(evaluate.UNET_WIDTHS, {"unet_small": 4}),
        ):
            assert evaluate.main(["--config", str(config), "--baselines"]) == 0

        out = capsys.readouterr().out
        # every scored model, and both regimes as columns
        for expected in ("persistence", "zero", "event_001", "event_002", "MEAN"):
            assert expected in out

    def test_the_initial_model_row_is_reproducible(self, tmp_path, mini_dataset):
        """The baseline has to be the model the run started from, every time.

        An unseeded draw makes "did adapting help?" meaningless, because the
        reference moves between invocations.
        """
        import torch as _torch

        from apeiron.config.configuration import DataCfg as _DataCfg

        cfg = well_cfg(mini_dataset)
        narrow = WellDataset(
            name="mini",
            fields=MINI.fields,
            spatial_resolution=MINI.spatial_resolution,
        )

        def draw():
            _torch.manual_seed(cfg.seed)
            return evaluate.build_model(narrow, 4).state_dict()[
                "encoder1.enc1conv1.weight"
            ]

        _torch.manual_seed(999)  # disturb the global RNG between draws
        first = draw()
        _torch.manual_seed(12345)
        assert _torch.equal(first, draw())
        assert isinstance(_DataCfg, type)

    def test_writes_the_matrix_to_csv(self, tmp_path, mini_dataset):
        for regime, seed in (("mini_tcool_0.30", 5), ("mini_tcool_1.00", 6)):
            write_regime(
                mini_dataset / MINI.name / "data" / "test" / f"{regime}.hdf5", seed
            )
        config = tmp_path / "mini.toml"
        config.write_text(
            "seed = 3\n"
            'device = "cpu"\n'
            "[model]\n"
            'name = "unet_small"\n'
            'ckpts_path = ""\n'
            "[data]\n"
            'name = "well:mini"\n'
            f'path = "{mini_dataset}"\n'
            "batch_size = 2\n"
            "[train]\n"
            "batch_size = 2\n"
            "num_workers = 0\n"
            "init_lr = 1e-3\n"
            "[drift_detection]\n"
            'detector_name = "PageHinkleyDetector"\n'
        )
        out_csv = tmp_path / "matrix.csv"
        narrow = WellDataset(
            name="mini",
            fields=MINI.fields,
            spatial_resolution=MINI.spatial_resolution,
        )
        with (
            patch.dict(well_datasets.REGISTRY, {"mini": narrow}),
            patch.dict(evaluate.UNET_WIDTHS, {"unet_small": 4}),
        ):
            evaluate.main(
                ["--config", str(config), "--baselines", "--out", str(out_csv)]
            )
        header, *body = out_csv.read_text().splitlines()
        assert header.startswith("model,mini_tcool_0.30,mini_tcool_1.00,mean")
        assert len(body) == 2


class TestEvaluateOverrides:
    def test_set_overrides_reach_the_config(
        self, registered, mini_dataset, tmp_path, capsys
    ):
        """--set passes through to build_config, so a job script can retarget
        the config without editing the TOML."""
        config = tmp_path / "mini.toml"
        config.write_text(
            "seed = 3\n"
            'device = "cpu"\n'
            "[model]\n"
            'name = "unet_small"\n'
            "[data]\n"
            'name = "well:mini"\n'
            f'path = "{mini_dataset}"\n'
            "batch_size = 2\n"
            "[train]\n"
            "batch_size = 2\n"
            "num_workers = 0\n"
            "init_lr = 1e-3\n"
            "[drift_detection]\n"
            'detector_name = "PageHinkleyDetector"\n'
        )
        with (
            patch.dict(well_datasets.REGISTRY, {MINI.name: MINI}),
            patch.dict(evaluate.UNET_WIDTHS, {"unet": 4}),
        ):
            assert (
                evaluate.main(
                    [
                        "--config",
                        str(config),
                        "--baselines",
                        "--set",
                        "model.name=unet",
                    ]
                )
                == 0
            )
        # The TOML says unet_small; the banner reporting unet proves the
        # override reached build_config.
        assert "init_features=4 (unet)" in capsys.readouterr().out


class TestPlot:
    def test_parse_metrics_recovers_trace_and_firing_batches(self, tmp_path):
        """A firing's logger step falls between eval steps; its batch index is
        where that step inserts into the eval sequence."""
        rows = [
            ("step", "metric", "value"),
            (1, "eval/vrmse", 1.0),
            (2, "eval/vrmse", 1.1),
            (3, "drift/detected", 0),
            (4, "eval/vrmse", 1.2),
            (5, "drift/detected", 1),
            (6, "eval/vrmse", 2.0),
            (7, "eval/mse", 9.9),
            (8, "drift/detected", 1),
        ]
        path = tmp_path / "metrics.csv"
        path.write_text("\n".join(",".join(str(c) for c in r) for r in rows))

        trace, fires = plot.parse_metrics(path)
        assert trace == [1.0, 1.1, 1.2, 2.0]
        # First fire lands after the third eval batch, second after the fourth.
        assert fires == [3, 4]
