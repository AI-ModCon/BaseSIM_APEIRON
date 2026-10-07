"""Apeiron harness for PolymathicAI's "The Well": one physical regime per window.

Every other example in this repository manufactures its drift. MNIST, CIFAR and
ImageNet each draw a random affine transform per stream window and push the
inputs away from what the model was trained on. That is a fine way to exercise
the machinery, but the drift is synthetic and its magnitude is a knob.

Here it is not. A Well dataset is a parameter sweep over a simulation -- the 2D
turbulent radiative layer varies a cooling time from ``tcool=0.03`` to
``tcool=3.16`` across nine files -- so putting one regime in each window walks
the model through genuine physical change, in physical order. The inputs are
untouched. What moves is the system being modelled.

What that gives the framework to detect::

    window 0   tcool=0.03    the regime the model adapts to first
    window 1   tcool=0.06    nearby; the detector may or may not fire
    ...
    window 8   tcool=3.16    a different flow, and the error says so

Each window's regime is downloaded when the window opens, converted once into a
memory-mappable array, and read back as a view, so a run touches as many
regimes as it gets to rather than the dataset. Earlier regimes stay on disk and
are replayed as historical data, which is what EWC anchors against.

The stream and the CL rounds both read the regime's ``train`` split; ``valid``
is kept for evaluation and ``test`` is left for scoring checkpoints afterwards.

Config::

    [model] name = "unet_small"
    [data]  name = "well:turbulent_radiative_layer_2D"
            path = "hf://datasets/polymathic-ai"   # or a local Well directory

Adding another Well dataset means adding a row to ``REGISTRY`` in
``examples/well/datasets.py``; this harness has no dataset-specific knowledge
left in it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from apeiron.config.configuration import Config
from apeiron.logger import get_logger
from apeiron.model.torch_model_harness import BaseModelHarness

from examples.well.datasets import RegimeCache, WellDataset, get_dataset, read
from examples.well.unet import UNetClassic

# Steps of history in, steps predicted out. Time folds into the channel
# dimension, so the model is a plain image-to-image map.
N_STEPS_INPUT = 4
N_STEPS_OUTPUT = 1

# The Well's U-Net baseline, at two widths. "unet" is the width their published
# checkpoints were trained at, so changing 48 means one will not load;
# "unet_small" is the same architecture narrowed until a training step fits on a
# laptop CPU, which costs the ability to start from their weights.
UNET_WIDTHS: Dict[str, int] = {
    "unet": 48,
    "unet_small": 16,
}

# Distinct per role and per window, because the framework seeds nothing
# globally: without this the shuffle would differ from run to run and the drift
# trace would not be repeatable.
ROLE_SEEDS = {"stream": 1, "train": 2, "hist": 3}

_EPS = 1e-7


def vrmse(y_hat: Tensor, y: Tensor) -> Tensor:
    """Variance-scaled RMSE, The Well's headline metric. Lower is better.

    Per sample and channel, the spatial error is divided by the target's own
    spatial variance, so fields of wildly different magnitude -- density in the
    tens, pressure near one -- contribute comparably.

    The square root is taken **per sample and channel, before averaging**, which
    is what The Well's own implementation does (``NRMSE`` applies ``sqrt`` to the
    normalised MSE tensor and leaves the reduction to the caller). Averaging the
    ratios first and taking one square root is a different number -- higher, by
    Jensen -- and would not be comparable with their published results.
    """
    spatial = tuple(range(2, y.ndim))
    ratio = (y_hat - y).pow(2).mean(dim=spatial) / (y.var(dim=spatial) + _EPS)
    return torch.sqrt(ratio).mean()


def mse(y_hat: Tensor, y: Tensor) -> Tensor:
    return torch.nn.functional.mse_loss(y_hat, y)


class FrozenBatchNormUNet(UNetClassic):
    """``UNetClassic`` whose BatchNorm layers never update their statistics.

    Every window here is a different physical regime, so batch statistics move
    with it and the running buffers chase whichever regime is current. That is
    a second, uncontrolled adaptation sitting underneath the one being
    measured: twenty forward passes in ``train()`` mode, with no optimizer at
    all, take The Well's published checkpoint from 0.204 to 1.569 VRMSE on the
    regime it was trained for.

    Putting a BatchNorm module in eval mode does two things: the forward pass
    uses the stored running statistics instead of the batch's, and the running
    statistics stop being updated. Keeping them there during training leaves
    the normalisation fixed and lets only the weights move.

    Adds no parameters and renames nothing, so a published checkpoint still
    loads into this with ``strict=True``.
    """

    def train(self, mode: bool = True) -> "FrozenBatchNormUNet":
        super().train(mode)
        if mode:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self


class RegimeFrames:
    """``[trajectory, time, channel, *spatial]`` for one regime, mapped.

    Nothing is read until a sample asks for it, which is what lets a regime
    larger than memory be a window.
    """

    def __init__(self, path: Path):
        self.path = path
        self._array = read(path)
        self.shape = tuple(self._array.shape)

    def steps(self, trajectory: int, start: int, count: int) -> np.ndarray:
        """``[count, channel, *spatial]`` starting at a time index."""
        return np.asarray(self._array[trajectory, start : start + count])


class RegimeDataset(Dataset):
    """Sliding windows of time within one regime.

    A sample is ``N_STEPS_INPUT`` consecutive frames stacked into channels,
    predicting the next ``N_STEPS_OUTPUT``.
    """

    def __init__(self, frames: RegimeFrames, mean: np.ndarray, std: np.ndarray):
        self.frames = frames
        spatial_axes = len(frames.shape) - 3
        self.mean = mean.reshape(1, -1, *([1] * spatial_axes))
        self.std = std.reshape(1, -1, *([1] * spatial_axes))
        n_traj, n_time = frames.shape[0], frames.shape[1]
        self.per_trajectory = n_time - N_STEPS_INPUT - N_STEPS_OUTPUT + 1
        if self.per_trajectory <= 0:
            raise ValueError(f"{frames.path}: too few time steps")
        self.length = n_traj * self.per_trajectory

    def __len__(self) -> int:
        return self.length

    def _normalised(self, trajectory: int, start: int, count: int) -> Tensor:
        block = self.frames.steps(trajectory, start, count)
        block = (block - self.mean) / self.std
        # Time folds into channels: [t, c, *spatial] -> [t*c, *spatial].
        flat = block.reshape(-1, *block.shape[2:])
        return torch.from_numpy(np.nan_to_num(flat, copy=True))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        trajectory, start = divmod(index, self.per_trajectory)
        x = self._normalised(trajectory, start, N_STEPS_INPUT)
        y = self._normalised(trajectory, start + N_STEPS_INPUT, N_STEPS_OUTPUT)
        return x, y


class WELL_UNET(BaseModelHarness):
    """One Well regime per window, a U-Net stepping the fields forward."""

    def __init__(self, cfg: Config, model: Optional[nn.Module] = None):
        if ":" not in cfg.data.name:
            raise ValueError(
                f"[data] name should be 'well:<dataset>', got {cfg.data.name!r}"
            )
        self.spec: WellDataset = get_dataset(cfg.data.name.split(":", 1)[1])

        # Seed before the weights are drawn. Nothing in the framework seeds a
        # torch RNG -- cfg.seed reaches no entry point -- so without this every
        # run starts from a different random model, and with an untrained model
        # the per-batch error *is* the drift signal. Two runs of this config
        # produced per-regime VRMSE of 1.34/1.33/1.28/1.30 and
        # 1.28/1.36/1.45/1.62: one flat, one rising. Detector settings tuned
        # against one would not transfer to the other.
        torch.manual_seed(cfg.seed)

        super().__init__(cfg=cfg, model=model or self._build_model(cfg, self.spec))

        self.logger = get_logger()
        self.cache = RegimeCache(self.spec, cfg.data.path)
        self.regimes = self.cache.regimes()
        self.logger.info(
            f"Well dataset {self.spec.name}: {len(self.regimes)} regimes, "
            f"{self.regimes[0]} .. {self.regimes[-1]}"
        )

        self._load_pretrained()

        self.eval_metrics = {"vrmse": vrmse, "mse": mse}
        self.window = -1
        self._stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._loaders: Dict[str, DataLoader] = {}

    # -- model --------------------------------------------------------------
    @staticmethod
    def _build_model(cfg: Config, spec: WellDataset) -> nn.Module:
        """The Well's own U-Net baseline, shaped by the dataset row."""
        if cfg.model.name not in UNET_WIDTHS:
            raise ValueError(
                f"[model] name should be one of {sorted(UNET_WIDTHS)}, "
                f"got {cfg.model.name!r}"
            )
        return FrozenBatchNormUNet(
            dim_in=N_STEPS_INPUT * spec.n_channels,
            dim_out=N_STEPS_OUTPUT * spec.n_channels,
            n_spatial_dims=spec.n_spatial_dims,
            spatial_resolution=spec.spatial_resolution,
            init_features=UNET_WIDTHS[cfg.model.name],
        )

    def _load_pretrained(self) -> None:
        """Start from published weights, if the config asks for them.

        ``.safetensors`` is what The Well publishes; a ``.pt`` is loaded the way
        the other examples load one. The width has to match: their checkpoints
        belong to ``name = "unet"``.
        """
        source = self.cfg.model.pretrained_path
        if not source:
            return
        path = Path(source)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(path)
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.cfg.device)
        self.logger.info(f"Loaded pretrained weights from {source}")

    def get_optmizer(self) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.init_lr)

    def get_criterion(self) -> Any:
        return nn.MSELoss()

    # -- regimes ------------------------------------------------------------
    def _regime(self, window: int) -> str:
        """The regime a window shows, holding at the last one if asked beyond."""
        return self.regimes[min(window, len(self.regimes) - 1)]

    def _statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._stats is None:
            self._stats = self.cache.statistics()
        return self._stats

    def _dataset(self, split: str, regime: str) -> RegimeDataset:
        """One regime as a dataset, fetching and converting it if needed."""
        mean, std = self._statistics()
        return RegimeDataset(RegimeFrames(self.cache.array(split, regime)), mean, std)

    def _loader(
        self, dataset: Dataset, batch_size: int, role: Optional[str] = None
    ) -> DataLoader:
        generator = None
        if role is not None:
            generator = torch.Generator()
            generator.manual_seed(
                self.cfg.seed * 1_000_003 + self.window * 101 + ROLE_SEEDS[role]
            )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=role is not None,
            generator=generator,
            num_workers=self.cfg.train.num_workers,
            drop_last=False,
        )

    # -- stream protocol ----------------------------------------------------
    def update_data_stream(self) -> None:
        """Open the next window: the next regime in the sweep.

        The Well publishes three splits and this harness uses them as three
        different things:

        ``train``
            Both the monitoring stream and what a CL round trains on. The
            monitoring loop scores each batch before any round touches it, so
            this is prequential -- test-then-train -- which is the usual
            protocol for a data stream. A round consumes about 5% of the
            window, so the overlap between what was scored and what was
            trained on is small.
        ``valid``
            Evaluation only: the pre/post round scores, and the past-task
            scores behind FWT and BWT. Never trained on.
        ``test``
            Not touched here at all. It is for scoring finished checkpoints
            afterwards, which is what ``examples/well/evaluate.py`` does.
        """
        self.window += 1
        regime = self._regime(self.window)
        self.logger.info(f"Window {self.window}: regime {regime}")

        train = self._dataset("train", regime)
        valid = self._dataset("valid", regime)
        self._loaders = {
            "stream": self._loader(train, self.cfg.data.batch_size, role="stream"),
            "train": self._loader(train, self.cfg.train.batch_size, role="train"),
            "valid": self._loader(valid, self.cfg.train.batch_size),
        }

    def get_stream_dataloader(self) -> DataLoader:
        return self._loaders["stream"]

    def get_train_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        return self._loaders["train"], self._loaders["valid"]

    def get_hist_dataloaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """Every regime seen before this one -- the replay anchor."""
        prior: List[str] = [self._regime(w) for w in range(self.window)]
        if not prior:
            return None, None
        train: Dataset = ConcatDataset([self._dataset("train", r) for r in prior])
        valid: Dataset = ConcatDataset([self._dataset("valid", r) for r in prior])
        return (
            self._loader(train, self.cfg.train.batch_size, role="hist"),
            self._loader(valid, self.cfg.train.batch_size),
        )
