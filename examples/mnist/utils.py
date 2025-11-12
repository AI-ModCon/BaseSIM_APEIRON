# examples/mnist/memory_efficient_utils.py
from __future__ import annotations
from typing import Tuple, Optional, Dict, Any, List
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF


def _base_tf(normalize: bool = True):
    t = [transforms.ToTensor()]
    if normalize:
        t.append(transforms.Normalize((0.1307,), (0.3081,)))
    return transforms.Compose(t)


def get_mnist_train(root: str = "./data", normalize: bool = True) -> datasets.MNIST:
    return datasets.MNIST(
        root, train=True, download=True, transform=_base_tf(normalize)
    )


def get_mnist_val(root: str = "./data", normalize: bool = True) -> datasets.MNIST:
    # Standard MNIST “test” split used as validation
    return datasets.MNIST(
        root, train=False, download=True, transform=_base_tf(normalize)
    )


class FixedAffine:
    """Apply one fixed affine to every sample (tensor) in this view."""

    def __init__(
        self, angle: float, scale: float, translate: Tuple[int, int], shear: float
    ):
        self.angle = float(angle)
        self.scale = float(scale)
        self.translate = (int(translate[0]), int(translate[1]))
        self.shear = float(shear)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1,H,W]
        return TF.affine(
            x,
            angle=self.angle,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
        )


def sample_aug(seed: int) -> Dict[str, Any]:
    """
    Sample a random affine transformation given a seed.

    Parameters
    ----------
    seed : int
        The seed for the random number generator.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the sampled angle, scale, translation and shear.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    angle = float(torch.rand(1, generator=g).item() * 180.0)
    scale = float(1.0 + torch.rand(1, generator=g).item())
    shear = angle
    translate = (int(scale), int(scale))
    return dict(angle=angle, scale=scale, translate=translate, shear=shear)


class TransformedView(Dataset):
    """A lightweight view that applies x_transform to every sample of a base dataset."""

    def __init__(self, base: Dataset, x_transform=None):
        self.base = base
        self.x_transform = x_transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        if self.x_transform is not None:
            x = self.x_transform(x)
        return x.squeeze(0), y  # CNN adds channel later


def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, drop_last=False)
    if num_workers > 0:
        kwargs.update(
            dict(
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
        )
    return DataLoader(ds, **kwargs)
