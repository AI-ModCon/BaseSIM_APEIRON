# examples/mnist/memory_efficient_utils.py
from __future__ import annotations
from typing import Tuple, Dict, Any
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
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
    """
    Builds a DataLoader from a given Dataset.

    Parameters
    ----------
    ds : Dataset
        The base dataset to build the DataLoader from.
    batch_size : int
        The batch size to use for the DataLoader.
    shuffle : bool
        If True, the DataLoader will shuffle the dataset.
    num_workers : int, optional
        Number of workers to use for data loading. Defaults to 4.
    pin_memory : bool, optional
        If True, the DataLoader will pin the memory for faster data loading. Defaults to True.
    persistent_workers : bool, optional
        If True, the DataLoader will use persistent workers. Defaults to True.
    prefetch_factor : int, optional
        Number of samples to prefetch for the DataLoader. Defaults to 2.

    Returns
    -------
    DataLoader
        The built DataLoader.
    """
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
    return DataLoader(ds, **kwargs)  # type: ignore[arg-type]


class CustomMnistData(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


def get_mnist_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Download and load the MNIST dataset with normalization.
    """
    my_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=my_transforms
    )
    [images, labels] = [list(t) for t in zip(*dataset)]
    images = torch.stack(images, dim=0)
    images = images.view(-1, 28, 28).float()
    labels = torch.tensor(labels)

    return images, labels


def filter_mnist_by_classes(
    images: torch.Tensor, labels: torch.Tensor, classes: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter MNIST dataset to include only specified classes.

    Args:
        images: Image tensor of shape [N, 28, 28]
        labels: Label tensor of shape [N]
        classes: List of class labels to keep (e.g., [0, 1, 2, 3, 4])

    Returns:
        tuple: Filtered (images, labels) containing only the specified classes
    """
    mask = torch.zeros(len(labels), dtype=torch.bool)
    for cls in classes:
        mask |= labels == cls

    filtered_images = images[mask]
    filtered_labels = labels[mask]

    return filtered_images, filtered_labels


def split_train_test(
    images: torch.Tensor, labels: torch.Tensor, train_ratio: float = 0.8
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Split data into training and test sets.

    Args:
        images: Image tensor of shape [N, 28, 28]
        labels: Label tensor of shape [N]
        train_ratio: Fraction of data to use for training (default: 0.8)

    Returns:
        tuple: ((train_images, train_labels), (test_images, test_labels))
    """
    n = len(images)
    n_train = int(train_ratio * n)

    # Random permutation for splitting
    perm = torch.randperm(n)
    idx_train = perm[:n_train]
    idx_test = perm[n_train:]

    train_images = images[idx_train]
    train_labels = labels[idx_train]
    test_images = images[idx_test]
    test_labels = labels[idx_test]

    return (train_images, train_labels), (test_images, test_labels)
