# examples/cifar10_vit/utils.py
from __future__ import annotations
from typing import Any, Dict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from config.configuration import Config
from examples.cifar.src import cnns, vision_transformers


def _base_tf(normalize: bool = True, size: int = 224):
    """
    Minimal ViT-B/16 preprocessing for CIFAR-10:
    - Resize to 224x224
    - ToTensor (scales to [0,1])
    - Normalize to mean/std typically used by ViT (map to [-1,1])
    """
    t = [
        transforms.Resize(
            (size, size), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
    ]
    if normalize:
        # ViT checkpoints commonly use mean=std=0.5 after rescaling to [0,1]
        t.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    return transforms.Compose(t)


def get_cifar_train(
    cfg: Config, root: str = "./data", normalize: bool = True
) -> datasets.CIFAR10:
    crop_size = 32
    if cfg.model.name.startswith("vit"):
        crop_size = 224

    if cfg.data.name == "cifar10":
        return datasets.CIFAR10(
            root,
            train=True,
            download=True,
            transform=_base_tf(normalize=normalize, size=crop_size),
        )
    elif cfg.data.name == "cifar100":
        return datasets.CIFAR100(
            root,
            train=True,
            download=True,
            transform=_base_tf(normalize=normalize, size=crop_size),
        )
    else:
        raise NotImplementedError


def get_cifar_val(
    cfg: Config, root: str = "./data", normalize: bool = True
) -> datasets.CIFAR10:
    crop_size = 32
    if cfg.model.name.startswith("vit"):
        crop_size = 224

    if cfg.data.name == "cifar10":
        return datasets.CIFAR10(
            root,
            train=False,
            download=True,
            transform=_base_tf(normalize=normalize, size=crop_size),
        )
    elif cfg.data.name == "cifar100":
        return datasets.CIFAR100(
            root,
            train=False,
            download=True,
            transform=_base_tf(normalize=normalize, size=crop_size),
        )
    else:
        raise NotImplementedError


class FixedAffine:
    """Apply one fixed affine to every sample (tensor) in this view.

    Accepts a list of augmentation dicts with keys: angle, scale, translate, shear.
    Uses the last augmentation in the list.
    """

    def __init__(self, aug_history: list[Dict[str, Any]]) -> None:
        if not aug_history:
            self.angle = 0.0
            self.scale = 1.0
            self.translate = (0, 0)
            self.shear = 0.0
        else:
            aug = aug_history[-1]
            self.angle = float(aug["angle"])
            self.scale = float(aug["scale"])
            self.translate = (int(aug["translate"][0]), int(aug["translate"][1]))
            self.shear = float(aug["shear"])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [3,H,W] for CIFAR-10 / ViT preprocessing
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
    angle = float(torch.rand(1, generator=g).item() * 180.0) / 10
    scale = float(1.0 + torch.rand(1, generator=g).item() / 10)
    print("Mutating the picture using an angle of", angle, "and a scale of", scale)
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
        return x, y  # keep 3-channel for ViT


def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    sampler: Sampler | None = None,
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
    sampler : Sampler or None, optional
        If provided, overrides shuffle and uses this sampler instead.

    Returns
    -------
    DataLoader
        The built DataLoader.
    """
    kwargs: dict[str, Any] = dict(batch_size=batch_size, drop_last=False)
    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = shuffle
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


def load_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Load a pre-trained model and modify its classifier to output the desired number of classes.

    Args:
        model_name (str): The name of the pre-trained model to load.
        num_classes (int): The number of classes to output from the model's classifier.

    Returns:
        nn.Module: The loaded model with the modified classifier.
    """
    Model: Any
    if model_name == "alexnet":
        Model = cnns.alexnet(pretrained=True)
        Model.classifier._modules["6"] = nn.Linear(4096, num_classes)
    elif model_name == "vgg11":
        Model = cnns.vgg11(pretrained=True)
        Model.classifier._modules["6"] = nn.Linear(4096, num_classes)
    elif model_name == "vgg16":
        Model = cnns.vgg16(pretrained=True)
        Model.classifier._modules["6"] = nn.Linear(4096, num_classes)
    elif model_name == "vgg19":
        Model = cnns.vgg19(pretrained=True)
        Model.classifier._modules["6"] = nn.Linear(4096, num_classes)
    elif model_name == "inception":
        Model = cnns.inception_v3(pretrained=True, aux_logits=False)
        Model.fc = nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "resnet18":
        Model = cnns.resnet18(pretrained=True)
        Model.fc = nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "resnet34":
        Model = cnns.resnet34(pretrained=True)
        Model.fc = nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        Model = cnns.resnet50(pretrained=True)
        Model.fc = nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "resnet101":
        Model = cnns.resnet101(pretrained=True)
        Model.fc = nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "resnext50_32x4d":
        Model = cnns.resnext50_32x4d(pretrained=True)
        Model.fc = nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "resnext101_32x8d":
        Model = cnns.resnext101_32x8d(pretrained=True)
        Model.fc = nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "densenet121":
        Model = cnns.densenet121(pretrained=True)
        Model.classifier = nn.Linear(1024, num_classes)
    elif model_name == "densenet169":
        Model = cnns.densenet169(pretrained=True)
        Model.classifier = nn.Linear(1664, num_classes)
    elif model_name == "densenet201":
        Model = cnns.densenet201(pretrained=True)
        Model.classifier = nn.Linear(1920, num_classes)
    elif model_name == "regnet_x_400mf":
        Model = cnns.regnet_x_400mf(pretrained=True)
        Model.fc = nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "regnet_x_8gf":
        Model = cnns.regnet_x_8gf(pretrained=True)
        Model.fc = nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "regnet_x_16gf":
        Model = cnns.regnet_x_16gf(pretrained=True)
        Model.fc = nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "vit16b":
        Model = vision_transformers.vit_b16(num_classes)
    elif model_name == "vit16l":
        Model = vision_transformers.vit_l16(num_classes)
    elif model_name == "vit32l":
        Model = vision_transformers.vit_l32(num_classes)
    elif model_name == "vit14h":
        Model = vision_transformers.vit_h14(num_classes)
    elif model_name == "vit14g":
        Model = vision_transformers.vit_g14(num_classes)
    else:
        raise NotImplementedError
    return Model
