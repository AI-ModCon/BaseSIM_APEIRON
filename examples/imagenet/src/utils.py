# examples/imagenet/src/utils.py
from __future__ import annotations
from typing import Dict, List, Any


import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from config.configuration import Config
from examples.cifar.src import cnns, vision_transformers


def _base_tf(normalize: bool = True, size: int = 224):
    """
    Standard ImageNet preprocessing for ViT:
    - Resize to 224x224
    - ToTensor (scales to [0,1])
    - Normalize using ImageNet mean/std
    """
    t = [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ]
    if normalize:
        # Standard ImageNet normalization
        t.append(
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )
    return transforms.Compose(t)


def get_imagenet_train(cfg: Config, normalize: bool = True) -> datasets.ImageFolder:
    """
    Load ImageNet training dataset from the configured data path.

    Expected directory structure:
        cfg.data.path/
            train/
                n01440764/
                    *.JPEG
                n01443537/
                    *.JPEG
                ...
    """
    crop_size = 224
    train_dir = os.path.join(cfg.data.path, "train")

    return datasets.ImageFolder(
        train_dir,
        transform=_base_tf(normalize=normalize, size=crop_size),
    )


def get_imagenet_val(cfg: Config, normalize: bool = True) -> datasets.ImageFolder:
    """
    Load ImageNet validation dataset from the configured data path.

    Expected directory structure:
        cfg.data.path/
            val/
                n01440764/
                    *.JPEG
                n01443537/
                    *.JPEG
                ...
    """
    crop_size = 224
    val_dir = os.path.join(cfg.data.path, "val")

    return datasets.ImageFolder(
        val_dir,
        transform=_base_tf(normalize=normalize, size=crop_size),
    )


class FixedAffine:
    """Apply chained affine transforms from aug_history to every sample."""

    def __init__(self, aug_history: List[Dict[str, Any]]):
        # Each dict has keys: angle, scale, translate, shear
        self.transforms = [
            {
                "angle": float(aug["angle"]),
                "scale": float(aug["scale"]),
                "translate": (int(aug["translate"][0]), int(aug["translate"][1])),
                "shear": float(aug["shear"]),
            }
            for aug in aug_history
        ]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, H, W]
        for t in self.transforms:
            x = TF.affine(
                x,
                angle=t["angle"],
                translate=t["translate"],
                scale=t["scale"],
                shear=t["shear"],
            )
        return x


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
    MAX_ANGLE = 10
    MAX_SCALE = 1.25
    MIN_SCALE = 0.75
    angle = float(torch.rand(1, generator=g).item() * MAX_ANGLE)
    scale = float(
        MIN_SCALE + (MAX_SCALE - MIN_SCALE) * torch.rand(1, generator=g).item()
    )
    shear = angle
    translate = (int(scale), int(scale))

    print(
        "Mutating the picture further using an angle of", angle, "and a scale of", scale
    )
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
        return x, y


def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 8,
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
        Number of workers to use for data loading. Defaults to 8.
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


def load_model(model_name: str, num_classes: int = 1000) -> nn.Module:
    """
    Load a pre-trained model and modify its classifier to output the desired number of classes.

    Args:
        model_name (str): The name of the pre-trained model to load.
        num_classes (int): The number of classes to output from the model's classifier.
                          Defaults to 1000 for ImageNet.

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
