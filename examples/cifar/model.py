# src/model/mnist_cnn_harness.py
import gc
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, ConcatDataset

from src.model.torch_model_harness import BaseModelHarness
from src.config.configuration import Config
from examples.cifar.src.utils import (
    get_cifar_train,
    get_cifar_val,
    FixedAffine,
    TransformedView,
    make_loader,
    sample_aug,
)
from examples.cifar.src.utils import load_model


class VisionModelCifar(nn.Module):
    """
    ViT-B/16 classifier for CIFAR-10 using a Hugging Face pretrained checkpoint.

    - Expects inputs already resized to 224x224 and normalized (see your CIFAR-10 ViT utils).
    - Outputs log-probabilities (for use with NLLLoss, mirroring your CNN).
    """

    def __init__(
        self,
        cfg: Config,
    ):
        super().__init__()
        self.is_vit = False
        if cfg.model.name.startswith("vit"):
            self.is_vit = True

        if cfg.data.name == "cifar10":
            num_classes = 10
        elif cfg.data.name == "cifar100":
            num_classes = 100
        else:
            raise NotImplementedError

        self.model = load_model(model_name=cfg.model.name, num_classes=num_classes)
        print(
            f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the VisionModelCifar (CNN or ViT).

        Args:
            x: [B, 3, H, W] tensor already preprocessed (224x224, normalized).

        Returns:
            [B, num_labels] log-probabilities (for NLLLoss).
        """
        # If we are using ViT, pixel_values is the correct input
        if self.is_vit:
            out = self.model(pixel_values=x, return_dict=True).logits
        else:
            # Otherwise, we are using a CNN and the input is just x
            _, out = self.model(x)

        return out


class CIFAR_VISION(BaseModelHarness):
    """
    Pattern:
      - One MNIST train dataset (train=True) and one MNIST val dataset (train=False)
      - Current drift params in self.cur_aug
      - Historical drifts in self.aug_history (previous iterations)
      - get_cur_data_loaders(): build loaders over FULL train/val using self.cur_aug
      - get_hist_data_loaders(): ConcatDataset over self.aug_history; then append self.cur_aug
    """

    def __init__(self, cfg: Config, model: nn.Module = None):

        super().__init__(cfg=cfg, model=VisionModelCifar(cfg=cfg))

        # FULL datasets (no index split)
        self.ds_train = get_cifar_train(cfg=cfg, normalize=True)
        self.ds_val = get_cifar_val(cfg=cfg, normalize=True)

        self.task_counter = 0
        self.cur_aug: Dict[str, Any] = {}
        self.aug_history: List[Dict[str, Any]] = []

        self._cur_train_loader: Optional[DataLoader] = None
        self._cur_val_loader: Optional[DataLoader] = None

    def _dispose_current_loaders(self):
        if self._cur_train_loader is not None:
            del self._cur_train_loader
            self._cur_train_loader = None
        if self._cur_val_loader is not None:
            del self._cur_val_loader
            self._cur_val_loader = None
        gc.collect()

    def get_optmizer(self) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.init_lr)

    def get_cur_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        self._dispose_current_loaders()

        # Deterministic per-iteration drift; “one affine for all samples”
        self.cur_aug = sample_aug(seed=self.cfg.seed + self.task_counter)
        tf = FixedAffine(**self.cur_aug)

        ds_train_tf = TransformedView(self.ds_train, x_transform=tf)
        ds_val_tf = TransformedView(self.ds_val, x_transform=tf)

        bs = self.cfg.train.batch_size
        nw = self.cfg.train.num_workers
        pin = torch.cuda.is_available()

        self._cur_train_loader = make_loader(
            ds_train_tf, bs, shuffle=True, num_workers=nw, pin_memory=pin
        )
        self._cur_val_loader = make_loader(
            ds_val_tf, bs, shuffle=False, num_workers=nw, pin_memory=pin
        )

        self.task_counter += 1
        return self._cur_train_loader, self._cur_val_loader

    def get_hist_data_loaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """
        If no history yet: add current drift to history and return (None, None).
        Else: return loaders over ConcatDataset of prior drifts, then append current drift to history.
        Effective dataset length = len(aug_history) * len(full_split).
        """
        if len(self.aug_history) == 0:
            if self.cur_aug:
                self.aug_history.append(self.cur_aug.copy())
            return None, None

        # Concatenate FULL train/val views for each historical drift
        train_views = [
            TransformedView(self.ds_train, x_transform=FixedAffine(**aug))
            for aug in self.aug_history
        ]
        val_views = [
            TransformedView(self.ds_val, x_transform=FixedAffine(**aug))
            for aug in self.aug_history
        ]

        ds_hist_train = ConcatDataset(train_views)
        ds_hist_val = ConcatDataset(val_views)

        bs = self.cfg.train.batch_size
        nw = getattr(self.cfg.data, "num_workers", 4)
        pin = torch.cuda.is_available()

        hist_train_loader = make_loader(
            ds_hist_train, bs, shuffle=True, num_workers=nw, pin_memory=pin
        )
        hist_val_loader = make_loader(
            ds_hist_val, bs, shuffle=False, num_workers=nw, pin_memory=pin
        )

        if self.cur_aug:
            self.aug_history.append(self.cur_aug.copy())

        return hist_train_loader, hist_val_loader

    def get_criterion(self):
        return torch.nn.CrossEntropyLoss()
