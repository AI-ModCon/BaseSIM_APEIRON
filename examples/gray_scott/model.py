# src/model/mnist_cnn_harness.py
import gc
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, ConcatDataset

from model.torch_model_harness import BaseModelHarness
from config.configuration import Config
from examples.gray_scott.utils import (
    get_gs_train,
    get_gs_val,
    FixedAffine,
    TransformedView,
    make_loader,
    sample_aug,
)


class Cnn(nn.Module):
    def __init__(self, num_classes=2):
        super(Cnn, self).__init__()

        # Feature Map Size After 3 Pooling Layers (425 / 8 = 53.125 -> 53)
        final_dim = 53

        # Convolutional Layers (425x425 -> 53x53)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 212x212
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 106x106
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 53x53
        )

        # Fully Connected Layers
        self.input_features_fc = 128 * final_dim * final_dim

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.input_features_fc, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


class GSimgCNN(BaseModelHarness):
    """
    Pattern:
      - One GSimg train dataset (train=True) and one MNIST val dataset (train=False)
      - Current drift params in self.cur_aug
      - Historical drifts in self.aug_history (previous iterations)
      - get_cur_data_loaders(): build loaders over FULL train/val using self.cur_aug
      - get_hist_data_loaders(): ConcatDataset over self.aug_history; then append self.cur_aug
    """

    def __init__(self, cfg: Config, model: nn.Module = Cnn()):
        super().__init__(cfg=cfg, model=model)

        # FULL datasets (no index split)
        self.ds_train = get_gs_train()
        self.ds_val = get_gs_val()

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

    def update_data_stream(self) -> None:
        self._dispose_current_loaders()

        # Deterministic per-iteration drift; “one affine for all samples”
        self.cur_aug = sample_aug(seed=self.cfg.seed + self.task_counter)
        tf = FixedAffine(**self.cur_aug)

        ds_train_tf = TransformedView(self.ds_train, x_transform=tf)
        ds_val_tf = TransformedView(self.ds_val, x_transform=tf)

        bs = self.cfg.train.batch_size
        nw = getattr(self.cfg.train, "num_workers", 4)
        pin = torch.cuda.is_available()

        self._cur_train_loader = make_loader(
            ds_train_tf, bs, shuffle=True, num_workers=nw, pin_memory=pin
        )
        self._cur_val_loader = make_loader(
            ds_val_tf, bs, shuffle=False, num_workers=nw, pin_memory=pin
        )

        self.task_counter += 1

        self.aug_history.append(self.cur_aug.copy())

    def get_cur_data_loaders(self) -> Tuple[DataLoader, DataLoader]:

        return self._cur_train_loader, self._cur_val_loader

    def get_hist_data_loaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """
        If no history yet: add current drift to history and return (None, None).
        Else: return loaders over ConcatDataset of prior drifts, then append current drift to history.
        Effective dataset length = len(aug_history) * len(full_split).
        """
        if self.task_counter == 1:
            return None, None

        # Concatenate FULL train/val views for each historical drift
        train_views = [
            TransformedView(self.ds_train, x_transform=FixedAffine(**aug))
            for aug in self.aug_history[:-1]
        ]
        val_views = [
            TransformedView(self.ds_val, x_transform=FixedAffine(**aug))
            for aug in self.aug_history[:-1]
        ]

        ds_hist_train: ConcatDataset[Any] = ConcatDataset(train_views)
        ds_hist_val: ConcatDataset[Any] = ConcatDataset(val_views)

        bs = self.cfg.train.batch_size
        nw = getattr(self.cfg.data, "num_workers", 4)
        pin = torch.cuda.is_available()

        hist_train_loader = make_loader(
            ds_hist_train, bs, shuffle=True, num_workers=nw, pin_memory=pin
        )
        hist_val_loader = make_loader(
            ds_hist_val, bs, shuffle=False, num_workers=nw, pin_memory=pin
        )

        return hist_train_loader, hist_val_loader

    def get_criterion(self):
        return torch.nn.CrossEntropyLoss()
