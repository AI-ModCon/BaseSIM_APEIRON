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
from examples.mnist.utils import (
    get_mnist_train,
    get_mnist_val,
    FixedAffine,
    TransformedView,
    make_loader,
    sample_aug,
)


class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1).float()  # expect [B, H, W]
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 3 * 3 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MNIST_CNN(BaseModelHarness):
    """
    Pattern:
      - One MNIST train dataset (train=True) and one MNIST val dataset (train=False)
      - Current drift params in self.cur_aug
      - Historical drifts in self.aug_history (previous iterations)
      - get_cur_data_loaders(): build loaders over FULL train/val using self.cur_aug
      - get_hist_data_loaders(): ConcatDataset over self.aug_history; then append self.cur_aug
    """

    def __init__(self, cfg: Config, model: nn.Module = Cnn()):
        super().__init__(cfg=cfg, model=model)

        # FULL datasets (no index split)
        self.ds_train = get_mnist_train("./data", normalize=True)
        self.ds_val = get_mnist_val("./data", normalize=True)

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

    # --- Current task (MNIST with one new global drift) ---
    def get_cur_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
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

        if self.cur_aug:
            self.aug_history.append(self.cur_aug.copy())

        return hist_train_loader, hist_val_loader

    def get_criterion(self):
        return torch.nn.NLLLoss()
