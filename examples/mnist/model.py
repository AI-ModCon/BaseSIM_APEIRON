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
from examples.mnist.utils import (
    get_mnist_train,
    get_mnist_val,
    FixedAffine,
    TransformedView,
    make_loader,
    sample_aug,
)
from evaluation.metrics import accuracy


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

        self.eval_metrics = [accuracy, self.get_criterion()]
        self.higher_is_better = [True, False]

        # Load pretrained weights if available
        pretrained_path = "./examples/mnist/mnist.pth"
        try:
            state_dict = torch.load(
                pretrained_path, map_location=cfg.device, weights_only=False
            )

            new_state_dict = {}
            for key, value in state_dict.items():
                new_state_dict[key] = value

            self.model.load_state_dict(new_state_dict)
            print(f"Loaded pretrained MNIST model from {pretrained_path}")
        except FileNotFoundError:
            print(
                f"Warning: Pretrained model not found at {pretrained_path}, using randomly initialized weights"
            )
        except Exception as e:
            print(f"Warning: Failed to load pretrained MNIST model: {e}")

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

    def get_cur_data_loaders(self):
        return self._cur_train_loader, self._cur_val_loader

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
        return torch.nn.NLLLoss()
