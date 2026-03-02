# examples/aeris/model.py
"""AERIS model harness for the BaseSim continuous-learning framework.

This harness wraps a 8-layer neural network trained to predict enthalpy per atom from a given fuel material."""

import gc
import torch
from typing import Tuple, Optional, List, Any, Mapping, cast
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset

from model.torch_model_harness import BaseModelHarness
from config.configuration import Config

from examples.aeris.utils import (
    load_datasets,
    make_loader,
    load_pretrained_model,
    split_into_windows,
)


# Aeris model architecture used for prediction
class AerisFullStructure(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        first_layer = min(1024, max(512, input_dim * 2))
        self.layers = nn.Sequential(
            nn.Linear(input_dim, first_layer),
            nn.ReLU(),
            nn.BatchNorm1d(first_layer),
            nn.Linear(first_layer, first_layer),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(first_layer, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)


# Fraction of each time window reserved for validation
_VAL_FRACTION: float = 0.2


class AERIS(BaseModelHarness):
    """
    Continuous-learning harness for the AERIS prediction model.
    """

    def __init__(self, cfg: Config):
        # ----- build model ---------------------------------------------------
        ckpt = load_pretrained_model(
            cfg.model.pretrained_path, cfg.model.name, device=cfg.device
        )

        # Checkpoint is a dict saved via torch.save(model_info, ...)
        input_dim_raw = ckpt.get("input_dim")
        if input_dim_raw is None:
            raise KeyError("Checkpoint missing required key: 'input_dim'")
        input_dim = int(cast(int, input_dim_raw))

        feature_names_raw = ckpt.get("feature_names")
        if feature_names_raw is None:
            raise KeyError("Checkpoint missing required key: 'feature_names'")
        feature_names = cast(List[str], feature_names_raw)

        scaler_raw = ckpt.get("scaler")
        if scaler_raw is None:
            raise KeyError("Checkpoint missing required key: 'scaler'")
        scaler = cast(Any, scaler_raw)

        state_raw = ckpt.get("model_state_dict")
        if state_raw is None:
            raise KeyError("Checkpoint missing required key: 'model_state_dict'")
        state = cast(Mapping[str, Any], state_raw)

        model = AerisFullStructure(input_dim=input_dim)
        model.load_state_dict(state)
        model.to(cfg.device)

        super().__init__(cfg=cfg, model=model)

        # ----- eval metrics (prediction) -------------------------------------
        self.eval_metrics = {"mae": self.mae_metric(), "loss": self.get_criterion()}
        self.higher_is_better = {"accuracy": False, "loss": False}

        # ----- data loaders  -------------------------------------
        X, y = load_datasets(cfg.data.path, cfg.data.name, feature_names)
        X_raw = torch.tensor(X, dtype=torch.float32)
        X_scaled: Tensor = scaler.transform(X_raw)
        y_raw = torch.tensor(y, dtype=torch.float32)
        # y is a 1D array of shape (N,), but model outputs (N, 1):
        if y_raw.ndim == 1:
            y_raw = y_raw.unsqueeze(1)

        self.windows = split_into_windows(X_scaled, y_raw)

        # ----- streaming state -----------------------------------------------
        self.window_idx: int = 0
        self.history_windows: List[Tuple[Tensor, Tensor]] = []

        self._cur_train_loader: Optional[DataLoader] = None
        self._cur_val_loader: Optional[DataLoader] = None

    def get_optmizer(self) -> Optimizer:  # noqa: D102  (spelling kept for ABC)
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.init_lr)

    def get_criterion(self):  # noqa: D102
        return nn.MSELoss()

    def mae_metric(self):
        return nn.L1Loss()

    def get_cur_data_loaders(self) -> Tuple[DataLoader, DataLoader]:  # noqa: D102
        assert self._cur_train_loader is not None and self._cur_val_loader is not None
        return self._cur_train_loader, self._cur_val_loader

    def get_hist_data_loaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """Return loaders over all previously-seen time windows.

        Returns ``(None, None)`` until at least two windows have been served.
        """
        if self.window_idx <= 1:
            return None, None

        # Concatenate all history windows
        hist_train_views: List[TensorDataset] = []
        hist_val_views: List[TensorDataset] = []

        for X_w, y_w in self.history_windows:
            n = X_w.shape[0]
            n_val = max(1, int(n * _VAL_FRACTION))
            n_train = n - n_val
            hist_train_views.append(TensorDataset(X_w[:n_train], y_w[:n_train]))
            hist_val_views.append(TensorDataset(X_w[n_train:], y_w[n_train:]))

        ds_hist_train: ConcatDataset[Any] = ConcatDataset(hist_train_views)
        ds_hist_val: ConcatDataset[Any] = ConcatDataset(hist_val_views)

        bs = self.cfg.train.batch_size
        nw = self.cfg.train.num_workers
        pin = torch.cuda.is_available()
        return (
            make_loader(
                ds_hist_train, bs, shuffle=True, num_workers=nw, pin_memory=pin
            ),
            make_loader(ds_hist_val, bs, shuffle=False, num_workers=nw, pin_memory=pin),
        )

    def update_data_stream(self) -> None:
        """Advance to the next chronological time window.

        The current window is added to the history, and new train/val loaders
        are built from the upcoming window.
        """
        self._dispose_current_loaders()

        if self.window_idx >= len(self.windows):
            print(
                f"Warning: All {len(self.windows)} time windows exhausted; "
                "wrapping around to the first window."
            )
            self.window_idx = 0

        X_w, y_w = self.windows[self.window_idx]

        # Archive previous window in history (skip the very first call)
        if self.window_idx > 0:
            prev_X, prev_y = self.windows[self.window_idx - 1]
            # Only add if not already stored (idempotency guard)
            if len(self.history_windows) < self.window_idx:
                self.history_windows.append((prev_X, prev_y))
        # Train / val split (last _VAL_FRACTION chronologically)
        n = X_w.shape[0]
        n_val = max(1, int(n * _VAL_FRACTION))
        n_train = n - n_val

        ds_train = TensorDataset(X_w[:n_train], y_w[:n_train])
        ds_val = TensorDataset(X_w[n_train:], y_w[n_train:])

        bs = self.cfg.train.batch_size
        nw = self.cfg.train.num_workers
        pin = torch.cuda.is_available()

        self._cur_train_loader = make_loader(
            ds_train, bs, shuffle=True, num_workers=nw, pin_memory=pin
        )
        self._cur_val_loader = make_loader(
            ds_val, bs, shuffle=False, num_workers=nw, pin_memory=pin
        )

        self.window_idx += 1

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _dispose_current_loaders(self) -> None:
        if self._cur_train_loader is not None:
            del self._cur_train_loader
            self._cur_train_loader = None
        if self._cur_val_loader is not None:
            del self._cur_val_loader
            self._cur_val_loader = None
        gc.collect()
