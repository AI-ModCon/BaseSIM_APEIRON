# examples/aeris/model.py
"""AERIS model harness for the BaseSim continuous-learning framework.

This harness wraps a 8-layer neural network trained to predict enthalpy per atom from a given fuel material."""

import gc
import math
import torch
import numpy as np
from typing import Tuple, Optional, List, Any
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset

from apeiron.model.torch_model_harness import BaseModelHarness
from apeiron.config.configuration import Config

from examples.aeris.utils import (
    load_datasets,
    make_loader,
    load_pretrained_model,
    split_into_windows,
)

# Aeris model architecture used for prediction
class AerisFullStructure(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        first_layer = min(1024, max(512, input_dim * 2))
        self.layers = nn.Sequential(
            nn.Linear(input_dim, first_layer), nn.ReLU(), nn.BatchNorm1d(first_layer),
            nn.Linear(first_layer, first_layer), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(first_layer, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def train(self, mode: bool = True):
        """Set training mode, but always keep BatchNorm layers in eval mode.

        The ``jvp_reg`` continual-learning updater runs this model through
        ``torch.func.jvp``/``grad`` via ``functional_call``. In training mode
        ``BatchNorm1d`` performs an in-place ``num_batches_tracked.add_(1)`` on a
        captured buffer, which functorch transforms forbid. Keeping BatchNorm in
        eval mode (frozen running stats) both avoids that crash and prevents the
        small, drifted CL batches from corrupting the normalization statistics.
        Dropout still follows ``mode`` normally.
        """
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
        return self


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

        # Expect full harness checkpoint format
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(
                f"Checkpoint '{cfg.model.name}' must be a full harness checkpoint "
                "with 'model_state_dict', 'feature_names', 'scaler', and 'input_dim'. "
                "Legacy weight-only checkpoints are no longer supported."
            )

        state_dict = ckpt["model_state_dict"]
        feature_names = ckpt["feature_names"]
        scaler = ckpt["scaler"]
        input_dim = int(ckpt["input_dim"])

        model = AerisFullStructure(input_dim=input_dim)
        model.load_state_dict(state_dict)
        model.to(cfg.device)
        model.eval()

        super().__init__(cfg=cfg, model=model)

        self._feature_names = feature_names
        self._scaler = scaler
        self._input_dim = input_dim

        # ----- data loaders  -------------------------------------
        X, y = load_datasets(cfg.data.path, cfg.data.name, feature_names, input_dim)
        # X shape: (n_samples, 245) y shape: (n_samples, 1)

        # scale (must match training)
        X_scaled = scaler.transform(X).astype(np.float32)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        self.windows = split_into_windows(X_tensor, y_tensor)
        print(f"Prepared {len(self.windows)} time windows for streaming. Each window has ~{self.windows[0][0].shape[0]} samples.")

        # ----- optional base/initial training data ---------------------------
        # The pre-drift training split the model was originally fit on. When
        # cfg.data.base_train_path is set it is blended into full retrains via
        # get_base_train_dataloaders() so the from-scratch model does not forget
        # the base distribution. Featurized + scaled exactly like the stream.
        self._base_train_ds: Optional[TensorDataset] = None
        self._base_val_ds: Optional[TensorDataset] = None
        base_path = getattr(cfg.data, "base_train_path", "")
        if base_path:
            Xb, yb = load_datasets(base_path, cfg.data.name, feature_names, input_dim)
            Xb_scaled = scaler.transform(Xb).astype(np.float32)
            Xb_t = torch.tensor(Xb_scaled, dtype=torch.float32)
            yb_t = torch.tensor(yb, dtype=torch.float32).view(-1, 1)
            # Shuffle once (seeded) so the val slice isn't a biased tail.
            gen = torch.Generator().manual_seed(cfg.seed)
            perm = torch.randperm(Xb_t.shape[0], generator=gen)
            Xb_t, yb_t = Xb_t[perm], yb_t[perm]
            n = Xb_t.shape[0]
            n_val = max(1, int(n * _VAL_FRACTION))
            n_train = n - n_val
            self._base_train_ds = TensorDataset(Xb_t[:n_train], yb_t[:n_train])
            self._base_val_ds = TensorDataset(Xb_t[n_train:], yb_t[n_train:])
            print(
                f"Loaded {n} base training samples from {base_path} "
                "(blended into full retrains)."
            )

        # ----- eval metrics (prediction) -------------------------------------
        self._y_var_ref = self._reference_variance(y_tensor)
        self.eval_metrics = {
            "mse": self.mse_metric(),
            "mae": self.get_criterion(),
            "r2": self.r2_metric(),
            "nrmse": self.nrmse_metric(),
        }
        self.higher_is_better = {
            "mse": False, "mae": False, "r2": True, "nrmse": False,
        }

        # ----- streaming state -----------------------------------------------
        self.window_idx: int = 0
        self.history_windows: List[Tuple[Tensor, Tensor]] = []

        self._cur_train_loader: Optional[DataLoader] = None
        self._cur_val_loader: Optional[DataLoader] = None
        self._cur_stream_loader: Optional[DataLoader] = None

    def _reference_variance(self, y_stream: Tensor) -> float:
        """Fixed denominator for R^2 / NRMSE metrics."""
        y_ref = (
            self._base_train_ds.tensors[1]
            if self._base_train_ds is not None
            else y_stream
        )
        var = float(y_ref.float().var(unbiased=False).item())
        if var <= 0.0:
            raise ValueError("Reference target variance is 0; R^2/NRMSE undefined.")
        return var

    def r2_metric(self):
        """1 - MSE/var_ref against a constant denominator.

        Being affine in MSE means the sample-weighted mean that
        BaseModelHarness.eval() computes is exactly the pooled R^2 -- which
        would NOT hold if each batch normalized by its own variance.
        """
        var_ref = self._y_var_ref

        def _r2(y_hat: Tensor, y: Tensor) -> Tensor:
            return 1.0 - torch.mean((y_hat - y) ** 2) / var_ref

        return _r2

    def nrmse_metric(self):
        """RMSE / std_ref, same reference as r2_metric."""
        std_ref = math.sqrt(self._y_var_ref)

        def _nrmse(y_hat: Tensor, y: Tensor) -> Tensor:
            return torch.sqrt(torch.mean((y_hat - y) ** 2)) / std_ref

        return _nrmse

    def get_optmizer(self) -> Optimizer:  # noqa: D102  (spelling kept for ABC)
        weight_decay = 1e-7
        return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.train.init_lr, weight_decay=weight_decay)

    def mse_metric(self):  # noqa: D102
        return nn.MSELoss()

    def get_criterion(self):
        return nn.L1Loss()

    def get_stream_dataloader(self):
        assert self._cur_stream_loader is not None
        return self._cur_stream_loader

    def get_train_dataloaders(self) -> Tuple[DataLoader, DataLoader]:  # noqa: D102
        assert self._cur_train_loader is not None and self._cur_val_loader is not None
        return self._cur_train_loader, self._cur_val_loader

    def get_hist_dataloaders(
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

    def get_base_train_dataloaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """Return (train, val) loaders over the initial training split.

        Returns ``(None, None)`` unless ``cfg.data.base_train_path`` was set.
        Loaders are built on demand (only during a full retrain) so no worker
        processes are held open for the rest of the run.
        """
        if self._base_train_ds is None:
            return None, None

        bs = self.cfg.train.batch_size
        nw = self.cfg.train.num_workers
        pin = torch.cuda.is_available()
        train_loader = make_loader(
            self._base_train_ds, bs, shuffle=True, num_workers=nw, pin_memory=pin
        )
        val_loader = (
            make_loader(
                self._base_val_ds, bs, shuffle=False, num_workers=nw, pin_memory=pin
            )
            if self._base_val_ds is not None
            else None
        )
        return train_loader, val_loader

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

        bs = self.cfg.data.batch_size
        self._cur_stream_loader = make_loader(
            ds_train, bs, shuffle=True, num_workers=nw, pin_memory=pin
        )
        self.window_idx += 1

    def build_checkpoint_payload(self) -> dict[str, Any]:
        """Save full harness checkpoint matching load_pretrained_model format.

        Overrides BaseModelHarness to include feature names, scaler, and input_dim
        so that saved checkpoints can be loaded directly without reference checkpoints.
        """
        return {
            "model_state_dict": self.model.state_dict(),
            "feature_names": self._feature_names,
            "scaler": self._scaler,
            "input_dim": self._input_dim,
        }

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
        if self._cur_stream_loader is not None:
            del self._cur_stream_loader
            self._cur_stream_loader = None
        gc.collect()
