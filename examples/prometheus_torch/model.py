from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset, DataLoader

from config.configuration import Config
from examples.prometheus_torch.utils import (
    SequenceDataset,
    compute_cumulative_stats,
    load_csvs,
    make_loader,
    make_sequence_windows,
)
from model.torch_model_harness import BaseModelHarness


class TorchTemporalModel(nn.Module):
    """PyTorch reproduction of the Keras TemporalPredict architecture.

    LSTM(128) -> Dropout -> LSTM(64) -> Dropout -> LSTM(32) -> Linear(n_targets)
    Output shape: ``(batch, seq_len, n_targets)``  (full-sequence prediction).
    """

    def __init__(self, n_features: int, n_targets: int, dropout: float = 0.1):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 128, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        self.head = nn.Linear(32, n_targets)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x)
        x, _ = self.lstm3(x)
        return self.head(x)


class PrometheusHarness(BaseModelHarness):
    """Continual-learning harness for the reproduced Prometheus temporal model.

    Each ``update_data_stream()`` call advances to the next training CSV,
    recomputes cumulative normalization stats from all CSVs seen so far,
    and builds windowed datasets with full-sequence targets matching the
    ``TorchTemporalModel`` output shape ``(batch, seq_len, n_targets)``.
    """

    FEATURE_COLS: List[str] = [
        "NRAD_RX_REG_POS",
        "NRAD_RX_SHIM1_POS",
        "NRAD_RX_SHIM2_POS",
        "total_rod_position",
        "NRAD_RX_PERIOD_Inverse",
        "NRAD_RX_REG_POS_dt",
        "NRAD_RX_REG_POS_dt2",
        "NRAD_RX_SHIM1_POS_dt",
        "NRAD_RX_SHIM1_POS_dt2",
        "NRAD_RX_SHIM2_POS_dt",
        "NRAD_RX_SHIM2_POS_dt2",
        "NRAD_RX_NMP1_PWR_integral",
    ]
    TARGET_COLS: List[str] = ["NRAD_RX_NMP1_PWR"]
    SEQUENCE_LENGTH: int = 10
    VAL_RATIO: float = 0.2

    def __init__(self, cfg: Config):
        n_features = len(self.FEATURE_COLS)
        n_targets = len(self.TARGET_COLS)
        model = TorchTemporalModel(n_features=n_features, n_targets=n_targets)
        super().__init__(cfg=cfg, model=model)

        self.eval_metrics: Dict[str, Any] = {"mse": self.get_criterion()}
        self.higher_is_better: Dict[str, bool] = {"mse": False}

        # Load pretrained weights
        pretrained_path = cfg.model.pretrained_path
        if pretrained_path:
            try:
                state_dict = torch.load(
                    pretrained_path, map_location=cfg.device, weights_only=False
                )
                self.model.load_state_dict(state_dict)
                print(f"Loaded pretrained PrometheusV2 model from {pretrained_path}")
            except FileNotFoundError:
                print(
                    f"Warning: Pretrained model not found at {pretrained_path}, "
                    "using randomly initialised weights."
                )
            except Exception as e:
                print(f"Warning: Failed to load pretrained model: {e}")

        # Load all training CSVs (one per file, sequential)
        train_dir = os.path.join(cfg.data.path, "train")
        self.train_dfs: List = load_csvs(train_dir)
        if not self.train_dfs:
            raise ValueError(f"No CSV files found in '{train_dir}'.")

        # State tracking
        self.task_counter: int = 0
        self._dfs_seen: List = []  # accumulates for cumulative stats
        self._all_cols = self.FEATURE_COLS + self.TARGET_COLS
        self._task_datasets: List[Tuple[SequenceDataset, SequenceDataset]] = []
        self._cur_train_loader: Optional[DataLoader] = None
        self._cur_val_loader: Optional[DataLoader] = None

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _dispose_current_loaders(self) -> None:
        if self._cur_train_loader is not None:
            del self._cur_train_loader
            self._cur_train_loader = None
        if self._cur_val_loader is not None:
            del self._cur_val_loader
            self._cur_val_loader = None
        gc.collect()

    def _make_loader(self, ds: SequenceDataset, shuffle: bool) -> DataLoader:
        return make_loader(
            ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.train.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # ------------------------------------------------------------------ #
    # BaseModelHarness interface                                           #
    # ------------------------------------------------------------------ #

    def get_optmizer(self) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.init_lr)

    def get_criterion(self) -> nn.MSELoss:
        return nn.MSELoss()

    def save_ckpt(self, event: int) -> str:
        """Save model checkpoint with a stats sidecar for reproduce_prometheus.py compare."""
        ckpt_path = super().save_ckpt(event)

        # Write cumulative normalization stats alongside the checkpoint
        stats = compute_cumulative_stats(self._dfs_seen, self._all_cols)
        sidecar = Path(ckpt_path).with_suffix(".stats.json")
        payload = {k: [float(mu), float(std)] for k, (mu, std) in stats.items()}
        with open(sidecar, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  stats sidecar -> {sidecar}")
        return ckpt_path

    def update_data_stream(self) -> None:
        """Advance to the next training CSV with cumulative normalization."""
        self._dispose_current_loaders()

        csv_idx = self.task_counter % len(self.train_dfs)
        current_df = self.train_dfs[csv_idx]
        self._dfs_seen.append(current_df)

        # Recompute cumulative stats from all CSVs seen so far
        stats = compute_cumulative_stats(self._dfs_seen, self._all_cols)

        label = current_df.attrs.get("source", f"csv{csv_idx:02d}")
        print(
            f"Prometheus: loading CSV {csv_idx + 1}/{len(self.train_dfs)} "
            f"[{label}] (stats from {len(self._dfs_seen)} file(s))"
        )

        # Build windowed dataset for the current CSV
        X, Y = make_sequence_windows(
            [current_df],
            self.FEATURE_COLS,
            self.TARGET_COLS,
            self.SEQUENCE_LENGTH,
            stats,
        )

        # 80/20 temporal split
        # This may need to be modified since it biases the start.
        n = len(X)
        n_val = max(1, int(n * self.VAL_RATIO))
        n_train = n - n_val

        ds_train = SequenceDataset(X[:n_train], Y[:n_train])
        ds_val = SequenceDataset(X[n_train:], Y[n_train:])
        self._task_datasets.append((ds_train, ds_val))

        self._cur_train_loader = self._make_loader(ds_train, shuffle=True)
        self._cur_val_loader = self._make_loader(ds_val, shuffle=False)

        self.task_counter += 1

    def get_cur_data_loaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        return self._cur_train_loader, self._cur_val_loader

    def get_hist_data_loaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """Return loaders over all prior task datasets. ``(None, None)`` if no history."""
        if self.task_counter <= 1:
            return None, None

        prior = self._task_datasets[:-1]
        hist_train: ConcatDataset = ConcatDataset([ds[0] for ds in prior])
        hist_val: ConcatDataset = ConcatDataset([ds[1] for ds in prior])

        return (
            self._make_loader(hist_train, shuffle=True),  # type: ignore[arg-type]
            self._make_loader(hist_val, shuffle=False),  # type: ignore[arg-type]
        )
