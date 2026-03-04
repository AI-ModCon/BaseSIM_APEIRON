# examples/prometheus/model.py
import gc
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset, DataLoader

from config.configuration import Config
from examples.prometheus.utils import (
    SequenceDataset,
    create_sequences,
    make_loader,
    read_csv_files,
)
from model.torch_model_harness import BaseModelHarness


class StackedLSTM(nn.Module):
    """
    Stacked LSTM architecture mirroring the TensorFlow implementation:

        LSTM(64, tanh) → LSTM(64, tanh) → Dropout(0.1)
            → Linear(64→32, tanh) → Linear(32→n_targets)

    For a single target the final activation is sigmoid (matching the original
    ``Dense(1, activation='sigmoid')``); for multiple targets no activation is
    applied (linear output).
    """

    def __init__(self, n_features: int, n_targets: int, seq_len: int = 30):
        super().__init__()
        self.seq_len = seq_len
        self.n_targets = n_targets

        # Two stacked LSTM layers (tanh is the default cell activation in PyTorch)
        self.lstm1 = nn.LSTM(n_features, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)

        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, n_targets)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, seq_len, n_features]
        out, _ = self.lstm1(x)                  # [B, seq_len, 64]
        _, (h_n, _) = self.lstm2(out)           # h_n: [1, B, 64]
        out = h_n.squeeze(0)                    # [B, 64]
        out = self.dropout(out)
        out = torch.tanh(self.fc1(out))         # [B, 32]
        out = self.fc2(out)                     # [B, n_targets]
        if self.n_targets == 1:
            out = torch.sigmoid(out)
        return out


class PrometheusHarness(BaseModelHarness):
    """
    Continual learning harness for the Prometheus reactor time-series task.

    Pattern:
      - CSV files under ``data.path`` are split into ``NUM_TASKS`` chunks.
      - Each ``update_data_stream()`` call advances to the next chunk,
        simulating operational drift arriving in successive batches.
      - ``get_hist_data_loaders()`` returns a ConcatDataset of all prior task
        datasets for experience replay.

    Customise the domain constants (feature/target columns, sequence length,
    etc.) by subclassing and overriding the class-level attributes below.
    """

    # ------------------------------------------------------------------ #
    # Domain constants – override in a subclass for different datasets    #
    # ------------------------------------------------------------------ #
    FEATURE_VARIABLES: List[str] = [
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
    TARGET_VARIABLES: List[str] = ["NRAD_RX_NMP1_PWR"]
    SEQUENCE_LENGTH: int = 30
    NUM_TASKS: int = 10      # how many iterative data chunks to create
    VAL_RATIO: float = 0.2   # fraction of each chunk held out for validation

    def __init__(self, cfg: Config):
        n_feat = len(self.FEATURE_VARIABLES)
        n_tgt = len(self.TARGET_VARIABLES)
        model = StackedLSTM(
            n_features=n_feat,
            n_targets=n_tgt,
            seq_len=self.SEQUENCE_LENGTH,
        )
        super().__init__(cfg=cfg, model=model)

        self.eval_metrics: Dict[str, Any] = {"mse": self.get_criterion()}
        self.higher_is_better: Dict[str, bool] = {"mse": False}

        # Load pretrained weights if a path is provided
        pretrained_path = cfg.model.pretrained_path
        if pretrained_path:
            try:
                state_dict = torch.load(
                    pretrained_path, map_location=cfg.device, weights_only=False
                )
                self.model.load_state_dict(state_dict)
                print(f"Loaded pretrained Prometheus model from {pretrained_path}")
            except FileNotFoundError:
                print(
                    f"Warning: Pretrained model not found at {pretrained_path}, "
                    "using randomly initialised weights."
                )
            except Exception as e:
                print(f"Warning: Failed to load pretrained model: {e}")

        # Read all CSV files and split into NUM_TASKS chunks
        data_path = cfg.data.path
        all_dfs = read_csv_files(data_path)
        if not all_dfs:
            raise ValueError(f"No CSV files found at '{data_path}'.")

        chunk_size = max(1, len(all_dfs) // self.NUM_TASKS)
        self.df_chunks: List[List] = [
            all_dfs[i : i + chunk_size]
            for i in range(0, len(all_dfs), chunk_size)
        ]

        self.task_counter: int = 0
        # Stores (train_ds, val_ds) for every task seen so far
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

    def _build_datasets(
        self, dfs: List
    ) -> Tuple[SequenceDataset, SequenceDataset]:
        """
        Build train / val datasets from a list of DataFrames.
        Sequences are created with per-DataFrame z-score normalisation.
        The last VAL_RATIO fraction is used as validation to preserve
        the temporal ordering of samples.
        """
        x, y = create_sequences(
            dfs,
            self.FEATURE_VARIABLES,
            self.TARGET_VARIABLES,
            self.SEQUENCE_LENGTH,
            normalize=True,
        )
        n = len(x)
        n_val = max(1, int(n * self.VAL_RATIO))
        n_train = n - n_val
        return (
            SequenceDataset(x[:n_train], y[:n_train]),
            SequenceDataset(x[n_train:], y[n_train:]),
        )

    def _make_loader(
        self, ds: SequenceDataset, shuffle: bool
    ) -> DataLoader:
        bs = self.cfg.train.batch_size
        nw = self.cfg.train.num_workers
        pin = torch.cuda.is_available()
        return make_loader(ds, bs, shuffle=shuffle, num_workers=nw, pin_memory=pin)

    # ------------------------------------------------------------------ #
    # BaseModelHarness interface                                           #
    # ------------------------------------------------------------------ #

    def get_optmizer(self) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.init_lr)

    def get_cur_data_loaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        return self._cur_train_loader, self._cur_val_loader

    def update_data_stream(self) -> None:
        """
        Advance to the next chunk of operational CSV data.
        Wraps around if more updates are requested than available chunks.
        """
        self._dispose_current_loaders()

        chunk_idx = self.task_counter % len(self.df_chunks)
        print(
            f"Prometheus: loading data chunk {chunk_idx + 1}/{len(self.df_chunks)} "
            f"({len(self.df_chunks[chunk_idx])} CSV file(s))"
        )

        ds_train, ds_val = self._build_datasets(self.df_chunks[chunk_idx])
        self._task_datasets.append((ds_train, ds_val))

        self._cur_train_loader = self._make_loader(ds_train, shuffle=True)
        self._cur_val_loader = self._make_loader(ds_val, shuffle=False)

        self.task_counter += 1

    def get_hist_data_loaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """
        Return loaders over all task datasets seen *before* the current one.
        Returns (None, None) on the very first task (no history yet).
        """
        if self.task_counter <= 1:
            return None, None

        # Exclude the most recent (current) task
        prior = self._task_datasets[:-1]
        hist_train: ConcatDataset = ConcatDataset([ds[0] for ds in prior])
        hist_val: ConcatDataset = ConcatDataset([ds[1] for ds in prior])

        bs = self.cfg.train.batch_size
        nw = self.cfg.train.num_workers
        pin = torch.cuda.is_available()

        return (
            make_loader(hist_train, bs, shuffle=True, num_workers=nw, pin_memory=pin),
            make_loader(hist_val, bs, shuffle=False, num_workers=nw, pin_memory=pin),
        )

    def get_criterion(self) -> nn.MSELoss:
        return nn.MSELoss()
