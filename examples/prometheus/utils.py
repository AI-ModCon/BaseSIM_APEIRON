# examples/prometheus/utils.py
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def read_csv_files(path: str) -> List[pd.DataFrame]:
    """
    Read all CSV files from a directory, sorted alphabetically.

    Parameters
    ----------
    path : str
        Directory containing CSV files.

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames, one per CSV file.
    """
    files = sorted(f for f in os.listdir(path) if f.endswith(".csv"))
    dfs = []
    for file in files:
        try:
            dfs.append(pd.read_csv(os.path.join(path, file)))
        except Exception as e:
            print(f"Failed to read {file}: {e}")
    return dfs


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard (z-score) normalization to all columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with each column normalized to zero mean and unit variance.
    """
    df = df.copy()
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std if std > 0 else 0.0
    return df


def create_sequences(
    dfs: List[pd.DataFrame],
    feature_cols: List[str],
    target_cols: List[str],
    seq_len: int,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sliding-window (x, y) sequence pairs suitable for LSTM input.

    Each label y is the target value at the last timestep of the window,
    matching the TF training convention: ``train_y[:, -1, :]``.

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        List of DataFrames (one per CSV file).
    feature_cols : list[str]
        Column names used as LSTM input features.
    target_cols : list[str]
        Column names to predict.
    seq_len : int
        Number of timesteps per input window.
    normalize : bool
        If True, apply per-DataFrame z-score normalization before windowing.

    Returns
    -------
    x : torch.Tensor, shape [N, seq_len, n_features]
    y : torch.Tensor, shape [N, n_targets]
    """
    x_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    all_cols = feature_cols + target_cols
    for df in dfs:
        df = df[all_cols].copy().dropna()
        if len(df) <= seq_len:
            continue
        if normalize:
            df = normalize_df(df)

        feat = df[feature_cols].values  # [T, n_feat]
        tgt = df[target_cols].values    # [T, n_tgt]

        for i in range(len(feat) - seq_len):
            x_all.append(feat[i : i + seq_len])
            y_all.append(tgt[i + seq_len - 1])  # label = last step of window

    if not x_all:
        raise ValueError(
            "No sequences created. Check data path and column names "
            f"(features={feature_cols}, targets={target_cols})."
        )

    x = torch.tensor(np.array(x_all), dtype=torch.float32)
    y = torch.tensor(np.array(y_all), dtype=torch.float32)
    return x, y


class SequenceDataset(Dataset):
    """Minimal Dataset wrapping pre-built (x, y) sequence tensors."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Build a DataLoader from a Dataset.

    Parameters
    ----------
    ds : Dataset
        The dataset to wrap.
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle samples each epoch.
    num_workers : int
        Number of worker processes for data loading.
    pin_memory : bool
        Pin host memory for faster GPU transfer.
    persistent_workers : bool
        Keep worker processes alive between iterations.
    prefetch_factor : int
        Number of batches to prefetch per worker.

    Returns
    -------
    DataLoader
    """
    kwargs: dict = dict(batch_size=batch_size, shuffle=shuffle, drop_last=False)
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
