from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def load_csvs(folder: str) -> List[pd.DataFrame]:
    """Read every .csv in *folder* (sorted) and stash the filename stem on
    ``df.attrs['source']`` for downstream labelling.
    """
    files = sorted(f for f in os.listdir(folder) if f.endswith(".csv"))
    dfs: List[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(os.path.join(folder, f))
        df.attrs["source"] = os.path.splitext(f)[0]
        dfs.append(df)
    return dfs


def compute_cumulative_stats(
    dfs: List[pd.DataFrame], cols: List[str]
) -> Dict[str, Tuple[float, float]]:
    """Return ``{col: (mean, std)}`` computed over the concatenation of *dfs*."""
    full = pd.concat([df[cols] for df in dfs], ignore_index=True)
    return {c: (float(full[c].mean()), float(full[c].std())) for c in cols}


def normalize(df: pd.DataFrame, stats: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Z-score normalize *df* using pre-computed *stats*."""
    out = df.copy()
    for c, (mu, std) in stats.items():
        if c in out.columns:
            out[c] = (out[c] - mu) / std
    return out


def make_sequence_windows(
    dfs: List[pd.DataFrame],
    feature_cols: List[str],
    target_cols: List[str],
    seq_len: int,
    stats: Dict[str, Tuple[float, float]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sliding-window sequences with full-sequence targets.

    Returns
    -------
    X : Tensor of shape ``(N, seq_len, n_features)``
    Y : Tensor of shape ``(N, seq_len, n_targets)``
    """
    all_cols = feature_cols + target_cols
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    for df in dfs:
        if len(df) <= seq_len:
            continue
        data = normalize(df[all_cols], stats)
        feat = data[feature_cols].to_numpy(dtype=np.float32)
        targ = data[target_cols].to_numpy(dtype=np.float32)
        n = len(data) - seq_len
        x_parts.append(np.stack([feat[i : i + seq_len] for i in range(n)]))
        y_parts.append(np.stack([targ[i : i + seq_len] for i in range(n)]))

    if not x_parts:
        raise ValueError(
            "No sequences created. Check data path and column names "
            f"(features={feature_cols}, targets={target_cols})."
        )

    X = torch.from_numpy(np.concatenate(x_parts))
    Y = torch.from_numpy(np.concatenate(y_parts))
    return X, Y


class SequenceDataset(Dataset):
    """Wraps pre-built ``(X, Y)`` tensors."""

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
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a DataLoader from a Dataset."""
    kwargs: dict = dict(batch_size=batch_size, shuffle=shuffle, drop_last=False)
    if num_workers > 0:
        kwargs.update(
            dict(
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
                prefetch_factor=2,
            )
        )
    return DataLoader(ds, **kwargs)  # type: ignore[arg-type]
