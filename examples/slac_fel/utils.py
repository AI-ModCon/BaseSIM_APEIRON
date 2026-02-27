# examples/slac_fel/utils.py
"""Data-loading utilities for the SLAC FEL continuous-learning example.

The data pipeline assumes that all heavy cleaning (archive pull, filtering,
exclusion windows, invalid-PV removal, column selection) has already been done
and the result saved as a single ``data.pkl`` file.  This module loads that
file, applies the saved min-max scalers, and slices the chronologically-ordered
data into non-overlapping time windows that ``SLAC_FEL.update_data_stream()``
will serve one at a time.

Expected directory layout (pointed to by ``cfg.data.path``)::

    <data_dir>/
        data.pkl               # pandas DataFrame, datetime-indexed, sorted
        input_scaler.pt        # botorch AffineInputTransform for inputs
        output_scaler.pt       # botorch AffineInputTransform for output
        feature_config.yml     # YAML listing input_variables / output_variables
"""

from __future__ import annotations

import os
from typing import List, Tuple

import pandas as pd
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FELDataset(Dataset):
    """Simple dataset wrapping pre-scaled input/output tensors."""

    def __init__(self, X: Tensor, y: Tensor) -> None:
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        self.X = X.float()
        self.y = y.float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# DataLoader helper
# ---------------------------------------------------------------------------
def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Build a ``DataLoader`` from a ``Dataset``.

    Parameters
    ----------
    ds:
        The base dataset.
    batch_size:
        Batch size.
    shuffle:
        Whether to shuffle.
    num_workers:
        Number of data-loading workers.
    pin_memory:
        Pin CUDA memory for faster transfers.
    persistent_workers:
        Keep worker processes alive between iterations.
    prefetch_factor:
        Samples to prefetch per worker.

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


# ---------------------------------------------------------------------------
# Feature-config helpers
# ---------------------------------------------------------------------------
def load_feature_config(data_path: str) -> Tuple[List[str], List[str]]:
    """Read ``feature_config.yml`` and return ``(input_cols, output_cols)``.

    The YAML file is expected to have top-level keys ``input_variables`` and
    ``output_variables``, each mapping variable names to metadata dicts.
    """
    cfg_path = os.path.join(data_path, "feature_config.yml")
    with open(cfg_path, "r") as fh:
        yml = yaml.safe_load(fh)
    input_cols = list(yml["input_variables"].keys())
    output_cols = list(yml["output_variables"].keys())
    return input_cols, output_cols


# ---------------------------------------------------------------------------
# Scaler helpers
# ---------------------------------------------------------------------------
def load_scalers(
    data_path: str, device: str = "cpu"
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load the saved BoTorch ``AffineInputTransform`` scalers.

    Parameters
    ----------
    data_path:
        Directory containing ``input_scaler.pt`` and ``output_scaler.pt``.
    device:
        Device to map the scalers to.

    Returns
    -------
    (input_scaler, output_scaler)
    """
    input_scaler = torch.load(
        os.path.join(data_path, "input_scaler.pt"),
        map_location=device,
        weights_only=False,
    )
    output_scaler = torch.load(
        os.path.join(data_path, "output_scaler.pt"),
        map_location=device,
        weights_only=False,
    )
    return input_scaler, output_scaler


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_fel_data(
    data_path: str, device: str = "cpu"
) -> Tuple[Tensor, Tensor, pd.Index]:
    """Load ``data.pkl``, apply scalers, and return scaled tensors + timestamps.

    The pickle is expected to be a ``pandas.DataFrame`` with:
    * a datetime index (already sorted chronologically),
    * columns matching those listed in ``feature_config.yml``.

    Parameters
    ----------
    data_path:
        Directory containing ``data.pkl``, scalers, and ``feature_config.yml``.
    device:
        Device string (used for scaler loading only; tensors stay on CPU here).

    Returns
    -------
    (X_scaled, y_scaled, timestamps)
        * ``X_scaled`` — ``[N, n_inputs]`` float32
        * ``y_scaled`` — ``[N, n_outputs]`` float32
        * ``timestamps`` — ``pd.Index`` (DatetimeIndex) of length N
    """
    df: pd.DataFrame = pd.read_pickle(os.path.join(data_path, "data.pkl"))

    # Ensure sorted by time
    df = df.sort_index()

    input_cols, output_cols = load_feature_config(data_path)
    input_scaler, output_scaler = load_scalers(data_path, device=device)

    X_raw = torch.tensor(df[input_cols].values, dtype=torch.float32)
    y_raw = torch.tensor(df[output_cols].values, dtype=torch.float32)

    X_scaled: Tensor = input_scaler.transform(X_raw)
    y_scaled: Tensor = output_scaler.transform(y_raw)

    return X_scaled, y_scaled, df.index


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

# Default number of samples per time window.  Can be overridden by the caller.
DEFAULT_WINDOW_SIZE: int = 5000


def split_into_windows(
    X: Tensor,
    y: Tensor,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> List[Tuple[Tensor, Tensor]]:
    """Split chronologically-ordered tensors into non-overlapping windows.

    Any leftover samples that don't fill a complete window are appended as
    a final (smaller) window so no data is discarded.

    Parameters
    ----------
    X:
        Input features ``[N, D]``.
    y:
        Targets ``[N, T]``.
    window_size:
        Number of samples per window.

    Returns
    -------
    List of ``(X_chunk, y_chunk)`` tuples.
    """
    n = X.shape[0]
    windows: List[Tuple[Tensor, Tensor]] = []
    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        windows.append((X[start:end], y[start:end]))
    return windows
