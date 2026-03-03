# examples/slac_fel/utils.py
"""Data-loading utilities for the SLAC FEL continuous-learning example.

The data pipeline assumes that all heavy cleaning (archive pull, filtering,
exclusion windows, invalid-PV removal, column selection) has already been done
and the results saved as individual pickle files (``data_1.pkl``,
``data_2.pkl``, …).  All pickle files are loaded, concatenated into a single
dataset in chronological order, and then split into fixed-size windows
controlled by ``cfg.data.window_size``.

If only a single ``data.pkl`` is present (legacy layout), it is loaded in its
entirety and split into fixed-size windows via ``window_size``.

Expected directory layout (pointed to by ``cfg.data.path``)::

    <data_dir>/
        data_1.pkl                     # pandas DataFrame, datetime-indexed, sorted
        data_2.pkl                     # ...
        ...
"""

from __future__ import annotations

import glob
import logging
import os
import re
import warnings
from typing import List, Tuple

import pandas as pd
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

_log = logging.getLogger(__name__)


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

    Args:
        ds: The base dataset.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of data-loading workers.
        pin_memory: Pin CUDA memory for faster transfers.
        persistent_workers: Keep worker processes alive between iterations.
        prefetch_factor: Samples to prefetch per worker.

    Returns:
        DataLoader built from *ds* with the given settings.
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

    Tries the new naming convention (``input_scaler.pt``) first, then
    falls back to the legacy names (``lcls_fel_input_scaler.pt``).

    Args:
        data_path: Directory containing the scaler ``.pt`` files.
        device: Device to map the scalers to.

    Returns:
        Tuple of ``(input_scaler, output_scaler)``.
    """
    # New names (train_fel_model.py v2) → legacy names (fallback)
    input_candidates = ["input_scaler.pt", "lcls_fel_input_scaler.pt"]
    output_candidates = ["output_scaler.pt", "lcls_fel_output_scaler.pt"]

    def _load_first(candidates: list[str]) -> torch.nn.Module:
        for name in candidates:
            path = os.path.join(data_path, name)
            if os.path.exists(path):
                return torch.load(path, map_location=device, weights_only=False)
        raise FileNotFoundError(f"No scaler found in {data_path}; tried {candidates}")

    input_scaler = _load_first(input_candidates)
    output_scaler = _load_first(output_candidates)
    return input_scaler, output_scaler


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_fel_data(
    data_path: str, device: str = "cpu"
) -> Tuple[Tensor, Tensor, pd.Index]:
    """Load a single ``data.pkl``, apply scalers, and return scaled tensors.

    Note that with the current workflow, prefer :func:`discover_window_files`
    + :func:`load_window_file` for per-file lazy loading.

    Args:
        data_path: Directory containing ``data.pkl``, scalers, and
            ``feature_config.yml``.
        device: Device string (used for scaler loading only; tensors stay on
            CPU here).

    Returns:
        Tuple of ``(X_scaled, y_scaled, timestamps)``.
    """
    df: pd.DataFrame = pd.read_pickle(os.path.join(data_path, "data.pkl"))

    # Ensure sorted by time
    df = df.sort_index()

    input_cols, output_cols = load_feature_config(data_path)
    input_scaler, output_scaler = load_scalers(data_path, device=device)

    # Drop rows with NaN in any input or output column
    all_cols = input_cols + output_cols
    n_before = len(df)
    df = df.dropna(subset=all_cols)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        pct = 100.0 * n_dropped / n_before
        msg = (
            f"[load_fel_data] Dropped {n_dropped}/{n_before} rows "
            f"({pct:.1f}%) containing NaN values"
        )
        warnings.warn(msg, stacklevel=2)
        _log.warning(msg)

    X_raw = torch.as_tensor(df[input_cols].values, dtype=torch.float32)
    y_raw = torch.as_tensor(df[output_cols].values, dtype=torch.float32)

    # TODO: maybe import botorch scalers differently to avoid the type: ignore here
    X_scaled: Tensor = input_scaler.transform(X_raw)  # type: ignore[operator]
    y_scaled: Tensor = output_scaler.transform(y_raw)  # type: ignore[operator]

    return X_scaled, y_scaled, df.index


# ---------------------------------------------------------------------------
# Per-file window discovery & lazy loading
# ---------------------------------------------------------------------------


def _natural_sort_key(path: str) -> Tuple[str, int]:
    """Sort key that orders ``data_1.pkl`` < ``data_2.pkl`` < ``data_10.pkl``.

    Falls back to lexicographic order if no numeric suffix is found.
    """
    basename = os.path.basename(path)
    m = re.search(r"(\d+)", basename)
    if m:
        return (basename[: m.start()], int(m.group(1)))
    return (basename, 0)


def discover_window_files(data_path: str) -> List[str]:
    """Return sorted paths to ``data_*.pkl`` files in *data_path*.

    Files are sorted by the numeric suffix so that ``data_1.pkl`` comes
    before ``data_2.pkl`` and ``data_10.pkl``.

    Args:
        data_path: Directory to search.

    Returns:
        List of absolute paths, one per window file, sorted by numeric suffix.
    """
    pattern = os.path.join(data_path, "data_*.pkl")
    paths = glob.glob(pattern)
    paths.sort(key=_natural_sort_key)
    return paths


def load_window_file(
    pkl_path: str,
    input_cols: List[str],
    output_cols: List[str],
    input_scaler: torch.nn.Module,
    output_scaler: torch.nn.Module,
) -> Tuple[Tensor, Tensor, pd.Index]:
    """Load a single window pickle, scale, and return ``(X, y, index)`` tensors.

    Args:
        pkl_path: Path to a single ``data_<N>.pkl`` file.
        input_cols: Column names for input features (from ``feature_config.yml``).
        output_cols: Column names for output targets.
        input_scaler: Pre-fitted scaler for inputs.
        output_scaler: Pre-fitted scaler for outputs.

    Returns:
        Tuple ``(X_scaled, y_scaled, index)`` where ``X_scaled`` is
        ``[N, n_inputs]`` float32, ``y_scaled`` is ``[N, n_outputs]`` float32,
        and ``index`` is the DataFrame index after NaN filtering.
    """
    df: pd.DataFrame = pd.read_pickle(pkl_path)
    df = df.sort_index()

    # Drop rows with NaN in any input or output column
    all_cols = input_cols + output_cols
    n_before = len(df)
    df = df.dropna(subset=all_cols)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        pct = 100.0 * n_dropped / n_before
        basename = os.path.basename(pkl_path)
        msg = (
            f"[load_window_file] {basename}: Dropped {n_dropped}/{n_before} rows "
            f"({pct:.1f}%) containing NaN values"
        )
        warnings.warn(msg, stacklevel=2)
        _log.warning(msg)

    X_raw = torch.as_tensor(df[input_cols].values, dtype=torch.float32)
    y_raw = torch.as_tensor(df[output_cols].values, dtype=torch.float32)

    X_scaled: Tensor = input_scaler.transform(X_raw)  # type: ignore[operator]
    y_scaled: Tensor = output_scaler.transform(y_raw)  # type: ignore[operator]

    return X_scaled, y_scaled, df.index


def load_all_window_files(
    data_path: str,
    input_cols: List[str],
    output_cols: List[str],
    input_scaler: torch.nn.Module,
    output_scaler: torch.nn.Module,
) -> Tuple[Tensor, Tensor, pd.Index]:
    """Load all ``data_*.pkl`` files, concatenate, and return scaled tensors.

    Each pickle is loaded and scaled individually via :func:`load_window_file`,
    then the results are concatenated into a single pair of tensors in
    chronological order (files are sorted by numeric suffix).

    Args:
        data_path: Directory containing ``data_*.pkl`` files.
        input_cols: Column names for input features.
        output_cols: Column names for output targets.
        input_scaler: Pre-fitted scaler for inputs.
        output_scaler: Pre-fitted scaler for outputs.

    Returns:
        Tuple ``(X_all, y_all, timestamps)`` with all windows concatenated
        along dim 0.  ``timestamps`` is the combined DataFrame index
        preserving the chronological order of every row.

    Raises:
        FileNotFoundError: If no ``data_*.pkl`` files are found in *data_path*.
    """
    window_files = discover_window_files(data_path)
    if not window_files:
        raise FileNotFoundError(f"No data_*.pkl files found in {data_path}")

    X_parts: List[Tensor] = []
    y_parts: List[Tensor] = []
    index_parts: List[pd.Index] = []
    total_samples = 0

    for pkl_path in window_files:
        X_w, y_w, idx = load_window_file(
            pkl_path, input_cols, output_cols, input_scaler, output_scaler
        )
        X_parts.append(X_w)
        y_parts.append(y_w)
        index_parts.append(idx)
        total_samples += X_w.shape[0]
        _log.info(
            "[load_all_window_files] Loaded %s: %d samples",
            os.path.basename(pkl_path),
            X_w.shape[0],
        )

    X_all = torch.cat(X_parts, dim=0)
    y_all = torch.cat(y_parts, dim=0)
    timestamps = index_parts[0].append(index_parts[1:])

    _log.info(
        "[load_all_window_files] Combined %d files → %d total samples",
        len(window_files),
        total_samples,
    )

    return X_all, y_all, timestamps


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

# Default number of samples per time window.  Can be overridden by the caller.
DEFAULT_WINDOW_SIZE: int = 5000


def split_into_windows(
    X: Tensor,
    y: Tensor,
    window_size: int = DEFAULT_WINDOW_SIZE,
    min_window_size: int = 2,
) -> List[Tuple[Tensor, Tensor]]:
    """Split chronologically-ordered tensors into non-overlapping windows.

    If the final chunk has fewer than *min_window_size* samples it is merged
    into the preceding window so that every window is large enough for a
    meaningful train/val split.

    Args:
        X: Input features ``[N, D]``.
        y: Targets ``[N, T]``.
        window_size: Number of samples per window.
        min_window_size: Minimum samples in a window.  A trailing chunk
            smaller than this is merged into the previous window.

    Returns:
        List of ``(X_chunk, y_chunk)`` tuples.
    """
    n = X.shape[0]
    windows: List[Tuple[Tensor, Tensor]] = []
    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        windows.append((X[start:end], y[start:end]))

    # Merge a too-small trailing window into the previous one
    if len(windows) > 1 and windows[-1][0].shape[0] < min_window_size:
        prev_X, prev_y = windows[-2]
        tail_X, tail_y = windows[-1]
        windows[-2] = (
            torch.cat([prev_X, tail_X], dim=0),
            torch.cat([prev_y, tail_y], dim=0),
        )
        windows.pop()

    return windows


def split_timestamps(
    timestamps: pd.Index,
    window_size: int = DEFAULT_WINDOW_SIZE,
    min_window_size: int = 2,
) -> List[pd.Index]:
    """Split a timestamp index to match :func:`split_into_windows`.

    Applies the same chunking and merging logic so that
    ``split_timestamps(ts, ws)[i]`` aligns row-for-row with
    ``split_into_windows(X, y, ws)[i]``.

    Args:
        timestamps: Row-aligned index (same length as the tensors).
        window_size: Number of samples per window.
        min_window_size: Merge trailing chunk if smaller than this.

    Returns:
        List of ``pd.Index`` slices, one per window.
    """
    n = len(timestamps)
    chunks: List[pd.Index] = []
    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        chunks.append(timestamps[start:end])

    if len(chunks) > 1 and len(chunks[-1]) < min_window_size:
        chunks[-2] = chunks[-2].append(chunks[-1])
        chunks.pop()

    return chunks
