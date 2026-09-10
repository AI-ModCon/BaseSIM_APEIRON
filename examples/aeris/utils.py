# examples/aeris/utils.py
"""Utility functions for the AERIS continuous-learning example.

Expected directory layout (pointed to by ``cfg.data.path``)::

    <data_dir>/
        dataset.csv            # data that will be parsed by the SIM framework
        aeris_model.pt        # AERIS pre-trained model

Featurization uses the *fast path* and must stay byte-for-byte consistent with
``examples/aeris/scripts/make_drift_split.py:build_features`` -- the same code
that produced the training features. 227 of the 233 features are pre-computed
columns pulled directly from the CSV; the remaining 6 lattice params are parsed
from the ``structure`` string. No matminer recompute (that path could disagree
with the pre-computed columns the model was trained on).
"""

import os
import glob
import re
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def load_pretrained_model(
    data_path: str, model_name: str, device: str = "cpu"
) -> dict[str, Any]:
    """Load the pretrained AERIS model.

    Parameters
    ----------
    data_path:
        Directory containing the model.
    model_name:
        The name of the pretrained model.
    device:
        Device to map the scalers to.

    Returns
    -------
    model_info = {
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'feature_names': feature_names,
        'scaler': scaler,
        'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2},
        'history': history
    }
    """
    ckpt = None
    if os.path.exists(data_path):
        ckpt = torch.load(
            os.path.join(data_path, model_name), map_location=device, weights_only=False
        )
    if ckpt is None:
        raise FileNotFoundError("No model found at path: " + data_path)
    return ckpt


# -----------------------------
# Fast-path featurization
# (mirrors scripts/make_drift_split.py so the harness featurizes inputs exactly
#  the way the model was trained)
# -----------------------------
LATTICE_KEYS = [
    "lattice_a", "lattice_b", "lattice_c",
    "lattice_alpha", "lattice_beta", "lattice_gamma",
]


def _parse_lattice(struct_str: Any) -> Dict[str, float]:
    """Extract the 6 lattice params from a pymatgen structure string."""
    r = {k: 0.0 for k in LATTICE_KEYS}
    s = str(struct_str)
    abc = re.search(r"abc\s*:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)", s)
    ang = re.search(r"angles\s*:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)", s)
    if abc:
        r["lattice_a"], r["lattice_b"], r["lattice_c"] = map(float, abc.groups())
    if ang:
        r["lattice_alpha"], r["lattice_beta"], r["lattice_gamma"] = map(float, ang.groups())
    return r


def _build_X_fast(df: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    """Assemble the ``(N, len(feature_names))`` matrix via the fast path.

    227 features are pulled directly from pre-computed CSV columns; the 6 lattice
    params are parsed from the ``structure`` string. Any feature not found is
    left at 0. Identical to ``make_drift_split.build_features``.
    """
    n = len(df)
    X = np.zeros((n, len(feature_names)), dtype=np.float32)
    col_idx = {f: j for j, f in enumerate(feature_names)}

    present = [f for f in feature_names if f in df.columns]
    sub = df[present].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    for k, f in enumerate(present):
        X[:, col_idx[f]] = sub[:, k]

    if "structure" in df.columns:
        lat = np.array(
            [list(_parse_lattice(s).values()) for s in df["structure"].tolist()],
            dtype=np.float32,
        )
        for k, f in enumerate(LATTICE_KEYS):
            if f in col_idx:
                X[:, col_idx[f]] = lat[:, k]

    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)


def load_datasets(
    data_path: str, dataset_name: str, feature_names: List[str], input_dim: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load the dataset used by the model.

    Features are assembled in the exact ``feature_names`` order via the fast
    path (pre-computed columns + parsed lattice params), matching how the model
    was trained. Rows with a missing target are dropped.

    Returns
    -------
    X: numpy.ndarray of shape (n_samples, n_features) dtype float32 (unscaled)
    y: numpy.ndarray of shape (n_samples, 1) dtype float32

    Note: scaling is intentionally NOT applied here. The caller (model harness)
    applies the saved scaler from the checkpoint via scaler.transform().
    """
    dataset_pattern = os.path.join(data_path)
    dataset_files: List[str] = glob.glob(dataset_pattern)
    if not dataset_files:
        raise FileNotFoundError(f"No dataset files matched pattern: {dataset_pattern}")

    dfs = [pd.read_csv(fp, low_memory=False) for fp in dataset_files]
    dataset: pd.DataFrame = pd.concat(dfs, ignore_index=True)

    target_col = "formation_energy_per_atom"
    if target_col not in dataset.columns:
        raise KeyError(f"Missing target column '{target_col}'")
    dataset = dataset[dataset[target_col].notna()].reset_index(drop=True)

    X = _build_X_fast(dataset, feature_names)
    y = dataset[target_col].to_numpy(np.float32).reshape(-1, 1)

    if X.shape[1] != input_dim:
        raise ValueError(
            f"Checkpoint input_dim={input_dim} but built X has {X.shape[1]} features."
        )

    return X, y


# Default number of samples per time window.  Can be overridden by the caller.
DEFAULT_WINDOW_SIZE: int = 500

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
