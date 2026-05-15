"""Data utilities for the acoustic scattering maze dataset."""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Dataset loading & complexity analysis
# ---------------------------------------------------------------------------


def load_tensor(path: str) -> Tensor:
    """Load a .pt file and return a 4-D tensor (seq, len, x, y).

    Supports two formats:

    1. **Legacy** — a bare tensor saved directly via ``torch.save(tensor, ...)``.
    2. **Prepared** — a dict produced by ``prepare_data.py`` with keys
       ``"pressure"`` (Tensor), ``"complexity"`` (Tensor), ``"metadata"``
       (dict).  Only the ``"pressure"`` tensor is returned here; call
       :func:`load_prepared` to access the full dict.
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict):
        data = raw["pressure"]
    else:
        data = raw

    if not isinstance(data, Tensor):
        raise TypeError(f"Expected Tensor, got {type(data)}")
    if data.ndim != 4:
        raise ValueError(
            f"Expected 4-D tensor (seq, len, x, y), got shape {data.shape}"
        )
    return data


def load_prepared(path: str) -> Tuple[Tensor, Tensor, dict]:
    """Load a prepared .pt file and return (pressure, complexity, metadata).

    If the file is a legacy bare tensor, complexity is computed from spatial
    variance and metadata is empty.
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict):
        pressure: Tensor = raw["pressure"]
        complexity: Tensor = raw.get("complexity", torch.zeros(pressure.shape[0]))
        metadata: dict = raw.get("metadata", {})
    elif isinstance(raw, Tensor):
        pressure = raw
        complexity = torch.zeros(pressure.shape[0])
        metadata = {}
    else:
        raise TypeError(f"Unexpected type in .pt file: {type(raw)}")

    if pressure.ndim != 4:
        raise ValueError(
            f"Expected 4-D tensor (seq, len, x, y), got shape {pressure.shape}"
        )
    return pressure, complexity, metadata


def compute_trajectory_complexity(
    data: Tensor,
    precomputed: Tensor | None = None,
) -> List[float]:
    """Per-trajectory complexity score.

    Parameters
    ----------
    data : Tensor
        Shape (seq, len, x, y).
    precomputed : Tensor | None
        If provided (shape ``(seq,)``), use these scores directly instead
        of computing spatial variance.  Non-zero values are trusted;
        all-zeros triggers the spatial-variance fallback.

    Returns
    -------
    list[float]
        One complexity score per trajectory.
    """
    if precomputed is not None and precomputed.abs().sum() > 0:
        return precomputed.tolist()

    # Fallback: spatial variance per trajectory, averaged over frames
    # data shape: (seq, len, x, y)
    spatial_var = data.var(dim=(-2, -1))  # (seq, len)
    return spatial_var.mean(dim=1).tolist()  # (seq,)


def sort_by_complexity(complexities: Sequence[float]) -> List[int]:
    """Return trajectory indices sorted simple → complex."""
    return sorted(range(len(complexities)), key=lambda i: complexities[i])


def split_test_set(
    sorted_indices: Sequence[int],
    test_fraction: float,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Split trajectory indices into train/test, sampling uniformly across complexity.

    Every ``1 / test_fraction``-th trajectory from the complexity-sorted list
    is assigned to the test set so that the test set spans the full complexity
    range.  Deterministic given *seed*.

    Parameters
    ----------
    sorted_indices : Sequence[int]
        Trajectory indices sorted by complexity (simple → complex).
    test_fraction : float
        Fraction of trajectories to hold out for test (e.g. 0.1 → 10 %).
    seed : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    (train_indices, test_indices)
    """
    if test_fraction <= 0.0:
        return list(sorted_indices), []

    step = max(1, round(1.0 / test_fraction))
    rng = random.Random(seed)
    # Random offset so the test set is not always the 0-th, step-th, …
    offset = rng.randint(0, step - 1)

    test: List[int] = []
    train: List[int] = []
    for i, idx in enumerate(sorted_indices):
        if (i - offset) % step == 0:
            test.append(idx)
        else:
            train.append(idx)
    return train, test


def split_into_brackets(sorted_indices: Sequence[int], n: int) -> List[List[int]]:
    """Split sorted indices into *n* contiguous brackets."""
    total = len(sorted_indices)
    base_size = total // n
    remainder = total % n
    brackets: List[List[int]] = []
    start = 0
    for i in range(n):
        end = start + base_size + (1 if i < remainder else 0)
        brackets.append(list(sorted_indices[start:end]))
        start = end
    return brackets


# ---------------------------------------------------------------------------
# Sliding-window dataset
# ---------------------------------------------------------------------------

_WINDOW = 4  # number of input frames


class FramePairDataset(Dataset):
    """Sliding-window dataset: 4 input frames → 1 target frame.

    Parameters
    ----------
    data : Tensor
        Full tensor (seq, len, x, y).
    traj_indices : list[int]
        Which trajectories to include.
    """

    def __init__(self, data: Tensor, traj_indices: List[int]):
        super().__init__()
        self.data = data
        self.pairs: List[Tuple[int, int]] = []
        for t_idx in traj_indices:
            n_frames = data.shape[1]
            for f in range(n_frames - _WINDOW):
                self.pairs.append((t_idx, f))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        t_idx, f_start = self.pairs[idx]
        # Input: 4 consecutive frames as channels → (4, x, y)
        inp = self.data[t_idx, f_start : f_start + _WINDOW]  # (4, x, y)
        # Target: next frame → (1, x, y)
        tgt = self.data[t_idx, f_start + _WINDOW].unsqueeze(0)  # (1, x, y)
        return inp.float(), tgt.float()


class SelectiveFramePairDataset(Dataset):
    """Wraps any indexable dataset, exposing only the selected indices."""

    def __init__(self, base: Dataset, selected_indices: List[int]):
        self.base = base
        self.selected = selected_indices

    def __len__(self) -> int:
        return len(self.selected)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.base[self.selected[idx]]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
