"""SOLPS 2D field map helpers (numpy only — safe for login-node plotting)."""

from __future__ import annotations

import numpy as np

SOLPS_ION_FIELD_NAMES = ("ne2d", "te2d", "ti2d")


def squeeze_solps_field_maps(arr: np.ndarray) -> np.ndarray:
    """Return (C, H, W) from MATEY outputs that may include batch/time dims."""
    out = np.asarray(arr, dtype=np.float32)
    while out.ndim > 3 and out.shape[0] == 1:
        out = out[0]
    while out.ndim > 3 and out.shape[1] == 1:
        out = out[:, 0, ...]
    if out.ndim == 2:
        out = out[np.newaxis, ...]
    if out.ndim != 3:
        raise ValueError(f"Expected field maps with 3 dims after squeeze, got {out.shape}")
    return out


def field_names_for_maps(maps: np.ndarray) -> list[str]:
    n_fields = min(len(SOLPS_ION_FIELD_NAMES), int(maps.shape[0]))
    return list(SOLPS_ION_FIELD_NAMES[:n_fields])
