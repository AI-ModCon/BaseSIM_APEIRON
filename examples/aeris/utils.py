# examples/aeris/utils.py
"""Utility functions for the AERIS continuous-learning example.

Expected directory layout (pointed to by ``cfg.data.path``)::

    <data_dir>/
        dataset.csv            # data that will be parsed by the SIM framework
        aeris_model.pt        # AERIS pre-trained model
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

from pymatgen.core.composition import Composition
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf


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


def _parse_formula(s: str) -> Dict[str, float]:
    parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", str(s).strip())
    if not parts:
        raise ValueError(f"Could not parse formula: {s}")
    comp: Dict[str, float] = {}
    for el, num in parts:
        comp[el] = float(num) if num else 1.0
    return comp


def _parse_structure_string(struct_str: str) -> Dict[str, float]:
    # minimal lattice extractor (compatible with training utils)
    result = {
        "lattice_a": np.nan,
        "lattice_b": np.nan,
        "lattice_c": np.nan,
        "lattice_alpha": np.nan,
        "lattice_beta": np.nan,
        "lattice_gamma": np.nan,
        "volume": np.nan,
        "density": np.nan,
        "nsites": np.nan,
        "spacegroup_number": np.nan,
    }
    if struct_str is None:
        return result
    s = str(struct_str)
    abc_pattern = r"abc\s*:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    angles_pattern = r"angles\s*:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    abc = re.search(abc_pattern, s)
    ang = re.search(angles_pattern, s)
    if abc:
        result["lattice_a"] = float(abc.group(1))
        result["lattice_b"] = float(abc.group(2))
        result["lattice_c"] = float(abc.group(3))
    if ang:
        result["lattice_alpha"] = float(ang.group(1))
        result["lattice_beta"] = float(ang.group(2))
        result["lattice_gamma"] = float(ang.group(3))
    # try volume
    vol_match = re.search(r"volume\s*[:=]\s*([\d.]+)", s)
    if vol_match:
        result["volume"] = float(vol_match.group(1))
    dens_match = re.search(r"density\s*[:=]\s*([\d.]+)", s)
    if dens_match:
        result["density"] = float(dens_match.group(1))
    sg_match = re.search(r"spacegroup(?:_number)?\s*[:=]\s*(\d+)", s)
    if sg_match:
        result["spacegroup_number"] = int(sg_match.group(1))
    nsites_match = re.search(r"nsites\s*[:=]\s*(\d+)", s)
    if nsites_match:
        result["nsites"] = int(nsites_match.group(1))
    return result


# -----------------------------
# Build X,y in *checkpoint feature order*
# -----------------------------
# optional numeric columns (if present in CSV) that we will include as features
OPTIONAL_NUMERIC_COLS = [
    'density_atomic', 'CN_max', 'CN_min', 'CN_avg',
    # add more if you know they exist & are useful
]

def _make_magpie_featurizer() -> MultipleFeaturizer:
    return MultipleFeaturizer([
        cf.Stoichiometry(),
        cf.ElementProperty.from_preset("magpie"),
        cf.ValenceOrbital(props=['avg']),
        cf.IonProperty(fast=True),
    ])

def _compute_magpie_df(compositions: pd.Series) -> pd.DataFrame:
    featurizer = _make_magpie_featurizer()

    comp_objs = []
    for s in compositions.astype(str).tolist():
        try:
            comp_objs.append(Composition(s))
        except Exception:
            comp_objs.append(None)

    base = pd.DataFrame({"comp_obj": comp_objs}, index=compositions.index)

    try:
        feat_df = featurizer.featurize_dataframe(
            base, col_id="comp_obj", ignore_errors=True, pbar=False, n_jobs=1
        )
    except TypeError:
        try:
            featurizer.set_n_jobs(1)
        except Exception:
            pass
        feat_df = featurizer.featurize_dataframe(
            base, col_id="comp_obj", ignore_errors=True, pbar=False
        )

    feat_df = feat_df.drop(columns=[c for c in feat_df.columns if c == "comp_obj"], errors="ignore")
    return feat_df

def _build_X_y_in_ckpt_order(
    df: pd.DataFrame,
    feature_names: List[str],
    target_col: Optional[str],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    required = ["composition", "structure"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}'")

    if target_col is not None and target_col not in df.columns:
        raise KeyError(f"Missing target column '{target_col}'")

    magpie_df = _compute_magpie_df(df["composition"])
    n = len(df)
    X = np.zeros((n, len(feature_names)), dtype=np.float32)
    y: Optional[np.ndarray] = None
    if target_col is not None:
        y = np.zeros((n, 1), dtype=np.float32)

    df2 = df.reset_index(drop=True)

    for i, row in df2.iterrows():
        comp_str = str(row["composition"])

        # 1) element fractions
        try:
            parsed = _parse_formula(comp_str)
            total = float(sum(parsed.values())) if parsed else 0.0
        except Exception:
            parsed, total = {}, 0.0

        elem_frac: Dict[str, float] = {}
        if total > 0:
            for el, cnt in parsed.items():
                elem_frac[el] = float(cnt) / total

        # 2) structure features
        struct_vals = _parse_structure_string(row.get("structure"))

        # 3) optional numeric cols
        opt_vals: Dict[str, float] = {}
        for c in OPTIONAL_NUMERIC_COLS:
            if c in df2.columns:
                v = row.get(c)
                try:
                    opt_vals[c] = float(v)
                except Exception:
                    opt_vals[c] = np.nan

        # 4) magpie row
        magpie_row = magpie_df.iloc[i].to_dict()

        # single lookup dict, then assemble in EXACT feature_names order
        value_by_name: Dict[str, float] = {}
        for el, frac in elem_frac.items():
            value_by_name[el] = float(frac)
        for k, v in struct_vals.items():
            try:
                value_by_name[k] = float(v)
            except Exception:
                pass
        for k, v in opt_vals.items():
            try:
                value_by_name[k] = float(v)
            except Exception:
                pass
        for k, v in magpie_row.items():
            try:
                value_by_name[k] = float(v)
            except Exception:
                pass

        X[i, :] = np.array([value_by_name.get(name, 0.0) for name in feature_names], dtype=np.float32)

        if y is not None:
            try:
                y[i, 0] = float(row[target_col])  # type: ignore[arg-type]
            except Exception:
                y[i, 0] = np.nan

    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    if y is not None:
        y = y.astype(np.float32)
        # drop rows where y is nan
        mask = ~np.isnan(y[:, 0])
        X = X[mask]
        y = y[mask]
    return X, y


def load_datasets(data_path: str, dataset_name: str, feature_names: List[str], input_dim: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load the dataset used by the model.

    This function attempts to *prefer* loading the exact columns listed in
    `feature_names` (in the same order). If those columns are present in the
    CSV(s), they are used directly (fast, deterministic). If not all feature
    columns are present, the function falls back to building feature vectors
    row-by-row using _build_feature_vector to preserve compatibility with older
    or alternate CSV formats.

    The function returns:
        X: numpy.ndarray of shape (n_samples, n_features) dtype float32
        y: numpy.ndarray of shape (n_samples,) dtype float32

    Note: scaling is intentionally NOT applied here. The caller (model harness)
    will apply the saved scaler from the checkpoint (if any) via scaler.transform().
    """

    # collect files
    dataset_pattern = os.path.join(data_path, dataset_name)
    dataset_files: List[str] = glob.glob(dataset_pattern)
    if not dataset_files:
        raise FileNotFoundError(f"No dataset files matched pattern: {dataset_pattern}")

    # read & concatenate CSV files
    dfs = []
    for file_path in dataset_files:
        dfs.append(pd.read_csv(file_path, low_memory=False))
    dataset: pd.DataFrame = pd.concat(dfs, ignore_index=True)

    target_col = 'formation_energy_per_atom'
    X_raw, y = _build_X_y_in_ckpt_order(dataset, feature_names=feature_names, target_col=target_col)
    #print("Prepared X:", X_raw.shape, "y:", None if y is None else y.shape, "num_features:", len(feature_names))

    if X_raw.shape[1] != input_dim:
        raise ValueError(f"Checkpoint input_dim={input_dim} but built X has {X_raw.shape[1]} features.")

    return X_raw, y

# Default number of samples per time window.  Can be overridden by the caller.
DEFAULT_WINDOW_SIZE: int = 100

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
