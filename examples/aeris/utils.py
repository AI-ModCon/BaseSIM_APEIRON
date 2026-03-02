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
from typing import Dict, List, Tuple, Any

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


def _apply_df_parse_formula_num(val):
    try:
        if pd.isna(val):
            return None
        parsed = _parse_formula(str(val))
        return int(sum(parsed.values()))
    except Exception:
        return None


def _apply_df_parse_formula_str(val):
    try:
        if pd.isna(val):
            return None
        parsed = _parse_formula(str(val))
        return "".join(f"{k}{v}" for k, v in sorted(parsed.items()))
    except Exception:
        return None


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


def _build_feature_vector(
    composition: str, features: Dict, feature_names: List[str]
) -> np.ndarray:
    comp = _parse_formula(composition)
    total_atoms = float(sum(comp.values()))
    # prepare composition fractions
    elem_frac = {k: v / total_atoms for k, v in comp.items()}

    # parse structure if string/dict
    struct_vals = {}
    if features is not None:
        for feature in features:
            if feature not in feature_names:
                continue
            struct_vals[feature] = features[feature]
        if "structure" in features:
            parsed_struct = _parse_structure_string(features["structure"])
            struct_vals.update(parsed_struct)

    # magpie
    feature_calculators = MultipleFeaturizer(
        [
            cf.Stoichiometry(),
            cf.ElementProperty.from_preset("magpie"),
            cf.ValenceOrbital(props=["avg"]),
            cf.IonProperty(fast=True),
        ]
    )

    comp_obj = Composition(composition)
    data = pd.DataFrame([{"comp_obj": comp_obj, "composition_reduced": composition}])

    # Calculate Magpie features.
    # IMPORTANT: when running under MPI, do NOT let matminer spawn multiprocessing pools
    # inside each rank (oversubscription/hangs). Force single-process.
    magpie_features_dict = {}
    try:
        # Some matminer versions support n_jobs; if yours does, keep it at 1.
        magpie_features = feature_calculators.featurize_dataframe(
            data, col_id="comp_obj", ignore_errors=True, pbar=False, n_jobs=1
        )
        magpie_features_dict = magpie_features.iloc[0].to_dict()
    except Exception:
        try:
            feature_calculators.set_n_jobs(1)
            feats = feature_calculators.featurize_many([Composition(composition)])
            magpie_features = pd.DataFrame(feats)
            magpie_features.index = [0]
            magpie_features_dict = magpie_features.iloc[0].to_dict()
        except Exception as e:
            print("Magpie featurizer failed, falling back to empty features:", repr(e))
            magpie_features_dict = {}

    vec = np.zeros(len(feature_names), dtype=np.float32)
    for i, name in enumerate(feature_names):
        # elemental features (assume single element name)
        if re.match(r"^[A-Z][a-z]?$", name) and name in elem_frac:
            vec[i] = float(elem_frac.get(name, 0.0))
            continue

        # structural features
        if name in struct_vals:
            vec[i] = float(struct_vals[name])
            continue

        # magpie features
        if name in magpie_features_dict:
            vec[i] = float(magpie_features_dict[name])
            continue

        # try numeric keys in struct_vals
        val = struct_vals.get(name)
        if val is None:
            v = struct_vals.get(name, 0.0)
            try:
                vec[i] = float(v)
            except Exception:
                vec[i] = 0.0
        else:
            try:
                vec[i] = float(val)
            except Exception:
                vec[i] = 0.0

    # Return a 1D feature vector (D,) instead of (1, D)
    vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
    return vec


def load_datasets(data_path: str, dataset_name: str, feature_names: List[str]):
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

    # Ensure target present
    if "formation_energy_per_atom" not in dataset.columns:
        raise KeyError("Required target column 'formation_energy_per_atom' not found in dataset")

    # Drop rows missing target
    dataset = dataset.dropna(subset=["formation_energy_per_atom"]).copy()

    # If all feature_names are present as columns, take that branch (preferred)
    all_present = all((fn in dataset.columns) for fn in feature_names)

    if all_present:
        # Select columns in the exact saved order
        X = dataset[feature_names].to_numpy(dtype=np.float32)

        # Replace infs / NaNs in numeric columns with column means (same as training)
        # (do not change dtype or drop rows here; keep alignment with model)
        numeric_mask = np.isfinite(X)
        # For each column replace non-finite with column mean (computed over finite rows)
        col_means = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        # Where a column is completely NaN/inf, set mean to 0.0
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        inds = np.where(~np.isfinite(X))
        if inds[0].size > 0:
            X[inds] = np.take(col_means, inds[1])

    else:
        # Fall back to building feature vectors row-by-row using the helper
        # This creates the same ordering as feature_names when possible (elemental
        # names are interpreted by _build_feature_vector).
        X_rows = []
        # Build a minimal features dict per row (this mirrors training's inputs)
        for _, row in dataset.iterrows():
            comp = row.get("composition_reduced", row.get("composition", None))
            features = {
                "composition": row.get("composition", None),
                "structure": row.get("structure", None),
                "spacegroup_number": row.get("spacegroup_number", None),
                "density_atomic": row.get("density_atomic", None),
                "CN_max": row.get("CN_max", None),
                "CN_min": row.get("CN_min", None),
                "CN_avg": row.get("CN_avg", None),
            }
            try:
                vec = _build_feature_vector(comp, features, feature_names)
            except Exception:
                # on failure, append zeros to avoid mismatched shapes
                vec = np.zeros(len(feature_names), dtype=np.float32)
            X_rows.append(vec)
        X = np.vstack(X_rows).astype(np.float32)

        # Replace any remaining nan/inf with column means
        col_means = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        inds = np.where(~np.isfinite(X))
        if inds[0].size > 0:
            X[inds] = np.take(col_means, inds[1])

    # Prepare target vector shape (N,)
    y = dataset["formation_energy_per_atom"].to_numpy(dtype=np.float32).reshape(-1, 1)
    assert X.shape[0] == y.shape[0], "Feature matrix and target vector must have same number of rows"

    return X, y


def load_datasets2(data_path: str, dataset_name: str, feature_names: List[str]):
    """Load the dataset that will be parsed, return features and ground truth.

    Parameters
    ----------
    data_path:
        Directory containing the datasets.
    dataset_name:
        The name or regular expression for the datasets
    feature_names:
        The features used by the model for prediction

    Returns
    -------
    input festures, output target values
    """
    dfs = []
    dataset_pattern = os.path.join(data_path, dataset_name)
    dataset_files: List[str] = glob.glob(dataset_pattern)
    if not dataset_files:
        raise FileNotFoundError(f"No dataset files matched pattern: {dataset_pattern}")
    for file_path in dataset_files:
        dfs.append(pd.read_csv(file_path, low_memory=False))
    dataset: pd.DataFrame = pd.concat(dfs, ignore_index=True)

    # Filter all entries that do not have a target value
    dataset = dataset.dropna(subset=["formation_energy_per_atom"]).copy()

    # Replace NaN/+inf/-inf in numeric columns (keep DataFrame type)
    num_cols = dataset.select_dtypes(include=[np.number]).columns
    dataset[num_cols] = dataset[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y = (
        dataset["formation_energy_per_atom"]
        .values.astype(np.float32)
        .reshape(-1, 1)
    )
    X = []
    for _, row in dataset.iterrows():
        composition = row["composition_reduced"]
        features = {
            "composition": row["composition"],
            "structure": row["structure"],
            "spacegroup_number": row["spacegroup_number"],
            "density_atomic": row["density_atomic"],
            "CN_max": row["CN_max"],
            "CN_min": row["CN_min"],
            "CN_avg": row["CN_avg"],
        }
        X.append(_build_feature_vector(composition, features, feature_names))

    print("X shape:", np.array(X).shape, "y shape:", y.shape)
    assert len(X) == len(y), (
        "The feature and target vectors do not have the same lenght"
    )
    return X, y


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
