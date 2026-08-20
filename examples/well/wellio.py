"""Read a `The Well <https://polymathic-ai.github.io/the_well>`_ HDF5 file into a
dense ``[time, channel, H, W]`` array.

The Well stores physical fields grouped by tensor order:

* ``t0_fields/<name>``  scalar fields   -- shape ``[n_traj, time, H, W]``
* ``t1_fields/<name>``  vector fields   -- shape ``[n_traj, time, H, W, d]``
* ``t2_fields/<name>``  tensor fields   -- shape ``[n_traj, time, H, W, d, d]``

This flattens all of them into one channel axis in a deterministic order (groups
``t0 -> t1 -> t2``; datasets sorted by name; trailing component dims expanded with
``_0``/``_1`` suffixes), so a channel layout is reproducible across files and runs.

Only stdlib + numpy + h5py are used, so this does not require the (heavy)
``the_well`` package -- just ``pip install h5py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import h5py

_FIELD_GROUPS = ("t0_fields", "t1_fields", "t2_fields")


@dataclass(frozen=True)
class WellTrajectory:
    """One trajectory's fields plus its provenance metadata."""

    fields: np.ndarray  # [time, channel, H, W], float32
    channels: tuple[str, ...]  # channel names, len == fields.shape[1]
    params: dict[str, float]  # simulation parameters (e.g. {"tcool": 0.03})
    times: np.ndarray  # [time] physical timestamps
    source: str  # file basename


def _expand_channels(name: str, arr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Expand one field dataset (already sliced to [time, H, W, *comp]) to channels.

    ``arr`` has the trajectory axis removed. A scalar field is ``[time, H, W]``
    (one channel); a vector/tensor field has trailing component dims that become
    separate channels named ``<name>_<i>[_<j>]``.
    """
    if arr.ndim == 3:  # [time, H, W] scalar field
        return [(name, arr)]
    comp_shape = arr.shape[3:]
    n_comp = int(np.prod(comp_shape))
    t, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
    flat = arr.reshape(t, h, w, n_comp)  # [time, H, W, prod(comp)]
    out = []
    for k in range(n_comp):
        idx = np.unravel_index(k, comp_shape)
        suffix = "_".join(str(i) for i in idx)
        out.append((f"{name}_{suffix}", flat[..., k]))
    return out


@dataclass(frozen=True)
class WellBundle:
    """All (or the first ``max``) trajectories of one file, plus provenance.

    ``trajectories`` are each ``[time, channel, H, W]`` and share ``channels`` /
    ``params`` (a file is one simulation regime, e.g. one ``tcool``). Stacking
    them makes bigger windows without mixing regimes.
    """

    trajectories: list[np.ndarray]
    channels: tuple[str, ...]
    params: dict[str, float]
    times: np.ndarray
    source: str

    @property
    def n_trajectories(self) -> int:
        return len(self.trajectories)


def _read_params_times(f: "h5py.File") -> tuple[dict[str, float], "np.ndarray | None"]:
    params: dict[str, float] = {}
    for p in _param_names(f):
        if p in f.attrs:
            params[p] = float(np.asarray(f.attrs[p]).reshape(-1)[0])
    times = (
        np.asarray(f["dimensions"]["time"], dtype=np.float32)
        if "dimensions" in f and "time" in f["dimensions"]
        else None
    )
    return params, times


def _assemble_trajectory(
    f: "h5py.File", trajectory: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Assemble one trajectory of an open file into ``[time, channel, H, W]``."""
    channels: list[str] = []
    planes: list[np.ndarray] = []
    for group in _FIELD_GROUPS:
        if group not in f:
            continue
        grp = f[group]
        for name in sorted(grp.keys()):
            arr = np.asarray(grp[name][trajectory], dtype=np.float32)  # [time, H, W, *]
            for ch_name, plane in _expand_channels(name, arr):
                channels.append(ch_name)
                planes.append(plane)
    if not planes:
        raise ValueError("no t0/t1/t2 fields found")
    fields = np.stack(planes, axis=0).transpose(1, 0, 2, 3)  # -> [time, C, H, W]
    return np.ascontiguousarray(fields), tuple(channels)


def read_well_file(path: str | Path, trajectory: int = 0) -> WellTrajectory:
    """Load one trajectory from a Well HDF5 file as ``[time, channel, H, W]``."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        params, times = _read_params_times(f)
        fields, channels = _assemble_trajectory(f, trajectory)
    if times is None:
        times = np.arange(fields.shape[0], dtype=np.float32)
    return WellTrajectory(
        fields=fields, channels=channels, params=params, times=times, source=path.name
    )


def read_all_trajectories(
    path: str | Path, max_trajectories: int | None = None
) -> WellBundle:
    """Load every (or the first ``max_trajectories``) trajectory of a file.

    Opens the file once and slices each trajectory lazily, so an 8-trajectory
    file is one open, not eight.
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        params, times = _read_params_times(f)
        n = int(np.asarray(f.attrs.get("n_trajectories", 1)).reshape(-1)[0])
        if max_trajectories:
            n = min(n, max_trajectories)
        trajs: list[np.ndarray] = []
        channels: tuple[str, ...] = ()
        for i in range(max(1, n)):
            fields, channels = _assemble_trajectory(f, i)
            trajs.append(fields)
    if times is None:
        times = np.arange(trajs[0].shape[0], dtype=np.float32)
    return WellBundle(
        trajectories=trajs,
        channels=channels,
        params=params,
        times=times,
        source=path.name,
    )


def _param_names(f: "h5py.File") -> list[str]:
    raw = f.attrs.get("simulation_parameters")
    if raw is None:
        return []
    if isinstance(raw, (bytes, str)):
        return [raw.decode() if isinstance(raw, bytes) else raw]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(raw)]


def channel_count(path: str | Path) -> int:
    """Number of channels a file yields (cheap: reads shapes, not the data)."""
    import h5py

    n = 0
    with h5py.File(path, "r") as f:
        for group in _FIELD_GROUPS:
            if group not in f:
                continue
            for name in f[group]:
                shape = f[group][name].shape  # [n_traj, time, H, W, *comp]
                comp = shape[4:]
                n += int(np.prod(comp)) if comp else 1
    return n


def infer_param(path: str | Path) -> tuple[str, float] | None:
    """Return the primary ``(param_name, value)`` used to order the drift stream."""
    import h5py

    with h5py.File(path, "r") as f:
        names = _param_names(f)
        if not names:
            return None
        name = names[0]
        if name in f.attrs:
            return name, float(np.asarray(f.attrs[name]).reshape(-1)[0])
    return None


def to_dict(traj: WellTrajectory) -> dict[str, Any]:
    """JSON-friendly provenance (no field data) for manifests."""
    return {
        "source": traj.source,
        "channels": list(traj.channels),
        "params": traj.params,
        "n_time": int(traj.fields.shape[0]),
        "grid": list(traj.fields.shape[2:]),
    }
