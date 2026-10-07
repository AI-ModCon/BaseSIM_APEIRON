"""Which Well datasets this example knows, and how to get one onto disk.

PolymathicAI's "The Well" is a collection of physics simulation datasets, each
published as a *parameter sweep*: one HDF5 file per value of some physical
parameter. That layout is the reason the collection is useful here. Walking the
files in parameter order walks a model through real physical regime change, so
the drift a run sees is not a perturbation applied to the inputs -- it is what
happens when the physics moves.

Three things vary from one Well dataset to the next, and all three live in a
:class:`WellDataset` row:

``fields``
    Which HDF5 groups and fields become which channels. The Well stores scalars
    under ``t0_fields`` and vectors under ``t1_fields``, and a vector needs a
    component index.
``spatial_resolution`` / ``n_spatial_dims``
    The grid. Checked against the file at conversion time, so a wrong row fails
    with the shape it actually found rather than somewhere downstream.
``order``
    How to turn a filename into the swept parameter, for sorting. This is the
    one most likely to need overriding, because the trailing number is only the
    right answer when the sweep has one parameter::

        turbulent_radiative_layer_tcool_0.06      -> 0.06   correct
        rayleigh_benard_Rayleigh_1e10_Prandtl_1   -> 1.0    sorts by Prandtl
        shear_flow_Reynolds_1e4_Schmidt_1e-1      -> 0.1    sorts by Schmidt
        viscoelastic_instability_AH               -> inf    no number at all

    A two-parameter sweep wants its own key function; a dataset whose files are
    named flow states rather than numbers wants an explicit sequence.

To add a dataset, add a row to :data:`REGISTRY`. Nothing else in this example
is dataset-specific.
"""

from __future__ import annotations

import json
import os
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import yaml

HF_HOST = "https://huggingface.co"

# Where regimes land when they are downloaded. The other examples auto-download
# to ./data, so this follows them.
CACHE_ROOT = Path("data/well")

# Bump when the conversion below changes in a way that alters the bytes. Arrays
# stamped with an older recipe are rebuilt rather than trusted.
CONVERSION_RECIPE = "well-hdf5->f32[traj,time,channel,*spatial]:v1"

RECIPE_SUFFIX = ".recipe"

# (HDF5 group, field name, component index or None for a scalar field)
Field = Tuple[str, str, Optional[int]]


def trailing_number(name: str) -> float:
    """The number at the end of a Well filename.

    ``turbulent_radiative_layer_tcool_0.06`` -> ``0.06``. Sorting on this
    rather than on the string keeps a sweep in physical order even when the
    filenames are not zero-padded. Names with no trailing number sort last,
    together, rather than raising -- which is the signal that the dataset needs
    its own ``order``.
    """
    try:
        return float(name.rsplit("_", 1)[-1])
    except ValueError:
        return float("inf")


@dataclass(frozen=True)
class WellDataset:
    """Everything this example needs to know about one Well dataset."""

    name: str
    fields: Tuple[Field, ...]
    spatial_resolution: Tuple[int, ...]
    n_spatial_dims: int = 2
    order: Callable[[str], float] = trailing_number

    @property
    def n_channels(self) -> int:
        return len(self.fields)

    @property
    def channel_names(self) -> List[str]:
        """Human-readable channel labels, for the record of what a run saw."""
        return [
            f"{field}" if component is None else f"{field}[{component}]"
            for _, field, component in self.fields
        ]


TURBULENT_RADIATIVE_LAYER_2D = WellDataset(
    name="turbulent_radiative_layer_2D",
    fields=(
        ("t0_fields", "density", None),
        ("t0_fields", "pressure", None),
        ("t1_fields", "velocity", 0),
        ("t1_fields", "velocity", 1),
    ),
    spatial_resolution=(128, 384),
    n_spatial_dims=2,
    # Nine files, tcool_0.03 through tcool_3.16. One parameter, at the end of
    # the name, so the default key is right.
    order=trailing_number,
)

REGISTRY: Dict[str, WellDataset] = {
    dataset.name: dataset for dataset in (TURBULENT_RADIATIVE_LAYER_2D,)
}


def get_dataset(name: str) -> WellDataset:
    """Look up a registry row, or say what is on offer."""
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY)) or "(none)"
        raise NotImplementedError(
            f"no Well dataset row for {name!r}; this example knows: {known}. "
            "Add one to REGISTRY in examples/well/datasets.py -- see the module "
            "docstring for what a row has to get right."
        ) from None


# -- mapped arrays ----------------------------------------------------------
#
# A window larger than memory has to be paged in by the operating system rather
# than read into the process, and that needs a file whose bytes *are* the array.
# HDF5 is generally not one, being chunked and often compressed, so the file
# that gets fetched is not the file that gets mapped. ``.npy`` is: the header is
# self-describing, the array is contiguous after it, numpy maps it with one
# call, and it is a single file, so it can be landed with one rename.


@contextmanager
def writing(dest: Path, shape: Sequence[int], dtype: Any) -> Iterator[np.memmap]:
    """Open a new mapped array for writing and land it atomically.

    The array is created at full size up front and filled through the returned
    view, so converting a file larger than memory never holds it. The write
    goes to a ``.part`` beside the destination and is renamed on success, so a
    file that exists is a file that finished.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(f"{dest.name}.part.{os.getpid()}")
    filled = False
    try:
        out = np.lib.format.open_memmap(
            part, mode="w+", dtype=np.dtype(dtype), shape=tuple(shape)
        )
        try:
            yield out
            out.flush()
            filled = True
        finally:
            # Drop the mapping before touching the file underneath it.
            del out
    finally:
        if filled:
            os.replace(part, dest)
        else:
            part.unlink(missing_ok=True)


def read(path: Path) -> np.memmap:
    """Map an array read-only. Pages fault in as they are touched."""
    mapped = np.load(path, mmap_mode="r")
    if not isinstance(mapped, np.memmap):  # pragma: no cover - defensive
        raise TypeError(f"{path} did not map to an array")
    return mapped


def convert_to_array(spec: WellDataset, source: Path, dest: Path) -> None:
    """Rewrite one regime's HDF5 as a mappable array.

    Shape ``[trajectory, time, channel, *spatial]``, float32, with the channel
    order of ``spec.fields``. Written a trajectory at a time, so a file larger
    than memory converts without being held.
    """
    with h5py.File(source, "r") as handle:
        group, field, _ = spec.fields[0]
        first = handle[f"{group}/{field}"]
        n_traj, n_time = first.shape[0], first.shape[1]
        spatial = tuple(first.shape[2 : 2 + spec.n_spatial_dims])
        if spatial != spec.spatial_resolution:
            raise ValueError(
                f"{source.name}: grid is {spatial}, but the {spec.name} row says "
                f"{spec.spatial_resolution}"
            )

        shape = (n_traj, n_time, spec.n_channels, *spatial)
        with writing(dest, shape, "float32") as out:
            for trajectory in range(n_traj):
                for channel, (group, field, component) in enumerate(spec.fields):
                    block = handle[f"{group}/{field}"][trajectory]
                    if component is not None:
                        block = block[..., component]
                    out[trajectory, :, channel] = block

    dest.with_name(dest.name + RECIPE_SUFFIX).write_text(CONVERSION_RECIPE)


# -- getting a regime onto disk ---------------------------------------------


def fetch(url: str, dest: Path) -> None:
    """Download one file, landing it atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(f"{dest.name}.part.{os.getpid()}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with part.open("wb") as handle:
                while chunk := response.read(1 << 22):
                    handle.write(chunk)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, dest)


class RegimeCache:
    """One Well dataset's regimes on local disk, converted for mapping.

    ``source`` is either ``hf://datasets/<owner>`` -- the published collection,
    downloaded a regime at a time -- or a local directory laid out the way The
    Well lays it out, in which case nothing is downloaded and nothing is
    deleted.

    A regime costs two downloads, train and valid, and is converted once. The
    HDF5 is removed after a successful conversion when it came from the
    network: nothing reads it again, it is half of what a regime costs on disk,
    and it can always be fetched back.
    """

    def __init__(self, spec: WellDataset, source: str, root: Optional[Path] = None):
        self.spec = spec
        self.source = source.rstrip("/")
        self.root = Path(root) if root is not None else CACHE_ROOT / spec.name

    @property
    def remote(self) -> bool:
        return self.source.startswith(("hf://", "http://", "https://"))

    def _url(self, subpath: str) -> str:
        """``hf://datasets/<owner>`` + ``data/train/x.hdf5`` -> a download URL."""
        if self.source.startswith("hf://"):
            owner = self.source[len("hf://") :].removeprefix("datasets/")
            return f"{HF_HOST}/datasets/{owner}/{self.spec.name}/resolve/main/{subpath}"
        return f"{self.source}/{self.spec.name}/{subpath}"

    def _local(self, subpath: str) -> Path:
        return Path(self.source) / self.spec.name / subpath

    def regimes(self) -> List[str]:
        """Regime names in physical order, without downloading anything."""
        if self.source.startswith("hf://"):
            owner = self.source[len("hf://") :].removeprefix("datasets/")
            url = (
                f"{HF_HOST}/api/datasets/{owner}/{self.spec.name}/tree/main/data/train"
            )
            with urllib.request.urlopen(url, timeout=60) as response:
                entries = [str(item["path"]) for item in json.load(response)]
        else:
            entries = [str(p) for p in self._local("data/train").glob("*.hdf5")]

        names = sorted(
            (Path(entry).stem for entry in entries if entry.endswith(".hdf5")),
            key=self.spec.order,
        )
        if not names:
            where = self.source if self.remote else self._local("data/train")
            raise FileNotFoundError(f"no {self.spec.name} HDF5 files under {where}")
        return names

    def _source_file(self, subpath: str) -> Path:
        """A path to ``subpath``, downloading it first if it is remote."""
        if not self.remote:
            local = self._local(subpath)
            if not local.exists():
                raise FileNotFoundError(local)
            return local
        cached = self.root / "download" / subpath
        if not cached.exists():
            fetch(self._url(subpath), cached)
        return cached

    def array(self, split: str, regime: str) -> Path:
        """The mapped array for one regime, built if it is not there yet."""
        dest = self.root / split / f"{regime}.npy"
        recipe = dest.with_name(dest.name + RECIPE_SUFFIX)
        if dest.exists() and recipe.exists():
            if recipe.read_text().strip() == CONVERSION_RECIPE:
                return dest

        hdf5 = self._source_file(f"data/{split}/{regime}.hdf5")
        convert_to_array(self.spec, hdf5, dest)
        if self.remote:
            # Re-fetchable, and nothing reads it again.
            hdf5.unlink(missing_ok=True)
        return dest

    def statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Dataset-wide per-channel mean and standard deviation.

        From The Well's own ``stats.yaml``, and fixed across regimes on
        purpose: normalising per regime would divide out the very shift the run
        is watching for.
        """
        stats = yaml.safe_load(self._source_file("stats.yaml").read_text())
        mean: List[float] = []
        std: List[float] = []
        for _, field, component in self.spec.fields:
            for key, target in (("mean", mean), ("std", std)):
                value = stats[key][field]
                target.append(float(value if component is None else value[component]))
        return np.asarray(mean, dtype="float32"), np.asarray(std, dtype="float32")
