"""Committed, immutable data windows backed by memory-mapped files.

Design
------
A window lives in one directory under the store root::

    <root>/<window_id>/
        x.npy          # inputs, one row per sample (np.save, C-contiguous)
        y.npy          # targets, one row per sample
        <name>_idx.npy # optional explicit index arrays for named splits
        manifest.json  # shapes, dtypes, splits, timestamps -- written LAST

*Immutability* is the whole point: a committed window never changes, so any
number of processes on any number of nodes can ``np.load(..., mmap_mode="r")``
the same file with no locking, and the OS page cache deduplicates within a node.

*Commit is atomic.* The writer stages everything in ``<root>/.tmp-*`` and then
``os.replace``\\s the directory into place (a single ``rename(2)``). A reader
either sees a fully-formed window or does not see it at all; a crash mid-write
leaves only a ``.tmp-*`` directory, which readers ignore. "Committed" is defined
as "the window directory exists and contains ``manifest.json``".

Reads never load a whole window into memory. :class:`WindowHandle` hands out
:class:`torch.utils.data.Dataset` objects that memory-map ``x.npy``/``y.npy``
and materialize one sample at a time, re-opening the memmap lazily inside each
DataLoader worker (memmaps must not cross a fork/spawn boundary).

Sharding
--------
Because sample order within a window does not matter for either monitoring
inference or data-parallel updates, :meth:`WindowHandle.dataset` accepts a
``shard=(rank, world_size)`` argument that returns a contiguous block of the
split for one rank. Contiguous (not strided) shards keep reads aligned with the
underlying parallel filesystem's stripes. This is the hook multi-node support
(a later phase) builds on; on a single process it is a no-op.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from apeiron.data.catalog import WindowCatalog

SCHEMA_VERSION = 1

# Directory prefix for in-progress (not yet committed) windows.
_TMP_PREFIX = ".tmp-"


def _to_numpy(a: Any) -> np.ndarray:
    """Coerce a torch tensor or array-like to a C-contiguous numpy array."""
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    return np.ascontiguousarray(a)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowSplit:
    """How a named split (e.g. ``"train"``/``"val"``) selects samples.

    Two schemes:

    * ``"range"`` -- a contiguous ``[lo, hi)`` half-open range. The common case:
      "the last 10% of the window is validation" is a range split.
    * ``"index"`` -- an explicit index array stored in ``<name>_idx.npy``. Use
      this only when the split is not contiguous (e.g. a random split).
    """

    scheme: Literal["range", "index"]
    lo: Optional[int] = None
    hi: Optional[int] = None
    idx_file: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "lo": self.lo,
            "hi": self.hi,
            "idx_file": self.idx_file,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WindowSplit":
        return cls(
            scheme=d["scheme"],
            lo=d.get("lo"),
            hi=d.get("hi"),
            idx_file=d.get("idx_file"),
        )

    def resolve(self, n_samples: int, window_dir: Path) -> np.ndarray:
        """Return the 1-D int index array this split selects, in order."""
        if self.scheme == "range":
            lo = 0 if self.lo is None else self.lo
            hi = n_samples if self.hi is None else self.hi
            return np.arange(lo, hi, dtype=np.int64)
        if self.scheme == "index":
            if not self.idx_file:
                raise ValueError("index split missing idx_file")
            return np.load(window_dir / self.idx_file).astype(np.int64, copy=False)
        raise ValueError(f"unknown split scheme: {self.scheme}")


@dataclass(frozen=True)
class WindowManifest:
    """Self-describing metadata for one committed window.

    ``seq`` is a monotonically increasing integer assigned at commit time; it is
    the canonical ordering of windows (window ids are opaque strings and need not
    sort). ``t_start``/``t_end`` are opaque timestamp strings describing the
    ``delta_t`` the window covers -- the store never parses them, but the catalog
    and time-range queries key on them.
    """

    window_id: str
    seq: int
    n_samples: int
    x_shape: tuple[int, ...]
    x_dtype: str
    y_shape: tuple[int, ...]
    y_dtype: str
    splits: dict[str, WindowSplit] = field(default_factory=dict)
    t_start: Optional[str] = None
    t_end: Optional[str] = None
    detected: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "seq": self.seq,
            "n_samples": self.n_samples,
            "x_shape": list(self.x_shape),
            "x_dtype": self.x_dtype,
            "y_shape": list(self.y_shape),
            "y_dtype": self.y_dtype,
            "splits": {k: v.to_dict() for k, v in self.splits.items()},
            "t_start": self.t_start,
            "t_end": self.t_end,
            "detected": self.detected,
            "extra": self.extra,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WindowManifest":
        return cls(
            window_id=d["window_id"],
            seq=int(d["seq"]),
            n_samples=int(d["n_samples"]),
            x_shape=tuple(d["x_shape"]),
            x_dtype=d["x_dtype"],
            y_shape=tuple(d["y_shape"]),
            y_dtype=d["y_dtype"],
            splits={
                k: WindowSplit.from_dict(v) for k, v in d.get("splits", {}).items()
            },
            t_start=d.get("t_start"),
            t_end=d.get("t_end"),
            detected=bool(d.get("detected", False)),
            extra=d.get("extra", {}),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
        )


# ---------------------------------------------------------------------------
# Memmap-backed dataset
# ---------------------------------------------------------------------------


class MemmapWindowDataset(Dataset):
    """A ``Dataset`` over a subset of one window's samples, via memmap.

    Holds only file paths and an index array; the memmaps are opened lazily on
    first access so the object is safe to pickle into DataLoader workers (each
    worker opens its own memmap). Each ``__getitem__`` copies a single sample out
    of the memmap into an owned tensor -- the copy is one sample wide, and it is
    what keeps the rest of the window on disk instead of in RAM.
    """

    def __init__(self, x_path: Path, y_path: Path, indices: np.ndarray) -> None:
        self._x_path = str(x_path)
        self._y_path = str(y_path)
        self._indices = np.asarray(indices, dtype=np.int64)
        self._x: Optional[np.memmap] = None
        self._y: Optional[np.memmap] = None

    def _ensure_open(self) -> None:
        if self._x is None:
            self._x = np.load(self._x_path, mmap_mode="r")
            self._y = np.load(self._y_path, mmap_mode="r")

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        assert self._x is not None and self._y is not None
        idx = int(self._indices[i])
        # np.array(...) forces a writable, contiguous copy so torch.from_numpy
        # neither warns about a read-only buffer nor aliases the memmap.
        x = torch.from_numpy(np.array(self._x[idx]))
        y = torch.from_numpy(np.array(self._y[idx]))
        return x, y


def _shard_indices(indices: np.ndarray, shard: Optional[tuple[int, int]]) -> np.ndarray:
    """Return the contiguous block of ``indices`` owned by ``rank`` of ``world``."""
    if shard is None:
        return indices
    rank, world = shard
    if world <= 0:
        raise ValueError(f"world_size must be positive, got {world}")
    if not (0 <= rank < world):
        raise ValueError(f"rank {rank} out of range for world_size {world}")
    n = indices.shape[0]
    # Contiguous, balanced blocks: the first (n % world) shards get one extra.
    base, rem = divmod(n, world)
    lo = rank * base + min(rank, rem)
    hi = lo + base + (1 if rank < rem else 0)
    return indices[lo:hi]


# ---------------------------------------------------------------------------
# Read handle
# ---------------------------------------------------------------------------


class WindowHandle:
    """Read-only view of one committed window.

    Cheap to construct (reads only the manifest); the sample data stays on disk
    until a returned dataset/loader is actually iterated.
    """

    def __init__(self, window_dir: Path, manifest: WindowManifest) -> None:
        self.dir = window_dir
        self.manifest = manifest

    @property
    def window_id(self) -> str:
        return self.manifest.window_id

    @property
    def x_path(self) -> Path:
        return self.dir / "x.npy"

    @property
    def y_path(self) -> Path:
        return self.dir / "y.npy"

    def split_names(self) -> list[str]:
        return list(self.manifest.splits.keys())

    def indices(
        self,
        split: str = "all",
        shard: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Resolve a split name to its ordered index array (optionally sharded).

        ``"all"`` selects every sample in order and is always available even if
        no splits were declared at commit time.
        """
        if split == "all":
            idx = np.arange(self.manifest.n_samples, dtype=np.int64)
        else:
            spec = self.manifest.splits.get(split)
            if spec is None:
                raise KeyError(
                    f"window {self.window_id!r} has no split {split!r}; "
                    f"available: {self.split_names() or ['all']}"
                )
            idx = spec.resolve(self.manifest.n_samples, self.dir)
        return _shard_indices(idx, shard)

    def dataset(
        self,
        split: str = "all",
        shard: Optional[tuple[int, int]] = None,
    ) -> MemmapWindowDataset:
        """A memmap-backed ``Dataset`` over ``split`` (optionally this rank's shard)."""
        return MemmapWindowDataset(self.x_path, self.y_path, self.indices(split, shard))

    def loader(
        self,
        split: str = "all",
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        shard: Optional[tuple[int, int]] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        """A ``DataLoader`` over ``split`` of this window."""
        return DataLoader(
            self.dataset(split, shard),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def load_full(self, split: str = "all") -> tuple[torch.Tensor, torch.Tensor]:
        """Read an entire split into memory as ``(x, y)`` tensors.

        Convenience for callers that genuinely want the whole split resident
        (e.g. building a small in-memory probe set); ordinary iteration should go
        through :meth:`loader` and stay memory-mapped.
        """
        idx = self.indices(split)
        x = np.load(self.x_path, mmap_mode="r")
        y = np.load(self.y_path, mmap_mode="r")
        return (
            torch.from_numpy(np.array(x[idx])),
            torch.from_numpy(np.array(y[idx])),
        )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class WindowStore:
    """A directory of committed, immutable data windows.

    Parameters
    ----------
    root:
        Directory holding the windows (created if missing).
    catalog:
        If ``True`` (default) maintain a sqlite :class:`WindowCatalog` at
        ``<root>/catalog.db`` and upsert each committed window into it. Set
        ``False`` for a pure filesystem store with no index.
    """

    def __init__(self, root: str | os.PathLike[str], catalog: bool = True) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        self._catalog: Optional["WindowCatalog"] = None
        if catalog:
            from apeiron.data.catalog import WindowCatalog

            self._catalog = WindowCatalog(self.root / "catalog.db")

    # -- introspection -----------------------------------------------------

    @property
    def catalog(self) -> Optional["WindowCatalog"]:
        return self._catalog

    def _committed_dirs(self) -> list[Path]:
        """Directories that hold a committed window (manifest present)."""
        out = []
        for p in self.root.iterdir():
            if not p.is_dir() or p.name.startswith(_TMP_PREFIX):
                continue
            if (p / "manifest.json").exists():
                out.append(p)
        return out

    def _read_manifest(self, window_dir: Path) -> WindowManifest:
        data = json.loads((window_dir / "manifest.json").read_text())
        return WindowManifest.from_dict(data)

    def list_manifests(self) -> list[WindowManifest]:
        """All committed manifests, ordered by ``seq`` (commit order)."""
        mans = [self._read_manifest(p) for p in self._committed_dirs()]
        mans.sort(key=lambda m: m.seq)
        return mans

    def window_ids(self) -> list[str]:
        return [m.window_id for m in self.list_manifests()]

    def __len__(self) -> int:
        return len(self._committed_dirs())

    def __contains__(self, window_id: str) -> bool:
        return (self.root / window_id / "manifest.json").exists()

    def window(self, window_id: str) -> WindowHandle:
        """Read handle for a committed window (raises ``KeyError`` if absent)."""
        d = self.root / window_id
        if not (d / "manifest.json").exists():
            raise KeyError(f"no committed window {window_id!r} in {self.root}")
        return WindowHandle(d, self._read_manifest(d))

    def windows(self) -> list[WindowHandle]:
        """Read handles for every committed window, in commit order."""
        return [WindowHandle(self.root / m.window_id, m) for m in self.list_manifests()]

    def __iter__(self) -> Iterator[WindowHandle]:
        return iter(self.windows())

    def _next_seq(self) -> int:
        existing = self.list_manifests()
        return (max((m.seq for m in existing), default=-1)) + 1

    # -- commit ------------------------------------------------------------

    def commit(
        self,
        x: Any,
        y: Any,
        *,
        window_id: Optional[str] = None,
        splits: Optional[dict[str, WindowSplit]] = None,
        val_fraction: Optional[float] = None,
        t_start: Optional[str] = None,
        t_end: Optional[str] = None,
        detected: bool = False,
        extra: Optional[dict[str, Any]] = None,
        seq: Optional[int] = None,
    ) -> WindowManifest:
        """Atomically write a new window and return its manifest.

        Parameters
        ----------
        x, y:
            Sample inputs and targets, first axis indexed by sample. Torch
            tensors or numpy arrays; stored in their given dtype (keep raw
            precision -- casting/normalizing belongs in the read path).
        window_id:
            Directory name; defaults to the zero-padded ``seq``. Must be unique.
        splits:
            Named splits (see :class:`WindowSplit`). Mutually exclusive with
            ``val_fraction``.
        val_fraction:
            Convenience for the common contiguous split: the trailing
            ``val_fraction`` of samples become ``"val"``, the rest ``"train"``.
        t_start, t_end:
            Opaque timestamps describing the window's ``delta_t``.
        detected:
            Whether a detector fired on this window (recorded for later queries).
        seq:
            Override the auto-assigned sequence number (advanced/testing).
        """
        x = _to_numpy(x)
        y = _to_numpy(y)
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x and y disagree on sample count: {x.shape[0]} vs {y.shape[0]}"
            )
        n = int(x.shape[0])

        if splits is not None and val_fraction is not None:
            raise ValueError("pass either splits or val_fraction, not both")
        if val_fraction is not None:
            if not (0.0 <= val_fraction < 1.0):
                raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
            n_val = int(round(n * val_fraction))
            n_train = n - n_val
            splits = {
                "train": WindowSplit("range", lo=0, hi=n_train),
                "val": WindowSplit("range", lo=n_train, hi=n),
            }
        splits = splits or {}
        for name, spec in splits.items():
            if spec.scheme == "index":
                raise ValueError(
                    f"split {name!r} uses the 'index' scheme; use "
                    "commit_indexed() so the index arrays are written to disk"
                )

        seq = self._next_seq() if seq is None else seq
        window_id = window_id or f"{seq:06d}"
        final = self.root / window_id
        if final.exists():
            raise FileExistsError(f"window {window_id!r} already exists at {final}")

        self._counter += 1
        tmp = self.root / f"{_TMP_PREFIX}{window_id}-{os.getpid()}-{self._counter}"
        if tmp.exists():
            _rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            np.save(tmp / "x.npy", x)
            np.save(tmp / "y.npy", y)
            manifest = WindowManifest(
                window_id=window_id,
                seq=seq,
                n_samples=n,
                x_shape=tuple(x.shape[1:]),
                x_dtype=str(x.dtype),
                y_shape=tuple(y.shape[1:]),
                y_dtype=str(y.dtype),
                splits=splits,
                t_start=t_start,
                t_end=t_end,
                detected=detected,
                extra=extra or {},
            )
            # manifest.json is written last inside the staging dir; the whole
            # directory then becomes visible in a single atomic rename.
            (tmp / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2))
            os.replace(tmp, final)
        except BaseException:
            _rmtree(tmp)
            raise

        if self._catalog is not None:
            self._catalog.upsert(manifest)
        return manifest

    def commit_indexed(
        self,
        x: Any,
        y: Any,
        *,
        train_idx: Sequence[int],
        val_idx: Sequence[int],
        window_id: Optional[str] = None,
        **kwargs: Any,
    ) -> WindowManifest:
        """Commit a window with explicit (possibly non-contiguous) train/val indices.

        The index arrays are stored in ``train_idx.npy``/``val_idx.npy`` and
        referenced by the manifest, so a random split survives on disk and BWT
        re-evaluation reproduces exactly the samples that were validated on.
        """
        # Stage the arrays through commit(): we resolve them to files by writing
        # them first into the staging dir, so pre-materialize via extra handling.
        x = _to_numpy(x)
        y = _to_numpy(y)
        n = int(x.shape[0])
        seq = kwargs.pop("seq", None)
        seq = self._next_seq() if seq is None else seq
        window_id = window_id or f"{seq:06d}"
        final = self.root / window_id
        if final.exists():
            raise FileExistsError(f"window {window_id!r} already exists at {final}")

        self._counter += 1
        tmp = self.root / f"{_TMP_PREFIX}{window_id}-{os.getpid()}-{self._counter}"
        if tmp.exists():
            _rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            np.save(tmp / "x.npy", x)
            np.save(tmp / "y.npy", y)
            np.save(tmp / "train_idx.npy", np.asarray(train_idx, dtype=np.int64))
            np.save(tmp / "val_idx.npy", np.asarray(val_idx, dtype=np.int64))
            splits = {
                "train": WindowSplit("index", idx_file="train_idx.npy"),
                "val": WindowSplit("index", idx_file="val_idx.npy"),
            }
            manifest = WindowManifest(
                window_id=window_id,
                seq=seq,
                n_samples=n,
                x_shape=tuple(x.shape[1:]),
                x_dtype=str(x.dtype),
                y_shape=tuple(y.shape[1:]),
                y_dtype=str(y.dtype),
                splits=splits,
                t_start=kwargs.get("t_start"),
                t_end=kwargs.get("t_end"),
                detected=bool(kwargs.get("detected", False)),
                extra=kwargs.get("extra", {}) or {},
            )
            (tmp / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2))
            os.replace(tmp, final)
        except BaseException:
            _rmtree(tmp)
            raise

        if self._catalog is not None:
            self._catalog.upsert(manifest)
        return manifest

    # -- maintenance -------------------------------------------------------

    def delete(self, window_id: str) -> None:
        """Remove a committed window from disk and the catalog.

        The caller is responsible for the retention policy -- in particular, a
        window still referenced by a task record must be copied out first (see
        the task-record store). This just does the deletion.
        """
        d = self.root / window_id
        if d.exists():
            _rmtree(d)
        if self._catalog is not None:
            self._catalog.remove(window_id)

    def rebuild_catalog(self) -> None:
        """Rebuild the catalog from the manifests on disk (recovery/repair)."""
        if self._catalog is None:
            from apeiron.data.catalog import WindowCatalog

            self._catalog = WindowCatalog(self.root / "catalog.db")
        self._catalog.rebuild(self.list_manifests())

    def close(self) -> None:
        if self._catalog is not None:
            self._catalog.close()


def _rmtree(path: Path) -> None:
    """Recursively remove a directory tree (best-effort, no shutil import cost)."""
    import shutil

    shutil.rmtree(path, ignore_errors=True)
