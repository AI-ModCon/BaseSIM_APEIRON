"""Task records for backward-transfer (forgetting) measurement.

A *task* is one drift event: the window the loop adapted to. To measure
forgetting later, the harness must be able to re-score the current model on that
window's validation split, and remember ``R[i][i]`` -- the score right after
adapting to it.

The costly way (and what the harness did before) is to copy the whole
validation split into RAM per task and retain up to N of them. This module
replaces that with a re-evaluable *reference* so the copy is avoided whenever the
data already lives on disk:

* :class:`WindowEvalSetRef` -- a pointer ``(window_id, split)`` into a committed
  :class:`~apeiron.data.window_store.WindowStore`. No copy: the eval set is
  re-derived from the immutable window on demand. This is the "big win".
* :class:`InMemoryEvalSet` -- the fallback for harnesses that generate data on
  the fly and have no window store. Holds the tensors, and *spills them to a
  ``.pt`` file* when persisted, so even this path is durable (a crashed run's
  forgetting history survives) and does not have to stay resident.

Either way a :class:`TaskRecord` is small metadata plus a reference, so the full
task history can be persisted as JSONL and reloaded on resume.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from apeiron.data.window_store import WindowStore


class EvalSetRef(ABC):
    """A re-evaluable reference to a task's frozen validation set."""

    kind: str

    @abstractmethod
    def loader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        """Return a fresh, deterministic (unshuffled) loader over the eval set."""

    @abstractmethod
    def to_dict(self, records_dir: Optional[Path]) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict.

        ``records_dir`` is where an in-memory ref may spill its tensors; a
        pointer ref ignores it.
        """

    @staticmethod
    def deserialize(
        d: dict[str, Any],
        store: Optional["WindowStore"],
        records_dir: Optional[Path],
    ) -> "EvalSetRef":
        """Reconstruct the concrete ref described by ``d`` (dispatches on kind)."""
        kind = d["kind"]
        if kind == "window":
            return WindowEvalSetRef.from_dict(d, store)
        if kind == "memory":
            return InMemoryEvalSet.from_dict(d, records_dir)
        raise ValueError(f"unknown eval-set ref kind: {kind!r}")


class WindowEvalSetRef(EvalSetRef):
    """Pointer into a committed window -- the no-copy task eval set."""

    kind = "window"

    def __init__(
        self, store: "WindowStore", window_id: str, split: str = "val"
    ) -> None:
        self._store = store
        self.window_id = window_id
        self.split = split

    def loader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        return self._store.window(self.window_id).loader(
            self.split, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    def to_dict(self, records_dir: Optional[Path]) -> dict[str, Any]:
        return {"kind": self.kind, "window_id": self.window_id, "split": self.split}

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], store: Optional["WindowStore"]
    ) -> "WindowEvalSetRef":
        if store is None:
            raise ValueError(
                "cannot restore a window eval-set ref without a WindowStore"
            )
        return cls(store, d["window_id"], d.get("split", "val"))


class InMemoryEvalSet(EvalSetRef):
    """A frozen eval set copied into memory (fallback for storeless harnesses).

    Spills to a ``.pt`` file on persist so the record is durable and the tensors
    need not stay resident once written.
    """

    kind = "memory"

    def __init__(
        self,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        spill_path: Optional[Path] = None,
    ) -> None:
        if x is None and spill_path is None:
            raise ValueError("InMemoryEvalSet needs tensors or a spill_path")
        self._x = x
        self._y = y
        self._spill_path = spill_path

    def _tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._x is None or self._y is None:
            assert self._spill_path is not None
            blob = torch.load(self._spill_path, weights_only=False)
            self._x, self._y = blob["x"], blob["y"]
        return self._x, self._y

    def loader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        x, y = self._tensors()
        return DataLoader(
            TensorDataset(x, y),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def to_dict(self, records_dir: Optional[Path]) -> dict[str, Any]:
        if self._spill_path is None:
            if records_dir is None:
                raise ValueError(
                    "cannot persist an in-memory eval set without a records_dir"
                )
            records_dir.mkdir(parents=True, exist_ok=True)
            # Stable name derived from object id would not survive a reload; the
            # caller (TaskRecord) supplies a deterministic name via event id.
            x, y = self._tensors()
            path = records_dir / f"evalset_{id(self):x}.pt"
            torch.save({"x": x, "y": y}, path)
            self._spill_path = path
        return {"kind": self.kind, "path": str(self._spill_path)}

    def spill(self, path: Path) -> None:
        """Write the tensors to ``path`` and record it as the backing file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        x, y = self._tensors()
        torch.save({"x": x, "y": y}, path)
        self._spill_path = path

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], records_dir: Optional[Path]
    ) -> "InMemoryEvalSet":
        return cls(spill_path=Path(d["path"]))


class TaskRecord:
    """One registered task: its eval-set reference plus its diagonal metrics.

    ``diagonal`` is ``R[i][i]`` -- the metric vector on this task's validation
    split measured right after adapting to it. ``window_id`` is the provenance
    pointer (which committed partition this task's data came from), or ``None``
    for storeless harnesses.
    """

    def __init__(
        self,
        event_id: int,
        diagonal: list[float],
        eval_ref: EvalSetRef,
        window_id: Optional[str] = None,
    ) -> None:
        self.event_id = event_id
        self.diagonal = list(diagonal)
        self.eval_ref = eval_ref
        self.window_id = window_id

    def to_dict(self, records_dir: Optional[Path]) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "diagonal": self.diagonal,
            "window_id": self.window_id,
            "eval_ref": self.eval_ref.to_dict(records_dir),
        }

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        store: Optional["WindowStore"],
        records_dir: Optional[Path],
    ) -> "TaskRecord":
        return cls(
            event_id=int(d["event_id"]),
            diagonal=[float(v) for v in d["diagonal"]],
            eval_ref=EvalSetRef.deserialize(d["eval_ref"], store, records_dir),
            window_id=d.get("window_id"),
        )


class TaskRecordStore:
    """Persist/restore the task-record list as JSONL under ``records_dir``.

    Durability is the point: without it the forgetting history lives only in the
    running process, so a crash on drift event N erases every frozen eval set and
    BWT can never be recomputed. With it, the records are a small file that a
    resumed run reloads.
    """

    def __init__(self, records_dir: Path) -> None:
        self.records_dir = Path(records_dir)
        self.jsonl_path = self.records_dir / "task_records.jsonl"

    def save(self, records: list[TaskRecord]) -> None:
        self.records_dir.mkdir(parents=True, exist_ok=True)
        # For in-memory refs, spill to a deterministic per-event file so a
        # reload finds them (id()-based names would not survive a restart).
        keep_spills: set[str] = set()
        for rec in records:
            if isinstance(rec.eval_ref, InMemoryEvalSet):
                spill = self.records_dir / f"evalset_{rec.event_id}.pt"
                rec.eval_ref.spill(spill)
                keep_spills.add(spill.name)
        # Prune spill files for tasks that have since been evicted, so disk use
        # tracks the live record set rather than the run's whole history.
        for stale in self.records_dir.glob("evalset_*.pt"):
            if stale.name not in keep_spills:
                stale.unlink(missing_ok=True)
        lines = [json.dumps(r.to_dict(self.records_dir)) for r in records]
        tmp = self.jsonl_path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""))
        tmp.replace(self.jsonl_path)

    def load(self, store: Optional["WindowStore"]) -> list[TaskRecord]:
        if not self.jsonl_path.exists():
            return []
        out: list[TaskRecord] = []
        for line in self.jsonl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(TaskRecord.from_dict(json.loads(line), store, self.records_dir))
        return out
