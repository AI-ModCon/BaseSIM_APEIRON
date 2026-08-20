"""Checkpoint storage with metric-aware retention and promotion.

Each saved checkpoint gets a JSON sidecar recording the metrics that were in
hand when it was written (post-CL current/historical accuracy, FWT, BWT). Once
checkpoints carry their metrics, "which ones do I keep" and "which one do I
serve" become rules evaluated over those sidecars rather than blind
newest-N-by-mtime:

* **Retention** decides which checkpoint files survive on disk.
* **Promotion** decides which surviving checkpoint the ``deployed`` pointer
  names -- the one a serving system should load. These are separate concerns:
  you usually retain generously but deploy the single most robust model.

Rule spec grammar
-----------------
A retention spec is one or more clauses joined by ``+``; the keep set is their
union. A clause is:

* ``fifo`` / ``latest`` -- the newest ``N`` by event (``N`` = ``max_ckpts`` when
  this is the only clause, else ``1``; override as ``latest:3``).
* ``max:<metric>`` / ``best:<metric>`` -- the top ``N`` by ``<metric>``
  descending (``best:2`` for two; default 1).
* ``min:<metric>`` -- the top ``N`` by ``<metric>`` ascending (for
  lower-is-better metrics such as loss or MAE).

Examples::

    "fifo"                                  # newest max_ckpts (default)
    "latest:1+max:test_hist_acc+max:test_curr_acc"   # last + best-hist + best-cur

A promotion spec (``deploy_rule``) is a single clause naming one winner, e.g.
``max:test_hist_acc`` or ``latest``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch


@dataclass(frozen=True)
class CheckpointRecord:
    """Metadata sidecar for one saved checkpoint."""

    event: int
    file: str
    metrics: dict[str, float] = field(default_factory=dict)
    window_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "file": self.file,
            "metrics": self.metrics,
            "window_id": self.window_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointRecord":
        return cls(
            event=int(d["event"]),
            file=d["file"],
            metrics={k: float(v) for k, v in d.get("metrics", {}).items()},
            window_id=d.get("window_id"),
        )


# ---------------------------------------------------------------------------
# Rule parsing
# ---------------------------------------------------------------------------


def _clause_selector(
    clause: str, max_ckpts: int, solo: bool
) -> Callable[[list[CheckpointRecord]], list[int]]:
    """Compile one clause into ``records -> ordered list of event ids to keep``."""
    clause = clause.strip()
    if not clause:
        raise ValueError("empty retention clause")

    name, _, arg = clause.partition(":")
    name = name.lower()

    if name in ("fifo", "latest"):
        n = int(arg) if arg else (max_ckpts if solo else 1)

        def sel_latest(recs: list[CheckpointRecord]) -> list[int]:
            ordered = sorted(recs, key=lambda r: r.event, reverse=True)
            return [r.event for r in ordered[: max(0, n)]]

        return sel_latest

    if name in ("max", "best", "min"):
        if not arg:
            raise ValueError(f"clause {clause!r} needs a metric, e.g. {name}:accuracy")
        metric = arg
        descending = name in ("max", "best")
        # A bare "max:acc" keeps the single best; "max:acc:2" keeps two.
        metric, _, count = metric.partition(":")
        n = int(count) if count else 1

        def sel_metric(recs: list[CheckpointRecord]) -> list[int]:
            have = [r for r in recs if metric in r.metrics]
            ordered = sorted(have, key=lambda r: r.metrics[metric], reverse=descending)
            return [r.event for r in ordered[: max(0, n)]]

        return sel_metric

    raise ValueError(f"unknown retention clause: {clause!r}")


def select_keep(records: list[CheckpointRecord], spec: str, max_ckpts: int) -> set[int]:
    """Event ids to keep under ``spec`` (union of clauses)."""
    clauses = [c for c in spec.split("+") if c.strip()]
    if not clauses:
        clauses = ["fifo"]
    solo = len(clauses) == 1
    keep: set[int] = set()
    for clause in clauses:
        keep.update(_clause_selector(clause, max_ckpts, solo)(records))
    return keep


def select_one(records: list[CheckpointRecord], spec: str) -> Optional[int]:
    """The single winning event id under a promotion ``spec`` (or None)."""
    if not spec.strip() or not records:
        return None
    picks = _clause_selector(spec.split("+")[0], max_ckpts=1, solo=False)(records)
    return picks[0] if picks else None


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class CheckpointStore:
    """Directory of model checkpoints plus their metric sidecars.

    Files:

    * ``drift_adaptation_<event>.pt``   -- ``state_dict``
    * ``drift_adaptation_<event>.json`` -- :class:`CheckpointRecord`
    * ``latest``   -- text pointer to the most recently written ``.pt``
    * ``deployed`` -- text pointer to the promoted ``.pt`` (if ``deploy_rule``)
    """

    def __init__(
        self,
        directory: str | Path,
        max_ckpts: int,
        retention: str = "fifo",
        deploy_rule: str = "",
    ) -> None:
        self.dir = Path(directory)
        self.max_ckpts = max_ckpts
        self.retention = retention or "fifo"
        self.deploy_rule = deploy_rule or ""

    # -- introspection -----------------------------------------------------

    def records(self) -> list[CheckpointRecord]:
        """All checkpoint records with a surviving ``.pt``, sorted by event."""
        out: list[CheckpointRecord] = []
        if not self.dir.exists():
            return out
        for sidecar in self.dir.glob("drift_adaptation_*.json"):
            try:
                rec = CheckpointRecord.from_dict(json.loads(sidecar.read_text()))
            except (json.JSONDecodeError, KeyError):
                continue
            if (self.dir / rec.file).exists():
                out.append(rec)
        out.sort(key=lambda r: r.event)
        return out

    def latest(self) -> Optional[str]:
        p = self.dir / "latest"
        return p.read_text().strip() if p.exists() else None

    def deployed(self) -> Optional[str]:
        p = self.dir / "deployed"
        return p.read_text().strip() if p.exists() else None

    # -- write -------------------------------------------------------------

    def save(
        self,
        state_dict: dict[str, Any],
        event: int,
        metrics: Optional[dict[str, float]] = None,
        window_id: Optional[str] = None,
    ) -> str:
        """Persist a checkpoint, then apply retention and promotion.

        Returns the path to the checkpoint written for ``event``.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        fname = f"drift_adaptation_{event}.pt"
        ckpt_path = self.dir / fname
        torch.save(state_dict, ckpt_path)

        record = CheckpointRecord(
            event=event,
            file=fname,
            metrics={k: float(v) for k, v in (metrics or {}).items()},
            window_id=window_id,
        )
        (self.dir / f"drift_adaptation_{event}.json").write_text(
            json.dumps(record.to_dict(), indent=2)
        )
        (self.dir / "latest").write_text(fname)

        self._apply_retention()
        self._apply_promotion()
        return str(ckpt_path)

    # -- policy ------------------------------------------------------------

    def _apply_retention(self) -> None:
        recs = self.records()
        if self.max_ckpts <= 0:
            return
        keep = select_keep(recs, self.retention, self.max_ckpts)
        for rec in recs:
            if rec.event not in keep:
                self._remove(rec)

    def _apply_promotion(self) -> None:
        if not self.deploy_rule:
            return
        winner = select_one(self.records(), self.deploy_rule)
        if winner is not None:
            (self.dir / "deployed").write_text(f"drift_adaptation_{winner}.pt")

    def _remove(self, rec: CheckpointRecord) -> None:
        (self.dir / rec.file).unlink(missing_ok=True)
        (self.dir / f"drift_adaptation_{rec.event}.json").unlink(missing_ok=True)
