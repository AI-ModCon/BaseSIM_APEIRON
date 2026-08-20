"""A small queryable index over committed window manifests.

The catalog answers "which windows do I need?" without globbing the filesystem
and re-reading every manifest: time-range queries for building a historic replay
load, "which windows did a detector fire on", full-retrain enumeration, etc. It
is a cache -- the manifests on disk are the source of truth -- so it can be
dropped and rebuilt at any time with :meth:`WindowStore.rebuild_catalog`.

Backed by stdlib ``sqlite3`` (no extra dependency). Single-writer: in a
multi-node run only rank 0 commits windows and writes the catalog; other ranks
read the manifests directly through the store.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

if TYPE_CHECKING:
    from apeiron.data.window_store import WindowManifest


_SCHEMA = """
CREATE TABLE IF NOT EXISTS windows (
    window_id  TEXT PRIMARY KEY,
    seq        INTEGER NOT NULL,
    n_samples  INTEGER NOT NULL,
    t_start    TEXT,
    t_end      TEXT,
    detected   INTEGER NOT NULL DEFAULT 0,
    x_dtype    TEXT,
    y_dtype    TEXT
);
CREATE INDEX IF NOT EXISTS idx_windows_seq ON windows(seq);
CREATE INDEX IF NOT EXISTS idx_windows_tstart ON windows(t_start);
CREATE INDEX IF NOT EXISTS idx_windows_detected ON windows(detected);
"""


class WindowCatalog:
    """SQLite index over window manifests, keyed by ``window_id``."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        # check_same_thread=False so the catalog can be read from a background
        # analysis thread; all writes still funnel through one process.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- writes ------------------------------------------------------------

    def upsert(self, manifest: "WindowManifest") -> None:
        """Insert or replace one window's row."""
        self._conn.execute(
            """
            INSERT INTO windows
                (window_id, seq, n_samples, t_start, t_end, detected, x_dtype, y_dtype)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(window_id) DO UPDATE SET
                seq=excluded.seq,
                n_samples=excluded.n_samples,
                t_start=excluded.t_start,
                t_end=excluded.t_end,
                detected=excluded.detected,
                x_dtype=excluded.x_dtype,
                y_dtype=excluded.y_dtype
            """,
            (
                manifest.window_id,
                manifest.seq,
                manifest.n_samples,
                manifest.t_start,
                manifest.t_end,
                int(manifest.detected),
                manifest.x_dtype,
                manifest.y_dtype,
            ),
        )
        self._conn.commit()

    def remove(self, window_id: str) -> None:
        self._conn.execute("DELETE FROM windows WHERE window_id = ?", (window_id,))
        self._conn.commit()

    def rebuild(self, manifests: Iterable["WindowManifest"]) -> None:
        """Drop all rows and re-index from the given manifests."""
        self._conn.execute("DELETE FROM windows")
        for m in manifests:
            self.upsert(m)
        self._conn.commit()

    # -- reads -------------------------------------------------------------

    def get(self, window_id: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM windows WHERE window_id = ?", (window_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def all(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM windows ORDER BY seq").fetchall()
        return [dict(r) for r in rows]

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM windows").fetchone()
        return int(row["n"])

    def query(
        self,
        *,
        t_start_gte: Optional[str] = None,
        t_end_lte: Optional[str] = None,
        detected: Optional[bool] = None,
        seq_gte: Optional[int] = None,
        seq_lt: Optional[int] = None,
        order: str = "seq",
        limit: Optional[int] = None,
    ) -> list[str]:
        """Return window ids matching the filters, ordered (default by ``seq``).

        Timestamp comparisons are lexicographic on the opaque ``t_start``/
        ``t_end`` strings, so ISO-8601 timestamps sort chronologically. This is
        the query behind historic-load creation: "give me every window whose
        ``delta_t`` falls in ``[t0, t1)``".
        """
        clauses: list[str] = []
        params: list[Any] = []
        if t_start_gte is not None:
            clauses.append("t_start >= ?")
            params.append(t_start_gte)
        if t_end_lte is not None:
            clauses.append("t_end <= ?")
            params.append(t_end_lte)
        if detected is not None:
            clauses.append("detected = ?")
            params.append(int(detected))
        if seq_gte is not None:
            clauses.append("seq >= ?")
            params.append(seq_gte)
        if seq_lt is not None:
            clauses.append("seq < ?")
            params.append(seq_lt)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        if order not in ("seq", "t_start", "t_end"):
            raise ValueError(f"unsupported order column: {order}")
        sql = f"SELECT window_id FROM windows {where} ORDER BY {order}"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [r["window_id"] for r in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "WindowCatalog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
