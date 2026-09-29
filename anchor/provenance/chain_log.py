"""Append-only, hash-chained SQLite log for provenance records.

Every record stores the SHA-256 hash of the preceding record's
serialised form, creating a tamper-evident chain.  If any historical
record is modified, ``verify_chain()`` will detect the first broken link.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chain_log (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    prev_hash   TEXT    NOT NULL,
    payload     TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    record_hash TEXT    NOT NULL
);
"""

_GENESIS_PREV_HASH = "0" * 64  # sentinel for the first record


def _hash_record(prev_hash: str, payload: str, timestamp: str) -> str:
    """Compute the SHA-256 digest that commits to (prev_hash ‖ payload ‖ timestamp)."""
    data = json.dumps(
        {"prev_hash": prev_hash, "payload": payload, "timestamp": timestamp},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class ChainLog:
    """Append-only, hash-chained provenance log backed by SQLite.

    Parameters
    ----------
    db_path:
        File path for the SQLite database.  Created if it does not exist.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    # ── public API ───────────────────────────────────────────────────────

    def append(self, payload: dict[str, Any]) -> int:
        """Append a new record and return its sequence number.

        The record's *prev_hash* is derived from the most recent existing
        record, or from the genesis sentinel if the chain is empty.
        """
        payload_str = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        timestamp = datetime.now(timezone.utc).isoformat()

        prev_hash = self._latest_hash()
        record_hash = _hash_record(prev_hash, payload_str, timestamp)

        cur = self._conn.execute(
            "INSERT INTO chain_log (prev_hash, payload, timestamp, record_hash) VALUES (?, ?, ?, ?)",
            (prev_hash, payload_str, timestamp, record_hash),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def verify_chain(self) -> tuple[bool, int | None]:
        """Walk the entire chain and verify every link.

        Returns
        -------
        (True, None)
            if the chain is intact.
        (False, seq)
            where *seq* is the sequence number of the first record whose
            stored hash does not match the recomputed hash.
        """
        rows = self._conn.execute(
            "SELECT seq, prev_hash, payload, timestamp, record_hash FROM chain_log ORDER BY seq"
        ).fetchall()

        expected_prev = _GENESIS_PREV_HASH

        for seq, prev_hash, payload, timestamp, record_hash in rows:
            # Check that prev_hash links to the previous record correctly
            if prev_hash != expected_prev:
                return False, seq

            # Check that the stored hash matches a fresh computation
            recomputed = _hash_record(prev_hash, payload, timestamp)
            if record_hash != recomputed:
                return False, seq

            expected_prev = record_hash

        return True, None

    def get_record(self, seq: int) -> dict[str, Any] | None:
        """Return a single record by sequence number, or *None*."""
        row = self._conn.execute(
            "SELECT seq, prev_hash, payload, timestamp, record_hash FROM chain_log WHERE seq = ?",
            (seq,),
        ).fetchone()
        if row is None:
            return None
        return {
            "seq": row[0],
            "prev_hash": row[1],
            "payload": json.loads(row[2]),
            "timestamp": row[3],
            "record_hash": row[4],
        }

    def length(self) -> int:
        """Return the number of records in the chain."""
        row = self._conn.execute("SELECT COUNT(*) FROM chain_log").fetchone()
        return row[0]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    # ── internals ────────────────────────────────────────────────────────

    def _latest_hash(self) -> str:
        """Return the record_hash of the most recent record, or the genesis sentinel."""
        row = self._conn.execute(
            "SELECT record_hash FROM chain_log ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else _GENESIS_PREV_HASH
