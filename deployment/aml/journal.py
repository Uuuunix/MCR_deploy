"""Chunk-level idempotency across partial Add retries."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path


class RequestConflict(ValueError):
    pass


@dataclass(frozen=True)
class JournalEntry:
    completed_units: int
    done: bool


class AddJournal:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS add_requests (
                request_id TEXT PRIMARY KEY,
                payload_hash TEXT NOT NULL,
                completed_units INTEGER NOT NULL DEFAULT 0,
                done INTEGER NOT NULL DEFAULT 0
            )"""
        )
        self._db.commit()

    def begin(self, request_id: str, payload_hash: str) -> JournalEntry:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO add_requests(request_id,payload_hash) VALUES (?,?)",
                (request_id, payload_hash),
            )
            row = self._db.execute(
                "SELECT payload_hash,completed_units,done FROM add_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if row[0] != payload_hash:
                raise RequestConflict("request_id was already used with different content")
            return JournalEntry(completed_units=int(row[1]), done=bool(row[2]))

    def advance(self, request_id: str, completed_units: int, *, done: bool = False) -> None:
        with self._lock, self._db:
            self._db.execute(
                """UPDATE add_requests SET completed_units=MAX(completed_units,?),
                   done=MAX(done,?) WHERE request_id=?""",
                (completed_units, int(done), request_id),
            )

    def close(self) -> None:
        self._db.close()
