"""Small SQLite catalog for isolated workspace storage locations."""

from __future__ import annotations

import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from mcr.workspaces.errors import WorkspaceConflict, WorkspaceNotFound
from mcr.workspaces.models import WorkspaceRecord

_WORKSPACE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class WorkspaceCatalog:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "catalog.sqlite3"
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS workspaces (
                   id TEXT PRIMARY KEY,
                   sqlite_path TEXT NOT NULL,
                   qdrant_path TEXT NOT NULL,
                   collection_name TEXT NOT NULL,
                   created_at TEXT NOT NULL
               )"""
        )
        self._connection.commit()

    @staticmethod
    def validate_id(workspace_id: str) -> str:
        value = workspace_id.strip()
        if not _WORKSPACE_ID.fullmatch(value):
            raise ValueError(
                "workspace_id must be 1-64 characters using letters, numbers, '.', '_' or '-'"
            )
        return value

    def register_default(
        self,
        *,
        sqlite_path: str | Path,
        qdrant_path: str | Path,
        collection_name: str = "mcr_memories",
    ) -> WorkspaceRecord:
        expected = (
            self._canonical(sqlite_path),
            self._canonical(qdrant_path),
            collection_name,
        )
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM workspaces WHERE id='default'"
            ).fetchone()
            if row is not None:
                actual = (
                    self._canonical(row["sqlite_path"]),
                    self._canonical(row["qdrant_path"]),
                    row["collection_name"],
                )
                if actual != expected:
                    raise WorkspaceConflict(
                        "default workspace already points to different SQLite/Qdrant storage"
                    )
                return self._from_row(row)
            now = datetime.now(timezone.utc).isoformat()
            self._connection.execute(
                "INSERT INTO workspaces VALUES ('default', ?, ?, ?, ?)",
                (str(sqlite_path), str(qdrant_path), collection_name, now),
            )
            row = self._connection.execute(
                "SELECT * FROM workspaces WHERE id='default'"
            ).fetchone()
        return self._from_row(row)

    def create(self, workspace_id: str) -> WorkspaceRecord:
        workspace_id = self.validate_id(workspace_id)
        with self._lock:
            existing = self._connection.execute(
                "SELECT * FROM workspaces WHERE id=?", (workspace_id,)
            ).fetchone()
        if existing is not None:
            return self._from_row(existing)
        directory = self.root / "workspaces" / workspace_id
        directory.mkdir(parents=True, exist_ok=True)
        sqlite_path = directory / "sources.sqlite3"
        qdrant_path = directory / "qdrant"
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO workspaces VALUES (?, ?, ?, ?, ?)",
                (
                    workspace_id,
                    str(sqlite_path),
                    str(qdrant_path),
                    "mcr_memories",
                    now,
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM workspaces WHERE id=?", (workspace_id,)
            ).fetchone()
        return self._from_row(row)

    def get(self, workspace_id: str) -> WorkspaceRecord:
        workspace_id = self.validate_id(workspace_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM workspaces WHERE id=?", (workspace_id,)
            ).fetchone()
        if row is None:
            raise WorkspaceNotFound(f"workspace {workspace_id!r} does not exist")
        return self._from_row(row)

    def list(self) -> list[WorkspaceRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM workspaces ORDER BY CASE id WHEN 'default' THEN 0 ELSE 1 END, created_at, id"
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def exists(self, workspace_id: str) -> bool:
        try:
            self.get(workspace_id)
        except WorkspaceNotFound:
            return False
        return True

    def close(self) -> None:
        self._connection.close()

    @staticmethod
    def _canonical(value: str | Path) -> str:
        text = str(value)
        return text if text == ":memory:" else str(Path(text).expanduser().resolve())

    @staticmethod
    def _from_row(row: sqlite3.Row) -> WorkspaceRecord:
        return WorkspaceRecord(
            id=row["id"],
            sqlite_path=row["sqlite_path"],
            qdrant_path=row["qdrant_path"],
            collection_name=row["collection_name"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

