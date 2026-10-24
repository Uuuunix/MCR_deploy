"""Workspace catalog models."""

from __future__ import annotations

from datetime import datetime

from mcr.memory.common import DomainModel, NonEmptyStr


class WorkspaceRecord(DomainModel):
    id: NonEmptyStr
    sqlite_path: NonEmptyStr
    qdrant_path: NonEmptyStr
    collection_name: NonEmptyStr = "mcr_memories"
    created_at: datetime

