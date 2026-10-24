"""Registration helper for the pre-workspace MCR stores."""

from __future__ import annotations

from pathlib import Path

from mcr.workspaces.catalog import WorkspaceCatalog
from mcr.workspaces.models import WorkspaceRecord


def register_legacy_default(
    catalog: WorkspaceCatalog,
    *,
    sqlite_path: str | Path,
    qdrant_path: str | Path,
    collection_name: str = "mcr_memories",
) -> WorkspaceRecord:
    return catalog.register_default(
        sqlite_path=sqlite_path,
        qdrant_path=qdrant_path,
        collection_name=collection_name,
    )

