"""Explicit per-user deletion for AML retention obligations; run with server stopped."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path

from mcr.workspaces import WorkspaceCatalog

from deployment.aml.service import workspace_id
from deployment.aml.settings import Settings


def purge_user(settings: Settings, user_id: str) -> bool:
    if not user_id.strip():
        raise ValueError("user_id is required")
    identity = workspace_id(user_id)
    root = (settings.data_root / "users").resolve()
    catalog = WorkspaceCatalog(root)
    try:
        if not catalog.exists(identity):
            return False
        record = catalog.get(identity)
        directory = (root / "workspaces" / identity).resolve()
        if directory.parent != (root / "workspaces").resolve():
            raise ValueError("unsafe workspace path")
        if (directory / "sources.sqlite3").resolve() != Path(record.sqlite_path).resolve():
            raise ValueError("workspace source path does not match catalog")
        if (directory / "qdrant").resolve() != Path(record.qdrant_path).resolve():
            raise ValueError("workspace Qdrant path does not match catalog")
        if directory.is_dir():
            shutil.rmtree(directory)
        # The catalog has no public delete operation; delete only this exact ID.
        with sqlite3.connect(catalog.path) as connection:
            connection.execute("DELETE FROM workspaces WHERE id=?", (identity,))
        return True
    finally:
        catalog.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete exactly one AML user's data")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--yes", action="store_true", help="Confirm irreversible deletion")
    args = parser.parse_args()
    if not args.yes:
        parser.error("pass --yes after stopping the AML server")
    print("deleted" if purge_user(Settings.from_environment(), args.user_id) else "not found")


if __name__ == "__main__":
    main()
