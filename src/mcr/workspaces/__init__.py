"""Runtime-creatable isolated MCR workspaces."""

from mcr.workspaces.catalog import WorkspaceCatalog
from mcr.workspaces.errors import WorkspaceConflict, WorkspaceError, WorkspaceNotFound
from mcr.workspaces.manager import WorkspaceManager
from mcr.workspaces.migration import register_legacy_default
from mcr.workspaces.models import WorkspaceRecord
from mcr.workspaces.runtime import WorkspaceRuntime

__all__ = [
    "WorkspaceCatalog",
    "WorkspaceConflict",
    "WorkspaceError",
    "WorkspaceManager",
    "WorkspaceNotFound",
    "WorkspaceRecord",
    "WorkspaceRuntime",
    "register_legacy_default",
]
