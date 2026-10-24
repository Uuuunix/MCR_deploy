"""Workspace management errors."""


class WorkspaceError(RuntimeError):
    """Base workspace error."""


class WorkspaceConflict(WorkspaceError):
    """Raised when an existing workspace points at different storage."""


class WorkspaceNotFound(WorkspaceError):
    """Raised when a workspace does not exist."""

