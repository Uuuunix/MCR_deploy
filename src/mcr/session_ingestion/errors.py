"""Automatic session-ingestion errors."""


class SubmissionInProgress(RuntimeError):
    """Raised when another worker still owns the submission lease."""

