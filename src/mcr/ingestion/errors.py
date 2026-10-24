"""Stable ingestion-pipeline exceptions."""


class IngestionError(RuntimeError):
    """Base ingestion failure."""


class JudgmentError(IngestionError):
    """The memory action judgment was invalid or unavailable."""


class IngestionPersistenceError(IngestionError):
    """Persistence or compensation failed."""


class RepairRequiredError(IngestionPersistenceError):
    """Compensation could not restore a verified safe state."""
