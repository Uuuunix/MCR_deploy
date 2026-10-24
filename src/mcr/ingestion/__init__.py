"""End-to-end immutable memory ingestion."""

from mcr.ingestion.errors import (
    IngestionError,
    IngestionPersistenceError,
    JudgmentError,
    RepairRequiredError,
)
from mcr.ingestion.judgment import MemoryJudge
from mcr.ingestion.models import (
    CandidateOutcome,
    IngestionRequest,
    IngestionResult,
    IngestionTrace,
    JudgmentPayload,
    MemoryDecision,
    RelationDecision,
)
from mcr.ingestion.orchestrator import MemoryIngestionService

__all__ = [
    "CandidateOutcome",
    "IngestionError",
    "IngestionPersistenceError",
    "IngestionRequest",
    "IngestionResult",
    "IngestionTrace",
    "JudgmentError",
    "JudgmentPayload",
    "MemoryDecision",
    "MemoryJudge",
    "MemoryIngestionService",
    "RelationDecision",
    "RepairRequiredError",
]
