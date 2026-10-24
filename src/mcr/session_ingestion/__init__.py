"""Automatic per-session turn, context, and ingestion orchestration."""

from mcr.session_ingestion.config import SessionIngestionConfig
from mcr.session_ingestion.errors import SubmissionInProgress
from mcr.session_ingestion.models import (
    AssignedMessage,
    SessionContextTrace,
    SessionIngestionRequest,
    SessionIngestionResult,
)
from mcr.session_ingestion.service import SessionIngestionService

__all__ = [
    "AssignedMessage",
    "SessionContextTrace",
    "SessionIngestionConfig",
    "SessionIngestionRequest",
    "SessionIngestionResult",
    "SessionIngestionService",
    "SubmissionInProgress",
]
