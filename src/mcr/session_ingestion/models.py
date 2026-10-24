"""Public contracts for automatic per-session ingestion."""

from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field

from mcr.ingestion import IngestionResult
from mcr.llm import ContentPart
from mcr.memory.common import DomainModel, NonEmptyStr


class SessionIngestionRequest(DomainModel):
    workspace_id: NonEmptyStr = "default"
    session_id: NonEmptyStr
    user_content: list[ContentPart] = Field(min_length=1)
    assistant_content: list[ContentPart] | None = Field(default=None, min_length=1)
    submission_id: UUID = Field(default_factory=uuid4)


class AssignedMessage(DomainModel):
    id: UUID
    role: Literal["user", "assistant"]
    turn_id: int | NonEmptyStr


class SessionContextTrace(DomainModel):
    summary_version: int | None = None
    recent_message_ids: list[UUID] = Field(default_factory=list)


class SessionIngestionResult(DomainModel):
    submission_id: UUID
    assigned_messages: list[AssignedMessage] = Field(min_length=1, max_length=2)
    context: SessionContextTrace
    result: IngestionResult
