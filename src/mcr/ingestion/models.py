"""Memory ingestion requests, decisions, traces, and outcomes."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from mcr.extraction import ConversationMessage
from mcr.memory import (
    MemoryContent,
    MemoryLink,
    MemoryMetadata,
    RetrievalDescription,
    SourceRelation,
)
from mcr.memory.common import DomainModel, NonEmptyStr
from mcr.source_store import MessageDraft


class IngestionRequest(DomainModel):
    session_id: NonEmptyStr
    current_messages: list[MessageDraft] = Field(min_length=1, max_length=2)
    summary: str | None = None
    recent_messages: list[ConversationMessage] = Field(default_factory=list)
    observation_time: datetime
    custom_instructions: str | None = None

    @model_validator(mode="after")
    def validate_current_shape(self) -> "IngestionRequest":
        roles = [message.role for message in self.current_messages]
        if roles not in (["user"], ["assistant"], ["user", "assistant"]):
            raise ValueError(
                "current_messages must be User-only, Assistant-only, or User+Assistant"
            )
        if self.observation_time.tzinfo is None or self.observation_time.utcoffset() is None:
            raise ValueError("observation_time must be timezone-aware")
        return self


class MemoryDecision(DomainModel):
    candidate: NonEmptyStr
    action: Literal["ADD", "SKIP"]
    duplicate_of: str | None = None

    @model_validator(mode="after")
    def validate_duplicate(self) -> "MemoryDecision":
        if self.action == "SKIP" and not self.duplicate_of:
            raise ValueError("SKIP requires duplicate_of")
        if self.action == "ADD" and self.duplicate_of is not None:
            raise ValueError("ADD cannot define duplicate_of")
        return self


class RelationDecision(DomainModel):
    left: NonEmptyStr
    right: NonEmptyStr
    reason: NonEmptyStr


class JudgmentPayload(DomainModel):
    decisions: list[MemoryDecision]
    relations: list[RelationDecision] = Field(default_factory=list)


class CandidateOutcome(DomainModel):
    candidate_id: UUID
    action: Literal["ADD", "SKIP"]
    memory_id: UUID
    duplicate_of: UUID | None = None
    related_memory_ids: list[UUID] = Field(default_factory=list)


class IngestionTrace(DomainModel):
    extraction: dict | None = None
    retrieved: dict[str, list[dict]] = Field(default_factory=dict)
    judgment_mode: Literal["batch", "per_candidate_fallback", "deterministic", "serial_batches", "additive"]
    judgment_requests: list[dict] = Field(default_factory=list)
    judgment_responses: list[str] = Field(default_factory=list)
    description_generation: list[dict] = Field(default_factory=list)
    persistence_steps: list[str] = Field(default_factory=list)
    batches: list[dict] = Field(default_factory=list)


class PendingMemory(DomainModel):
    """A memory between x extraction and post-admission D generation."""

    schema_version: Literal[2] = 2
    id: UUID
    x: MemoryContent
    meta: MemoryMetadata
    descriptions: list[RetrievalDescription] = Field(default_factory=list)
    relations: list[SourceRelation] = Field(min_length=1)
    links: list[MemoryLink] = Field(default_factory=list)


class IngestionResult(DomainModel):
    operation_id: UUID
    message_ids: list[UUID]
    mode: Literal["batch", "per_candidate_fallback", "deterministic", "serial_batches", "additive"]
    outcomes: list[CandidateOutcome]
    trace: IngestionTrace
