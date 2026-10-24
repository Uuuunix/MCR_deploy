"""Contracts for turning retrieved memories into generation context."""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from mcr.extraction import ConversationMessage
from mcr.memory.common import DomainModel, NonEmptyStr
from mcr.retrieval import RetrievedMemory


class MemoryIntegrationRequest(DomainModel):
    workspace_id: NonEmptyStr = "default"
    session_id: NonEmptyStr
    exchange_id: UUID
    current_user: ConversationMessage
    retrieved_memories: list[RetrievedMemory] = Field(default_factory=list)


class IntegrationImageRef(DomainModel):
    source_message_id: NonEmptyStr
    part_index: int = Field(ge=0)
    media_type: NonEmptyStr
    memory_ids: list[UUID] = Field(default_factory=list)


class MemoryIntegrationResult(DomainModel):
    memory_ids: list[UUID] = Field(default_factory=list)
    context_text: str = ""
    summary: str = ""
    residuals: list[str] = Field(default_factory=list)
    dropped_memory_ids: list[UUID] = Field(default_factory=list)
    image_refs: list[IntegrationImageRef] = Field(default_factory=list)
    input_image_refs: list[IntegrationImageRef] = Field(default_factory=list)
    prompt_text: str = ""
    metadata: dict = Field(default_factory=dict)

