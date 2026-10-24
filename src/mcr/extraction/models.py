"""Input, candidate, and trace schemas for memory extraction."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, JsonValue, field_validator, model_validator

from mcr.llm import ContentPart, LLMMessage, LLMUsage
from mcr.memory import MemoryContent, RetrievalDescription, SourceRelation
from mcr.memory.common import DomainModel, NonEmptyStr


class ConversationMessage(DomainModel):
    id: NonEmptyStr
    role: Literal["user", "assistant"]
    content: list[ContentPart] = Field(min_length=1)
    created_at: datetime | None = None
    turn_id: int | NonEmptyStr | None = None

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("created_at must be timezone-aware")
        return value

    @field_validator("turn_id")
    @classmethod
    def validate_turn_id(cls, value: int | str | None) -> int | str | None:
        if isinstance(value, bool):
            raise ValueError("turn_id must be an integer or string")
        if isinstance(value, int) and value < 0:
            raise ValueError("integer turn_id must be non-negative")
        return value


class ExtractionRequest(DomainModel):
    session_id: NonEmptyStr
    summary: str | None = None
    recent_messages: list[ConversationMessage] = Field(default_factory=list)
    current_user: ConversationMessage | None = None
    current_assistant: ConversationMessage | None = None
    observation_time: datetime
    custom_instructions: str | None = None
    existing_memories: list["ExistingMemoryContext"] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_pair(self) -> "ExtractionRequest":
        if self.current_user is None and self.current_assistant is None:
            raise ValueError("at least one current message is required")
        if self.current_user is not None and self.current_user.role != "user":
            raise ValueError("current_user must have role 'user'")
        if self.current_assistant is not None and self.current_assistant.role != "assistant":
            raise ValueError("current_assistant must have role 'assistant'")
        if self.observation_time.tzinfo is None or self.observation_time.utcoffset() is None:
            raise ValueError("observation_time must be timezone-aware")

        current_ids = {self.current_user.id} if self.current_user is not None else set()
        if self.current_assistant is not None:
            if (
                self.current_user is not None
                and self.current_assistant.id == self.current_user.id
            ):
                raise ValueError("current pair message IDs must be different")
            current_ids.add(self.current_assistant.id)
        recent_ids = [message.id for message in self.recent_messages]
        if len(recent_ids) != len(set(recent_ids)):
            raise ValueError("recent message IDs must be unique")
        overlap = current_ids.intersection(recent_ids)
        if overlap:
            raise ValueError("current messages cannot also appear in recent context")
        return self


class ExtractedSourceReference(DomainModel):
    source_message_id: NonEmptyStr
    part_index: int = Field(ge=0)


class ExistingMemoryContext(DomainModel):
    id: UUID
    text: NonEmptyStr


class ExtractedMemoryLink(DomainModel):
    memory_id: UUID
    reason: NonEmptyStr


class ExtractedMemory(DomainModel):
    """Minimal, untrusted x output requested from the LLM.

    ``image_refs`` is the only provenance decision delegated to the model: it
    tells MCR which current images actually contributed to this x. Text
    provenance, metadata, IDs, D, embeddings, and links are built by MCR.
    ``source_refs``/``source_message_ids`` remain readable for old checkpoints.
    """

    kind: Literal["event", "fact"]
    content: NonEmptyStr
    image_refs: list[ExtractedSourceReference] = Field(default_factory=list)
    source_refs: list[ExtractedSourceReference] = Field(default_factory=list)
    source_message_ids: list[NonEmptyStr] = Field(default_factory=list)


class ExtractionPayload(DomainModel):
    memories: list[ExtractedMemory] = Field(default_factory=list)


class CandidateMetadata(DomainModel):
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime
    observed_at: datetime
    session_id: NonEmptyStr
    source_message_ids: list[NonEmptyStr] = Field(min_length=1)
    source_turn_ids: list[int | NonEmptyStr] = Field(default_factory=list)
    provider: NonEmptyStr
    model: NonEmptyStr
    request_id: str | None = None

    @field_validator("created_at", "observed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("candidate timestamps must be timezone-aware")
        return value


class CandidateMemory(DomainModel):
    x: MemoryContent
    meta: CandidateMetadata
    descriptions: list[RetrievalDescription] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    relations: list[SourceRelation] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    links: list[ExtractedMemoryLink] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )


class ExtractionTrace(DomainModel):
    messages: list[LLMMessage]
    provider_request: dict[str, JsonValue] | None = None
    raw_response: str
    elapsed_ms: float = Field(ge=0)
    provider: NonEmptyStr
    model: NonEmptyStr
    request_id: str | None = None
    usage: LLMUsage | None = None
    effective_recent_message_ids: list[str] = Field(default_factory=list)
    schema_repair_used: bool = False
    response_attempts: list[str] = Field(default_factory=list)
    response_diagnostics: list[dict[str, JsonValue]] = Field(default_factory=list)
    discarded_candidates: list[dict[str, JsonValue]] = Field(default_factory=list)
    # Kept for trace compatibility with runs created before x/r and D were
    # split. New extraction runs do not use a detached visual-grounding call.
    visual_grounding_response: str | None = None
    visual_grounding_request: dict[str, JsonValue] | None = None
    visual_grounding_usage: LLMUsage | None = None
    visual_grounding_elapsed_ms: float = Field(default=0.0, ge=0)
    visual_grounding_error: str | None = None


class ExtractionResult(DomainModel):
    candidates: list[CandidateMemory]
    trace: ExtractionTrace

