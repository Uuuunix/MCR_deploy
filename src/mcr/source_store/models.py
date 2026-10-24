"""Raw immutable conversation source models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator

from mcr.llm import ContentPart
from mcr.memory.common import DomainModel, NonEmptyStr


class MessageDraft(DomainModel):
    role: Literal["user", "assistant"]
    turn_id: int | NonEmptyStr
    content: list[ContentPart] = Field(min_length=1)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class StoredMessagePart(DomainModel):
    message_id: UUID
    part_index: int = Field(ge=0)
    part_type: Literal["text", "image"]
    text_content: str | None = None
    image_bytes: bytes | None = Field(default=None, repr=False)
    media_type: str | None = None
    content_hash: NonEmptyStr
    source_metadata: dict = Field(default_factory=dict)


class StoredMessage(DomainModel):
    id: UUID
    session_id: NonEmptyStr
    turn_id: int | str
    role: Literal["user", "assistant"]
    created_at: datetime
    content_hash: NonEmptyStr
    parts: list[StoredMessagePart]


class IngestionRunRecord(DomainModel):
    id: UUID
    idempotency_key: NonEmptyStr
    session_id: NonEmptyStr
    request_hash: NonEmptyStr
    status: NonEmptyStr
    stage: NonEmptyStr
    result: dict | None = None
    error: dict | None = None
    checkpoint: dict | None = None
    created_at: datetime
    updated_at: datetime


class SessionSubmissionRecord(DomainModel):
    session_id: NonEmptyStr
    submission_id: UUID
    request_hash: NonEmptyStr
    status: Literal["processing", "completed", "failed"]
    message_ids: list[UUID] = Field(min_length=1, max_length=2)
    ingestion_run_id: UUID | None = None
    worker_token: UUID | None = None
    lease_expires_at: datetime | None = None
    result: dict | None = None
    error: dict | None = None
    created_at: datetime
    updated_at: datetime
    claimed: bool = False


class AgentExchangeRecord(DomainModel):
    id: UUID
    request_id: UUID
    session_id: NonEmptyStr
    user_request_hash: NonEmptyStr
    assistant_request_hash: str | None = None
    status: Literal["pending", "processing", "completed", "failed"]
    stage: NonEmptyStr
    user_message_id: UUID
    assistant_message_id: UUID | None = None
    before_result: dict | None = None
    after_result: dict | None = None
    error: dict | None = None
    created_at: datetime
    updated_at: datetime
