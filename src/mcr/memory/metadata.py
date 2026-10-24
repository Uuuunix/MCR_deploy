"""Conversation and provenance metadata for a memory."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, JsonValue, field_validator

from mcr.memory.common import DomainModel, NonEmptyStr


class MemoryMetadata(DomainModel):
    """Metadata locating a memory in its originating conversation."""

    created_at: datetime
    session_id: NonEmptyStr
    turn_id: int | NonEmptyStr
    extra: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @field_validator("turn_id")
    @classmethod
    def validate_turn_id(cls, value: int | str) -> int | str:
        if isinstance(value, bool):
            raise ValueError("turn_id must be an integer or string")
        if isinstance(value, int) and value < 0:
            raise ValueError("integer turn_id must be non-negative")
        return value

