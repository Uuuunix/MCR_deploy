"""Persistent and provider-neutral summary models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator

from mcr.llm import LLMMessage, LLMUsage
from mcr.memory.common import DomainModel, NonEmptyStr


class SessionSummary(DomainModel):
    session_id: NonEmptyStr
    text: NonEmptyStr
    version: int = Field(gt=0)
    summarized_through_sequence: int = Field(gt=0)
    provider: NonEmptyStr
    model: NonEmptyStr
    request_id: str | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("summary timestamps must be timezone-aware")
        return value


class SummaryJob(DomainModel):
    id: UUID
    session_id: NonEmptyStr
    from_sequence: int = Field(gt=0)
    through_sequence: int = Field(gt=0)
    status: Literal["pending", "running", "complete", "failed"]
    attempts: int = Field(ge=0)
    error: dict | None = None
    available_at: datetime
    claimed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator("available_at", "claimed_at", "created_at", "updated_at")
    @classmethod
    def require_job_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("job timestamps must be timezone-aware")
        return value


class SummaryTrace(DomainModel):
    messages: list[LLMMessage]
    provider_request: dict | None = None
    raw_response: str
    elapsed_ms: float = Field(ge=0)
    provider: NonEmptyStr
    model: NonEmptyStr
    request_id: str | None = None
    usage: LLMUsage | None = None


class SummaryResult(DomainModel):
    text: NonEmptyStr
    trace: SummaryTrace


class SummaryStatus(DomainModel):
    session_id: NonEmptyStr
    summary: SessionSummary | None = None
    submission_count: int = Field(ge=0)
    pending_submission_count: int = Field(ge=0)
    active_job: SummaryJob | None = None
