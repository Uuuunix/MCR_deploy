"""Public contracts for the Agent-facing memory lifecycle."""

from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field

from mcr.ingestion import IngestionResult
from mcr.llm import ContentPart
from mcr.memory.common import DomainModel, NonEmptyStr
from mcr.retrieval import RetrievedMemory
from mcr.context_integration.models import MemoryIntegrationResult


class AssignedLifecycleMessage(DomainModel):
    id: UUID
    role: Literal["user", "assistant"]
    turn_id: int | str


class BeforeGenerationRequest(DomainModel):
    session_id: NonEmptyStr
    user_content: list[ContentPart] = Field(min_length=1)
    top_k: int = Field(default=20, gt=0, le=100)
    score_threshold: float | None = None
    request_id: UUID = Field(default_factory=uuid4)
    integration_strategy: Literal["default", "query_aware"] = "default"


class BeforeGenerationTrace(DomainModel):
    retrieved: list[RetrievedMemory] = Field(default_factory=list)
    integration_metadata: dict = Field(default_factory=dict)
    integration_prompt_text: str = ""
    errors: list[dict] = Field(default_factory=list)


class BeforeGenerationResult(DomainModel):
    exchange_id: UUID
    request_id: UUID
    workspace_id: NonEmptyStr = "default"
    user_message: AssignedLifecycleMessage
    memories: list[RetrievedMemory] = Field(default_factory=list)
    context_text: str = ""
    integration: MemoryIntegrationResult | None = None
    trace: BeforeGenerationTrace


class AfterGenerationRequest(DomainModel):
    exchange_id: UUID
    assistant_content: list[ContentPart] | None = Field(default=None, min_length=1)


class LifecycleContextTrace(DomainModel):
    summary_version: int | None = None
    recent_message_ids: list[UUID] = Field(default_factory=list)


class AfterGenerationResult(DomainModel):
    exchange_id: UUID
    workspace_id: NonEmptyStr = "default"
    status: Literal["completed"] = "completed"
    assigned_messages: list[AssignedLifecycleMessage] = Field(min_length=1, max_length=2)
    context: LifecycleContextTrace
    result: IngestionResult


class ExchangeIngestionResult(DomainModel):
    before: BeforeGenerationResult
    after: AfterGenerationResult

