"""HTTP contracts for the local extraction workbench."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field

from mcr.extraction.models import ExtractionRequest, ExtractionResult
from mcr.llm import LLMMessage
from mcr.ingestion import IngestionRequest, IngestionResult
from mcr.memory import MemoryItem
from mcr.memory.common import DomainModel
from mcr.retrieval import RetrievalQuery, RetrievedMemory
from mcr.summarization import SummaryJob, SummaryStatus
from mcr.session_ingestion import SessionIngestionResult
from mcr.agent_lifecycle import BeforeGenerationResult, AfterGenerationResult
from mcr.llm import ContentPart
from mcr.memory.common import NonEmptyStr
from mcr.workspaces import WorkspaceRecord
from mcr.source_store import StoredMessage


class DebugExtractionRequest(DomainModel):
    extraction: ExtractionRequest
    recent_window_size: int = Field(default=8, ge=0, le=100)


class PromptPreviewResponse(DomainModel):
    messages: list[LLMMessage]
    effective_recent_message_ids: list[str]


class DebugExtractionResponse(DomainModel):
    result: ExtractionResult


class DebugIngestionRequest(DomainModel):
    ingestion: IngestionRequest


class DebugIngestionResponse(DomainModel):
    result: IngestionResult


class SessionIngestionResponse(DomainModel):
    result: SessionIngestionResult


class DebugSearchRequest(DomainModel):
    workspace_id: NonEmptyStr = "default"
    query: RetrievalQuery
    top_k: int = Field(default=20, gt=0, le=100)


class DebugSearchResponse(DomainModel):
    results: list[RetrievedMemory]


class MemoryLibraryResponse(DomainModel):
    items: list[MemoryItem]
    total: int = Field(ge=0)
    page: int = Field(gt=0)
    page_size: int = Field(gt=0)
    total_pages: int = Field(ge=0)

    @classmethod
    def from_items(
        cls,
        items: list[MemoryItem],
        *,
        page: int,
        page_size: int,
    ) -> "MemoryLibraryResponse":
        total = len(items)
        start = (page - 1) * page_size
        return cls(
            items=items[start : start + page_size],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=math.ceil(total / page_size) if total else 0,
        )


class SourcePartView(DomainModel):
    part_index: int
    part_type: str
    text_content: str | None = None
    media_type: str | None = None
    content_hash: str


class SourceMessageView(DomainModel):
    id: UUID
    session_id: str
    turn_id: int | str
    role: str
    created_at: datetime
    content_hash: str
    parts: list[SourcePartView]
    memories: list[MemoryItem] = Field(default_factory=list)

    @classmethod
    def from_message(
        cls, message: StoredMessage, memories: list[MemoryItem]
    ) -> "SourceMessageView":
        return cls(
            id=message.id,
            session_id=message.session_id,
            turn_id=message.turn_id,
            role=message.role,
            created_at=message.created_at,
            content_hash=message.content_hash,
            parts=[
                SourcePartView(
                    part_index=part.part_index,
                    part_type=part.part_type,
                    text_content=part.text_content,
                    media_type=part.media_type,
                    content_hash=part.content_hash,
                )
                for part in message.parts
            ],
            memories=memories,
        )


class SourceMessageLibraryResponse(DomainModel):
    items: list[SourceMessageView]
    total: int = Field(ge=0)
    page: int = Field(gt=0)
    page_size: int = Field(gt=0)
    total_pages: int = Field(ge=0)


class HealthResponse(DomainModel):
    ready: bool = True
    live_extraction_configured: bool
    live_ingestion_configured: bool = False
    memory_tools_configured: bool = False
    summarization_configured: bool = False
    provider: str | None = None
    model: str | None = None


class SummaryStatusResponse(DomainModel):
    status: SummaryStatus


class SummaryRetryResponse(DomainModel):
    job: SummaryJob | None = None


class BeforeGenerationHttpRequest(DomainModel):
    workspace_id: NonEmptyStr = "default"
    session_id: NonEmptyStr
    user_content: list[ContentPart] = Field(min_length=1)
    top_k: int = Field(default=20, gt=0, le=100)
    score_threshold: float | None = None
    request_id: UUID | None = None
    integration_strategy: Literal["default", "query_aware"] = "default"


class BeforeGenerationHttpResponse(DomainModel):
    result: BeforeGenerationResult


class AfterGenerationHttpRequest(DomainModel):
    workspace_id: NonEmptyStr = "default"
    exchange_id: UUID
    assistant_content: list[ContentPart] | None = Field(default=None, min_length=1)


class AfterGenerationHttpResponse(DomainModel):
    result: AfterGenerationResult


class WorkspaceCreateRequest(DomainModel):
    workspace_id: NonEmptyStr


class WorkspaceResponse(DomainModel):
    workspace: WorkspaceRecord


class WorkspaceListResponse(DomainModel):
    workspaces: list[WorkspaceRecord]

