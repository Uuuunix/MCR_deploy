"""FastAPI routes for prompt preview and live extraction."""

from __future__ import annotations

import math
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response

from mcr.debug_ui.schemas import (
    AfterGenerationHttpRequest,
    AfterGenerationHttpResponse,
    BeforeGenerationHttpRequest,
    BeforeGenerationHttpResponse,
    DebugExtractionRequest,
    DebugExtractionResponse,
    DebugIngestionRequest,
    DebugIngestionResponse,
    DebugSearchRequest,
    DebugSearchResponse,
    HealthResponse,
    MemoryLibraryResponse,
    SourceMessageLibraryResponse,
    SourceMessageView,
    PromptPreviewResponse,
    SummaryRetryResponse,
    SummaryStatusResponse,
    SessionIngestionResponse,
    WorkspaceCreateRequest,
    WorkspaceListResponse,
    WorkspaceResponse,
)
from mcr.agent_lifecycle import AfterGenerationRequest, BeforeGenerationRequest
from mcr.extraction import MemoryExtractor, build_extraction_prompt
from mcr.extraction.errors import ExtractionError, ExtractionResponseError
from mcr.llm import LLMClient, LLMError
from mcr.ingestion import IngestionError, MemoryIngestionService
from mcr.persistence import PersistenceError
from mcr.qdrant import QdrantMemoryRepository
from mcr.retrieval import RetrievalConfig, RetrievalError, UnifiedMemoryRetriever
from mcr.embedding import EmbeddingError
from mcr.source_store import SourceStoreError
from mcr.summarization import SummaryCoordinator
from mcr.session_ingestion import (
    SessionIngestionRequest,
    SessionIngestionService,
    SubmissionInProgress,
)
from mcr.workspaces import (
    WorkspaceError,
    WorkspaceManager,
    WorkspaceNotFound,
    WorkspaceRuntime,
)


def create_router(
    llm: LLMClient | None,
    provider: str | None = None,
    model: str | None = None,
    ingestion_service: MemoryIngestionService | None = None,
    retriever: UnifiedMemoryRetriever | None = None,
    repository: QdrantMemoryRepository | None = None,
    summary_coordinator: SummaryCoordinator | None = None,
    session_ingestion_service: SessionIngestionService | None = None,
    workspace_manager: WorkspaceManager[WorkspaceRuntime] | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api")

    async def get_runtime(workspace_id: str) -> WorkspaceRuntime | None:
        if workspace_manager is None:
            return None
        try:
            runtime = await workspace_manager.open(workspace_id)
            await runtime.start()
            return runtime
        except WorkspaceNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except WorkspaceError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            live_extraction_configured=llm is not None,
            live_ingestion_configured=(
                ingestion_service is not None or workspace_manager is not None
            ),
            memory_tools_configured=(
                (retriever is not None and repository is not None)
                or workspace_manager is not None
            ),
            summarization_configured=(
                summary_coordinator is not None or workspace_manager is not None
            ),
            provider=provider,
            model=model,
        )

    @router.post("/prompt-preview", response_model=PromptPreviewResponse)
    async def prompt_preview(payload: DebugExtractionRequest) -> PromptPreviewResponse:
        messages, recent = build_extraction_prompt(
            payload.extraction, payload.recent_window_size
        )
        return PromptPreviewResponse(
            messages=messages,
            effective_recent_message_ids=[message.id for message in recent],
        )

    @router.post("/extractions", response_model=DebugExtractionResponse)
    async def extract(payload: DebugExtractionRequest) -> DebugExtractionResponse:
        if llm is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Live extraction is not configured. Check MCR_MODEL_BACKEND "
                    "and its LLM settings, then restart the debug server."
                ),
            )
        extractor = MemoryExtractor(
            llm,
            recent_window_size=payload.recent_window_size,
            description_llm=llm,
        )
        try:
            result = await extractor.extract_with_trace(payload.extraction)
        except (ExtractionError, LLMError) as exc:
            detail = (
                exc.debug_detail()
                if isinstance(exc, ExtractionResponseError)
                else str(exc)
            )
            raise HTTPException(status_code=502, detail=detail) from exc
        return DebugExtractionResponse(result=result)

    @router.post("/ingestions", response_model=DebugIngestionResponse)
    async def ingest(payload: DebugIngestionRequest) -> DebugIngestionResponse:
        if ingestion_service is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Live ingestion is not configured. Set SiliconFlow model/key "
                    "and embedding dimension, then restart the debug server."
                ),
            )
        try:
            result = await ingestion_service.ingest(payload.ingestion)
        except (
            IngestionError,
            ExtractionError,
            LLMError,
            PersistenceError,
            EmbeddingError,
            SourceStoreError,
        ) as exc:
            detail = (
                exc.debug_detail()
                if isinstance(exc, ExtractionResponseError)
                else str(exc)
            )
            raise HTTPException(status_code=502, detail=detail) from exc
        return DebugIngestionResponse(result=result)

    @router.get("/session-ingestions/progress")
    async def session_ingestion_progress(
        session_id: str = Query(min_length=1),
        workspace_id: str = Query(default="default"),
    ) -> dict:
        runtime = await get_runtime(workspace_id)
        service = (
            runtime.session_ingestion if runtime is not None else session_ingestion_service
        )
        if service is None:
            raise HTTPException(status_code=503, detail="Session ingestion is not configured")
        return {
            "progress": service.get_progress(session_id)
        }

    @router.post("/session-ingestions", response_model=SessionIngestionResponse)
    async def session_ingest(
        payload: SessionIngestionRequest,
    ) -> SessionIngestionResponse:
        runtime = await get_runtime(payload.workspace_id)
        service = (
            runtime.session_ingestion if runtime is not None else session_ingestion_service
        )
        if service is None:
            raise HTTPException(
                status_code=503,
                detail="Automatic session ingestion is not configured.",
            )
        try:
            result = await service.ingest(payload)
        except SubmissionInProgress as exc:
            raise HTTPException(
                status_code=202,
                detail={
                    "status": "processing",
                    "submission_id": str(exc),
                },
                headers={"Retry-After": "2"},
            ) from exc
        except (
            IngestionError,
            ExtractionError,
            LLMError,
            PersistenceError,
            EmbeddingError,
            SourceStoreError,
            ValueError,
        ) as exc:
            detail = (
                exc.debug_detail()
                if isinstance(exc, ExtractionResponseError)
                else str(exc)
            )
            raise HTTPException(status_code=502, detail=detail) from exc
        return SessionIngestionResponse(result=result)

    @router.post("/search", response_model=DebugSearchResponse)
    async def search(payload: DebugSearchRequest) -> DebugSearchResponse:
        runtime = await get_runtime(payload.workspace_id)
        active_retriever = runtime.retriever if runtime is not None else retriever
        if active_retriever is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Memory search is not configured. Check MCR_MODEL_BACKEND "
                    "and its embedding settings, then restart the debug server."
                ),
            )
        try:
            results = await active_retriever.search(
                payload.query,
                RetrievalConfig(result_top_k=payload.top_k),
            )
        except (RetrievalError, EmbeddingError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return DebugSearchResponse(results=results)

    @router.get("/memories", response_model=MemoryLibraryResponse)
    async def memories(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
        memory_id: UUID | None = None,
        session_id: str | None = None,
        kind: str | None = None,
        facet_type: str | None = None,
        workspace_id: str = Query(default="default"),
    ) -> MemoryLibraryResponse:
        runtime = await get_runtime(workspace_id)
        active_repository = runtime.repository if runtime is not None else repository
        if active_repository is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Memory library is not configured. Set MCR_EMBEDDING_DIMENSION "
                    "and MCR_QDRANT_PATH, then restart the debug server."
                ),
            )
        try:
            items = await active_repository.list_memories()
        except PersistenceError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        if memory_id is not None:
            items = [item for item in items if item.id == memory_id]
        if session_id:
            items = [item for item in items if item.meta.session_id == session_id]
        if kind:
            expected = kind.casefold().strip()
            items = [item for item in items if item.x.kind.casefold() == expected]
        if facet_type:
            expected = facet_type.casefold().strip()
            items = [
                item
                for item in items
                if any(
                    value.facet_type.casefold() == expected
                    for value in item.descriptions
                )
            ]
        return MemoryLibraryResponse.from_items(
            items,
            page=page,
            page_size=page_size,
        )

    async def source_context(workspace_id: str):
        runtime = await get_runtime(workspace_id)
        if runtime is None:
            raise HTTPException(
                status_code=503,
                detail="Source-message inspection requires a workspace runtime.",
            )
        return runtime.source_store, runtime.repository

    @router.get(
        "/source-messages", response_model=SourceMessageLibraryResponse
    )
    async def source_messages(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
        session_id: str | None = None,
        workspace_id: str = Query(default="default"),
    ) -> SourceMessageLibraryResponse:
        sources, active_repository = await source_context(workspace_id)
        messages, total = sources.list_messages(
            session_id=session_id,
            offset=(page - 1) * page_size,
            limit=page_size,
        )
        memories = await active_repository.list_memories()
        by_source: dict[str, list] = {}
        for memory in memories:
            for source_id in {relation.source_id for relation in memory.relations}:
                by_source.setdefault(source_id, []).append(memory)
        return SourceMessageLibraryResponse(
            items=[
                SourceMessageView.from_message(
                    message, by_source.get(str(message.id), [])
                )
                for message in messages
            ],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=math.ceil(total / page_size) if total else 0,
        )

    @router.get("/source-messages/{message_id}", response_model=SourceMessageView)
    async def source_message(
        message_id: UUID,
        workspace_id: str = Query(default="default"),
    ) -> SourceMessageView:
        sources, active_repository = await source_context(workspace_id)
        try:
            message = sources.get_message(message_id)
        except SourceStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        memories = [
            memory
            for memory in await active_repository.list_memories()
            if any(relation.source_id == str(message_id) for relation in memory.relations)
        ]
        return SourceMessageView.from_message(message, memories)

    @router.get("/source-messages/{message_id}/parts/{part_index}")
    async def source_image(
        message_id: UUID,
        part_index: int,
        workspace_id: str = Query(default="default"),
    ) -> Response:
        sources, _ = await source_context(workspace_id)
        try:
            message = sources.get_message(message_id)
        except SourceStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        part = next(
            (part for part in message.parts if part.part_index == part_index), None
        )
        if part is None or part.part_type != "image" or part.image_bytes is None:
            raise HTTPException(status_code=404, detail="source image part not found")
        return Response(
            content=part.image_bytes,
            media_type=part.media_type or "application/octet-stream",
        )

    @router.get(
        "/summaries/{session_id}", response_model=SummaryStatusResponse
    )
    async def summary_status(
        session_id: str, workspace_id: str = Query(default="default")
    ) -> SummaryStatusResponse:
        runtime = await get_runtime(workspace_id)
        coordinator = (
            runtime.summary_coordinator if runtime is not None else summary_coordinator
        )
        if coordinator is None:
            raise HTTPException(
                status_code=503,
                detail="Session summarization is not configured.",
            )
        return SummaryStatusResponse(
            status=coordinator.get_status(session_id)
        )

    @router.post(
        "/summaries/{session_id}/retry", response_model=SummaryRetryResponse
    )
    async def retry_summary(
        session_id: str, workspace_id: str = Query(default="default")
    ) -> SummaryRetryResponse:
        runtime = await get_runtime(workspace_id)
        coordinator = (
            runtime.summary_coordinator if runtime is not None else summary_coordinator
        )
        if coordinator is None:
            raise HTTPException(
                status_code=503,
                detail="Session summarization is not configured.",
            )
        return SummaryRetryResponse(
            job=coordinator.retry_failed(session_id)
        )

    @router.post(
        "/before-generation", response_model=BeforeGenerationHttpResponse
    )
    async def before_generation(
        payload: BeforeGenerationHttpRequest,
    ) -> BeforeGenerationHttpResponse:
        runtime = await get_runtime(payload.workspace_id)
        if runtime is None:
            raise HTTPException(
                status_code=503,
                detail="Agent lifecycle is not configured.",
            )
        values = payload.model_dump(exclude={"workspace_id", "request_id"})
        if payload.request_id is not None:
            values["request_id"] = payload.request_id
        try:
            result = await runtime.lifecycle.before_generation(
                BeforeGenerationRequest(**values)
            )
        except (SourceStoreError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return BeforeGenerationHttpResponse(result=result)

    @router.post(
        "/after-generation", response_model=AfterGenerationHttpResponse
    )
    async def after_generation(
        payload: AfterGenerationHttpRequest,
    ) -> AfterGenerationHttpResponse:
        runtime = await get_runtime(payload.workspace_id)
        if runtime is None:
            raise HTTPException(
                status_code=503,
                detail="Agent lifecycle is not configured.",
            )
        try:
            result = await runtime.lifecycle.after_generation(
                AfterGenerationRequest(
                    exchange_id=payload.exchange_id,
                    assistant_content=payload.assistant_content,
                )
            )
        except (
            IngestionError,
            ExtractionError,
            LLMError,
            PersistenceError,
            EmbeddingError,
            SourceStoreError,
            ValueError,
        ) as exc:
            detail = (
                exc.debug_detail()
                if isinstance(exc, ExtractionResponseError)
                else str(exc)
            )
            raise HTTPException(status_code=502, detail=detail) from exc
        return AfterGenerationHttpResponse(result=result)

    @router.get("/workspaces", response_model=WorkspaceListResponse)
    async def list_workspaces() -> WorkspaceListResponse:
        if workspace_manager is None:
            return WorkspaceListResponse(workspaces=[])
        return WorkspaceListResponse(workspaces=workspace_manager.list())

    @router.post("/workspaces", response_model=WorkspaceResponse)
    async def create_workspace(
        payload: WorkspaceCreateRequest,
    ) -> WorkspaceResponse:
        if workspace_manager is None:
            raise HTTPException(status_code=503, detail="Workspace manager is not configured.")
        try:
            workspace = workspace_manager.create(payload.workspace_id)
        except (WorkspaceError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return WorkspaceResponse(workspace=workspace)

    return router

