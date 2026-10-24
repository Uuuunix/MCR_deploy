"""Application factory for the local extraction workbench."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mcr.debug_ui.routes import create_router
from mcr.llm import LLMClient
from mcr.ingestion import MemoryIngestionService
from mcr.qdrant import QdrantMemoryRepository
from mcr.retrieval import UnifiedMemoryRetriever
from mcr.summarization import SummaryCoordinator, SummaryWorker
from mcr.session_ingestion import SessionIngestionService
from mcr.workspaces import WorkspaceManager, WorkspaceRuntime

_STATIC_DIR = Path(__file__).with_name("static")


def create_app(
    llm: LLMClient | None = None,
    provider: str | None = None,
    model: str | None = None,
    ingestion_service: MemoryIngestionService | None = None,
    retriever: UnifiedMemoryRetriever | None = None,
    repository: QdrantMemoryRepository | None = None,
    summary_coordinator: SummaryCoordinator | None = None,
    summary_worker: SummaryWorker | None = None,
    session_ingestion_service: SessionIngestionService | None = None,
    workspace_manager: WorkspaceManager[WorkspaceRuntime] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if summary_worker is not None:
            await summary_worker.start()
        if workspace_manager is not None:
            runtime = await workspace_manager.open("default")
            await runtime.start()
        try:
            yield
        finally:
            if workspace_manager is not None:
                await workspace_manager.aclose()
            if summary_worker is not None:
                await summary_worker.stop()

    app = FastAPI(
        title="MCR Extraction Workbench",
        description="Local-only prompt and candidate-memory debugger",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(
        create_router(
            llm,
            provider=provider,
            model=model,
            ingestion_service=ingestion_service,
            retriever=retriever,
            repository=repository,
            summary_coordinator=summary_coordinator,
            session_ingestion_service=session_ingestion_service,
            workspace_manager=workspace_manager,
        )
    )
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    return app

