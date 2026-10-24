"""Concrete services bound to one workspace's isolated stores."""

from __future__ import annotations

from dataclasses import dataclass, field

from qdrant_client import AsyncQdrantClient

from mcr.agent_lifecycle import AgentMemoryLifecycle
from mcr.ingestion import MemoryIngestionService
from mcr.qdrant import QdrantMemoryRepository
from mcr.retrieval import UnifiedMemoryRetriever
from mcr.session_ingestion import SessionIngestionService
from mcr.source_store import SQLiteSourceStore
from mcr.summarization import SummaryCoordinator, SummaryWorker
from mcr.workspaces.models import WorkspaceRecord


@dataclass
class WorkspaceRuntime:
    record: WorkspaceRecord
    source_store: SQLiteSourceStore
    qdrant_client: AsyncQdrantClient
    repository: QdrantMemoryRepository
    retriever: UnifiedMemoryRetriever
    ingestion: MemoryIngestionService
    lifecycle: AgentMemoryLifecycle
    session_ingestion: SessionIngestionService
    summary_coordinator: SummaryCoordinator
    summary_worker: SummaryWorker
    _started: bool = field(default=False, init=False)

    async def start(self) -> None:
        if not self._started:
            await self.summary_worker.start()
            self._started = True

    async def before_generation(self, request):
        return await self.lifecycle.before_generation(request)

    async def after_generation(self, request):
        return await self.lifecycle.after_generation(request)

    async def ingest_exchange(self, **kwargs):
        return await self.lifecycle.ingest_exchange(**kwargs)

    def ingest_exchange_sync(self, **kwargs):
        return self.lifecycle.ingest_exchange_sync(**kwargs)

    async def aclose(self) -> None:
        if self._started:
            await self.summary_worker.stop()
            self._started = False
        await self.qdrant_client.close()
        self.source_store.close()

