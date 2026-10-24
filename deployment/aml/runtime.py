"""Isolated MCR runtime for one AML user, without altering MCR core."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient

from mcr.extraction import MemoryExtractor
from mcr.ingestion import MemoryIngestionService, MemoryJudge
from mcr.qdrant import QdrantConfig, QdrantMemoryRepository
from mcr.retrieval import UnifiedMemoryRetriever
from mcr.source_store import SQLiteSourceStore
from mcr.summarization import SessionSummarizer, SummaryConfig, SummaryCoordinator, SummaryWorker
from mcr.workspaces import WorkspaceCatalog, WorkspaceManager

from deployment.aml.providers import AcademicEmbeddings, AcademicLLM
from deployment.aml.settings import Settings


@dataclass
class AMLWorkspace:
    source_store: SQLiteSourceStore
    qdrant_client: AsyncQdrantClient
    retriever: UnifiedMemoryRetriever
    ingestion: MemoryIngestionService
    summary_worker: SummaryWorker

    async def aclose(self) -> None:
        await self.summary_worker.stop()
        await self.qdrant_client.close()
        self.source_store.close()


def build_manager(settings: Settings) -> tuple[WorkspaceCatalog, WorkspaceManager[AMLWorkspace]]:
    settings.require_providers()
    llm = AcademicLLM(
        settings.openai_api_key, settings.openai_base_url, model=settings.llm_model
    )
    embedding = AcademicEmbeddings(
        settings.dashscope_api_key,
        settings.dashscope_base_url,
        llm,
        dimension=settings.embedding_dimension,
    )
    catalog = WorkspaceCatalog(settings.data_root / "users")
    summary_config = SummaryConfig(batch_size=5, poll_seconds=1.0, max_attempts=3)

    async def build(record) -> AMLWorkspace:
        client = AsyncQdrantClient(path=record.qdrant_path)
        config = QdrantConfig(
            collection_name=record.collection_name,
            vector_size=embedding.dimension,
            embedding_model=embedding.model_id,
        )
        repository = QdrantMemoryRepository(client, config, embedding)
        retriever = UnifiedMemoryRetriever(client, config, embedding)
        sources = SQLiteSourceStore(record.sqlite_path)
        coordinator = SummaryCoordinator(sources, summary_config)
        worker = SummaryWorker(sources, SessionSummarizer(llm), summary_config)
        ingestion = MemoryIngestionService(
            sources,
            MemoryExtractor(
                llm,
                recent_window_size=settings.recent_window_size,
                description_llm=llm,
            ),
            embedding,
            retriever,
            repository,
            MemoryJudge(llm),
            summary_coordinator=coordinator,
        )
        await worker.start()
        return AMLWorkspace(sources, client, retriever, ingestion, worker)

    return catalog, WorkspaceManager(catalog, build)
