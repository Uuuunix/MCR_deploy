"""CLI for the loopback-only debug server with an OS-assigned port."""

from __future__ import annotations

import argparse
import os
import socket

import uvicorn
from qdrant_client import AsyncQdrantClient

from mcr.debug_ui.app import create_app
from mcr.llm.providers.siliconflow import SiliconFlowClient, SiliconFlowConfig
from mcr.llm.providers.vllm import VLLMClient, VLLMConfig
from mcr.embedding.providers.siliconflow import (
    SiliconFlowEmbeddingClient,
    SiliconFlowEmbeddingConfig,
)
from mcr.embedding.providers.vllm import VLLMEmbeddingClient, VLLMEmbeddingConfig
from mcr.extraction import MemoryExtractor
from mcr.agent_lifecycle import AgentMemoryLifecycle
from mcr.context_integration import QueryAwareMemoryContextIntegrator
from mcr.ingestion import MemoryIngestionService, MemoryJudge
from mcr.qdrant import QdrantConfig, QdrantMemoryRepository
from mcr.retrieval import UnifiedMemoryRetriever
from mcr.source_store import SQLiteSourceStore
from mcr.summarization import (
    SessionSummarizer,
    SummaryConfig,
    SummaryCoordinator,
    SummaryWorker,
)
from mcr.session_ingestion import SessionIngestionConfig, SessionIngestionService
from mcr.workspaces import (
    WorkspaceCatalog,
    WorkspaceManager,
    WorkspaceRuntime,
    register_legacy_default,
)

LOOPBACK_HOST = "127.0.0.1"


def bind_server_socket(port: int = 0) -> socket.socket:
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((LOOPBACK_HOST, port))
    sock.listen(2048)
    return sock


def build_app_from_environment():
    backend = os.getenv("MCR_MODEL_BACKEND", "siliconflow").strip().lower()
    if backend == "vllm":
        model = os.getenv("MCR_VLLM_LLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        dimension = int(os.getenv("MCR_VLLM_EMBEDDING_DIMENSION", "2048"))
        llm = VLLMClient(VLLMConfig(
            model=model,
            base_url=os.getenv("MCR_VLLM_LLM_BASE_URL", "http://127.0.0.1:18000/v1"),
            max_tokens=int(os.getenv("MCR_LLM_MAX_TOKENS", "8192")),
        ))
        embedding = VLLMEmbeddingClient(VLLMEmbeddingConfig(
            model=os.getenv("MCR_VLLM_EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B"),
            base_url=os.getenv("MCR_VLLM_EMBEDDING_BASE_URL", "http://127.0.0.1:43227/v1"),
            dimension=dimension,
        ))
        sqlite_path = os.getenv("MCR_VLLM_SQLITE_PATH", "/tmp/mcr-debug-vllm.sqlite3")
        qdrant_path = os.getenv("MCR_VLLM_QDRANT_PATH", "/tmp/mcr-debug-vllm-qdrant")
        workspace_root = os.getenv("MCR_VLLM_WORKSPACE_ROOT", "/tmp/mcr-debug-vllm-workspaces")
        summary_llm = VLLMClient(VLLMConfig(
            model=os.getenv("MCR_VLLM_SUMMARY_MODEL", model),
            base_url=os.getenv("MCR_VLLM_LLM_BASE_URL", "http://127.0.0.1:18000/v1"),
            max_tokens=int(os.getenv("MCR_SUMMARY_MAX_TOKENS", "1024")),
        ))
    elif backend == "siliconflow":
        model = os.getenv("MCR_LLM_MODEL")
        api_key = os.getenv("SILICONFLOW_API_KEY")
        dimension_text = os.getenv("MCR_EMBEDDING_DIMENSION")
        if not model or not api_key:
            return create_app(model=model, provider="siliconflow" if model else None)
        llm = SiliconFlowClient(SiliconFlowConfig(
            model=model,
            max_tokens=int(os.getenv("MCR_LLM_MAX_TOKENS", "8192")),
        ))
        if not dimension_text:
            return create_app(llm, provider="siliconflow", model=model)
        dimension = int(dimension_text)
        embedding = SiliconFlowEmbeddingClient(SiliconFlowEmbeddingConfig(
            dimension=dimension,
            model=os.getenv("MCR_EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-8B"),
        ))
        sqlite_path = os.getenv("MCR_SQLITE_PATH", "/tmp/mcr-debug.sqlite3")
        qdrant_path = os.getenv("MCR_QDRANT_PATH", "/tmp/mcr-debug-qdrant")
        workspace_root = os.getenv("MCR_WORKSPACE_ROOT", "/tmp/mcr-debug-workspaces")
        summary_llm = SiliconFlowClient(SiliconFlowConfig(
            model=os.getenv("MCR_SUMMARY_MODEL", model),
            max_tokens=int(os.getenv("MCR_SUMMARY_MAX_TOKENS", "1024")),
        ))
    else:
        raise ValueError("MCR_MODEL_BACKEND must be 'siliconflow' or 'vllm'")
    catalog = WorkspaceCatalog(workspace_root)
    register_legacy_default(
        catalog,
        sqlite_path=sqlite_path,
        qdrant_path=qdrant_path,
    )
    summary_config = SummaryConfig(
        batch_size=int(os.getenv("MCR_SUMMARY_BATCH_SIZE", "5")),
        poll_seconds=float(os.getenv("MCR_SUMMARY_POLL_SECONDS", "1.0")),
        max_attempts=int(os.getenv("MCR_SUMMARY_MAX_ATTEMPTS", "3")),
        max_tokens=int(os.getenv("MCR_SUMMARY_MAX_TOKENS", "1024")),
    )
    session_config = SessionIngestionConfig.from_environment()

    async def build_workspace(record):
        qdrant_client = AsyncQdrantClient(path=record.qdrant_path)
        qdrant_config = QdrantConfig(
            collection_name=record.collection_name,
            vector_size=dimension,
            embedding_model=embedding.model_id,
        )
        repository = QdrantMemoryRepository(
            qdrant_client, qdrant_config, embedding
        )
        retriever = UnifiedMemoryRetriever(
            qdrant_client, qdrant_config, embedding
        )
        source_store = SQLiteSourceStore(record.sqlite_path)
        summary_coordinator = SummaryCoordinator(source_store, summary_config)
        summary_worker = SummaryWorker(
            source_store,
            SessionSummarizer(summary_llm),
            summary_config,
        )
        ingestion = MemoryIngestionService(
            source_store,
            MemoryExtractor(
                llm,
                recent_window_size=session_config.recent_window_size,
                description_llm=llm,
            ),
            embedding,
            retriever,
            repository,
            MemoryJudge(llm),
            summary_coordinator=summary_coordinator,
        )
        lifecycle = AgentMemoryLifecycle(
            source_store,
            ingestion,
            retriever,
            query_aware_integrator=QueryAwareMemoryContextIntegrator(llm, source_store),
            summary_coordinator=summary_coordinator,
            recent_window_size=session_config.recent_window_size,
            workspace_id=record.id,
        )
        session_ingestion = SessionIngestionService(
            source_store,
            ingestion,
            summary_coordinator=summary_coordinator,
            config=session_config,
            lifecycle_service=lifecycle,
        )
        runtime = WorkspaceRuntime(
            record=record,
            source_store=source_store,
            qdrant_client=qdrant_client,
            repository=repository,
            retriever=retriever,
            ingestion=ingestion,
            lifecycle=lifecycle,
            session_ingestion=session_ingestion,
            summary_coordinator=summary_coordinator,
            summary_worker=summary_worker,
        )
        await runtime.start()
        return runtime

    workspace_manager = WorkspaceManager(catalog, build_workspace)
    return create_app(
        llm,
        provider=backend,
        model=model,
        workspace_manager=workspace_manager,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MCR extraction workbench")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="loopback port; defaults to an OS-assigned free port",
    )
    args = parser.parse_args()
    sock = bind_server_socket(args.port)
    actual_port = sock.getsockname()[1]
    print(f"MCR debug server running at http://{LOOPBACK_HOST}:{actual_port}", flush=True)
    config = uvicorn.Config(
        build_app_from_environment(),
        log_level="info",
        access_log=True,
    )
    uvicorn.Server(config).run(sockets=[sock])


if __name__ == "__main__":
    main()

