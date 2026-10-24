"""Two-stage memory lifecycle for external Agents."""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone

from mcr.agent_lifecycle.models import (
    AfterGenerationRequest,
    AfterGenerationResult,
    AssignedLifecycleMessage,
    BeforeGenerationRequest,
    BeforeGenerationResult,
    BeforeGenerationTrace,
    ExchangeIngestionResult,
    LifecycleContextTrace,
)
from mcr.context_integration import (
    CriticalIntegrationError,
    DefaultMemoryContextIntegrator,
    MemoryContextIntegrationError,
    MemoryContextIntegrator,
    MemoryIntegrationRequest,
)
from mcr.extraction import ConversationMessage
from mcr.ingestion import IngestionRequest, MemoryIngestionService
from mcr.llm import ContentPart, ImagePart, TextPart
from mcr.memory import ImageSource
from mcr.retrieval import RetrievalConfig, RetrievalQuery, UnifiedMemoryRetriever
from mcr.source_store import MessageDraft, SQLiteSourceStore, StoredMessage
from mcr.summarization import SummaryCoordinator


class AgentMemoryLifecycle:
    def __init__(
        self,
        source_store: SQLiteSourceStore,
        ingestion_service: MemoryIngestionService,
        retriever: UnifiedMemoryRetriever,
        *,
        integrator: MemoryContextIntegrator | None = None,
        query_aware_integrator: MemoryContextIntegrator | None = None,
        summary_coordinator: SummaryCoordinator | None = None,
        recent_window_size: int = 8,
        workspace_id: str = "default",
    ) -> None:
        if recent_window_size < 0:
            raise ValueError("recent_window_size must be non-negative")
        self._sources = source_store
        self._ingestion = ingestion_service
        self._retriever = retriever
        self._integrator = integrator or DefaultMemoryContextIntegrator()
        self._query_aware_integrator = query_aware_integrator
        self._summaries = summary_coordinator
        self.recent_window_size = recent_window_size
        self.workspace_id = workspace_id

    async def before_generation(
        self, request: BeforeGenerationRequest
    ) -> BeforeGenerationResult:
        observed_at = datetime.now(timezone.utc)
        exchange, user, created = await self._sources.begin_agent_exchange(
            session_id=request.session_id,
            request_id=request.request_id,
            user_content=request.user_content,
            created_at=observed_at,
        )
        if not created and exchange.before_result is not None:
            return BeforeGenerationResult.model_validate(exchange.before_result)

        current_user = self._to_conversation(user)
        trace = BeforeGenerationTrace()
        selected = []
        context_text = ""
        integrated = None
        try:
            query = self._retrieval_query(current_user)
            retrieved = await self._retriever.search(
                query,
                RetrievalConfig(
                    result_top_k=request.top_k,
                    score_threshold=request.score_threshold,
                ),
            )
            trace.retrieved = retrieved
            integrator = (
                self._query_aware_integrator
                if request.integration_strategy == "query_aware"
                else self._integrator
            )
            if integrator is None:
                raise CriticalIntegrationError("query-aware integration is not configured")
            integrated = await integrator.integrate(
                MemoryIntegrationRequest(
                    workspace_id=self.workspace_id,
                    session_id=request.session_id,
                    exchange_id=exchange.id,
                    current_user=current_user,
                    retrieved_memories=retrieved,
                )
            )
            selected = self._select_integrated_memories(retrieved, integrated.memory_ids)
            context_text = integrated.context_text
            trace.integration_metadata = integrated.metadata
            trace.integration_prompt_text = integrated.prompt_text
        except CriticalIntegrationError:
            raise
        except Exception as exc:
            trace.errors.append({"type": type(exc).__name__, "message": str(exc)})

        result = BeforeGenerationResult(
            exchange_id=exchange.id,
            request_id=request.request_id,
            workspace_id=self.workspace_id,
            user_message=self._assigned(user),
            memories=selected,
            context_text=context_text,
            integration=integrated,
            trace=trace,
        )
        self._sources.save_exchange_before_result(
            exchange.id, result.model_dump(mode="json")
        )
        return result

    async def after_generation(
        self, request: AfterGenerationRequest
    ) -> AfterGenerationResult:
        existing = self._sources.get_agent_exchange(request.exchange_id)
        if existing.status == "completed" and existing.after_result is not None:
            # Validate content identity even on a completed replay.
            await self._sources.attach_exchange_assistant(
                request.exchange_id, request.assistant_content, existing.updated_at
            )
            return AfterGenerationResult.model_validate(existing.after_result)

        observed_at = datetime.now(timezone.utc)
        try:
            exchange, assistant = await self._sources.attach_exchange_assistant(
                request.exchange_id,
                request.assistant_content,
                observed_at,
            )
            user = self._sources.get_message(exchange.user_message_id)
            current = [user] + ([assistant] if assistant is not None else [])
            recent = self._sources.get_recent_messages(
                exchange.session_id,
                limit=self.recent_window_size,
                exclude_ids=[message.id for message in current],
            )
            summary = (
                self._summaries.get_summary(exchange.session_id)
                if self._summaries is not None
                else None
            )
            ingestion = await self._ingestion.ingest(
                IngestionRequest(
                    session_id=exchange.session_id,
                    current_messages=[self._to_draft(message) for message in current],
                    recent_messages=[self._to_conversation(message) for message in recent],
                    observation_time=observed_at,
                )
            )
            result = AfterGenerationResult(
                exchange_id=exchange.id,
                workspace_id=self.workspace_id,
                assigned_messages=[self._assigned(message) for message in current],
                context=LifecycleContextTrace(
                    summary_version=summary.version if summary is not None else None,
                    recent_message_ids=[message.id for message in recent],
                ),
                result=ingestion,
            )
            self._sources.complete_agent_exchange(
                exchange.id, result.model_dump(mode="json")
            )
            return result
        except Exception as exc:
            self._sources.fail_agent_exchange(
                request.exchange_id,
                {"type": type(exc).__name__, "message": str(exc)},
            )
            raise

    async def ingest_exchange(
        self,
        *,
        session_id: str,
        user: list[ContentPart],
        assistant: list[ContentPart] | None = None,
        request_id=None,
        top_k: int = 20,
        score_threshold: float | None = None,
    ) -> ExchangeIngestionResult:
        before_payload = {
            "session_id": session_id,
            "user_content": user,
            "top_k": top_k,
            "score_threshold": score_threshold,
        }
        if request_id is not None:
            before_payload["request_id"] = request_id
        before = await self.before_generation(BeforeGenerationRequest(**before_payload))
        after = await self.after_generation(
            AfterGenerationRequest(
                exchange_id=before.exchange_id,
                assistant_content=assistant,
            )
        )
        return ExchangeIngestionResult(before=before, after=after)

    def ingest_exchange_sync(self, **kwargs) -> ExchangeIngestionResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ingest_exchange(**kwargs))
        raise RuntimeError(
            "ingest_exchange_sync cannot run inside an active event loop; "
            "await ingest_exchange instead"
        )

    @staticmethod
    def _select_integrated_memories(retrieved, memory_ids):
        if len(memory_ids) != len(set(memory_ids)):
            raise MemoryContextIntegrationError(
                "integrator returned duplicate memory IDs"
            )
        by_id = {item.memory.id: item for item in retrieved}
        unknown = [memory_id for memory_id in memory_ids if memory_id not in by_id]
        if unknown:
            raise MemoryContextIntegrationError(
                f"integrator returned memory IDs outside retrieval: {unknown}"
            )
        return [by_id[memory_id] for memory_id in memory_ids]

    @staticmethod
    def _retrieval_query(message: ConversationMessage) -> RetrievalQuery:
        texts = [part.text for part in message.content if isinstance(part, TextPart)]
        images = [
            ImageSource(url=part.url, file_path=part.file_path, data_url=part.data_url)
            for part in message.content
            if isinstance(part, ImagePart)
        ]
        return RetrievalQuery(texts=texts, images=images)

    @staticmethod
    def _assigned(message: StoredMessage) -> AssignedLifecycleMessage:
        return AssignedLifecycleMessage(
            id=message.id, role=message.role, turn_id=message.turn_id
        )

    @classmethod
    def _to_draft(cls, message: StoredMessage) -> MessageDraft:
        converted = cls._to_conversation(message)
        return MessageDraft(
            role=message.role,
            turn_id=message.turn_id,
            content=converted.content,
            created_at=message.created_at,
        )

    @staticmethod
    def _to_conversation(message: StoredMessage) -> ConversationMessage:
        content = []
        for part in message.parts:
            if part.part_type == "text":
                content.append(TextPart(text=part.text_content or ""))
                continue
            encoded = base64.b64encode(part.image_bytes or b"").decode("ascii")
            content.append(
                ImagePart(
                    data_url=f"data:{part.media_type};base64,{encoded}",
                    media_type=part.media_type,
                )
            )
        return ConversationMessage(
            id=str(message.id),
            role=message.role,
            content=content,
            created_at=message.created_at,
            turn_id=message.turn_id,
        )

