"""End-to-end extraction, admission, linking, and persistence orchestration."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from copy import deepcopy
from datetime import timezone
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from mcr.embedding import MultimodalEmbeddingClient
from mcr.extraction import (
    CandidateMemory,
    ConversationMessage,
    ExtractionRequest,
    ExtractionResult,
    ExtractionTrace,
    MemoryExtractor,
    build_extraction_prompt,
)
from mcr.ingestion.errors import IngestionError, IngestionPersistenceError, JudgmentError, RepairRequiredError
from mcr.ingestion.additive import additive_judgment
from mcr.ingestion.judgment import JudgmentResult, MemoryJudge
from mcr.ingestion.models import (
    CandidateOutcome,
    IngestionRequest,
    IngestionResult,
    IngestionTrace,
    JudgmentPayload,
    MemoryDecision,
    PendingMemory,
)
from mcr.llm import ImagePart, TextPart
from mcr.memory import (
    ImageSource,
    MemoryItem,
    MemoryLink,
    MemoryMetadata,
    RetrievalDescription,
    memory_relation_id,
)
from mcr.qdrant import QdrantMemoryRepository
from mcr.retrieval import (
    EmbeddedRetrievalQuery,
    RetrievalConfig,
    RetrievedMemory,
    UnifiedMemoryRetriever,
)
from mcr.source_store import SQLiteSourceStore, SourceStoreError, StoredMessage
from mcr.summarization import SummaryCoordinator


class MemoryIngestionService:
    def __init__(
        self,
        source_store: SQLiteSourceStore,
        extractor: MemoryExtractor,
        embedder: MultimodalEmbeddingClient,
        retriever: UnifiedMemoryRetriever,
        repository: QdrantMemoryRepository,
        judge: MemoryJudge,
        summary_coordinator: SummaryCoordinator | None = None,
        related_memory_top_k: int = 10,
        write_lock_timeout_seconds: float = 30.0,
        judgment_batch_max_candidates: int = 5,
        judgment_batch_max_chars: int = 16000,
        admission_strategy: Literal["additive", "legacy"] = "legacy",
    ) -> None:
        if related_memory_top_k <= 0:
            raise ValueError("related_memory_top_k must be positive")
        if judgment_batch_max_candidates <= 0 or judgment_batch_max_chars <= 0:
            raise ValueError("judgment batch limits must be positive")
        self._sources = source_store
        self._extractor = extractor
        self._embedder = embedder
        self._retriever = retriever
        self._repository = repository
        self._judge = judge
        self._summaries = summary_coordinator
        self.related_memory_top_k = related_memory_top_k
        self.write_lock_timeout_seconds = write_lock_timeout_seconds
        self.judgment_batch_max_candidates = judgment_batch_max_candidates
        self.judgment_batch_max_chars = judgment_batch_max_chars
        self.admission_strategy = admission_strategy

    async def ingest(self, request: IngestionRequest) -> IngestionResult:
        identity = "|".join(
            f"{message.role}:{message.turn_id}" for message in request.current_messages
        )
        idempotency_key = f"{request.session_id}:{identity}"

        stored = await self._sources.save_messages(
            request.session_id, request.current_messages
        )
        request_hash = hashlib.sha256(
            "|".join(message.content_hash for message in stored).encode()
        ).hexdigest()
        run = self._sources.get_or_create_run(
            idempotency_key, request.session_id, request_hash
        )
        if self._summaries is not None:
            stored_summary = self._summaries.get_summary(request.session_id)
            summary_text = stored_summary.text if stored_summary is not None else None
            self._summaries.register_submission(
                run.id,
                request.session_id,
                [message.id for message in stored],
            )
        else:
            summary_text = request.summary
        if run.status == "complete" and run.result is not None:
            return IngestionResult.model_validate(run.result)

        try:
            self._sources.update_run(
                run.id, status="running", stage="messages_saved"
            )
            extraction_request = self._extraction_request(
                request, stored, summary=summary_text
            )
            checkpoint = run.checkpoint
            strategy = (
                checkpoint.get("admission_strategy", "legacy")
                if checkpoint is not None else self.admission_strategy
            )
            if checkpoint is not None:
                prompt_messages, recent = build_extraction_prompt(
                    extraction_request, self._extractor.recent_window_size
                )
                extraction = ExtractionResult(
                    candidates=[
                        CandidateMemory.model_validate(item)
                        for item in checkpoint["candidates"]
                    ],
                    trace=ExtractionTrace(
                        messages=prompt_messages,
                        effective_recent_message_ids=[item.id for item in recent],
                        **checkpoint["trace"],
                    ),
                )
            else:
                extraction = await self._extractor.extract_with_trace(extraction_request)
                checkpoint_trace = extraction.trace.model_dump(
                    mode="json", exclude={"messages", "effective_recent_message_ids"}
                )
                self._redact_data_urls(checkpoint_trace)
                checkpoint = {
                    "candidates": [
                        item.model_dump(mode="json") for item in extraction.candidates
                    ],
                    "trace": checkpoint_trace,
                    "completed_batches": [],
                    "prepared_batch": None,
                    "admission_strategy": strategy,
                    "existing_memories": [
                        item.model_dump(mode="json")
                        for item in extraction_request.existing_memories
                    ],
                }
                self._sources.save_run_checkpoint(run.id, checkpoint)
            self._hydrate_image_relations(extraction.candidates, stored)
            memories = self._build_memories(run.id, extraction.candidates)

            extraction_dump = extraction.model_dump(mode="json")
            self._redact_data_urls(extraction_dump)
            trace = IngestionTrace(
                extraction=extraction_dump,
                judgment_mode="additive" if strategy == "additive" else "deterministic",
            )
            if not memories:
                result = IngestionResult(
                    operation_id=run.id,
                    message_ids=[message.id for message in stored],
                    mode=trace.judgment_mode,
                    outcomes=[],
                    trace=trace,
                )
                self._complete_run(run.id, result)
                return result

            # Old additive checkpoints may already contain a fully persisted
            # exchange. New admissions always retrieve per candidate and use
            # the LLM judge after extraction, regardless of the stored name.
            completed = (checkpoint or {}).get("completed_batches") or []
            if strategy == "additive" and completed and "trace" not in completed[0]:
                outcomes = await self._ingest_additive(
                    run.id, extraction.candidates, memories, checkpoint or {}, trace
                )
            else:
                outcomes = await self._ingest_batches(
                    run.id, memories, checkpoint or {}, trace
                )
            result = IngestionResult(
                operation_id=run.id,
                message_ids=[message.id for message in stored],
                mode=trace.judgment_mode,
                outcomes=outcomes,
                trace=trace,
            )
            self._complete_run(run.id, result)
            return result
        except Exception as exc:
            self._sources.update_run(
                run.id,
                status="repair_required" if isinstance(exc, RepairRequiredError) else "failed",
                stage="failed",
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise

    async def _ingest_additive(
        self,
        run_id: UUID,
        candidates: list[CandidateMemory],
        memories: list[PendingMemory],
        checkpoint: dict,
        trace: IngestionTrace,
    ) -> list[CandidateOutcome]:
        completed = checkpoint.get("completed_batches") or []
        if completed:
            trace.judgment_mode = "additive"
            return [
                CandidateOutcome.model_validate(item)
                for item in completed[0]["outcomes"]
            ]

        existing_by_text = {
            " ".join(item["text"].casefold().split()): UUID(item["id"])
            for item in checkpoint.get("existing_memories", [])
        }
        kept = []
        kept_candidates = []
        seen_texts = set()
        for candidate, memory in zip(candidates, memories, strict=True):
            key = " ".join(memory.x.content.casefold().split())
            if key in existing_by_text or key in seen_texts:
                continue
            seen_texts.add(key)
            kept.append(memory)
            kept_candidates.append(candidate)

        if not kept:
            trace.judgment_mode = "additive"
            checkpoint["completed_batches"] = [{"start": 0, "end": len(memories), "outcomes": []}]
            self._sources.save_run_checkpoint(run_id, checkpoint)
            return []

        # D calls are independent. Bound concurrency to avoid flooding a shared
        # local model when one exchange contains many distinct memories.
        semaphore = asyncio.Semaphore(5)

        async def generate(memory):
            async with semaphore:
                return await self._extractor.generate_descriptions(memory.x)

        generated = await asyncio.gather(*(generate(memory) for memory in kept))
        for index, (memory, description_result) in enumerate(zip(kept, generated, strict=True)):
            memory.descriptions = description_result.descriptions
            trace.description_generation.append({
                "candidate": f"candidate-{index}",
                "raw_response": description_result.raw_response,
                "provider_request": description_result.provider_request,
                "elapsed_ms": description_result.elapsed_ms,
                "error": description_result.error,
            })
            self._assign_description_ids(memory)

        persisted = []
        for memory in kept:
            value = memory.model_dump(mode="python", exclude={"relations"})
            value["relations"] = memory.relations
            persisted.append(MemoryItem.model_validate(value))
        visual_vectors = await self._embed_candidates(persisted)
        self._sources.update_run(run_id, status="running", stage="d_embedded")
        judgment = additive_judgment(kept_candidates)
        outcomes = await self._persist(
            run_id, persisted, visual_vectors, {}, judgment, trace
        )
        trace.judgment_mode = "additive"
        trace.batches.append({
            "start": 0, "end": len(memories), "judgment_mode": "additive",
            "outcomes": [item.model_dump(mode="json") for item in outcomes],
        })
        checkpoint["completed_batches"] = [{
            "start": 0, "end": len(memories),
            "outcomes": [item.model_dump(mode="json") for item in outcomes],
        }]
        self._sources.save_run_checkpoint(run_id, checkpoint)
        return outcomes

    def _complete_run(self, run_id: UUID, result: IngestionResult) -> None:
        self._sources.update_run(
            run_id,
            status="complete",
            stage="persisted",
            result=result.model_dump(mode="json"),
        )

    async def _ingest_batches(
        self,
        run_id: UUID,
        memories: list[PendingMemory],
        checkpoint: dict,
        trace: IngestionTrace,
    ) -> list[CandidateOutcome]:
        completed = checkpoint.setdefault("completed_batches", [])
        outcomes = []
        cursor = 0
        for batch in completed:
            if batch["start"] != cursor or batch["end"] > len(memories):
                raise IngestionError("batch checkpoint is not contiguous")
            outcomes.extend(CandidateOutcome.model_validate(item) for item in batch["outcomes"])
            self._append_batch_trace(trace, batch["trace"])
            cursor = batch["end"]

        if cursor == len(memories):
            trace.judgment_mode = "serial_batches" if len(completed) > 1 else (
                completed[0]["trace"]["judgment_mode"] if completed else "deterministic"
            )
            return outcomes

        x_vectors = await self._embedder.embed_texts(
            [memory.x.content for memory in memories]
        )
        if len(x_vectors) != len(memories):
            raise IngestionError("embedding client returned the wrong candidate vector count")
        self._sources.update_run(run_id, status="running", stage="x_embedded")

        max_size = self.judgment_batch_max_candidates
        single_failures = 0
        while cursor < len(memories):
            prepared = checkpoint.get("prepared_batch")
            if prepared is not None:
                if prepared["start"] != cursor or prepared["end"] > len(memories):
                    raise IngestionError("prepared batch checkpoint has an invalid range")
                batch_memories = memories[cursor:prepared["end"]]
                for memory, descriptions in zip(
                    batch_memories, prepared["descriptions"], strict=True
                ):
                    memory.descriptions = [
                        RetrievalDescription.model_validate(item) for item in descriptions
                    ]
                judgment = JudgmentResult(
                    payload=JudgmentPayload.model_validate(prepared["judgment"]["payload"]),
                    mode=prepared["judgment"]["mode"],
                    requests=prepared["judgment"]["requests"],
                    responses=prepared["judgment"]["responses"],
                    existing_aliases={
                        alias: UUID(value)
                        for alias, value in prepared["judgment"]["existing_aliases"].items()
                    },
                )
                batch_trace = prepared["trace"]
            else:
                batch_memories, retrieved = await self._select_batch(
                    memories, x_vectors, cursor, max_size
                )
                try:
                    judgment = await self._judge_memories(batch_memories, retrieved)
                    judgment = self._collapse_literal_duplicates(batch_memories, judgment)
                except JudgmentError:
                    if len(batch_memories) > 1:
                        max_size = max(1, len(batch_memories) // 2)
                        continue
                    single_failures += 1
                    if single_failures < 2:
                        continue
                    raise
                single_failures = 0
                batch_trace = {
                    "start": cursor,
                    "end": cursor + len(batch_memories),
                    "retrieved": {
                        f"candidate-{cursor + int(alias.removeprefix('candidate-'))}": [
                            item.model_dump(mode="json") for item in results
                        ]
                        for alias, results in retrieved.items()
                    },
                    "judgment_mode": judgment.mode,
                    "judgment_requests": judgment.requests,
                    "judgment_responses": judgment.responses,
                    "description_generation": [],
                    "persistence_steps": [],
                }
                decisions = {
                    decision.candidate: decision
                    for decision in judgment.payload.decisions
                }
                missing = [
                    (index, memory.x)
                    for index, memory in enumerate(batch_memories)
                    if decisions[f"candidate-{index}"].action == "ADD"
                    and not memory.descriptions
                ]
                if missing:
                    generated = await asyncio.gather(
                        *(self._extractor.generate_descriptions(x) for _, x in missing)
                    )
                    for (index, _), result in zip(missing, generated, strict=True):
                        batch_memories[index].descriptions = result.descriptions
                        batch_trace["description_generation"].append({
                            "candidate": f"candidate-{cursor + index}",
                            "raw_response": result.raw_response,
                            "provider_request": result.provider_request,
                            "elapsed_ms": result.elapsed_ms,
                            "error": result.error,
                        })
                prepared = {
                    "start": cursor,
                    "end": cursor + len(batch_memories),
                    "descriptions": [
                        [item.model_dump(mode="json") for item in memory.descriptions]
                        for memory in batch_memories
                    ],
                    "judgment": {
                        "payload": judgment.payload.model_dump(mode="json"),
                        "mode": judgment.mode,
                        "requests": judgment.requests,
                        "responses": judgment.responses,
                        "existing_aliases": {
                            alias: str(memory_id)
                            for alias, memory_id in judgment.existing_aliases.items()
                        },
                    },
                    "trace": batch_trace,
                }
                checkpoint["prepared_batch"] = prepared
                self._sources.save_run_checkpoint(run_id, checkpoint)

            decisions = {
                decision.candidate: decision for decision in judgment.payload.decisions
            }
            add_memories = []
            for index, memory in enumerate(batch_memories):
                if decisions[f"candidate-{index}"].action != "ADD":
                    continue
                self._assign_description_ids(memory)
                value = memory.model_dump(mode="python", exclude={"relations"})
                value["relations"] = memory.relations
                persisted = MemoryItem.model_validate(value)
                batch_memories[index] = persisted
                add_memories.append(persisted)

            visual_vectors = await self._embed_candidates(add_memories)
            self._sources.update_run(run_id, status="running", stage="d_embedded")
            batch_persistence_trace = IngestionTrace(judgment_mode=judgment.mode)
            batch_outcomes = await self._persist(
                run_id, batch_memories, visual_vectors, {}, judgment,
                batch_persistence_trace, candidate_offset=cursor,
            )
            batch_trace["persistence_steps"] = batch_persistence_trace.persistence_steps
            batch_trace["outcomes"] = [
                item.model_dump(mode="json") for item in batch_outcomes
            ]
            completed.append({
                "start": cursor,
                "end": prepared["end"],
                "outcomes": batch_trace["outcomes"],
                "trace": batch_trace,
            })
            checkpoint["prepared_batch"] = None
            self._sources.save_run_checkpoint(run_id, checkpoint)
            outcomes.extend(batch_outcomes)
            self._append_batch_trace(trace, batch_trace)
            cursor = prepared["end"]
            max_size = self.judgment_batch_max_candidates
            self._sources.update_run(
                run_id, status="running", stage=f"batch_{cursor}_persisted"
            )

        trace.judgment_mode = "serial_batches" if len(completed) > 1 else (
            completed[0]["trace"]["judgment_mode"] if completed else "deterministic"
        )
        return outcomes

    async def _select_batch(self, memories, x_vectors, start: int, max_size: int):
        selected = []
        retrieved = {}
        seen_existing = set()
        estimated_chars = 1000
        for index in range(start, min(len(memories), start + max_size)):
            results = await self._retriever.search_vectors(
                EmbeddedRetrievalQuery(text_vectors=[x_vectors[index]], vision_vectors=[]),
                RetrievalConfig(result_top_k=self.related_memory_top_k),
            )
            additional = 500 + len(memories[index].x.content)
            for result in results:
                if result.memory.id not in seen_existing:
                    additional += 350 + len(result.memory.x.content)
                    additional += sum(len(item.text) for item in result.memory.descriptions)
            if selected and estimated_chars + additional > self.judgment_batch_max_chars:
                break
            selected.append(memories[index])
            retrieved[f"candidate-{len(selected) - 1}"] = results
            estimated_chars += additional
            seen_existing.update(item.memory.id for item in results)
        return selected, retrieved

    @staticmethod
    def _append_batch_trace(trace: IngestionTrace, batch: dict) -> None:
        trace.batches.append({
            "start": batch["start"],
            "end": batch["end"],
            "judgment_mode": batch["judgment_mode"],
            "outcomes": batch.get("outcomes"),
        })
        trace.retrieved.update(batch.get("retrieved", {}))
        trace.judgment_requests.extend(batch.get("judgment_requests", []))
        trace.judgment_responses.extend(batch.get("judgment_responses", []))
        trace.description_generation.extend(batch.get("description_generation", []))
        trace.persistence_steps.extend(batch.get("persistence_steps", []))

    @staticmethod
    def _redact_data_urls(value) -> None:
        if isinstance(value, dict):
            for key, item in list(value.items()):
                if isinstance(item, str) and item.startswith("data:image/"):
                    value[key] = "[image data URL omitted]"
                else:
                    MemoryIngestionService._redact_data_urls(item)
        elif isinstance(value, list):
            for item in value:
                MemoryIngestionService._redact_data_urls(item)

    @staticmethod
    def _stored_to_conversation(message: StoredMessage) -> ConversationMessage:
        content = []
        for part in message.parts:
            if part.part_type == "text":
                content.append(TextPart(text=part.text_content or ""))
            else:
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

    def _extraction_request(
        self,
        request: IngestionRequest,
        stored: list[StoredMessage],
        *,
        summary: str | None,
    ) -> ExtractionRequest:
        current = [self._stored_to_conversation(message) for message in stored]
        return ExtractionRequest(
            session_id=request.session_id,
            summary=summary,
            recent_messages=request.recent_messages,
            current_user=next((item for item in current if item.role == "user"), None),
            current_assistant=next(
                (item for item in current if item.role == "assistant"), None
            ),
            observation_time=request.observation_time,
            custom_instructions=request.custom_instructions,
        )

    @staticmethod
    def _hydrate_image_relations(candidates, stored: list[StoredMessage]) -> None:
        sources = {str(message.id): message for message in stored}
        for candidate in candidates:
            for relation in candidate.relations:
                if relation.source_type != "image":
                    continue
                message = sources.get(relation.source_id)
                part_index = (relation.locator or {}).get("part_index")
                if message is None or not isinstance(part_index, int):
                    raise IngestionError("image relation does not resolve to a current source")
                if part_index < 0 or part_index >= len(message.parts):
                    raise IngestionError("image relation part_index is out of range")
                part = message.parts[part_index]
                if part.part_type != "image" or not part.image_bytes or not part.media_type:
                    raise IngestionError("image relation source bytes are unavailable")
                encoded = base64.b64encode(part.image_bytes).decode("ascii")
                relation.image_source = ImageSource(
                    data_url=f"data:{part.media_type};base64,{encoded}"
                )

    @staticmethod
    def _build_memories(run_id: UUID, candidates) -> list[PendingMemory]:
        memories = []
        for ordinal, candidate in enumerate(candidates):
            if not candidate.relations:
                raise IngestionError("every extracted candidate requires source relations")
            memory_id = uuid5(NAMESPACE_URL, f"mcr:ingestion:{run_id}:candidate:{ordinal}")
            for relation_index, relation in enumerate(candidate.relations):
                relation.id = uuid5(
                    NAMESPACE_URL,
                    f"mcr:{memory_id}:source:{relation.source_id}:{relation.locator}:{relation_index}",
                )
            source_turns = candidate.meta.source_turn_ids
            turn_id = "+".join(str(value) for value in source_turns) or "unknown"
            memories.append(
                PendingMemory(
                    schema_version=2,
                    id=memory_id,
                    x=candidate.x,
                    meta=MemoryMetadata(
                        created_at=candidate.meta.created_at,
                        session_id=candidate.meta.session_id,
                        turn_id=turn_id,
                        extra={
                            "observed_at": candidate.meta.observed_at.isoformat(),
                            "source_message_ids": candidate.meta.source_message_ids,
                            "extraction_provider": candidate.meta.provider,
                            "extraction_model": candidate.meta.model,
                        },
                    ),
                    descriptions=candidate.descriptions,
                    relations=candidate.relations,
                )
            )
        return memories

    @staticmethod
    def _assign_description_ids(memory) -> None:
        for item_index, item in enumerate(memory.descriptions):
            item.id = uuid5(
                NAMESPACE_URL,
                f"mcr:{memory.id}:description:{item_index}:{item.facet_type}",
            )

    @staticmethod
    def _collapse_literal_duplicates(memories, judgment: JudgmentResult) -> JudgmentResult:
        decisions = {
            decision.candidate: decision
            for decision in judgment.payload.decisions
        }
        canonical = {}
        for index, memory in enumerate(memories):
            alias = f"candidate-{index}"
            key = " ".join(memory.x.content.casefold().split())
            if key in canonical:
                decisions[alias] = MemoryDecision(
                    candidate=alias,
                    action="SKIP",
                    duplicate_of=canonical[key],
                )
            else:
                canonical[key] = alias
        return JudgmentResult(
            payload=judgment.payload.model_copy(
                update={"decisions": list(decisions.values())}
            ),
            mode=judgment.mode,
            requests=judgment.requests,
            responses=judgment.responses,
            existing_aliases=judgment.existing_aliases,
        )

    async def _embed_candidates(
        self, memories: list[MemoryItem]
    ) -> dict[UUID, dict[UUID, list[float]]]:
        descriptions = [item for memory in memories for item in memory.descriptions]
        text_vectors = await self._embedder.embed_texts(
            [description.text for description in descriptions]
        )
        if len(text_vectors) != len(descriptions):
            raise IngestionError("embedding client returned the wrong text vector count")
        for description, vector in zip(descriptions, text_vectors, strict=True):
            description.embedding = vector

        unique_images = {}
        for memory in memories:
            for relation in memory.relations:
                if relation.source_type == "image":
                    if relation.image_source is None:
                        raise IngestionError("image relation is missing its stored image source")
                    key = (relation.source_id, json.dumps(relation.locator, sort_keys=True))
                    unique_images.setdefault(key, relation.image_source)
        image_keys = list(unique_images)
        image_vectors = await self._embedder.embed_images(
            [unique_images[key] for key in image_keys]
        )
        if len(image_vectors) != len(image_keys):
            raise IngestionError("embedding client returned the wrong image vector count")
        by_source = dict(zip(image_keys, image_vectors, strict=True))
        result = {}
        for memory in memories:
            result[memory.id] = {}
            for relation in memory.relations:
                if relation.source_type == "image":
                    key = (relation.source_id, json.dumps(relation.locator, sort_keys=True))
                    result[memory.id][relation.id] = by_source[key]
        return result

    async def _judge_memories(
        self,
        memories: list[MemoryItem],
        retrieved: dict[str, list[RetrievedMemory]],
    ) -> JudgmentResult:
        if len(memories) == 1 and not retrieved.get("candidate-0"):
            return JudgmentResult(
                payload=JudgmentPayload(
                    decisions=[{"candidate": "candidate-0", "action": "ADD"}]
                ),
                mode="deterministic",
                requests=[],
                responses=[],
                existing_aliases={},
            )
        return await self._judge.judge(memories, retrieved, allow_fallback=False)

    async def _acquire_write_lock(self, operation_id: UUID) -> None:
        deadline = asyncio.get_running_loop().time() + self.write_lock_timeout_seconds
        while True:
            try:
                self._sources.acquire_write_lock(operation_id)
                return
            except SourceStoreError:
                if asyncio.get_running_loop().time() >= deadline:
                    raise IngestionPersistenceError("timed out waiting for global write lock")
                await asyncio.sleep(0.05)

    async def _persist(
        self,
        operation_id: UUID,
        memories: list[MemoryItem],
        visual_vectors: dict[UUID, dict[UUID, list[float]]],
        retrieved: dict[str, list[RetrievedMemory]],
        judgment: JudgmentResult,
        trace: IngestionTrace,
        candidate_offset: int = 0,
    ) -> list[CandidateOutcome]:
        decisions = {decision.candidate: decision for decision in judgment.payload.decisions}
        # Deterministically collapse literal duplicates even if the judge missed them.
        canonical = {}
        for index, memory in enumerate(memories):
            alias = f"candidate-{index}"
            key = " ".join(memory.x.content.casefold().split())
            if key in canonical:
                decisions[alias] = MemoryDecision(
                    candidate=alias, action="SKIP", duplicate_of=canonical[key]
                )
            else:
                canonical[key] = alias

        alias_ids = {
            f"candidate-{index}": memory.id for index, memory in enumerate(memories)
        }
        alias_ids.update(judgment.existing_aliases)

        def resolve(alias: str, trail: set[str] | None = None) -> UUID:
            trail = set() if trail is None else trail
            if alias in trail:
                raise IngestionError("duplicate_of contains a cycle")
            if alias not in alias_ids:
                raise IngestionError(f"unknown memory alias: {alias}")
            decision = decisions.get(alias)
            if decision is None or decision.action == "ADD" or alias.startswith("existing-"):
                return alias_ids[alias]
            return resolve(decision.duplicate_of or "", trail | {alias})

        add_aliases = {
            alias for alias, decision in decisions.items() if decision.action == "ADD"
        }
        existing_memories = {}
        for memory_id in judgment.existing_aliases.values():
            existing = await self._repository.get_memory(memory_id)
            if existing is None:
                raise IngestionError(f"retrieved memory {memory_id} disappeared")
            existing_memories[memory_id] = deepcopy(existing)

        relation_reasons = {}
        for relation in judgment.payload.relations:
            left = resolve(relation.left)
            right = resolve(relation.right)
            if left == right:
                continue
            if not (relation.left in add_aliases or relation.right in add_aliases):
                continue
            relation_reasons[tuple(sorted((left, right), key=str))] = relation.reason

        memory_by_id = {memory.id: memory for memory in memories}
        mutable = {**existing_memories, **memory_by_id}
        touched_old = set()
        for (left, right), reason in relation_reasons.items():
            if left not in mutable or right not in mutable:
                raise IngestionError("relation target is not available for persistence")
            relation_id = memory_relation_id(left, right)
            existing_link = next(
                (
                    link
                    for link in [*mutable[left].links, *mutable[right].links]
                    if link.relation_id == relation_id
                ),
                None,
            )
            final_reason = existing_link.reason if existing_link else reason
            for source, target in ((left, right), (right, left)):
                if not any(link.target_memory_id == target for link in mutable[source].links):
                    mutable[source].links.append(
                        MemoryLink(
                            relation_id=relation_id,
                            target_memory_id=target,
                            reason=final_reason,
                        )
                    )
                if source in existing_memories:
                    touched_old.add(source)

        add_memories = [
            memory_by_id[alias_ids[alias]]
            for alias in sorted(add_aliases)
        ]
        # A prepared batch can be replayed after its Qdrant writes succeeded
        # but before SQLite recorded completion. Its deterministic IDs make
        # those writes recoverable without creating another memory.
        previously_written = {}
        for memory in add_memories:
            previous = await self._repository.get_memory(memory.id)
            if previous is not None:
                if previous.x != memory.x:
                    raise RepairRequiredError(
                        f"memory {memory.id} has conflicting replay content"
                    )
                previously_written[memory.id] = previous
                desired = {link.target_memory_id: link for link in previous.links}
                desired.update({link.target_memory_id: link for link in memory.links})
                memory.links = list(desired.values())
        before_links = {
            memory_id: deepcopy(existing_memories[memory_id].links)
            for memory_id in touched_old
        }
        before_links.update({
            memory_id: deepcopy(previous.links)
            for memory_id, previous in previously_written.items()
        })

        await self._acquire_write_lock(operation_id)
        written = []
        try:
            trace.persistence_steps.append("write_lock_acquired")
            # Refresh and merge old links only after acquiring the global lock.
            for memory_id in touched_old:
                current = await self._repository.get_memory(memory_id)
                if current is None:
                    raise IngestionPersistenceError(f"related memory {memory_id} disappeared")
                before_links[memory_id] = deepcopy(current.links)
                desired = {
                    link.target_memory_id: link
                    for link in existing_memories[memory_id].links
                }
                desired.update({link.target_memory_id: link for link in current.links})
                existing_memories[memory_id].links = list(desired.values())

            for memory in add_memories:
                if memory.id in previously_written:
                    await self._repository.update_links(memory.id, memory.links)
                else:
                    await self._repository.save(memory, visual_vectors.get(memory.id, {}))
                    written.append(memory.id)
            trace.persistence_steps.append("new_memories_written")
            for memory_id in touched_old:
                await self._repository.update_links(
                    memory_id, existing_memories[memory_id].links
                )
            trace.persistence_steps.append("reverse_links_written")

            for memory in add_memories:
                stored = await self._repository.get_memory(memory.id)
                if stored is None or {link.target_memory_id for link in stored.links} != {
                    link.target_memory_id for link in memory.links
                }:
                    raise IngestionPersistenceError(
                        f"memory {memory.id} failed write verification"
                    )
            for memory_id in touched_old:
                stored = await self._repository.get_memory(memory_id)
                expected = {
                    link.target_memory_id for link in existing_memories[memory_id].links
                }
                if stored is None or not expected.issubset(
                    {link.target_memory_id for link in stored.links}
                ):
                    raise IngestionPersistenceError(
                        f"memory {memory_id} failed reverse-link verification"
                    )
            trace.persistence_steps.append("verified")
        except Exception as exc:
            try:
                for memory_id in written:
                    await self._repository.delete_memory(memory_id)
                for memory_id, links in before_links.items():
                    await self._repository.update_links(memory_id, links)
                trace.persistence_steps.append("rolled_back")
            except Exception as rollback_exc:
                trace.persistence_steps.append("repair_required")
                raise RepairRequiredError(
                    "ingestion failed and compensation could not be verified"
                ) from rollback_exc
            raise IngestionPersistenceError("ingestion persistence was rolled back") from exc
        finally:
            self._sources.release_write_lock(operation_id)

        related = {memory.id: [link.target_memory_id for link in memory.links] for memory in memories}
        outcomes = []
        for index, memory in enumerate(memories):
            alias = f"candidate-{index}"
            decision = decisions[alias]
            final_id = resolve(alias)
            outcome = CandidateOutcome(
                candidate_id=memory.id,
                action=decision.action,
                memory_id=final_id,
                duplicate_of=final_id if decision.action == "SKIP" else None,
                related_memory_ids=related.get(final_id, related.get(memory.id, [])),
            )
            outcomes.append(outcome)
            self._sources.save_candidate(
                operation_id,
                memory.id,
                candidate_offset + index,
                memory.model_dump(mode="json"),
                outcome.model_dump(mode="json"),
                "persisted" if decision.action == "ADD" else "skipped",
            )
        return outcomes
