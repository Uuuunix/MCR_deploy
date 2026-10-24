"""AML Add/Search semantics, idempotency and user isolation."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone

from mcr.ingestion import IngestionRequest
from mcr.retrieval import RetrievalConfig
from mcr.source_store import MessageDraft

from deployment.aml.conversion import (
    MAX_ADD_IMAGE_BYTES,
    MAX_SEARCH_IMAGE_BYTES,
    memory_to_wire,
    message_time,
    retrieval_query,
    source_content,
    stored_to_conversation,
)
from deployment.aml.journal import AddJournal
from deployment.aml.runtime import build_manager
from deployment.aml.schemas import AddRequest, AddResponse, SearchRequest, SearchResponse
from deployment.aml.settings import Settings


def workspace_id(user_id: str) -> str:
    return "u_" + hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:40]


def exchange_units(messages):
    """Preserve order; pair an adjacent user/assistant, otherwise ingest a singleton."""
    units = []
    index = 0
    while index < len(messages):
        if (messages[index][0] == "user" and index + 1 < len(messages)
                and messages[index + 1][0] == "assistant"):
            units.append(messages[index:index + 2])
            index += 2
        else:
            units.append(messages[index:index + 1])
            index += 1
    return units


class AMLService:
    def __init__(self, settings: Settings, *, manager=None, catalog=None, journal=None):
        self.settings = settings
        if manager is None or catalog is None:
            catalog, manager = build_manager(settings)
        self.catalog = catalog
        self.manager = manager
        self.journal = journal or AddJournal(settings.data_root / "add_journal.sqlite3")
        # Qdrant local mode must not be opened by multiple concurrent writers.
        self._add_lock = asyncio.Lock()

    async def add(self, request: AddRequest) -> AddResponse:
        serialized = request.model_dump_json().encode("utf-8")
        if len(serialized) > MAX_ADD_IMAGE_BYTES:
            raise ValueError("Add request exceeds 30 MiB")
        prepared = []
        total_image_bytes = 0
        fallback = datetime.now(timezone.utc)
        for index, message in enumerate(request.messages):
            parts, count = source_content(message.content)
            total_image_bytes += count
            prepared.append((message.role, parts, message_time(
                message, fallback + timedelta(microseconds=index)
            )))
        if total_image_bytes > MAX_ADD_IMAGE_BYTES:
            raise ValueError("Add images exceed the 30 MiB decoded total")
        payload_hash = hashlib.sha256(serialized).hexdigest()
        journal_key = hashlib.sha256(
            f"{request.user_id}\0{request.request_id}".encode("utf-8")
        ).hexdigest()
        units = exchange_units(prepared)
        result = AddResponse(
            request_id=request.request_id, user_id=request.user_id, session_id=request.session_id
        )
        async with self._add_lock:
            entry = self.journal.begin(journal_key, payload_hash)
            if entry.done:
                return result
            user_workspace = workspace_id(request.user_id)
            self.manager.create(user_workspace)
            runtime = await self.manager.open(user_workspace)
            for unit_index in range(entry.completed_units, len(units)):
                unit = units[unit_index]
                turn_id = (
                    "aml-" + hashlib.sha256(request.request_id.encode("utf-8")).hexdigest()[:24]
                    + f"-{unit_index}"
                )
                # A previous failed attempt can have saved this unit's immutable
                # raw messages before extraction failed. They are current, not history.
                recent = [item for item in runtime.source_store.get_recent_messages(
                    request.session_id, limit=self.settings.recent_window_size + 2
                ) if str(item.turn_id) != turn_id]
                if self.settings.recent_window_size:
                    recent = recent[-self.settings.recent_window_size:]
                else:
                    recent = []
                drafts = [
                    MessageDraft(role=role, turn_id=turn_id, content=parts, created_at=when)
                    for role, parts, when in unit
                ]
                await runtime.ingestion.ingest(IngestionRequest(
                    session_id=request.session_id,
                    current_messages=drafts,
                    recent_messages=[stored_to_conversation(item) for item in recent],
                    observation_time=max(when for _, _, when in unit),
                ))
                self.journal.advance(journal_key, unit_index + 1)
            self.journal.advance(journal_key, len(units), done=True)
        return result

    async def search(self, request: SearchRequest) -> SearchResponse:
        query = retrieval_query(request.query)
        user_workspace = workspace_id(request.user_id)
        if not self.manager.exists(user_workspace):
            return SearchResponse(data=[])
        runtime = await self.manager.open(user_workspace)
        results = await runtime.retriever.search(
            query,
            RetrievalConfig(result_top_k=request.top_k, vector_top_k=max(1000, request.top_k * 10)),
        )
        output = []
        image_bytes = 0
        for item in results:
            wire, size = memory_to_wire(item, runtime.source_store)
            image_bytes += size
            if image_bytes > MAX_SEARCH_IMAGE_BYTES:
                raise ValueError("Search response images exceed 30 MiB")
            output.append(wire)
        response = SearchResponse(data=output)
        if len(response.model_dump_json().encode("utf-8")) > MAX_SEARCH_IMAGE_BYTES:
            raise ValueError("Search response exceeds 30 MiB")
        return response

    async def aclose(self) -> None:
        await self.manager.aclose()
        self.catalog.close()
        self.journal.close()
