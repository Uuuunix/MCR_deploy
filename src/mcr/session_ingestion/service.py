"""High-level session workflow around the existing ingestion pipeline."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from mcr.extraction import ConversationMessage
from mcr.ingestion import IngestionRequest, MemoryIngestionService
from mcr.llm import ImagePart, TextPart
from mcr.session_ingestion.config import SessionIngestionConfig
from mcr.session_ingestion.errors import SubmissionInProgress
from mcr.session_ingestion.models import (
    AssignedMessage,
    SessionContextTrace,
    SessionIngestionRequest,
    SessionIngestionResult,
)
from mcr.source_store import MessageDraft, SQLiteSourceStore, StoredMessage
from mcr.summarization import SummaryCoordinator

if TYPE_CHECKING:
    from mcr.agent_lifecycle import AgentMemoryLifecycle


class SessionIngestionService:
    def __init__(
        self,
        source_store: SQLiteSourceStore,
        ingestion_service: MemoryIngestionService,
        *,
        summary_coordinator: SummaryCoordinator | None = None,
        config: SessionIngestionConfig | None = None,
        lifecycle_service: AgentMemoryLifecycle | None = None,
    ) -> None:
        self._sources = source_store
        self._ingestion = ingestion_service
        self._summaries = summary_coordinator
        self.config = config or SessionIngestionConfig()
        self._lifecycle = lifecycle_service

    def get_progress(self, session_id: str) -> dict | None:
        return self._sources.latest_ingestion_progress(session_id)

    async def ingest(self, request: SessionIngestionRequest) -> SessionIngestionResult:
        if self._lifecycle is not None:
            exchange = await self._lifecycle.ingest_exchange(
                session_id=request.session_id,
                user=request.user_content,
                assistant=request.assistant_content,
                request_id=request.submission_id,
            )
            after = exchange.after
            return SessionIngestionResult(
                submission_id=request.submission_id,
                assigned_messages=[
                    AssignedMessage(
                        id=message.id,
                        role=message.role,
                        turn_id=message.turn_id,
                    )
                    for message in after.assigned_messages
                ],
                context=SessionContextTrace(
                    summary_version=after.context.summary_version,
                    recent_message_ids=after.context.recent_message_ids,
                ),
                result=after.result,
            )
        request_hash = self._request_hash(request)
        worker_token = uuid4()
        observed_at = datetime.now(timezone.utc)
        messages = [("user", request.user_content)]
        if request.assistant_content is not None:
            messages.append(("assistant", request.assistant_content))
        submission, stored = await self._sources.begin_session_submission(
            session_id=request.session_id,
            submission_id=request.submission_id,
            request_hash=request_hash,
            messages=messages,
            created_at=observed_at,
            worker_token=worker_token,
            lease_seconds=self.config.lease_seconds,
        )
        if submission.status == "completed" and submission.result is not None:
            return SessionIngestionResult.model_validate(submission.result)
        if not submission.claimed:
            raise SubmissionInProgress(str(request.submission_id))

        heartbeat = asyncio.create_task(
            self._heartbeat(request.session_id, request.submission_id, worker_token)
        )
        try:
            recent = self._sources.get_recent_messages(
                request.session_id,
                limit=self.config.recent_window_size,
                exclude_ids=[message.id for message in stored],
            )
            summary = (
                self._summaries.get_summary(request.session_id)
                if self._summaries is not None
                else None
            )
            drafts = [
                MessageDraft(
                    role=message.role,
                    turn_id=message.turn_id,
                    content=(
                        request.user_content
                        if message.role == "user"
                        else request.assistant_content or []
                    ),
                    created_at=message.created_at,
                )
                for message in stored
            ]
            identity = "|".join(
                f"{message.role}:{message.turn_id}" for message in drafts
            )
            ingestion_hash = hashlib.sha256(
                "|".join(message.content_hash for message in stored).encode()
            ).hexdigest()
            run = self._sources.get_or_create_run(
                f"{request.session_id}:{identity}",
                request.session_id,
                ingestion_hash,
            )
            self._sources.attach_submission_run(
                request.session_id,
                request.submission_id,
                worker_token,
                run.id,
            )
            result = await self._ingestion.ingest(
                IngestionRequest(
                    session_id=request.session_id,
                    current_messages=drafts,
                    recent_messages=[self._to_conversation(message) for message in recent],
                    observation_time=observed_at,
                )
            )
            if result.operation_id != run.id:
                raise RuntimeError("ingestion service returned an unexpected operation ID")
            response = SessionIngestionResult(
                submission_id=request.submission_id,
                assigned_messages=[
                    AssignedMessage(
                        id=message.id,
                        role=message.role,
                        turn_id=message.turn_id,
                    )
                    for message in stored
                ],
                context=SessionContextTrace(
                    summary_version=summary.version if summary is not None else None,
                    recent_message_ids=[message.id for message in recent],
                ),
                result=result,
            )
            self._sources.complete_session_submission(
                request.session_id,
                request.submission_id,
                worker_token,
                response.model_dump(mode="json"),
            )
            return response
        except Exception as exc:
            self._sources.fail_session_submission(
                request.session_id,
                request.submission_id,
                worker_token,
                {"type": type(exc).__name__, "message": str(exc)},
            )
            raise
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass

    async def _heartbeat(
        self, session_id: str, submission_id: UUID, worker_token: UUID
    ) -> None:
        while True:
            await asyncio.sleep(self.config.heartbeat_seconds)
            if not self._sources.heartbeat_session_submission(
                session_id,
                submission_id,
                worker_token,
                self.config.lease_seconds,
            ):
                return

    @staticmethod
    def _request_hash(request: SessionIngestionRequest) -> str:
        payload = request.model_dump(mode="json", exclude={"submission_id"})
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    @staticmethod
    def _to_conversation(message: StoredMessage) -> ConversationMessage:
        content = []
        for part in message.parts:
            if part.part_type == "text":
                content.append(TextPart(text=part.text_content or ""))
                continue
            if not part.image_bytes or not part.media_type:
                raise ValueError(
                    f"stored image is unavailable: message={message.id} part={part.part_index}"
                )
            encoded = base64.b64encode(part.image_bytes).decode("ascii")
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
