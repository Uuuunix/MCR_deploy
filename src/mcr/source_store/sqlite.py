"""SQLite implementation of immutable source-message persistence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from mcr.llm import ContentPart, ImagePart, TextPart
from mcr.source_store.errors import ImmutableSourceConflict, SourceStoreError
from mcr.source_store.images import DefaultImageResolver, ImageResolver, ResolvedImage
from mcr.source_store.models import (
    AgentExchangeRecord,
    IngestionRunRecord,
    MessageDraft,
    SessionSubmissionRecord,
    StoredMessage,
    StoredMessagePart,
)
from mcr.source_store.schema import SCHEMA

if TYPE_CHECKING:
    from mcr.summarization.models import SessionSummary, SummaryJob, SummaryStatus


class SQLiteSourceStore:
    def __init__(
        self,
        path: str | Path,
        image_resolver: ImageResolver | None = None,
    ) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._resolver = image_resolver or DefaultImageResolver()
        self._lock = threading.RLock()
        self._write_connections: dict[UUID, sqlite3.Connection] = {}
        self._memory_write_owner: UUID | None = None
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.executescript(SCHEMA)
        columns = {
            row[1] for row in self._connection.execute("PRAGMA table_info(messages)")
        }
        if "input_hash" not in columns:
            self._connection.execute("ALTER TABLE messages ADD COLUMN input_hash TEXT")
        run_columns = {
            row[1]
            for row in self._connection.execute("PRAGMA table_info(ingestion_runs)")
        }
        if "checkpoint_json" not in run_columns:
            self._connection.execute(
                "ALTER TABLE ingestion_runs ADD COLUMN checkpoint_json TEXT"
            )
        self._connection.commit()

    @staticmethod
    def _input_hash(content: list[ContentPart]) -> str:
        return hashlib.sha256(
            json.dumps(
                [part.model_dump(mode="json") for part in content],
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()

    async def _materialize_content(self, content: list[ContentPart]) -> list[tuple]:
        parts = []
        for index, part in enumerate(content):
            if isinstance(part, TextPart):
                data = part.text.encode("utf-8")
                parts.append((index, "text", part.text, None, None, data, {}))
            elif isinstance(part, ImagePart):
                image = await self._resolver.resolve(part)
                if not image.data:
                    raise SourceStoreError("image source is empty")
                parts.append(
                    (
                        index,
                        "image",
                        None,
                        image.data,
                        image.media_type,
                        image.data,
                        image.source_metadata,
                    )
                )
        return parts

    async def save_messages(
        self, session_id: str, drafts: list[MessageDraft]
    ) -> list[StoredMessage]:
        materialized = []
        for draft in drafts:
            input_hash = self._input_hash(draft.content)
            with self._lock:
                existing = self._connection.execute(
                    "SELECT * FROM messages WHERE session_id=? AND turn_id=? AND role=?",
                    (session_id, str(draft.turn_id), draft.role),
                ).fetchone()
            if existing is not None and existing["input_hash"] == input_hash:
                materialized.append((draft, None, input_hash, existing["id"]))
                continue
            # Image inputs can legitimately change representation while retaining
            # identical bytes (for example file_path on first save and data_url
            # when a stored message is reconstructed for extraction).  In that
            # case the provider-neutral input hash differs, so materialize the
            # parts and let _save_materialized compare the canonical byte hash.
            parts = await self._materialize_content(draft.content)
            materialized.append((draft, parts, input_hash, None))
        return self._save_materialized(session_id, materialized)

    def _save_materialized(self, session_id: str, materialized: list[tuple]) -> list[StoredMessage]:
        results = []
        with self._lock, self._connection:
            for draft, parts, input_hash, existing_id in materialized:
                if existing_id is not None:
                    results.append(self.get_message(existing_id))
                    self._advance_turn_state_locked(session_id, draft.turn_id)
                    continue
                digest = hashlib.sha256()
                for index, part_type, text, image, media_type, data, metadata in parts:
                    digest.update(str(index).encode())
                    digest.update(part_type.encode())
                    digest.update(data)
                    digest.update((media_type or "").encode())
                content_hash = digest.hexdigest()
                existing = self._connection.execute(
                    "SELECT * FROM messages WHERE session_id=? AND turn_id=? AND role=?",
                    (session_id, str(draft.turn_id), draft.role),
                ).fetchone()
                if existing:
                    if existing["content_hash"] != content_hash:
                        raise ImmutableSourceConflict(
                            "immutable message identity was submitted with different content"
                        )
                    results.append(self.get_message(existing["id"]))
                    self._advance_turn_state_locked(session_id, draft.turn_id)
                    continue

                message_id = uuid4()
                self._connection.execute(
                    """INSERT INTO messages
                       (id, session_id, turn_id, role, created_at, input_hash, content_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(message_id), session_id, str(draft.turn_id), draft.role,
                        draft.created_at.isoformat(), input_hash, content_hash,
                    ),
                )
                for index, part_type, text, image, media_type, data, metadata in parts:
                    self._connection.execute(
                        "INSERT INTO message_parts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            str(message_id), index, part_type, text, image, media_type,
                            hashlib.sha256(data).hexdigest(),
                            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                        ),
                    )
                results.append(self.get_message(message_id))
                self._advance_turn_state_locked(session_id, draft.turn_id)
        return results

    def _advance_turn_state_locked(self, session_id: str, turn_id: int | str) -> None:
        try:
            numeric_turn = int(turn_id)
        except (TypeError, ValueError):
            return
        if str(numeric_turn) != str(turn_id) or numeric_turn < 0:
            return
        self._connection.execute(
            """INSERT INTO session_turn_state(session_id, next_turn) VALUES (?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                 next_turn=MAX(session_turn_state.next_turn, excluded.next_turn)""",
            (session_id, numeric_turn + 1),
        )

    def _next_turn_locked(self, session_id: str) -> int:
        state = self._connection.execute(
            "SELECT next_turn FROM session_turn_state WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if state is not None:
            return int(state["next_turn"])
        rows = self._connection.execute(
            "SELECT turn_id FROM messages WHERE session_id=?", (session_id,)
        ).fetchall()
        numeric = []
        for row in rows:
            try:
                value = int(row["turn_id"])
            except (TypeError, ValueError):
                continue
            if str(value) == row["turn_id"] and value >= 0:
                numeric.append(value)
        return max(numeric, default=0) + 1

    def _insert_materialized_message_locked(
        self,
        *,
        session_id: str,
        role: str,
        turn_id: int,
        input_hash: str,
        parts: list[tuple],
        created_at: datetime,
    ) -> UUID:
        digest = hashlib.sha256()
        for index, part_type, _text, _image, media_type, data, _metadata in parts:
            digest.update(str(index).encode())
            digest.update(part_type.encode())
            digest.update(data)
            digest.update((media_type or "").encode())
        message_id = uuid4()
        self._connection.execute(
            """INSERT INTO messages
               (id, session_id, turn_id, role, created_at, input_hash, content_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(message_id), session_id, str(turn_id), role,
                created_at.isoformat(), input_hash, digest.hexdigest(),
            ),
        )
        for index, part_type, text, image, media_type, data, metadata in parts:
            self._connection.execute(
                "INSERT INTO message_parts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(message_id), index, part_type, text, image, media_type,
                    hashlib.sha256(data).hexdigest(),
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                ),
            )
        self._advance_turn_state_locked(session_id, turn_id)
        return message_id

    async def begin_agent_exchange(
        self,
        *,
        session_id: str,
        request_id: UUID,
        user_content: list[ContentPart],
        created_at: datetime,
    ) -> tuple[AgentExchangeRecord, StoredMessage, bool]:
        """Create one durable User message and its exchange, idempotently."""
        request_hash = self._input_hash(user_content)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM agent_exchanges WHERE request_id=?",
                (str(request_id),),
            ).fetchone()
        if row is not None:
            if row["user_request_hash"] != request_hash or row["session_id"] != session_id:
                raise ImmutableSourceConflict(
                    "request_id was reused with different User content or session"
                )
            record = self._exchange_from_row(row)
            return record, self.get_message(record.user_message_id), False

        parts = await self._materialize_content(user_content)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM agent_exchanges WHERE request_id=?",
                (str(request_id),),
            ).fetchone()
            if row is not None:
                if row["user_request_hash"] != request_hash or row["session_id"] != session_id:
                    raise ImmutableSourceConflict(
                        "request_id was reused with different User content or session"
                    )
                record = self._exchange_from_row(row)
                return record, self.get_message(record.user_message_id), False
            turn = self._next_turn_locked(session_id)
            message_id = self._insert_materialized_message_locked(
                session_id=session_id,
                role="user",
                turn_id=turn,
                input_hash=request_hash,
                parts=parts,
                created_at=created_at,
            )
            exchange_id = uuid4()
            self._connection.execute(
                """INSERT INTO agent_exchanges
                   (id, request_id, session_id, user_request_hash,
                    assistant_request_hash, status, stage, user_message_id,
                    assistant_message_id, before_result_json, after_result_json,
                    error_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, NULL, 'pending', 'user_saved', ?, NULL,
                           NULL, NULL, NULL, ?, ?)""",
                (
                    str(exchange_id), str(request_id), session_id, request_hash,
                    str(message_id), now, now,
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM agent_exchanges WHERE id=?", (str(exchange_id),)
            ).fetchone()
        return self._exchange_from_row(row), self.get_message(message_id), True

    def save_exchange_before_result(self, exchange_id: UUID, result: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """UPDATE agent_exchanges
                   SET before_result_json=?, stage='before_completed', error_json=NULL,
                       updated_at=? WHERE id=?""",
                (json.dumps(result, ensure_ascii=False), now, str(exchange_id)),
            )
        if cursor.rowcount != 1:
            raise SourceStoreError(f"agent exchange {exchange_id} does not exist")

    def get_agent_exchange(self, exchange_id: UUID | str) -> AgentExchangeRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM agent_exchanges WHERE id=?", (str(exchange_id),)
            ).fetchone()
        if row is None:
            raise SourceStoreError(f"agent exchange {exchange_id} does not exist")
        return self._exchange_from_row(row)

    async def attach_exchange_assistant(
        self,
        exchange_id: UUID,
        assistant_content: list[ContentPart] | None,
        created_at: datetime,
    ) -> tuple[AgentExchangeRecord, StoredMessage | None]:
        assistant_hash = (
            self._input_hash(assistant_content)
            if assistant_content is not None
            else "__none__"
        )
        record = self.get_agent_exchange(exchange_id)
        if record.assistant_request_hash is not None:
            if record.assistant_request_hash != assistant_hash:
                raise ImmutableSourceConflict(
                    "completed exchange was submitted with different Assistant content"
                )
            assistant = (
                self.get_message(record.assistant_message_id)
                if record.assistant_message_id is not None
                else None
            )
            return record, assistant
        if record.status == "completed":
            raise ImmutableSourceConflict("completed exchange is immutable")

        parts = (
            await self._materialize_content(assistant_content)
            if assistant_content is not None
            else None
        )
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM agent_exchanges WHERE id=?", (str(exchange_id),)
            ).fetchone()
            if row is None:
                raise SourceStoreError(f"agent exchange {exchange_id} does not exist")
            if row["assistant_request_hash"] is not None:
                if row["assistant_request_hash"] != assistant_hash:
                    raise ImmutableSourceConflict(
                        "exchange was submitted with different Assistant content"
                    )
                current = self._exchange_from_row(row)
                assistant = (
                    self.get_message(current.assistant_message_id)
                    if current.assistant_message_id is not None
                    else None
                )
                return current, assistant
            message_id = None
            if parts is not None:
                turn = self._next_turn_locked(record.session_id)
                message_id = self._insert_materialized_message_locked(
                    session_id=record.session_id,
                    role="assistant",
                    turn_id=turn,
                    input_hash=assistant_hash,
                    parts=parts,
                    created_at=created_at,
                )
            self._connection.execute(
                """UPDATE agent_exchanges
                   SET assistant_request_hash=?, assistant_message_id=?,
                       status='processing', stage='assistant_saved', error_json=NULL,
                       updated_at=? WHERE id=?""",
                (assistant_hash, str(message_id) if message_id else None, now, str(exchange_id)),
            )
            row = self._connection.execute(
                "SELECT * FROM agent_exchanges WHERE id=?", (str(exchange_id),)
            ).fetchone()
        current = self._exchange_from_row(row)
        return current, self.get_message(message_id) if message_id else None

    def complete_agent_exchange(self, exchange_id: UUID, result: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """UPDATE agent_exchanges
                   SET status='completed', stage='completed', after_result_json=?,
                       error_json=NULL, updated_at=? WHERE id=?""",
                (json.dumps(result, ensure_ascii=False), now, str(exchange_id)),
            )
        if cursor.rowcount != 1:
            raise SourceStoreError(f"agent exchange {exchange_id} does not exist")

    def fail_agent_exchange(self, exchange_id: UUID, error: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            self._connection.execute(
                """UPDATE agent_exchanges
                   SET status='failed', stage='failed', error_json=?, updated_at=?
                   WHERE id=? AND status!='completed'""",
                (json.dumps(error, ensure_ascii=False), now, str(exchange_id)),
            )

    async def begin_session_submission(
        self,
        *,
        session_id: str,
        submission_id: UUID,
        request_hash: str,
        messages: list[tuple[str, list[ContentPart]]],
        created_at: datetime,
        worker_token: UUID,
        lease_seconds: float,
    ) -> tuple[SessionSubmissionRecord, list[StoredMessage]]:
        """Atomically claim a submission and allocate one turn per new message."""
        if not messages:
            raise ValueError("a session submission requires at least one message")
        now_dt = datetime.now(timezone.utc)
        with self._lock:
            existing = self._connection.execute(
                "SELECT * FROM session_submissions WHERE session_id=? AND submission_id=?",
                (session_id, str(submission_id)),
            ).fetchone()
        if existing is not None:
            if existing["request_hash"] != request_hash:
                raise ImmutableSourceConflict(
                    "submission_id was reused with different content"
                )
            lease = (
                datetime.fromisoformat(existing["lease_expires_at"])
                if existing["lease_expires_at"]
                else None
            )
            if existing["status"] == "completed" or (
                existing["status"] == "processing"
                and lease is not None
                and lease > now_dt
            ):
                record = self._submission_from_row(existing, claimed=False)
                return record, [self.get_message(value) for value in record.message_ids]
            lease_expires = (now_dt + timedelta(seconds=lease_seconds)).isoformat()
            with self._lock, self._connection:
                cursor = self._connection.execute(
                    """UPDATE session_submissions
                       SET status='processing', worker_token=?, lease_expires_at=?,
                           error_json=NULL, updated_at=?
                       WHERE session_id=? AND submission_id=?
                         AND (status='failed' OR lease_expires_at IS NULL OR lease_expires_at<=?)""",
                    (
                        str(worker_token), lease_expires, now_dt.isoformat(),
                        session_id, str(submission_id), now_dt.isoformat(),
                    ),
                )
                refreshed = self._connection.execute(
                    "SELECT * FROM session_submissions WHERE session_id=? AND submission_id=?",
                    (session_id, str(submission_id)),
                ).fetchone()
            record = self._submission_from_row(
                refreshed, claimed=cursor.rowcount == 1
            )
            return record, [self.get_message(value) for value in record.message_ids]
        materialized = [
            (role, content, self._input_hash(content), await self._materialize_content(content))
            for role, content in messages
        ]
        now = now_dt.isoformat()
        lease_expires = (now_dt + timedelta(seconds=lease_seconds)).isoformat()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM session_submissions WHERE session_id=? AND submission_id=?",
                (session_id, str(submission_id)),
            ).fetchone()
            if row is not None:
                if row["request_hash"] != request_hash:
                    raise ImmutableSourceConflict(
                        "submission_id was reused with different content"
                    )
                claimed = False
                lease = (
                    datetime.fromisoformat(row["lease_expires_at"])
                    if row["lease_expires_at"]
                    else None
                )
                if row["status"] == "failed" or (
                    row["status"] == "processing"
                    and (lease is None or lease <= now_dt)
                ):
                    cursor = self._connection.execute(
                        """UPDATE session_submissions
                           SET status='processing', worker_token=?, lease_expires_at=?,
                               error_json=NULL, updated_at=?
                           WHERE session_id=? AND submission_id=?
                             AND (status='failed' OR lease_expires_at IS NULL OR lease_expires_at<=?)""",
                        (
                            str(worker_token), lease_expires, now,
                            session_id, str(submission_id), now,
                        ),
                    )
                    claimed = cursor.rowcount == 1
                refreshed = self._connection.execute(
                    "SELECT * FROM session_submissions WHERE session_id=? AND submission_id=?",
                    (session_id, str(submission_id)),
                ).fetchone()
                record = self._submission_from_row(refreshed, claimed=claimed)
                return record, [self.get_message(value) for value in record.message_ids]

            state = self._connection.execute(
                "SELECT next_turn FROM session_turn_state WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if state is None:
                existing_turns = self._connection.execute(
                    "SELECT turn_id FROM messages WHERE session_id=?",
                    (session_id,),
                ).fetchall()
                numeric_turns = []
                for existing in existing_turns:
                    try:
                        value = int(existing["turn_id"])
                    except (TypeError, ValueError):
                        continue
                    if str(value) == existing["turn_id"] and value >= 0:
                        numeric_turns.append(value)
                next_turn = max(numeric_turns, default=0) + 1
            else:
                next_turn = int(state["next_turn"])

            message_ids = []
            for offset, (role, _content, input_hash, parts) in enumerate(materialized):
                turn = next_turn + offset
                message_id = uuid4()
                digest = hashlib.sha256()
                for index, part_type, _text, _image, media_type, data, _metadata in parts:
                    digest.update(str(index).encode())
                    digest.update(part_type.encode())
                    digest.update(data)
                    digest.update((media_type or "").encode())
                self._connection.execute(
                    """INSERT INTO messages
                       (id, session_id, turn_id, role, created_at, input_hash, content_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(message_id), session_id, str(turn), role,
                        created_at.isoformat(), input_hash, digest.hexdigest(),
                    ),
                )
                for index, part_type, text, image, media_type, data, metadata in parts:
                    self._connection.execute(
                        "INSERT INTO message_parts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            str(message_id), index, part_type, text, image, media_type,
                            hashlib.sha256(data).hexdigest(),
                            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                        ),
                    )
                message_ids.append(message_id)

            self._connection.execute(
                """INSERT INTO session_turn_state(session_id, next_turn) VALUES (?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET next_turn=excluded.next_turn""",
                (session_id, next_turn + len(message_ids)),
            )
            self._connection.execute(
                """INSERT INTO session_submissions
                   (session_id, submission_id, request_hash, status, message_ids_json,
                    ingestion_run_id, worker_token, lease_expires_at, result_json,
                    error_json, created_at, updated_at)
                   VALUES (?, ?, ?, 'processing', ?, NULL, ?, ?, NULL, NULL, ?, ?)""",
                (
                    session_id, str(submission_id), request_hash,
                    json.dumps([str(value) for value in message_ids]),
                    str(worker_token), lease_expires, now, now,
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM session_submissions WHERE session_id=? AND submission_id=?",
                (session_id, str(submission_id)),
            ).fetchone()
        return self._submission_from_row(row, claimed=True), [
            self.get_message(value) for value in message_ids
        ]

    def get_recent_messages(
        self,
        session_id: str,
        *,
        limit: int,
        exclude_ids: list[UUID] | None = None,
    ) -> list[StoredMessage]:
        if limit < 0:
            raise ValueError("recent message limit must be non-negative")
        if limit == 0:
            return []
        excluded = {str(value) for value in (exclude_ids or [])}
        with self._lock:
            rows = self._connection.execute(
                """SELECT id FROM messages WHERE session_id=?
                   ORDER BY created_at DESC, rowid DESC""",
                (session_id,),
            ).fetchall()
        ids = [row["id"] for row in rows if row["id"] not in excluded][:limit]
        return [self.get_message(value) for value in reversed(ids)]

    def list_messages(
        self,
        *,
        session_id: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[StoredMessage], int]:
        """List immutable source messages newest first for debug inspection."""
        if offset < 0:
            raise ValueError("message offset must be non-negative")
        if limit <= 0:
            raise ValueError("message limit must be positive")
        where = " WHERE session_id=?" if session_id else ""
        params: tuple = (session_id,) if session_id else ()
        with self._lock:
            total = int(
                self._connection.execute(
                    f"SELECT COUNT(*) FROM messages{where}", params  # noqa: S608
                ).fetchone()[0]
            )
            rows = self._connection.execute(
                f"""SELECT id FROM messages{where}
                    ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?""",  # noqa: S608
                (*params, limit, offset),
            ).fetchall()
        return [self.get_message(row["id"]) for row in rows], total

    def heartbeat_session_submission(
        self, session_id: str, submission_id: UUID, worker_token: UUID, lease_seconds: float
    ) -> bool:
        now_dt = datetime.now(timezone.utc)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """UPDATE session_submissions SET lease_expires_at=?, updated_at=?
                   WHERE session_id=? AND submission_id=? AND status='processing'
                     AND worker_token=?""",
                (
                    (now_dt + timedelta(seconds=lease_seconds)).isoformat(),
                    now_dt.isoformat(), session_id, str(submission_id), str(worker_token),
                ),
            )
        return cursor.rowcount == 1

    def attach_submission_run(
        self, session_id: str, submission_id: UUID, worker_token: UUID, run_id: UUID
    ) -> None:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """UPDATE session_submissions SET ingestion_run_id=?, updated_at=?
                   WHERE session_id=? AND submission_id=? AND status='processing'
                     AND worker_token=?""",
                (
                    str(run_id), datetime.now(timezone.utc).isoformat(), session_id,
                    str(submission_id), str(worker_token),
                ),
            )
        if cursor.rowcount != 1:
            raise SourceStoreError("session submission lease was lost")

    def complete_session_submission(
        self, session_id: str, submission_id: UUID, worker_token: UUID, result: dict
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """UPDATE session_submissions
                   SET status='completed', result_json=?, error_json=NULL,
                       lease_expires_at=NULL, updated_at=?
                   WHERE session_id=? AND submission_id=? AND status='processing'
                     AND worker_token=?""",
                (
                    json.dumps(result, ensure_ascii=False), now, session_id,
                    str(submission_id), str(worker_token),
                ),
            )
        if cursor.rowcount != 1:
            raise SourceStoreError("session submission lease was lost")

    def fail_session_submission(
        self, session_id: str, submission_id: UUID, worker_token: UUID, error: dict
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            self._connection.execute(
                """UPDATE session_submissions
                   SET status='failed', error_json=?, lease_expires_at=NULL, updated_at=?
                   WHERE session_id=? AND submission_id=? AND status='processing'
                     AND worker_token=?""",
                (
                    json.dumps(error, ensure_ascii=False), now, session_id,
                    str(submission_id), str(worker_token),
                ),
            )

    def get_message(self, message_id: UUID | str) -> StoredMessage:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM messages WHERE id=?", (str(message_id),)
            ).fetchone()
            if row is None:
                raise SourceStoreError(f"message {message_id} does not exist")
            part_rows = self._connection.execute(
                "SELECT * FROM message_parts WHERE message_id=? ORDER BY part_index",
                (str(message_id),),
            ).fetchall()
        parts = [
            StoredMessagePart(
                message_id=UUID(part["message_id"]),
                part_index=part["part_index"],
                part_type=part["part_type"],
                text_content=part["text_content"],
                image_bytes=part["image_bytes"],
                media_type=part["media_type"],
                content_hash=part["content_hash"],
                source_metadata=json.loads(part["source_metadata_json"]),
            )
            for part in part_rows
        ]
        return StoredMessage(
            id=UUID(row["id"]),
            session_id=row["session_id"],
            turn_id=self._normalize_turn_id(row["turn_id"]),
            role=row["role"],
            created_at=datetime.fromisoformat(row["created_at"]),
            content_hash=row["content_hash"],
            parts=parts,
        )

    @staticmethod
    def _normalize_turn_id(value: str) -> int | str:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return value
        return numeric if str(numeric) == value and numeric >= 0 else value

    def close(self) -> None:
        with self._lock:
            for connection in self._write_connections.values():
                connection.close()
            self._write_connections.clear()
            self._memory_write_owner = None
            self._connection.close()

    def get_or_create_run(
        self, idempotency_key: str, session_id: str, request_hash: str
    ) -> IngestionRunRecord:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM ingestion_runs WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if row is not None:
                if row["request_hash"] != request_hash:
                    raise ImmutableSourceConflict(
                        "idempotency key was reused with a different request"
                    )
                return self._run_from_row(row)
            run_id = uuid4()
            self._connection.execute(
                """INSERT INTO ingestion_runs
                   (id, idempotency_key, session_id, request_hash, status, stage,
                    result_json, error_json, checkpoint_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(run_id), idempotency_key, session_id, request_hash,
                    "running", "created", None, None, None, now, now,
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM ingestion_runs WHERE id=?", (str(run_id),)
            ).fetchone()
        return self._run_from_row(row)

    def save_run_checkpoint(self, run_id: UUID, checkpoint: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """UPDATE ingestion_runs SET checkpoint_json=?, updated_at=? WHERE id=?""",
                (json.dumps(checkpoint, ensure_ascii=False), now, str(run_id)),
            )
        if cursor.rowcount != 1:
            raise SourceStoreError(f"ingestion run {run_id} does not exist")

    def latest_ingestion_progress(self, session_id: str) -> dict | None:
        """Small, read-only status suitable for a live evaluation monitor."""
        with self._lock:
            row = self._connection.execute(
                """SELECT id, status, stage, checkpoint_json, updated_at
                   FROM ingestion_runs WHERE session_id=?
                   ORDER BY created_at DESC, rowid DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        checkpoint = json.loads(row["checkpoint_json"]) if row["checkpoint_json"] else {}
        completed = checkpoint.get("completed_batches", [])
        prepared = checkpoint.get("prepared_batch")
        return {
            "operation_id": row["id"],
            "status": row["status"],
            "stage": row["stage"],
            "candidate_count": len(checkpoint.get("candidates", [])),
            "completed_batches": len(completed),
            "completed_candidates": completed[-1]["end"] if completed else 0,
            "prepared_batch": (
                {"start": prepared["start"], "end": prepared["end"]}
                if prepared is not None else None
            ),
            "updated_at": row["updated_at"],
        }

    def update_run(
        self,
        run_id: UUID,
        *,
        status: str,
        stage: str,
        result: dict | None = None,
        error: dict | None = None,
    ) -> IngestionRunRecord:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE ingestion_runs SET status=?, stage=?, result_json=?, error_json=?, updated_at=? WHERE id=?",
                (
                    status,
                    stage,
                    json.dumps(result, ensure_ascii=False) if result is not None else None,
                    json.dumps(error, ensure_ascii=False) if error is not None else None,
                    now,
                    str(run_id),
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM ingestion_runs WHERE id=?", (str(run_id),)
            ).fetchone()
        if row is None:
            raise SourceStoreError(f"ingestion run {run_id} does not exist")
        return self._run_from_row(row)

    def get_session_summary(self, session_id: str) -> SessionSummary | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM session_summaries WHERE session_id=?",
                (session_id,),
            ).fetchone()
        return self._summary_from_row(row) if row is not None else None

    def register_summary_submission(
        self,
        run_id: UUID,
        session_id: str,
        message_ids: list[UUID],
        batch_size: int,
    ) -> int:
        """Register one API submission and enqueue the next full session batch."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not message_ids:
            raise ValueError("a summary submission requires at least one message")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            run = self._connection.execute(
                "SELECT session_id FROM ingestion_runs WHERE id=?",
                (str(run_id),),
            ).fetchone()
            if run is None:
                raise SourceStoreError(f"ingestion run {run_id} does not exist")
            if run["session_id"] != session_id:
                raise SourceStoreError("summary submission session does not match ingestion run")

            existing = self._connection.execute(
                "SELECT session_id, sequence FROM summary_submissions WHERE ingestion_run_id=?",
                (str(run_id),),
            ).fetchone()
            if existing is not None:
                if existing["session_id"] != session_id:
                    raise SourceStoreError("ingestion run is already registered to another session")
                sequence = existing["sequence"]
                registered = self._connection.execute(
                    """SELECT message_id FROM summary_submission_messages
                       WHERE session_id=? AND submission_sequence=?
                       ORDER BY ordinal""",
                    (session_id, sequence),
                ).fetchall()
                if [row["message_id"] for row in registered] != [
                    str(message_id) for message_id in message_ids
                ]:
                    raise SourceStoreError(
                        "summary submission was replayed with different messages"
                    )
                self._enqueue_next_summary_job(self._connection, session_id, batch_size, now)
                return sequence

            for message_id in message_ids:
                message = self._connection.execute(
                    "SELECT session_id FROM messages WHERE id=?",
                    (str(message_id),),
                ).fetchone()
                if message is None:
                    raise SourceStoreError(f"message {message_id} does not exist")
                if message["session_id"] != session_id:
                    raise SourceStoreError("summary message belongs to another session")

            row = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence "
                "FROM summary_submissions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            sequence = int(row["next_sequence"])
            self._connection.execute(
                "INSERT INTO summary_submissions VALUES (?, ?, ?, ?)",
                (session_id, sequence, str(run_id), now),
            )
            for ordinal, message_id in enumerate(message_ids):
                self._connection.execute(
                    "INSERT INTO summary_submission_messages VALUES (?, ?, ?, ?)",
                    (session_id, sequence, ordinal, str(message_id)),
                )
            self._enqueue_next_summary_job(self._connection, session_id, batch_size, now)
        return sequence

    @staticmethod
    def _enqueue_next_summary_job(
        connection: sqlite3.Connection,
        session_id: str,
        batch_size: int,
        now: str,
    ) -> None:
        active = connection.execute(
            """SELECT 1 FROM summary_jobs
               WHERE session_id=? AND status IN ('pending', 'running', 'failed')
               LIMIT 1""",
            (session_id,),
        ).fetchone()
        if active is not None:
            return
        summary = connection.execute(
            "SELECT summarized_through_sequence FROM session_summaries WHERE session_id=?",
            (session_id,),
        ).fetchone()
        covered = int(summary[0]) if summary is not None else 0
        count = connection.execute(
            """SELECT COUNT(*) FROM summary_submissions
               WHERE session_id=? AND sequence>?""",
            (session_id, covered),
        ).fetchone()[0]
        if count < batch_size:
            return
        start = covered + 1
        end = covered + batch_size
        contiguous = connection.execute(
            """SELECT COUNT(*) FROM summary_submissions
               WHERE session_id=? AND sequence BETWEEN ? AND ?""",
            (session_id, start, end),
        ).fetchone()[0]
        if contiguous != batch_size:
            raise SourceStoreError("summary submission sequence contains a gap")
        connection.execute(
            """INSERT OR IGNORE INTO summary_jobs
               (id, session_id, from_sequence, through_sequence, status, attempts,
                error_json, available_at, claimed_at, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'pending', 0, NULL, ?, NULL, ?, ?)""",
            (str(uuid4()), session_id, start, end, now, now, now),
        )

    def recover_summary_jobs(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            self._connection.execute(
                """UPDATE summary_jobs
                   SET status='pending', claimed_at=NULL, available_at=?, updated_at=?
                   WHERE status='running'""",
                (now, now),
            )

    def claim_summary_job(self) -> SummaryJob | None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            row = self._connection.execute(
                """SELECT * FROM summary_jobs
                   WHERE status='pending' AND available_at<=?
                   ORDER BY created_at, session_id LIMIT 1""",
                (now,),
            ).fetchone()
            if row is None:
                return None
            self._connection.execute(
                """UPDATE summary_jobs
                   SET status='running', attempts=attempts+1, claimed_at=?, updated_at=?
                   WHERE id=? AND status='pending'""",
                (now, now, row["id"]),
            )
            claimed = self._connection.execute(
                "SELECT * FROM summary_jobs WHERE id=?",
                (row["id"],),
            ).fetchone()
        return self._job_from_row(claimed)

    def get_summary_job_messages(self, job: SummaryJob) -> list[StoredMessage]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT sm.message_id
                   FROM summary_submission_messages sm
                   JOIN summary_submissions ss
                     ON ss.session_id=sm.session_id
                    AND ss.sequence=sm.submission_sequence
                   JOIN messages m ON m.id=sm.message_id
                   WHERE sm.session_id=?
                     AND sm.submission_sequence BETWEEN ? AND ?
                     AND ss.session_id=? AND m.session_id=?
                   ORDER BY sm.submission_sequence, sm.ordinal""",
                (
                    job.session_id,
                    job.from_sequence,
                    job.through_sequence,
                    job.session_id,
                    job.session_id,
                ),
            ).fetchall()
        messages = [self.get_message(row["message_id"]) for row in rows]
        if not messages:
            raise SourceStoreError("summary job contains no messages")
        if any(message.session_id != job.session_id for message in messages):
            raise SourceStoreError("summary job contains a cross-session message")
        return messages

    def complete_summary_job(
        self,
        job: SummaryJob,
        *,
        expected_version: int,
        text: str,
        provider: str,
        model: str,
        request_id: str | None,
        batch_size: int,
    ) -> SessionSummary:
        normalized = text.strip()
        if not normalized:
            raise ValueError("summary text cannot be empty")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            current = self._connection.execute(
                "SELECT * FROM session_summaries WHERE session_id=?",
                (job.session_id,),
            ).fetchone()
            current_version = int(current["version"]) if current is not None else 0
            covered = (
                int(current["summarized_through_sequence"])
                if current is not None
                else 0
            )
            if current_version != expected_version:
                raise SourceStoreError("summary version changed while job was running")
            if job.from_sequence != covered + 1:
                raise SourceStoreError("summary job does not continue the session cursor")
            job_row = self._connection.execute(
                "SELECT status, session_id FROM summary_jobs WHERE id=?",
                (str(job.id),),
            ).fetchone()
            if job_row is None or job_row["status"] != "running":
                raise SourceStoreError("summary job is not running")
            if job_row["session_id"] != job.session_id:
                raise SourceStoreError("summary job session changed")

            created_at = current["created_at"] if current is not None else now
            version = current_version + 1
            self._connection.execute(
                """INSERT INTO session_summaries
                   (session_id, summary_text, version, summarized_through_sequence,
                    provider, model, request_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                     summary_text=excluded.summary_text,
                     version=excluded.version,
                     summarized_through_sequence=excluded.summarized_through_sequence,
                     provider=excluded.provider,
                     model=excluded.model,
                     request_id=excluded.request_id,
                     updated_at=excluded.updated_at""",
                (
                    job.session_id,
                    normalized,
                    version,
                    job.through_sequence,
                    provider,
                    model,
                    request_id,
                    created_at,
                    now,
                ),
            )
            self._connection.execute(
                """UPDATE summary_jobs
                   SET status='complete', error_json=NULL, updated_at=?
                   WHERE id=?""",
                (now, str(job.id)),
            )
            self._enqueue_next_summary_job(
                self._connection, job.session_id, batch_size, now
            )
            row = self._connection.execute(
                "SELECT * FROM session_summaries WHERE session_id=?",
                (job.session_id,),
            ).fetchone()
        return self._summary_from_row(row)

    def fail_summary_job(
        self,
        job: SummaryJob,
        error: dict,
        *,
        max_attempts: int,
    ) -> SummaryJob:
        now_dt = datetime.now(timezone.utc)
        final = job.attempts >= max_attempts
        status = "failed" if final else "pending"
        delay = min(60, 2 ** max(job.attempts - 1, 0))
        available = (now_dt + timedelta(seconds=delay)).isoformat()
        now = now_dt.isoformat()
        with self._lock, self._connection:
            self._connection.execute(
                """UPDATE summary_jobs
                   SET status=?, error_json=?, available_at=?, claimed_at=NULL,
                       updated_at=?
                   WHERE id=? AND session_id=?""",
                (
                    status,
                    json.dumps(error, ensure_ascii=False),
                    available,
                    now,
                    str(job.id),
                    job.session_id,
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM summary_jobs WHERE id=? AND session_id=?",
                (str(job.id), job.session_id),
            ).fetchone()
        if row is None:
            raise SourceStoreError("summary job does not exist")
        return self._job_from_row(row)

    def retry_failed_summary(self, session_id: str) -> SummaryJob | None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            row = self._connection.execute(
                """SELECT * FROM summary_jobs
                   WHERE session_id=? AND status='failed'
                   ORDER BY from_sequence LIMIT 1""",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            self._connection.execute(
                """UPDATE summary_jobs
                   SET status='pending', attempts=0, error_json=NULL,
                       available_at=?, claimed_at=NULL, updated_at=?
                   WHERE id=? AND session_id=?""",
                (now, now, row["id"], session_id),
            )
            updated = self._connection.execute(
                "SELECT * FROM summary_jobs WHERE id=? AND session_id=?",
                (row["id"], session_id),
            ).fetchone()
        return self._job_from_row(updated)

    def get_summary_status(self, session_id: str) -> SummaryStatus:
        from mcr.summarization.models import SummaryStatus

        with self._lock:
            summary_row = self._connection.execute(
                "SELECT * FROM session_summaries WHERE session_id=?",
                (session_id,),
            ).fetchone()
            submission_count = int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM summary_submissions WHERE session_id=?",
                    (session_id,),
                ).fetchone()[0]
            )
            covered = (
                int(summary_row["summarized_through_sequence"])
                if summary_row is not None
                else 0
            )
            job_row = self._connection.execute(
                """SELECT * FROM summary_jobs
                   WHERE session_id=? AND status IN ('pending', 'running', 'failed')
                   ORDER BY from_sequence LIMIT 1""",
                (session_id,),
            ).fetchone()
        return SummaryStatus(
            session_id=session_id,
            summary=self._summary_from_row(summary_row) if summary_row else None,
            submission_count=submission_count,
            pending_submission_count=max(0, submission_count - covered),
            active_job=self._job_from_row(job_row) if job_row else None,
        )

    def save_candidate(
        self,
        run_id: UUID,
        candidate_id: UUID,
        ordinal: int,
        candidate: dict,
        decision: dict | None,
        status: str,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO ingestion_candidates
                   (run_id, candidate_id, ordinal, candidate_json, decision_json, status)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, candidate_id) DO UPDATE SET
                     candidate_json=excluded.candidate_json,
                     decision_json=excluded.decision_json,
                     status=excluded.status""",
                (
                    str(run_id), str(candidate_id), ordinal,
                    json.dumps(candidate, ensure_ascii=False),
                    json.dumps(decision, ensure_ascii=False) if decision is not None else None,
                    status,
                ),
            )

    def acquire_write_lock(self, operation_id: UUID) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if self.path == ":memory:":
                if self._memory_write_owner in (None, operation_id):
                    self._memory_write_owner = operation_id
                    return
                raise SourceStoreError(
                    "the global Qdrant ingestion write lock is busy"
                )
            if operation_id in self._write_connections:
                return
            connection = sqlite3.connect(self.path, timeout=0, isolation_level=None)
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """INSERT INTO ingestion_write_lock
                       (lock_name, owner_operation_id, acquired_at, heartbeat_at)
                       VALUES ('qdrant-write', ?, ?, ?)
                       ON CONFLICT(lock_name) DO UPDATE SET
                         owner_operation_id=excluded.owner_operation_id,
                         acquired_at=excluded.acquired_at,
                         heartbeat_at=excluded.heartbeat_at""",
                    (str(operation_id), now, now),
                )
            except sqlite3.OperationalError as exc:
                connection.close()
                raise SourceStoreError(
                    "the global Qdrant ingestion write lock is busy"
                ) from exc
            self._write_connections[operation_id] = connection

    def release_write_lock(self, operation_id: UUID) -> None:
        with self._lock:
            if self.path == ":memory:":
                if self._memory_write_owner == operation_id:
                    self._memory_write_owner = None
                return
            connection = self._write_connections.pop(operation_id, None)
            if connection is None:
                return
            try:
                connection.execute(
                    "DELETE FROM ingestion_write_lock WHERE lock_name='qdrant-write' AND owner_operation_id=?",
                    (str(operation_id),),
                )
                connection.execute("COMMIT")
            finally:
                connection.close()

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> IngestionRunRecord:
        return IngestionRunRecord(
            id=UUID(row["id"]),
            idempotency_key=row["idempotency_key"],
            session_id=row["session_id"],
            request_hash=row["request_hash"],
            status=row["status"],
            stage=row["stage"],
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=json.loads(row["error_json"]) if row["error_json"] else None,
            checkpoint=(
                json.loads(row["checkpoint_json"])
                if "checkpoint_json" in row.keys() and row["checkpoint_json"]
                else None
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _submission_from_row(
        row: sqlite3.Row, *, claimed: bool = False
    ) -> SessionSubmissionRecord:
        return SessionSubmissionRecord(
            session_id=row["session_id"],
            submission_id=UUID(row["submission_id"]),
            request_hash=row["request_hash"],
            status=row["status"],
            message_ids=[UUID(value) for value in json.loads(row["message_ids_json"])],
            ingestion_run_id=(
                UUID(row["ingestion_run_id"]) if row["ingestion_run_id"] else None
            ),
            worker_token=UUID(row["worker_token"]) if row["worker_token"] else None,
            lease_expires_at=(
                datetime.fromisoformat(row["lease_expires_at"])
                if row["lease_expires_at"]
                else None
            ),
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=json.loads(row["error_json"]) if row["error_json"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            claimed=claimed,
        )

    @staticmethod
    def _exchange_from_row(row: sqlite3.Row) -> AgentExchangeRecord:
        return AgentExchangeRecord(
            id=UUID(row["id"]),
            request_id=UUID(row["request_id"]),
            session_id=row["session_id"],
            user_request_hash=row["user_request_hash"],
            assistant_request_hash=row["assistant_request_hash"],
            status=row["status"],
            stage=row["stage"],
            user_message_id=UUID(row["user_message_id"]),
            assistant_message_id=(
                UUID(row["assistant_message_id"])
                if row["assistant_message_id"] else None
            ),
            before_result=(
                json.loads(row["before_result_json"])
                if row["before_result_json"] else None
            ),
            after_result=(
                json.loads(row["after_result_json"])
                if row["after_result_json"] else None
            ),
            error=json.loads(row["error_json"]) if row["error_json"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _summary_from_row(row: sqlite3.Row) -> SessionSummary:
        from mcr.summarization.models import SessionSummary

        return SessionSummary(
            session_id=row["session_id"],
            text=row["summary_text"],
            version=row["version"],
            summarized_through_sequence=row["summarized_through_sequence"],
            provider=row["provider"],
            model=row["model"],
            request_id=row["request_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> SummaryJob:
        from mcr.summarization.models import SummaryJob

        return SummaryJob(
            id=UUID(row["id"]),
            session_id=row["session_id"],
            from_sequence=row["from_sequence"],
            through_sequence=row["through_sequence"],
            status=row["status"],
            attempts=row["attempts"],
            error=json.loads(row["error_json"]) if row["error_json"] else None,
            available_at=datetime.fromisoformat(row["available_at"]),
            claimed_at=(
                datetime.fromisoformat(row["claimed_at"])
                if row["claimed_at"]
                else None
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
