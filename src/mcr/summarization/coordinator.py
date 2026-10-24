"""Session summary registration, reads, and retry coordination."""

from __future__ import annotations

from uuid import UUID

from mcr.source_store.sqlite import SQLiteSourceStore
from mcr.summarization.config import SummaryConfig
from mcr.summarization.models import SessionSummary, SummaryJob, SummaryStatus


class SummaryCoordinator:
    def __init__(self, store: SQLiteSourceStore, config: SummaryConfig) -> None:
        self.store = store
        self.config = config

    def get_summary(self, session_id: str) -> SessionSummary | None:
        return self.store.get_session_summary(session_id)

    def register_submission(
        self,
        run_id: UUID,
        session_id: str,
        message_ids: list[UUID],
    ) -> int:
        return self.store.register_summary_submission(
            run_id,
            session_id,
            message_ids,
            self.config.batch_size,
        )

    def get_status(self, session_id: str) -> SummaryStatus:
        return self.store.get_summary_status(session_id)

    def retry_failed(self, session_id: str) -> SummaryJob | None:
        return self.store.retry_failed_summary(session_id)
