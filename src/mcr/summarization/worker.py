"""Durable asynchronous worker for incremental session summaries."""

from __future__ import annotations

import asyncio
from contextlib import suppress

from mcr.source_store.sqlite import SQLiteSourceStore
from mcr.summarization.config import SummaryConfig
from mcr.summarization.summarizer import SessionSummarizer


class SummaryWorker:
    def __init__(
        self,
        store: SQLiteSourceStore,
        summarizer: SessionSummarizer,
        config: SummaryConfig,
    ) -> None:
        self.store = store
        self.summarizer = summarizer
        self.config = config
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self.store.recover_summary_jobs()
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="mcr-summary-worker")

    async def stop(self) -> None:
        self._stop.set()
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _run(self) -> None:
        while not self._stop.is_set():
            handled = await self.run_once()
            if handled:
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.config.poll_seconds
                )
            except TimeoutError:
                pass

    async def run_once(self) -> bool:
        job = self.store.claim_summary_job()
        if job is None:
            return False
        try:
            previous = self.store.get_session_summary(job.session_id)
            expected_version = previous.version if previous is not None else 0
            messages = self.store.get_summary_job_messages(job)
            result = await self.summarizer.summarize(previous, messages)
            self.store.complete_summary_job(
                job,
                expected_version=expected_version,
                text=result.text,
                provider=result.trace.provider,
                model=result.trace.model,
                request_id=result.trace.request_id,
                batch_size=self.config.batch_size,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.store.fail_summary_job(
                job,
                {"type": type(exc).__name__, "message": str(exc)},
                max_attempts=self.config.max_attempts,
            )
        return True
