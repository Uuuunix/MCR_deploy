"""One-call implementation of incremental multimodal summarization."""

from __future__ import annotations

from time import perf_counter

from mcr.llm import LLMClient
from mcr.source_store.models import StoredMessage
from mcr.summarization.models import SessionSummary, SummaryResult, SummaryTrace
from mcr.summarization.parser import parse_summary_response
from mcr.summarization.prompt import build_summary_prompt


class SessionSummarizer:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def summarize(
        self,
        previous: SessionSummary | None,
        messages: list[StoredMessage],
    ) -> SummaryResult:
        prompt = build_summary_prompt(previous, messages)
        started = perf_counter()
        response = await self._llm.complete(prompt)
        elapsed_ms = (perf_counter() - started) * 1000
        text = parse_summary_response(response.text)
        return SummaryResult(
            text=text,
            trace=SummaryTrace(
                messages=prompt,
                provider_request=response.request,
                raw_response=response.text,
                elapsed_ms=elapsed_ms,
                provider=response.provider,
                model=response.model,
                request_id=response.request_id,
                usage=response.usage,
            ),
        )
