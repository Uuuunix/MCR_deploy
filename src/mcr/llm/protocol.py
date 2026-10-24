"""Provider-independent async LLM port."""

from __future__ import annotations

from typing import Protocol

from mcr.llm.models import LLMMessage, LLMResponse, ResponseFormat


class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[LLMMessage],
        response_format: ResponseFormat | None = None,
    ) -> LLMResponse:
        """Complete a multimodal chat request."""
        ...

