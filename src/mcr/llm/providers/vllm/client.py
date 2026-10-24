"""Provider-neutral chat client backed by a local vLLM server."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from openai import AsyncOpenAI

from mcr.llm.errors import LLMRequestError
from mcr.llm.models import ImagePart, LLMMessage, LLMResponse, LLMUsage, ResponseFormat, TextPart
from mcr.llm.providers.siliconflow.images import image_part_to_url
from mcr.llm.providers.vllm.config import VLLMConfig


class VLLMClient:
    def __init__(self, config: VLLMConfig, client: AsyncOpenAI | Any | None = None) -> None:
        self.config = config
        self._client = client or AsyncOpenAI(
            api_key="EMPTY",
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    @staticmethod
    def _message_to_api(message: LLMMessage) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_part_to_url(part), "detail": part.detail},
                })
        return {"role": message.role, "content": content}

    @staticmethod
    def _safe_request(params: dict[str, Any]) -> dict[str, Any]:
        messages = []
        for message in params["messages"]:
            content = []
            for part in message["content"]:
                if part["type"] == "image_url":
                    image = dict(part["image_url"])
                    if image["url"].startswith("data:"):
                        image["url"] = "[image data URL omitted]"
                    content.append({"type": "image_url", "image_url": image})
                else:
                    content.append(dict(part))
            messages.append({"role": message["role"], "content": content})
        return {**params, "messages": messages}

    async def complete(
        self, messages: list[LLMMessage], response_format: ResponseFormat | None = None
    ) -> LLMResponse:
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": [self._message_to_api(message) for message in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if response_format is not None:
            params["response_format"] = response_format.model_dump(mode="json")
        try:
            response = await self._client.chat.completions.create(**params)
            if not response.choices or response.choices[0].message is None:
                raise ValueError("provider response contains no choices")
            usage = None
            if response.usage is not None:
                usage = LLMUsage(
                    prompt_tokens=getattr(response.usage, "prompt_tokens", None),
                    completion_tokens=getattr(response.usage, "completion_tokens", None),
                    total_tokens=getattr(response.usage, "total_tokens", None),
                )
            return LLMResponse(
                text=response.choices[0].message.content or "",
                provider="vllm",
                model=response.model or self.config.model,
                request_id=getattr(response, "id", None),
                usage=usage,
                finish_reason=getattr(response.choices[0], "finish_reason", None),
                request=self._safe_request(params),
            )
        except Exception as exc:
            raise LLMRequestError(f"vLLM chat completion failed: {exc}") from exc

    async def complete_stream(
        self,
        messages: list[LLMMessage],
        response_format: ResponseFormat | None,
        on_delta: Callable[[str], bool],
    ) -> LLMResponse:
        """Stream candidate JSON; the consumer can stop a generation loop early.

        Unlike ordinary completions, candidate extraction does not request the
        configured fixed output-token ceiling. The model/server context still
        bounds generation, and complete objects can be recovered on truncation.
        """
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": [self._message_to_api(message) for message in messages],
            "temperature": self.config.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if response_format is not None:
            params["response_format"] = response_format.model_dump(mode="json")
        chunks: list[str] = []
        finish_reason = None
        usage = None
        request_id = None
        model = self.config.model
        stream = None
        try:
            client = (
                self._client.with_options(max_retries=0)
                if hasattr(self._client, "with_options")
                else self._client
            )
            stream = await client.chat.completions.create(**params)
            async for event in stream:
                request_id = getattr(event, "id", None) or request_id
                model = getattr(event, "model", None) or model
                event_usage = getattr(event, "usage", None)
                if event_usage is not None:
                    usage = LLMUsage(
                        prompt_tokens=getattr(event_usage, "prompt_tokens", None),
                        completion_tokens=getattr(event_usage, "completion_tokens", None),
                        total_tokens=getattr(event_usage, "total_tokens", None),
                    )
                for choice in getattr(event, "choices", []):
                    finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                    delta = getattr(getattr(choice, "delta", None), "content", None)
                    if delta:
                        chunks.append(delta)
                        if on_delta(delta):
                            finish_reason = "client_repetition_stop"
                            return LLMResponse(
                                text="".join(chunks), provider="vllm", model=model,
                                request_id=request_id, usage=usage,
                                finish_reason=finish_reason,
                                request=self._safe_request(params),
                            )
        except Exception as exc:
            if not chunks:
                raise LLMRequestError(f"vLLM chat completion failed: {exc}") from exc
            finish_reason = "stream_error"
        finally:
            if stream is not None and hasattr(stream, "close"):
                await stream.close()
        return LLMResponse(
            text="".join(chunks), provider="vllm", model=model,
            request_id=request_id, usage=usage, finish_reason=finish_reason,
            request=self._safe_request(params),
        )
