"""SiliconFlow implementation of the provider-neutral LLM protocol."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from mcr.llm.errors import LLMConfigurationError, LLMRequestError
from mcr.llm.models import (
    ImagePart,
    LLMMessage,
    LLMResponse,
    LLMUsage,
    ResponseFormat,
    TextPart,
)
from mcr.llm.providers.siliconflow.config import SiliconFlowConfig
from mcr.llm.providers.siliconflow.images import image_part_to_url


class SiliconFlowClient:
    """Call SiliconFlow through its OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        config: SiliconFlowConfig,
        client: AsyncOpenAI | Any | None = None,
    ) -> None:
        self.config = config
        if client is None:
            if config.api_key is None:
                raise LLMConfigurationError(
                    "SiliconFlow API key is required; set SILICONFLOW_API_KEY"
                )
            client = AsyncOpenAI(
                api_key=config.api_key.get_secret_value(),
                base_url=config.base_url,
                timeout=config.timeout_seconds,
                max_retries=config.max_retries,
            )
        self._client = client

    @staticmethod
    def _message_to_api(message: LLMMessage) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, TextPart):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_part_to_url(part),
                            "detail": part.detail,
                        },
                    }
                )
        return {"role": message.role, "content": parts}

    @staticmethod
    def _safe_request(params: dict[str, Any]) -> dict[str, Any]:
        safe = {
            "model": params["model"],
            "temperature": params["temperature"],
            "max_tokens": params["max_tokens"],
            "messages": [],
        }
        if "response_format" in params:
            safe["response_format"] = params["response_format"]
        for message in params["messages"]:
            safe_parts = []
            for part in message["content"]:
                if part["type"] == "image_url":
                    image = dict(part["image_url"])
                    if image["url"].startswith("data:"):
                        image["url"] = "[image data URL omitted]"
                    safe_parts.append({"type": "image_url", "image_url": image})
                else:
                    safe_parts.append(dict(part))
            safe["messages"].append({"role": message["role"], "content": safe_parts})
        return safe

    async def complete(
        self,
        messages: list[LLMMessage],
        response_format: ResponseFormat | None = None,
    ) -> LLMResponse:
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": [self._message_to_api(message) for message in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        # SiliconFlow does not reliably support JSON mode for VL chat models;
        # with image input it can return a provider-side 500. The extraction
        # prompt and schema-repair pass still enforce structured output.
        if response_format is not None and "-VL-" not in self.config.model.upper():
            params["response_format"] = response_format.model_dump(mode="json")

        try:
            response = await self._client.chat.completions.create(**params)
            choices = getattr(response, "choices", None)
            if not choices:
                raise ValueError("provider response contains no choices")
            message = getattr(choices[0], "message", None)
            if message is None:
                raise ValueError("provider response choice contains no message")
            usage = None
            if getattr(response, "usage", None) is not None:
                usage = LLMUsage(
                    prompt_tokens=getattr(response.usage, "prompt_tokens", None),
                    completion_tokens=getattr(response.usage, "completion_tokens", None),
                    total_tokens=getattr(response.usage, "total_tokens", None),
                )
            return LLMResponse(
                text=getattr(message, "content", None) or "",
                provider="siliconflow",
                model=getattr(response, "model", None) or self.config.model,
                request_id=getattr(response, "id", None),
                usage=usage,
                finish_reason=getattr(choices[0], "finish_reason", None),
                request=self._safe_request(params),
            )
        except LLMRequestError:
            raise
        except Exception as exc:
            raise LLMRequestError(
                f"SiliconFlow chat completion failed: {exc}"
            ) from exc

