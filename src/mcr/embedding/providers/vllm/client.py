"""Qwen3-VL multimodal embeddings through vLLM's chat-embedding API."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI
from openai.types import CreateEmbeddingResponse

from mcr.embedding.errors import EmbeddingRequestError
from mcr.embedding.normalize import normalize_vector
from mcr.embedding.providers.siliconflow.images import image_source_to_value
from mcr.embedding.providers.vllm.config import VLLMEmbeddingConfig
from mcr.memory import ImageSource


class VLLMEmbeddingClient:
    def __init__(
        self, config: VLLMEmbeddingConfig, client: AsyncOpenAI | Any | None = None
    ) -> None:
        self.config = config
        self._client = client or AsyncOpenAI(
            api_key="EMPTY",
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    @property
    def model_id(self) -> str:
        return self.config.model

    @property
    def dimension(self) -> int:
        return self.config.dimension

    async def _embed(self, content: list[dict[str, Any]]) -> list[float]:
        # vLLM's Qwen3-VL example uses a chat-form embedding request. The
        # instruction and unfinished assistant turn must match for text/images.
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": self.config.instruction}]},
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            ],
            "encoding_format": "float",
            "continue_final_message": True,
            "add_special_tokens": True,
        }
        try:
            response = await self._client.post(
                "/embeddings", body=body, cast_to=CreateEmbeddingResponse
            )
            if len(response.data) != 1:
                raise ValueError(f"expected one vector, got {len(response.data)}")
            return normalize_vector(response.data[0].embedding, self.dimension)
        except Exception as exc:
            raise EmbeddingRequestError(f"vLLM embedding request failed: {exc}") from exc

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [await self._embed([{"type": "text", "text": text}]) for text in texts]

    async def embed_images(self, images: list[ImageSource]) -> list[list[float]]:
        return [
            await self._embed([
                {"type": "image_url", "image_url": {"url": image_source_to_value(image)}},
                {"type": "text", "text": ""},
            ])
            for image in images
        ]
