"""SiliconFlow multimodal embedding adapter."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from mcr.embedding.errors import EmbeddingConfigurationError, EmbeddingRequestError
from mcr.embedding.normalize import normalize_vector
from mcr.embedding.providers.siliconflow.config import SiliconFlowEmbeddingConfig
from mcr.embedding.providers.siliconflow.images import image_source_to_value
from mcr.memory import ImageSource


class SiliconFlowEmbeddingClient:
    def __init__(
        self,
        config: SiliconFlowEmbeddingConfig,
        client: AsyncOpenAI | Any | None = None,
    ) -> None:
        self.config = config
        if client is None:
            if config.api_key is None:
                raise EmbeddingConfigurationError(
                    "SiliconFlow API key is required; set SILICONFLOW_API_KEY"
                )
            client = AsyncOpenAI(
                api_key=config.api_key.get_secret_value(),
                base_url=config.base_url,
                timeout=config.timeout_seconds,
                max_retries=config.max_retries,
            )
        self._client = client

    @property
    def model_id(self) -> str:
        return self.config.model

    @property
    def dimension(self) -> int:
        return self.config.dimension

    async def _embed(self, inputs: list[dict[str, str]]) -> list[list[float]]:
        if not inputs:
            return []
        try:
            response = await self._client.embeddings.create(
                model=self.config.model,
                input=inputs,
                dimensions=self.config.dimension,
                encoding_format="float",
            )
            data = sorted(response.data, key=lambda item: item.index)
            if len(data) != len(inputs):
                raise EmbeddingRequestError(
                    f"SiliconFlow returned {len(data)} vectors for {len(inputs)} inputs"
                )
            if [item.index for item in data] != list(range(len(inputs))):
                raise EmbeddingRequestError(
                    "SiliconFlow returned invalid embedding indexes"
                )
            return [normalize_vector(item.embedding, self.dimension) for item in data]
        except EmbeddingRequestError:
            raise
        except Exception as exc:
            raise EmbeddingRequestError(
                f"SiliconFlow embedding request failed: {exc}"
            ) from exc

    async def _embed_separately(
        self, inputs: list[dict[str, str]]
    ) -> list[list[float]]:
        vectors = []
        for value in inputs:
            [vector] = await self._embed([value])
            vectors.append(vector)
        return vectors

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self._embed_separately([{"text": text} for text in texts])

    async def embed_images(self, images: list[ImageSource]) -> list[list[float]]:
        return await self._embed_separately(
            [{"image": image_source_to_value(image)} for image in images]
        )
