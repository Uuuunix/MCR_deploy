"""Provider-neutral multimodal embedding port."""

from __future__ import annotations

from typing import Protocol

from mcr.memory import ImageSource


class MultimodalEmbeddingClient(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_images(self, images: list[ImageSource]) -> list[list[float]]: ...
