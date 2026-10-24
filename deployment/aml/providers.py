"""OpenAI-compatible LLM and text-embedding-v4 provider adapters."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from mcr.embedding import normalize_vector
from mcr.embedding.providers.siliconflow.images import image_source_to_value
from mcr.llm import ImagePart, LLMMessage, LLMResponse, LLMUsage, ResponseFormat, TextPart
from mcr.llm.providers.siliconflow.images import image_part_to_url
from mcr.memory import ImageSource


CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-v4"
_CAPTION_PROMPT = (
    "Describe the visible image faithfully for later memory retrieval. "
    "Name identifiable objects, readable text, distinguishing visual details, "
    "and spatial relationships. Do not infer unseen facts or use outside knowledge. "
    "Output one compact descriptive paragraph."
)


class AcademicLLM:
    def __init__(
        self, api_key: str, base_url: str, *, model: str = CHAT_MODEL, client: Any | None = None
    ) -> None:
        if not model.strip():
            raise ValueError("LLM model ID must not be empty")
        self._model = model
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=600)

    async def complete(
        self, messages: list[LLMMessage], response_format: ResponseFormat | None = None
    ) -> LLMResponse:
        api_messages = []
        for message in messages:
            parts = []
            for part in message.content:
                if isinstance(part, TextPart):
                    parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": image_part_to_url(part), "detail": part.detail},
                    })
            api_messages.append({"role": message.role, "content": parts})
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "temperature": 0,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format.model_dump(mode="json")
        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=choice.message.content or "",
            provider="openai",
            model=self._model,
            request_id=getattr(response, "id", None),
            finish_reason=getattr(choice, "finish_reason", None),
            usage=LLMUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            ) if usage is not None else None,
            # Never retain image Data URIs or evaluation text in request traces.
            request=None,
        )


class AcademicEmbeddings:
    """Image -> configured LLM caption -> text-embedding-v4, not native vision embeddings."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        llm: AcademicLLM,
        *,
        dimension: int = 1024,
        client: Any | None = None,
    ) -> None:
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=600)
        self._llm = llm
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return EMBEDDING_MODEL

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), 10):
            batch = texts[start:start + 10]
            response = await self._client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
                dimensions=self.dimension,
                encoding_format="float",
            )
            data = sorted(response.data, key=lambda item: item.index)
            if len(data) != len(batch) or [item.index for item in data] != list(range(len(batch))):
                raise ValueError("text-embedding-v4 returned an invalid vector batch")
            vectors.extend(normalize_vector(item.embedding, self.dimension) for item in data)
        return vectors

    async def embed_images(self, images: list[ImageSource]) -> list[list[float]]:
        descriptions = []
        for image in images:
            response = await self._llm.complete([
                LLMMessage(role="user", content=[
                    TextPart(text=_CAPTION_PROMPT),
                    ImagePart(data_url=image_source_to_value(image))
                    if image.data_url is not None
                    else ImagePart(url=image.url) if image.url is not None
                    else ImagePart(file_path=image.file_path),
                ])
            ])
            caption = response.text.strip()
            if not caption:
                raise ValueError("configured LLM returned an empty image description")
            descriptions.append(caption)
        return await self.embed_texts(descriptions)
