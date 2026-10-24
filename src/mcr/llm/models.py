"""Provider-neutral multimodal LLM request and response models."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from mcr.memory.common import DomainModel, NonEmptyStr

ALLOWED_IMAGE_MEDIA_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/gif"}
)


class TextPart(DomainModel):
    type: Literal["text"] = "text"
    text: NonEmptyStr


class ImagePart(DomainModel):
    type: Literal["image"] = "image"
    url: str | None = None
    data_url: str | None = None
    file_path: Path | None = None
    media_type: str | None = None
    detail: Literal["auto", "low", "high"] = "auto"

    @model_validator(mode="after")
    def validate_source(self) -> "ImagePart":
        sources = [self.url is not None, self.data_url is not None, self.file_path is not None]
        if sum(sources) != 1:
            raise ValueError("image part must define exactly one source")
        if self.url is not None and not self.url.startswith(("http://", "https://")):
            raise ValueError("image URL must use http or https")
        if self.media_type is not None and self.media_type not in ALLOWED_IMAGE_MEDIA_TYPES:
            raise ValueError(f"unsupported image media type: {self.media_type}")
        if self.data_url is not None:
            prefix, separator, encoded = self.data_url.partition(",")
            if (
                not separator
                or not prefix.startswith("data:image/")
                or not prefix.endswith(";base64")
            ):
                raise ValueError("image data_url must be a base64 image data URL")
            data_media_type = prefix.removeprefix("data:").removesuffix(";base64")
            if data_media_type not in ALLOWED_IMAGE_MEDIA_TYPES:
                raise ValueError(f"unsupported image media type: {data_media_type}")
            if self.media_type is not None and self.media_type != data_media_type:
                raise ValueError("media_type does not match image data_url")
            try:
                base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("image data_url contains invalid base64") from exc
        return self


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class LLMMessage(DomainModel):
    role: Literal["system", "user", "assistant"]
    content: list[ContentPart] = Field(min_length=1)


class ResponseFormat(DomainModel):
    type: Literal["json_object"] = "json_object"


class LLMUsage(DomainModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class LLMResponse(DomainModel):
    text: str
    provider: NonEmptyStr
    model: NonEmptyStr
    request_id: str | None = None
    usage: LLMUsage | None = None
    finish_reason: str | None = None
    request: dict[str, JsonValue] | None = None

