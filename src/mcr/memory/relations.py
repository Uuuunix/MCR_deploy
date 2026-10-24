"""Links from a memory back to its original text or image source."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, JsonValue, model_validator

from mcr.memory.common import DomainModel, NonEmptyStr


class ImageSource(DomainModel):
    """An accessible image source used to create a visual vector."""

    url: str | None = None
    file_path: Path | None = None
    data_url: str | None = None

    @model_validator(mode="after")
    def require_exactly_one_source(self) -> "ImageSource":
        sources = (self.url is not None, self.file_path is not None, self.data_url is not None)
        if sum(sources) != 1:
            raise ValueError("image_source must define exactly one of url, file_path, data_url")
        if self.url is not None and not self.url.startswith(("http://", "https://")):
            raise ValueError("image URL must use http or https")
        if self.data_url is not None:
            prefix, separator, encoded = self.data_url.partition(",")
            if not separator or not prefix.startswith("data:image/") or not prefix.endswith(";base64"):
                raise ValueError("image data_url must be a base64 image data URL")
            try:
                base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("image data_url contains invalid base64") from exc
        return self


class SourceRelation(DomainModel):
    """Traceable relation between a memory and an upstream source."""

    id: UUID = Field(default_factory=uuid4)
    source_type: Literal["text", "image"]
    source_id: NonEmptyStr
    relation_type: NonEmptyStr = "derived_from"
    locator: dict[str, JsonValue] | None = None
    content_hash: NonEmptyStr | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)
    image_source: ImageSource | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_image_source(self) -> "SourceRelation":
        if self.image_source is not None and self.source_type != "image":
            raise ValueError("image_source is only valid for image relations")
        return self

