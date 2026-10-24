"""Materialize image parts into immutable bytes."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from mcr.llm import ImagePart
from mcr.source_store.errors import SourceImageError


@dataclass(frozen=True)
class ResolvedImage:
    data: bytes
    media_type: str
    source_metadata: dict[str, str]


class ImageResolver(Protocol):
    async def resolve(self, image: ImagePart) -> ResolvedImage: ...


class DefaultImageResolver:
    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def resolve(self, image: ImagePart) -> ResolvedImage:
        return await asyncio.to_thread(self._resolve_sync, image)

    def _resolve_sync(self, image: ImagePart) -> ResolvedImage:
        try:
            if image.data_url is not None:
                header, encoded = image.data_url.split(",", 1)
                media_type = header.removeprefix("data:").removesuffix(";base64")
                return ResolvedImage(
                    base64.b64decode(encoded, validate=True),
                    media_type,
                    {"input_type": "data_url"},
                )
            if image.file_path is not None:
                data = image.file_path.read_bytes()
                media_type = image.media_type or mimetypes.guess_type(image.file_path.name)[0]
                if not media_type:
                    raise SourceImageError("image media type is unknown")
                return ResolvedImage(
                    data,
                    media_type,
                    {"input_type": "file", "file_name": image.file_path.name},
                )
            if image.url is not None:
                with urllib.request.urlopen(image.url, timeout=self.timeout_seconds) as response:
                    data = response.read()
                    media_type = response.headers.get_content_type()
                resolved = ResolvedImage(
                    data,
                    media_type,
                    {"input_type": "url", "url": image.url},
                )
                if not resolved.media_type.startswith("image/"):
                    raise SourceImageError("URL did not return an image media type")
                return resolved
        except SourceImageError:
            raise
        except Exception as exc:
            raise SourceImageError("failed to materialize image source") from exc
        raise SourceImageError("image part has no source")
