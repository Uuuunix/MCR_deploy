"""Convert image sources into SiliconFlow-compatible values."""

from __future__ import annotations

import base64
import mimetypes

from mcr.embedding.errors import EmbeddingConfigurationError
from mcr.memory import ImageSource


def image_source_to_value(source: ImageSource) -> str:
    if source.url is not None:
        return source.url
    if source.data_url is not None:
        return source.data_url

    path = source.file_path
    if path is None or not path.is_file():
        raise EmbeddingConfigurationError(f"image path is not a readable file: {path}")
    media_type = mimetypes.guess_type(path.name)[0]
    if media_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise EmbeddingConfigurationError(
            f"unsupported image media type for {path}: {media_type or 'unknown'}"
        )
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise EmbeddingConfigurationError(f"failed to read image file: {path}") from exc
    return f"data:{media_type};base64,{encoded}"
