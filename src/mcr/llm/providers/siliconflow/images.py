"""Image conversion for SiliconFlow's OpenAI-compatible API."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from mcr.llm.errors import LLMConfigurationError
from mcr.llm.models import ALLOWED_IMAGE_MEDIA_TYPES, ImagePart


def image_part_to_url(part: ImagePart) -> str:
    if part.url is not None:
        return part.url
    if part.data_url is not None:
        return part.data_url

    path = Path(part.file_path or "")
    if not path.is_file():
        raise LLMConfigurationError(f"image path is not a readable file: {path}")
    media_type = part.media_type or mimetypes.guess_type(path.name)[0]
    if media_type not in ALLOWED_IMAGE_MEDIA_TYPES:
        raise LLMConfigurationError(
            f"unsupported image media type for {path}: {media_type or 'unknown'}"
        )
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise LLMConfigurationError(f"failed to read image file: {path}") from exc
    return f"data:{media_type};base64,{encoded}"

