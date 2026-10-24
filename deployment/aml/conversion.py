"""Lossless AML text/image parsing and source-image reconstruction."""

from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
from uuid import UUID

from mcr.extraction import ConversationMessage
from mcr.llm import ImagePart, TextPart
from mcr.memory import ImageSource
from mcr.retrieval import RetrievalQuery
from mcr.source_store import StoredMessage

from deployment.aml.schemas import AMLImagePart, AMLMessage, AMLPart, AMLTextPart, SearchMemory

ALLOWED_MEDIA = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_ADD_IMAGE_BYTES = 30 * 1024 * 1024
MAX_SEARCH_IMAGE_BYTES = 30 * 1024 * 1024


def decode_image_url(value: str) -> tuple[str, bytes]:
    prefix, comma, encoded = value.partition(",")
    media_type = prefix.removeprefix("data:").removesuffix(";base64")
    if not comma or not prefix.startswith("data:") or not prefix.endswith(";base64"):
        raise ValueError("image_url.url must be a Base64 image Data URI")
    if media_type not in ALLOWED_MEDIA:
        raise ValueError("image must be JPEG, PNG, or WebP")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("image_url.url contains invalid Base64") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("decoded image must be between 1 byte and 10 MiB")
    return media_type, raw


def source_content(parts: str | list[AMLPart]) -> tuple[list[TextPart | ImagePart], int]:
    if isinstance(parts, str):
        return [TextPart(text=parts)], 0
    output = []
    image_bytes = 0
    for part in parts:
        if isinstance(part, AMLTextPart):
            output.append(TextPart(text=part.text))
        elif isinstance(part, AMLImagePart):
            media_type, raw = decode_image_url(part.image_url.url)
            image_bytes += len(raw)
            output.append(ImagePart(data_url=part.image_url.url, media_type=media_type))
    return output, image_bytes


def message_time(message: AMLMessage, fallback: datetime) -> datetime:
    if message.timestamp is None:
        return fallback
    try:
        return datetime.fromtimestamp(message.timestamp / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError("timestamp is outside the supported Unix millisecond range") from exc


def stored_to_conversation(message: StoredMessage) -> ConversationMessage:
    parts = []
    for part in message.parts:
        if part.part_type == "text":
            parts.append(TextPart(text=part.text_content or ""))
        elif part.image_bytes and part.media_type:
            uri = f"data:{part.media_type};base64,{base64.b64encode(part.image_bytes).decode('ascii')}"
            parts.append(ImagePart(data_url=uri, media_type=part.media_type))
        else:
            raise ValueError("stored image is unavailable")
    return ConversationMessage(
        id=str(message.id),
        role=message.role,
        content=parts,
        created_at=message.created_at,
        turn_id=message.turn_id,
    )


def retrieval_query(value: str | list[AMLPart]) -> RetrievalQuery:
    if isinstance(value, str):
        return RetrievalQuery(texts=[value])
    texts = []
    images = []
    for part in value:
        if isinstance(part, AMLTextPart):
            texts.append(part.text)
        else:
            decode_image_url(part.image_url.url)
            images.append(ImageSource(data_url=part.image_url.url))
    return RetrievalQuery(texts=texts, images=images)


def memory_to_wire(retrieved, source_store) -> tuple[SearchMemory, int]:
    memory = retrieved.memory
    image_parts = []
    total_bytes = 0
    seen = set()
    for relation in memory.relations:
        if relation.source_type != "image":
            continue
        locator = relation.locator or {}
        key = (relation.source_id, locator.get("part_index"))
        if key in seen:
            continue
        seen.add(key)
        message = source_store.get_message(UUID(relation.source_id))
        matching = [
            item for item in message.parts
            if item.part_index == locator.get("part_index") and item.part_type == "image"
        ]
        if len(matching) != 1 or not matching[0].image_bytes or not matching[0].media_type:
            raise ValueError("a memory-linked source image is unavailable")
        part = matching[0]
        total_bytes += len(part.image_bytes)
        if len(part.image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("a memory-linked source image exceeds 10 MiB")
        uri = f"data:{part.media_type};base64,{base64.b64encode(part.image_bytes).decode('ascii')}"
        image_parts.append(AMLImagePart(type="image_url", image_url={"url": uri}))
    content = memory.x.content
    if image_parts:
        wire_content = [AMLTextPart(type="text", text=content), *image_parts]
    else:
        wire_content = content
    return SearchMemory(
        id=str(memory.id),
        content=wire_content,
        score=retrieved.score,
        created_at=memory.meta.created_at.isoformat().replace("+00:00", "Z"),
    ), total_bytes
