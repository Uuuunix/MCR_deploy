"""Strict response parsing with one fenced-JSON compatibility fallback."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from pydantic import ValidationError

from mcr.extraction.errors import ExtractionResponseError
from mcr.extraction.models import ExtractionPayload, ExtractedMemory

_JSON_FENCE = re.compile(r"\A\s*```(?:json)?\s*\n?(.*?)\n?```\s*\Z", re.DOTALL | re.IGNORECASE)


def _resolve_unambiguous_image_refs(
    raw: object, image_parts_by_message: Mapping[str, list[int]] | None
) -> object:
    """Accept a bare current message ID only when it identifies one image."""
    if not image_parts_by_message or not isinstance(raw, dict):
        return raw
    memories = raw.get("memories")
    if not isinstance(memories, list):
        return raw
    for memory in memories:
        if not isinstance(memory, dict) or not isinstance(memory.get("image_refs"), list):
            continue
        memory["image_refs"] = [
            {"source_message_id": ref, "part_index": image_parts_by_message[ref][0]}
            if isinstance(ref, str) and len(image_parts_by_message.get(ref, [])) == 1
            else ref
            for ref in memory["image_refs"]
        ]
    return raw


def _discard_legacy_extraction_fields(raw: object) -> object:
    """Ignore obsolete LLM fields rather than retrying a useful extraction."""
    if isinstance(raw, dict) and isinstance(raw.get("memories"), list):
        for memory in raw["memories"]:
            if isinstance(memory, dict):
                memory.pop("attributes", None)
                memory.pop("links", None)
    return raw


def parse_extraction_response(
    text: str, *, image_parts_by_message: Mapping[str, list[int]] | None = None
) -> ExtractionPayload:
    candidate = text.strip()
    fence = _JSON_FENCE.fullmatch(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        raw = _resolve_unambiguous_image_refs(
            _discard_legacy_extraction_fields(json.loads(candidate)),
            image_parts_by_message,
        )
    except json.JSONDecodeError as exc:
        raise ExtractionResponseError(
            "LLM response is not valid JSON",
            raw_response=text,
            issues=[
                {
                    "loc": ["json", exc.lineno, exc.colno],
                    "msg": exc.msg,
                    "type": "json_invalid",
                }
            ],
        ) from exc
    try:
        return ExtractionPayload.model_validate(raw)
    except ValidationError as exc:
        # Pydantic may place the original ValueError instance in ``ctx`` for
        # model validators.  Extraction issues cross both an LLM repair prompt
        # and an HTTP boundary, so keep only JSON-safe validation details.
        issues = exc.errors(
            include_url=False,
            include_input=False,
            include_context=False,
        )
        raise ExtractionResponseError(
            "LLM response does not match extraction schema",
            raw_response=text,
            issues=issues,
        ) from exc


def deduplicate_memories(memories: list[ExtractedMemory]) -> list[ExtractedMemory]:
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    result = []
    for memory in memories:
        key = (
            memory.kind.casefold().strip(),
            " ".join(memory.content.casefold().split()),
            tuple(
                sorted(
                    memory.source_message_ids
                    or [
                        f"{ref.source_message_id}:{ref.part_index}"
                        for ref in memory.source_refs
                    ]
                )
            ),
        )
        if key not in seen:
            seen.add(key)
            result.append(memory)
    return result


def salvage_valid_memories(
    text: str,
    *,
    image_parts_by_message: Mapping[str, list[int]] | None = None,
) -> tuple[ExtractionPayload, list[dict[str, object]]]:
    """Validate memories independently after whole-response repair is exhausted."""

    candidate = text.strip()
    fence = _JSON_FENCE.fullmatch(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        raw = _resolve_unambiguous_image_refs(
            json.loads(candidate), image_parts_by_message
        )
    except json.JSONDecodeError:
        return ExtractionPayload(), []
    if not isinstance(raw, dict) or not isinstance(raw.get("memories"), list):
        return ExtractionPayload(), []

    valid = []
    discarded = []
    for index, memory in enumerate(raw["memories"]):
        try:
            valid.append(ExtractedMemory.model_validate(memory))
        except ValidationError as exc:
            issues = exc.errors(
                include_url=False,
                include_input=False,
                include_context=False,
            )
            for issue in issues:
                issue["loc"] = list(issue.get("loc", ()))
            discarded.append(
                {
                    "index": index,
                    "content": memory.get("content") if isinstance(memory, dict) else None,
                    "reason": "candidate does not match extraction schema",
                    "issues": issues,
                }
            )
    return ExtractionPayload(memories=valid), discarded

