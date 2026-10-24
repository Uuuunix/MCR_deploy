"""Compact visual entity grounding before structured memory extraction."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from mcr.extraction.models import ExtractionRequest
from mcr.llm import ImagePart, LLMClient, LLMMessage, LLMUsage, TextPart


VISUAL_GROUNDING_PROMPT = """You are MCR's visual entity grounder.

Inspect every supplied image independently from any surrounding assistant claim.
For each image, identify the primary visible objects, places, documents, or
events using the most specific defensible common name. Also state a useful
broader functional category and normal purpose when visually supported. Do not
settle for vague labels such as item, thing, device, tool, contraption, or
kitchen gadget when the category is recognizable. Do not infer purchases,
ownership, prices, dates, or user intent; the downstream memory extractor will
combine visual identity with text evidence.

Return only compact JSON:
{"images":[{"source_message_id":"...","part_index":0,"entities":[{"name":"...","broader_type":"...","purpose":"..."}]}]}
"""


@dataclass(frozen=True)
class VisualGroundingResult:
    text: str | None = None
    request: dict | None = None
    usage: LLMUsage | None = None
    elapsed_ms: float = 0.0
    error: str | None = None


def current_images(request: ExtractionRequest) -> list[tuple[str, int, ImagePart]]:
    images = []
    for message in (request.current_user, request.current_assistant):
        if message is None:
            continue
        for part_index, part in enumerate(message.content):
            if isinstance(part, ImagePart):
                images.append((message.id, part_index, part))
    return images


async def ground_current_images(
    llm: LLMClient | None,
    request: ExtractionRequest,
) -> VisualGroundingResult:
    images = current_images(request)
    if llm is None or not images:
        return VisualGroundingResult()

    messages = [
        LLMMessage(role="system", content=[TextPart(text=VISUAL_GROUNDING_PROMPT)])
    ]
    for source_message_id, part_index, image in images:
        messages.append(
            LLMMessage(
                role="user",
                content=[
                    TextPart(
                        text=(
                            f"source_message_id={source_message_id!r}; "
                            f"part_index={part_index}"
                        )
                    ),
                    image,
                ],
            )
        )

    started = perf_counter()
    try:
        response = await llm.complete(messages)
    except Exception as exc:  # visual grounding is additive, never a hard dependency
        return VisualGroundingResult(
            elapsed_ms=(perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )
    return VisualGroundingResult(
        text=response.text.strip() or None,
        request=response.request,
        usage=response.usage,
        elapsed_ms=(perf_counter() - started) * 1000,
    )
