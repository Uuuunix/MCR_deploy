"""Generate flat Core + Dynamic retrieval descriptions from one x."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

from pydantic import Field, ValidationError

from mcr.llm import LLMClient, LLMMessage, LLMUsage, ResponseFormat, TextPart
from mcr.memory import MemoryContent, RetrievalDescription
from mcr.memory.common import DomainModel, NonEmptyStr


DESCRIPTION_SYSTEM_PROMPT = """You generate retrieval descriptions D for one immutable memory x.

The supplied x is the complete and only source of truth. Do not access or infer
information from conversation history, external knowledge, or unstated common
sense. Never rewrite, split, merge, or otherwise modify x.

Return one core description that preserves x's complete meaning, followed by
any number of dynamic descriptions covering distinct query intents supported by
x. Dynamic descriptions should focus on different useful aspects such as an
entity, amount, time, location, purpose, state, or relationship only when that
information exists in x. Do not follow a fixed category list.

Every description must be independently understandable: repeat the relevant
entity and event anchor instead of writing fragments such as "the price is 35"
or "the place is Wuhan". Remove paraphrase-only duplicates. Preserve exact
amounts, currencies, quantities, dates, locations, and named entities.

Return only this JSON shape:
{"facets":[{"facet_type":"core","text":"<complete retrieval description>"},{"facet_type":"dynamic","text":"<complementary retrieval description>"}]}

facet_type must be core or dynamic. Do not return candidate_index, domain,
attributes, IDs, embeddings, scores, or explanations. Angle-bracket values are
schema placeholders and must never be copied.
"""

_JSON_FENCE = re.compile(
    r"\A\s*```(?:json)?\s*\n?(.*?)\n?```\s*\Z", re.DOTALL | re.IGNORECASE
)


@dataclass
class DescriptionGenerationResult:
    descriptions: list[RetrievalDescription]
    messages: list[LLMMessage] = field(default_factory=list)
    raw_response: str | None = None
    provider_request: dict | None = None
    usage: LLMUsage | None = None
    elapsed_ms: float = 0.0
    error: str | None = None


class _GeneratedFacet(DomainModel):
    facet_type: Literal["core", "dynamic"]
    text: NonEmptyStr


class _GeneratedFacetPayload(DomainModel):
    facets: list[_GeneratedFacet] = Field(default_factory=list)


def build_description_prompt(x: MemoryContent) -> list[LLMMessage]:
    return [
        LLMMessage(
            role="system",
            content=[TextPart(text=DESCRIPTION_SYSTEM_PROMPT)],
        ),
        LLMMessage(
            role="user",
            content=[
                TextPart(
                    text=json.dumps(
                        {"content": x.content},
                        ensure_ascii=False,
                    )
                )
            ],
        ),
    ]


def fallback_descriptions(x: MemoryContent) -> list[RetrievalDescription]:
    return [RetrievalDescription(facet_type="core", text=x.content)]


def parse_description_response(
    text: str, x: MemoryContent
) -> list[RetrievalDescription]:
    return _parse_description_response(text, x)[0]


def _parse_description_response(
    text: str, x: MemoryContent
) -> tuple[list[RetrievalDescription], str | None]:
    candidate = text.strip()
    fence = _JSON_FENCE.fullmatch(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        raw = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return fallback_descriptions(x), "invalid JSON; used x as core"
    try:
        payload = _GeneratedFacetPayload.model_validate(raw)
    except ValidationError:
        return fallback_descriptions(x), "invalid description schema; used x as core"

    core: RetrievalDescription | None = None
    dynamic: list[RetrievalDescription] = []
    seen: set[str] = set()
    for value in payload.facets:
        facet_type = value.facet_type
        facet_text = value.text.strip()
        key = " ".join(facet_text.casefold().split())
        if key in seen:
            continue
        seen.add(key)
        if facet_type == "core" and core is None:
            core = RetrievalDescription(facet_type="core", text=facet_text)
        else:
            dynamic.append(
                RetrievalDescription(facet_type="dynamic", text=facet_text)
            )

    if core is None:
        fallback = fallback_descriptions(x)[0]
        fallback_key = " ".join(fallback.text.casefold().split())
        dynamic = [
            item
            for item in dynamic
            if " ".join(item.text.casefold().split()) != fallback_key
        ]
        core = fallback
    return [core, *dynamic], None


async def generate_descriptions(
    llm: LLMClient, x: MemoryContent
) -> DescriptionGenerationResult:
    messages = build_description_prompt(x)
    started = perf_counter()
    try:
        response = await llm.complete(messages, ResponseFormat())
    except Exception as exc:
        return DescriptionGenerationResult(
            descriptions=fallback_descriptions(x),
            messages=messages,
            elapsed_ms=(perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )
    descriptions, parse_error = _parse_description_response(response.text, x)
    return DescriptionGenerationResult(
        descriptions=descriptions,
        messages=messages,
        raw_response=response.text,
        provider_request=response.request,
        usage=response.usage,
        elapsed_ms=(perf_counter() - started) * 1000,
        error=parse_error,
    )
