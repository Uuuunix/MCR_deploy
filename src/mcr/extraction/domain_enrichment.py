"""Generate retrieval-oriented D values after multimodal x/r discovery."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from time import perf_counter

from pydantic import ValidationError

from mcr.extraction.errors import ExtractionResponseError
from mcr.extraction.models import DomainEnrichment, DomainEnrichmentPayload
from mcr.llm import LLMClient, LLMMessage, LLMUsage, ResponseFormat, TextPart
from mcr.memory import MemoryContent
from mcr.memory.domains import MemoryDomain


DOMAIN_SYSTEM_PROMPT = """You generate retrieval descriptions D for memory candidates whose x values already exist.

You MUST NOT add, remove, merge, split, or rewrite candidates. Return flat d_j
facet records for every candidate_index supplied by MCR. MCR groups them into D
objects. The supplied x is the complete and only source of truth. Do not create
facts absent from x and do not request or infer details from raw messages.

Classify each x into any applicable subset of: biography, experiences,
preferences, social_circle, work, psychometrics. Emit each domain at most once.
Every emitted domain D requires a retrieval-oriented core facet and may contain
additional atomic facets. A d_j must not copy x.content verbatim. It must repeat
enough of the entity or event anchor to remain independently understandable.
Do not include id or embedding; MCR generates them.

Every d_j is {"facet_type":"...","text":"...","attributes":{...}}.
Use only these facet types and exact attribute keys; omit unknown keys:
- core: subject, predicate, object
- entity_type: entity, type
- attribute: entity, name, value, unit
- quantity: entity, amount, unit
- cost: item, amount, currency
- payment_method: item, method
- time: event, value, normalized_value, precision
- location: event, place, city, country
- participant: event, person, role
- relationship: person, relationship_type, target
- emotion: event, emotion, intensity
- evaluation: target, sentiment, strength
- preference: target, polarity, strength
- purpose: action, purpose
- status: subject, status
- outcome: action, outcome
- frequency: action, frequency
- duration: event, value, unit
- score: framework, trait, value, scale

Numeric limits: intensity and strength are 0..1; polarity is -1..1. sentiment is
positive, neutral, negative, or mixed. time.precision is exact, day, week,
month, year, relative, or unknown. attributes is always required. Preserve
exact amounts, currencies, quantities, dates, locations, and entity types.

Return only a flat list with no nested domains/items:
{"facets":[{"candidate_index":0,"domain":"<applicable domain>","facet_type":"core","text":"<retrieval description derived from this x>","attributes":{}}]}
Angle-bracket values are schema placeholders. Never copy them or introduce
example facts. Ground every d_j in its exact candidate x.
"""

_JSON_FENCE = re.compile(
    r"\A\s*```(?:json)?\s*\n?(.*?)\n?```\s*\Z", re.DOTALL | re.IGNORECASE
)

_FACET_ATTRIBUTE_KEYS = {
    "core": {"subject", "predicate", "object"},
    "entity_type": {"entity", "type"},
    "attribute": {"entity", "name", "value", "unit"},
    "quantity": {"entity", "amount", "unit"},
    "cost": {"item", "amount", "currency"},
    "payment_method": {"item", "method"},
    "time": {"event", "value", "normalized_value", "precision"},
    "location": {"event", "place", "city", "country"},
    "participant": {"event", "person", "role"},
    "relationship": {"person", "relationship_type", "target"},
    "emotion": {"event", "emotion", "intensity"},
    "evaluation": {"target", "sentiment", "strength"},
    "preference": {"target", "polarity", "strength"},
    "purpose": {"action", "purpose"},
    "status": {"subject", "status"},
    "outcome": {"action", "outcome"},
    "frequency": {"action", "frequency"},
    "duration": {"event", "value", "unit"},
    "score": {"framework", "trait", "value", "scale"},
}
_FACET_TYPE_ALIASES = {
    "event": "core",
    "fact": "core",
    "description": "core",
}


def _normalize_facet(item: dict) -> dict:
    """Tolerate common small-model schema drift without another LLM call."""

    normalized = dict(item)
    facet_type = normalized.get("facet_type")
    if isinstance(facet_type, str):
        facet_type = _FACET_TYPE_ALIASES.get(facet_type, facet_type)
        normalized["facet_type"] = facet_type
    attributes = normalized.get("attributes")
    if isinstance(attributes, dict) and facet_type in _FACET_ATTRIBUTE_KEYS:
        allowed = _FACET_ATTRIBUTE_KEYS[facet_type]
        normalized["attributes"] = {
            key: value for key, value in attributes.items() if key in allowed
        }
    elif not isinstance(attributes, dict):
        normalized["attributes"] = {}
    return normalized


@dataclass
class DomainGenerationResult:
    domains_by_index: dict[int, list]
    messages: list[LLMMessage] = field(default_factory=list)
    raw_response: str | None = None
    provider_request: dict | None = None
    usage: LLMUsage | None = None
    elapsed_ms: float = 0.0
    response_attempts: list[str] = field(default_factory=list)
    fallback_count: int = 0


def build_domain_prompt(
    indexed_memories: list[tuple[int, MemoryContent]],
) -> list[LLMMessage]:
    candidates = [
        {
            "candidate_index": index,
            "x": {
                "kind": x.kind,
                "content": x.content,
                "attributes": x.attributes,
            },
        }
        for index, x in indexed_memories
    ]
    messages = [
        LLMMessage(role="system", content=[TextPart(text=DOMAIN_SYSTEM_PROMPT)]),
        LLMMessage(
            role="user",
            content=[
                TextPart(
                    text=(
                        "<CANDIDATES>\n"
                        + json.dumps(candidates, ensure_ascii=False)
                        + "\n</CANDIDATES>"
                    )
                )
            ],
        ),
    ]
    messages.append(
        LLMMessage(
            role="user",
            content=[
                TextPart(
                    text=(
                        "Generate D for every supplied candidate_index. Do not "
                        "change x and return only the JSON object."
                    )
                )
            ],
        )
    )
    return messages


def parse_domain_response(text: str) -> DomainEnrichmentPayload:
    candidate = text.strip()
    fence = _JSON_FENCE.fullmatch(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        raw = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ExtractionResponseError(
            "D generation response is not valid JSON",
            raw_response=text,
            issues=[{"loc": ["json", exc.lineno, exc.colno], "msg": exc.msg}],
        ) from exc
    if isinstance(raw, dict) and isinstance(raw.get("facets"), list):
        grouped: dict[tuple[int, str], list[dict]] = {}
        for facet_index, facet in enumerate(raw["facets"]):
            if not isinstance(facet, dict):
                raise ExtractionResponseError(
                    "D generation response does not match schema",
                    raw_response=text,
                    issues=[{"loc": ["facets", facet_index], "msg": "facet must be an object"}],
                )
            reserved = [name for name in ("id", "embedding") if name in facet]
            if reserved:
                raise ExtractionResponseError(
                    "D generation contains fields reserved for MCR",
                    raw_response=text,
                    issues=[
                        {
                            "loc": ["facets", facet_index, name],
                            "msg": f"{name} is generated by MCR",
                        }
                        for name in reserved
                    ],
                )
            candidate_index = facet.get("candidate_index")
            domain = facet.get("domain")
            if not isinstance(candidate_index, int) or candidate_index < 0:
                raise ExtractionResponseError(
                    "D generation response does not match schema",
                    raw_response=text,
                    issues=[{"loc": ["facets", facet_index, "candidate_index"], "msg": "must be a non-negative integer"}],
                )
            if not isinstance(domain, str):
                raise ExtractionResponseError(
                    "D generation response does not match schema",
                    raw_response=text,
                    issues=[{"loc": ["facets", facet_index, "domain"], "msg": "must be a domain string"}],
                )
            item = {
                key: value
                for key, value in facet.items()
                if key not in {"candidate_index", "domain"}
            }
            item = _normalize_facet(item)
            grouped.setdefault((candidate_index, domain), []).append(item)
        by_candidate: dict[int, list] = {}
        try:
            for (candidate_index, domain), items in grouped.items():
                memory_domain = MemoryDomain.model_validate(
                    {"domain": domain, "items": items}
                )
                by_candidate.setdefault(candidate_index, []).append(memory_domain)
            return DomainEnrichmentPayload(
                enrichments=[
                    DomainEnrichment(candidate_index=index, domains=domains)
                    for index, domains in sorted(by_candidate.items())
                ]
            )
        except ValidationError as exc:
            raise ExtractionResponseError(
                "D generation response does not match schema",
                raw_response=text,
                issues=exc.errors(
                    include_url=False, include_input=False, include_context=False
                ),
            ) from exc

    reserved_issues = []
    if isinstance(raw, dict) and isinstance(raw.get("enrichments"), list):
        for enrichment_index, enrichment in enumerate(raw["enrichments"]):
            if not isinstance(enrichment, dict):
                continue
            for domain_index, domain in enumerate(enrichment.get("domains", [])):
                if not isinstance(domain, dict):
                    continue
                for item_index, item in enumerate(domain.get("items", [])):
                    if not isinstance(item, dict):
                        continue
                    for field_name in ("id", "embedding"):
                        if field_name in item:
                            reserved_issues.append(
                                {
                                    "loc": [
                                        "enrichments",
                                        enrichment_index,
                                        "domains",
                                        domain_index,
                                        "items",
                                        item_index,
                                        field_name,
                                    ],
                                    "msg": (
                                        f"{field_name} is generated by MCR and "
                                        "must not be supplied by the LLM"
                                    ),
                                }
                            )
    if reserved_issues:
        raise ExtractionResponseError(
            "D generation contains fields reserved for MCR",
            raw_response=text,
            issues=reserved_issues,
        )
    if isinstance(raw, dict) and isinstance(raw.get("enrichments"), list):
        for enrichment in raw["enrichments"]:
            if not isinstance(enrichment, dict):
                continue
            for domain in enrichment.get("domains", []):
                if not isinstance(domain, dict) or not isinstance(domain.get("items"), list):
                    continue
                domain["items"] = [
                    _normalize_facet(item) if isinstance(item, dict) else item
                    for item in domain["items"]
                ]
    try:
        return DomainEnrichmentPayload.model_validate(raw)
    except ValidationError as exc:
        raise ExtractionResponseError(
            "D generation response does not match schema",
            raw_response=text,
            issues=exc.errors(
                include_url=False, include_input=False, include_context=False
            ),
        ) from exc


def _validate_indices(
    payload: DomainEnrichmentPayload, expected: set[int]
) -> dict[int, list]:
    indices = [item.candidate_index for item in payload.enrichments]
    if len(indices) != len(set(indices)):
        raise ExtractionResponseError("D generation returned duplicate candidate_index")
    actual = set(indices)
    if actual != expected:
        raise ExtractionResponseError(
            "D generation candidate_index coverage mismatch",
            issues=[
                {
                    "loc": ["enrichments"],
                    "msg": (
                        f"expected {sorted(expected)}, received {sorted(actual)}"
                    ),
                }
            ],
        )
    return {item.candidate_index: item.domains for item in payload.enrichments}


async def generate_domains(
    llm: LLMClient,
    indexed_memories: list[tuple[int, MemoryContent]],
) -> DomainGenerationResult:
    if not indexed_memories:
        return DomainGenerationResult(domains_by_index={})

    messages = build_domain_prompt(indexed_memories)
    started = perf_counter()
    response = await llm.complete(messages, ResponseFormat())
    attempts = [response.text]
    expected = {index for index, _ in indexed_memories}
    try:
        batch_payload = parse_domain_response(response.text)
        indices = [item.candidate_index for item in batch_payload.enrichments]
        if len(indices) != len(set(indices)) or not set(indices).issubset(expected):
            raise ExtractionResponseError(
                "D generation returned duplicate or unexpected candidate_index"
            )
        domains = {
            item.candidate_index: item.domains
            for item in batch_payload.enrichments
        }
    except ExtractionResponseError:
        # Invalid batch JSON/schema cannot be trusted, so retry every candidate.
        domains = {}

    # Preserve every valid batch result. Only missing candidates fall back to
    # individual calls, preventing one omission from multiplying latency.
    missing = [pair for pair in indexed_memories if pair[0] not in domains]
    for indexed_memory in missing:
        single_messages = build_domain_prompt([indexed_memory])
        single_response = await llm.complete(single_messages, ResponseFormat())
        attempts.append(single_response.text)
        try:
            payload = parse_domain_response(single_response.text)
            domains.update(_validate_indices(payload, {indexed_memory[0]}))
            continue
        except ExtractionResponseError as first_error:
            repair_messages = [
                *single_messages,
                LLMMessage(
                    role="assistant",
                    content=[TextPart(text=single_response.text or "[empty response]")],
                ),
                LLMMessage(
                    role="user",
                    content=[
                        TextPart(
                            text=(
                                "Correct only the JSON syntax and D schema. Keep "
                                "the same candidate_index and supported meaning. "
                                "Return a flat {facets:[...]} list; never emit "
                                "nested domains/items. Each facet attributes "
                                "object may contain only "
                                "the flat keys allowed for its facet_type; never "
                                "nest another facet inside attributes. Return only "
                                "the corrected JSON object. Validation errors: "
                                + json.dumps(
                                    first_error.issues,
                                    ensure_ascii=False,
                                    default=str,
                                )
                            )
                        )
                    ],
                ),
            ]
            repaired = await llm.complete(repair_messages, ResponseFormat())
            attempts.append(repaired.text)
            payload = parse_domain_response(repaired.text)
            domains.update(_validate_indices(payload, {indexed_memory[0]}))

    return DomainGenerationResult(
        domains_by_index=domains,
        messages=messages,
        raw_response=response.text,
        provider_request=response.request,
        usage=response.usage,
        elapsed_ms=(perf_counter() - started) * 1000,
        response_attempts=attempts,
        fallback_count=len(missing),
    )
