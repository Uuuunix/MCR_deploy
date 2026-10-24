"""One-pass, query-aware compression of retrieved multimodal memories."""

from __future__ import annotations

import base64
import json
import time

from mcr.context_integration.default import DefaultMemoryContextIntegrator
from mcr.context_integration.errors import (
    CriticalIntegrationError,
    SourceImageUnavailable,
)
from mcr.context_integration.models import (
    IntegrationImageRef,
    MemoryIntegrationRequest,
    MemoryIntegrationResult,
)
from mcr.llm import ImagePart, LLMClient, LLMMessage, ResponseFormat, TextPart
from mcr.source_store import SQLiteSourceStore


SYSTEM_PROMPT = """You integrate retrieved long-term memories for a question.
Use the query and ALL provided memories, including their retrieval descriptions D and original images.
In ONE pass, drop memories irrelevant to the query and compress the relevant evidence into:
1. summary: a concise abstract account of the relevant information.
2. residuals: query-relevant details not already stated in the summary (especially exact numbers, units, objects, dates, locations, uncertainty and contradictions).
Never infer that two memories concern the same transaction or object without explicit evidence.
Never invent facts, image identities, memory IDs, or external knowledge. D is a retrieval aid;
when D conflicts with x or the original image, treat x/image as stronger evidence and preserve uncertainty.
Output JSON only: {"kept_memory_ids":["uuid"],"summary":"...","residuals":[{"text":"...","memory_ids":["uuid"]}]}.
An empty kept_memory_ids list, empty summary, and empty residuals are valid if nothing is relevant.
"""

MAX_FORMAT_RETRIES = 3
RAW_RESPONSE_EXCERPT_LENGTH = 2000


def _parse_json(text: str) -> dict:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("integration response must be an object")
    return decoded


class QueryAwareMemoryContextIntegrator:
    def __init__(self, llm: LLMClient, sources: SQLiteSourceStore) -> None:
        self._llm = llm
        self._sources = sources
        self._fallback = DefaultMemoryContextIntegrator()

    def _images(self, request: MemoryIntegrationRequest):
        references: dict[tuple[str, int], IntegrationImageRef] = {}
        image_parts: dict[tuple[str, int], ImagePart] = {}
        for hit in request.retrieved_memories:
            for relation in hit.memory.relations:
                if relation.source_type != "image":
                    continue
                index = (relation.locator or {}).get("part_index")
                if not isinstance(index, int) or index < 0:
                    raise SourceImageUnavailable(
                        f"memory {hit.memory.id}: image relation lacks part_index"
                    )
                key = (relation.source_id, index)
                if key not in references:
                    try:
                        message = self._sources.get_message(relation.source_id)
                        part = next(
                            (part for part in message.parts if part.part_index == index), None
                        )
                    except Exception as exc:
                        raise SourceImageUnavailable(
                            f"memory {hit.memory.id}: image {key} cannot be read: {exc}"
                        ) from exc
                    if part is None or part.part_type != "image" or not part.image_bytes:
                        raise SourceImageUnavailable(
                            f"memory {hit.memory.id}: image {key} has no original bytes"
                        )
                    media_type = part.media_type or "image/png"
                    try:
                        image_parts[key] = ImagePart(
                            data_url=f"data:{media_type};base64,"
                            + base64.b64encode(part.image_bytes).decode("ascii"),
                            detail="high",
                        )
                    except ValueError as exc:
                        raise SourceImageUnavailable(
                            f"memory {hit.memory.id}: unsupported image {key}: {exc}"
                        ) from exc
                    references[key] = IntegrationImageRef(
                        source_message_id=relation.source_id,
                        part_index=index,
                        media_type=media_type,
                    )
                if hit.memory.id not in references[key].memory_ids:
                    references[key].memory_ids.append(hit.memory.id)
        return references, image_parts

    @staticmethod
    def _prompt(request: MemoryIntegrationRequest, refs: dict) -> str:
        query_text = "\n".join(
            part.text for part in request.current_user.content if isinstance(part, TextPart)
        )
        memories = []
        for hit in request.retrieved_memories:
            memory = hit.memory
            memories.append({
                "id": str(memory.id),
                "kind": memory.x.kind,
                "x": memory.x.content,
                "created_at": memory.meta.created_at.isoformat(),
                "score": hit.score,
                "D": [
                    {"type": item.facet_type, "text": item.text}
                    for item in memory.descriptions
                ],
                "image_refs": [
                    {"source_message_id": key[0], "part_index": key[1]}
                    for key, ref in refs.items() if memory.id in ref.memory_ids
                ],
            })
        return json.dumps(
            {"query": query_text, "memories": memories}, ensure_ascii=False
        )

    async def integrate(self, request: MemoryIntegrationRequest) -> MemoryIntegrationResult:
        if not request.retrieved_memories:
            return MemoryIntegrationResult(metadata={"strategy": "query_aware", "memory_count": 0})
        started = time.monotonic()
        refs, image_parts = self._images(request)
        payload_text = self._prompt(request, refs)
        prompt_text = SYSTEM_PROMPT + "\n\nUSER PAYLOAD:\n" + payload_text
        query_images = [
            part for part in request.current_user.content if isinstance(part, ImagePart)
        ]
        parts = [TextPart(text=payload_text), *query_images, *image_parts.values()]
        format_errors: list[str] = []
        last_raw_response = ""
        try:
            for attempt in range(MAX_FORMAT_RETRIES + 1):
                correction = (
                    "\nYour previous response was malformed JSON. "
                    "Return one valid JSON object only, with correctly escaped strings "
                    "and all required commas and brackets."
                    if attempt else ""
                )
                response = await self._llm.complete(
                    [
                        LLMMessage(
                            role="system",
                            content=[TextPart(text=SYSTEM_PROMPT + correction)],
                        ),
                        LLMMessage(role="user", content=parts),
                    ],
                    response_format=ResponseFormat(),
                )
                try:
                    payload = _parse_json(response.text)
                    break
                except ValueError as exc:
                    format_errors.append(f"{type(exc).__name__}: {exc}")
                    last_raw_response = response.text[:RAW_RESPONSE_EXCERPT_LENGTH]
                    if attempt == MAX_FORMAT_RETRIES:
                        raise
            by_id = {str(hit.memory.id): hit.memory.id for hit in request.retrieved_memories}
            raw_ids = payload.get("kept_memory_ids")
            raw_ids = raw_ids if isinstance(raw_ids, list) else []
            ignored_refs: list[str] = []
            memory_ids = []
            for value in raw_ids:
                if isinstance(value, str) and value in by_id:
                    if by_id[value] not in memory_ids:
                        memory_ids.append(by_id[value])
                else:
                    ignored_refs.append(str(value))
            summary = payload.get("summary", "")
            summary = summary.strip() if isinstance(summary, str) else ""
            raw_residuals = payload.get("residuals")
            raw_residuals = raw_residuals if isinstance(raw_residuals, list) else []
            residuals = []
            for item in raw_residuals:
                if isinstance(item, str):
                    text = item.strip()
                    references = []
                elif isinstance(item, dict):
                    text = item.get("text", "")
                    text = text.strip() if isinstance(text, str) else ""
                    references = item.get("memory_ids") or []
                    references = references if isinstance(references, list) else []
                else:
                    continue
                if text:
                    residuals.append(text)
                # A residual may mention a valid memory the model omitted from
                # kept_memory_ids. Preserve that evidence instead of rejecting
                # the entire response; unknown references are merely ignored.
                for value in references:
                    if isinstance(value, str) and value in by_id:
                        if by_id[value] not in memory_ids:
                            memory_ids.append(by_id[value])
                    else:
                        ignored_refs.append(str(value))
            if not memory_ids and (summary or residuals):
                # The text is usable but its IDs are not. Keep all source
                # memories so the answer model still receives original images.
                memory_ids = list(by_id.values())
                ignored_refs.append("no resolvable selected memory IDs; kept Top-K")
            selected = set(memory_ids)
            context = "\n".join(
                segment for segment in [
                    "<mcr-relevant-memories>",
                    "摘要：" + summary if summary else "",
                    "补充细节：\n" + "\n".join(f"- {x}" for x in residuals) if residuals else "",
                    "</mcr-relevant-memories>",
                ] if segment
            ) if memory_ids and (summary or residuals) else ""
            if memory_ids and not context:
                selected_request = request.model_copy(update={
                    "retrieved_memories": [
                        hit for hit in request.retrieved_memories if hit.memory.id in selected
                    ]
                })
                context = (await self._fallback.integrate(selected_request)).context_text
            return MemoryIntegrationResult(
                memory_ids=memory_ids,
                context_text=context,
                summary=summary,
                residuals=residuals,
                dropped_memory_ids=[hit.memory.id for hit in request.retrieved_memories if hit.memory.id not in selected],
                image_refs=[ref for ref in refs.values() if selected.intersection(ref.memory_ids)],
                input_image_refs=list(refs.values()),
                prompt_text=prompt_text,
                metadata={
                    "strategy": "query_aware", "model": response.model,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "format_retries": len(format_errors),
                    "format_errors": format_errors,
                    "ignored_memory_references": ignored_refs,
                    "image_refs": [
                        {"source_message_id": ref.source_message_id, "part_index": ref.part_index}
                        for ref in refs.values()
                    ],
                },
            )
        except Exception as exc:
            try:
                fallback = await self._fallback.integrate(request)
            except Exception as fallback_exc:
                raise CriticalIntegrationError(
                    f"integration and fallback failed: {exc}; {fallback_exc}"
                ) from fallback_exc
            return fallback.model_copy(update={
                "image_refs": list(refs.values()),
                "input_image_refs": list(refs.values()),
                "prompt_text": prompt_text,
                "metadata": {
                    "strategy": "deterministic_fallback",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "format_retries": min(len(format_errors), MAX_FORMAT_RETRIES),
                    "format_errors": format_errors,
                    "raw_response_excerpt": last_raw_response,
                },
            })
