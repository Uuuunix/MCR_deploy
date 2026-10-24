"""Orchestration of prompt construction, LLM invocation, and provenance."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from time import perf_counter

from mcr.extraction.errors import ExtractionResponseError
from mcr.extraction.models import (
    CandidateMemory,
    CandidateMetadata,
    ExtractedMemory,
    ExtractionRequest,
    ExtractionResult,
    ExtractionTrace,
)
from mcr.extraction.candidate_discovery import build_candidate_prompt
from mcr.extraction.description_generation import generate_descriptions
from mcr.extraction.parser import deduplicate_memories, parse_extraction_response
from mcr.extraction.streaming import CandidateStreamAccumulator
from mcr.llm import ImagePart, LLMClient, LLMMessage, ResponseFormat, TextPart
from mcr.memory import ImageSource, MemoryContent, SourceRelation


def _image_reference_error(
    item: ExtractedMemory, source_messages: dict[str, object]
) -> str | None:
    unknown_legacy_ids = set(item.source_message_ids).difference(source_messages)
    if unknown_legacy_ids:
        return (
            "candidate cites non-current source message IDs: "
            + ", ".join(sorted(unknown_legacy_ids))
        )
    seen_refs = set()
    references = item.image_refs or item.source_refs
    for reference in references:
        key = (reference.source_message_id, reference.part_index)
        if key in seen_refs:
            return "candidate image refs must be unique"
        seen_refs.add(key)
        if reference.source_message_id not in source_messages:
            return "candidate image ref cites a non-current source message"
        message = source_messages[reference.source_message_id]
        if reference.part_index >= len(message.content):
            return "candidate image ref part_index is out of range"
        if not isinstance(message.content[reference.part_index], ImagePart):
            # Legacy source_refs often cited text. MCR now binds all current
            # text deterministically, so those entries are harmlessly ignored.
            if item.image_refs:
                return "candidate image ref must point to an image part"
    return None


class MemoryExtractor:
    def __init__(
        self,
        llm: LLMClient,
        recent_window_size: int = 8,
        schema_repair_attempts: int = 1,
        description_llm: LLMClient | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if recent_window_size < 0:
            raise ValueError("recent_window_size must be non-negative")
        if schema_repair_attempts < 0:
            raise ValueError("schema_repair_attempts must be non-negative")
        self._llm = llm
        self._description_llm = description_llm or llm
        self.recent_window_size = recent_window_size
        self.schema_repair_attempts = schema_repair_attempts
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def extract(self, request: ExtractionRequest) -> list[CandidateMemory]:
        return (await self.extract_with_trace(request)).candidates

    async def generate_descriptions(self, x: MemoryContent):
        """Generate flat D from one admitted x and no other context."""
        return await generate_descriptions(self._description_llm, x)

    async def _complete_candidates(self, messages: list[LLMMessage]):
        stream_method = getattr(self._llm, "complete_stream", None)
        if stream_method is None:
            response = await self._llm.complete(messages, ResponseFormat())
            return response, response.text, None
        accumulator = CandidateStreamAccumulator()
        response = await stream_method(messages, ResponseFormat(), accumulator.feed)
        parse_text = response.text
        if response.finish_reason in {
            "client_repetition_stop", "length", "stream_error"
        }:
            parse_text = accumulator.recovered_json() or response.text
        return response, parse_text, accumulator

    async def extract_with_trace(self, request: ExtractionRequest) -> ExtractionResult:
        messages, recent = build_candidate_prompt(request, self.recent_window_size)
        source_messages = {
            message.id: message
            for message in (request.current_user, request.current_assistant)
            if message is not None
        }
        image_parts_by_message = {
            message.id: [
                index
                for index, part in enumerate(message.content)
                if isinstance(part, ImagePart)
            ]
            for message in source_messages.values()
        }
        started = perf_counter()
        response, parse_text, stream = await self._complete_candidates(messages)
        response_attempts = [response.text]
        response_diagnostics = [_response_diagnostic(response, 1, stream)]
        schema_repair_used = False
        try:
            payload = parse_extraction_response(
                parse_text, image_parts_by_message=image_parts_by_message
            )
            if stream is None and _repeats_memory_content(parse_text):
                raise ExtractionResponseError(
                    "LLM response repeats the same candidate memories",
                    raw_response=response.text,
                )
        except ExtractionResponseError as first_error:
            last_error = first_error
            for _ in range(self.schema_repair_attempts):
                schema_repair_used = True
                if response.finish_reason == "length" or _repeats_memory_content(response.text):
                    # A truncated repetition cannot be repaired by appending a
                    # formatting instruction to the same long context. Keep
                    # current multimodal evidence, but remove historical text
                    # from this exceptional retry.
                    compact_request = request.model_copy(
                        update={"summary": None, "recent_messages": []}
                    )
                    repair_messages, _ = build_candidate_prompt(compact_request, 0)
                    repair_instruction = (
                        "The previous answer repeated facts or hit the output limit. "
                        "Use ONLY the CURRENT_SOURCE user message as the source of "
                        "new memories. Include each distinct supported event or fact "
                        "once, then stop. Do not reconstruct SUMMARY or RECENT_CONTEXT. "
                        "Return one complete JSON object."
                    )
                else:
                    repair_messages = list(messages)
                    repair_instruction = (
                        "Generate a fresh response from the original CURRENT_SOURCE. "
                        "The previous response was invalid; do not reuse it. "
                        "Include each distinct current-source event or fact once, "
                        "never copy memories from SUMMARY or RECENT_CONTEXT, and "
                        "return one complete valid JSON object only. "
                        "Validation errors: "
                        f"{json.dumps(last_error.issues[:3], ensure_ascii=False, default=str)}"
                    )
                repair_messages.append(
                    LLMMessage(role="user", content=[TextPart(text=repair_instruction)])
                )
                response, parse_text, stream = await self._complete_candidates(repair_messages)
                response_attempts.append(response.text)
                response_diagnostics.append(
                    _response_diagnostic(response, len(response_attempts), stream)
                )
                try:
                    payload = parse_extraction_response(
                        parse_text, image_parts_by_message=image_parts_by_message
                    )
                    if stream is None and _repeats_memory_content(parse_text):
                        raise ExtractionResponseError(
                            "LLM response repeats the same candidate memories",
                            raw_response=response.text,
                        )
                    break
                except ExtractionResponseError as exc:
                    last_error = exc
            else:
                last_error.response_diagnostics = response_diagnostics
                raise last_error
        elapsed_ms = (perf_counter() - started) * 1000
        extracted = deduplicate_memories(payload.memories)

        # The model sometimes copies image locations from RECENT_CONTEXT into
        # text-only current submissions. They cannot provide current-image
        # provenance, but the candidate's text can still be valid evidence.
        if not any(image_parts_by_message.values()):
            for item in extracted:
                item.image_refs = []

        discarded_candidates = []
        source_valid = []
        for item_index, item in enumerate(extracted):
            source_error = _image_reference_error(item, source_messages)
            if source_error is not None:
                discarded_candidates.append(
                    {"index": item_index, "content": item.content, "reason": source_error}
                )
            else:
                source_valid.append(item)
        if extracted and not source_valid:
            reasons = "; ".join(str(item["reason"]) for item in discarded_candidates)
            raise ExtractionResponseError(
                "all extracted candidates failed source validation: " + reasons
            )
        extracted = source_valid

        candidates = []
        for item_index, item in enumerate(extracted):
            relations = []
            # r binds the current message pair deterministically for text.
            for source_id, message in source_messages.items():
                for part_index, part in enumerate(message.content):
                    if isinstance(part, TextPart):
                        relations.append(
                            SourceRelation(
                                source_type="text",
                                source_id=source_id,
                                locator={"part_index": part_index},
                            )
                        )

            # Only images explicitly marked relevant by the multimodal x call
            # are attached to this memory.
            seen_refs = set()
            for reference in (item.image_refs or item.source_refs):
                key = (reference.source_message_id, reference.part_index)
                if key in seen_refs:
                    continue
                seen_refs.add(key)
                message = source_messages[reference.source_message_id]
                part = message.content[reference.part_index]
                if not isinstance(part, ImagePart):
                    continue
                relations.append(
                    SourceRelation(
                        source_type="image",
                        source_id=reference.source_message_id,
                        locator={"part_index": reference.part_index},
                        image_source=ImageSource(
                            url=part.url,
                            file_path=part.file_path,
                            data_url=part.data_url,
                        ),
                    )
                )

            cited_ids = list(source_messages)
            source_turn_ids = [
                source_messages[source_id].turn_id
                for source_id in cited_ids
                if source_messages[source_id].turn_id is not None
            ]
            candidates.append(
                CandidateMemory(
                    x=MemoryContent(
                        kind=item.kind,
                        content=item.content,
                    ),
                    meta=CandidateMetadata(
                        created_at=self._clock(),
                        observed_at=request.observation_time,
                        session_id=request.session_id,
                        source_message_ids=cited_ids,
                        source_turn_ids=source_turn_ids,
                        provider=response.provider,
                        model=response.model,
                        request_id=response.request_id,
                    ),
                    descriptions=[],
                    relations=relations,
                )
            )

        if extracted and not candidates:
            reasons = "; ".join(
                str(item["reason"]) for item in discarded_candidates
            )
            raise ExtractionResponseError(
                "all extracted candidates failed source validation: " + reasons
            )

        return ExtractionResult(
            candidates=candidates,
            trace=ExtractionTrace(
                messages=messages,
                provider_request=response.request,
                raw_response=response.text,
                elapsed_ms=elapsed_ms,
                provider=response.provider,
                model=response.model,
                request_id=response.request_id,
                usage=response.usage,
                effective_recent_message_ids=[message.id for message in recent],
                schema_repair_used=schema_repair_used,
                response_attempts=response_attempts,
                response_diagnostics=response_diagnostics,
                discarded_candidates=discarded_candidates,
            ),
        )


def _response_diagnostic(
    response, attempt: int, stream: CandidateStreamAccumulator | None = None
) -> dict[str, object]:
    diagnostic = {
        "attempt": attempt,
        "finish_reason": response.finish_reason,
        "completion_tokens": (
            response.usage.completion_tokens if response.usage is not None else None
        ),
        "response_chars": len(response.text),
    }
    if stream is not None:
        diagnostic["stream_complete_unique_memories"] = stream.complete_count
        diagnostic["stream_repetition_detected"] = stream.repetition_detected
    return diagnostic


_CONTENT_STRING = re.compile(r'"content"\s*:\s*"((?:\\.|[^"\\])*)"')


def _repeats_memory_content(text: str) -> bool:
    """Detect a generation loop without imposing a candidate count limit."""
    values = Counter(match.group(1).casefold().strip() for match in _CONTENT_STRING.finditer(text))
    return any(count >= 3 for count in values.values())

