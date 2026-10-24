"""One exchange-level retrieval and ADD-only admission helpers."""

from __future__ import annotations

import base64

from mcr.extraction import CandidateMemory
from mcr.ingestion.judgment import JudgmentResult
from mcr.ingestion.models import JudgmentPayload, MemoryDecision, RelationDecision
from mcr.retrieval import RetrievalQuery
from mcr.source_store import StoredMessage
from mcr.memory import ImageSource


def exchange_retrieval_query(messages: list[StoredMessage]) -> RetrievalQuery | None:
    """Embed the current User/AI exchange once, including its actual images."""
    text_parts = []
    images = []
    for message in messages:
        for part in message.parts:
            if part.part_type == "text" and part.text_content and part.text_content.strip():
                text_parts.append(f"{message.role}: {part.text_content.strip()}")
            elif part.part_type == "image" and part.image_bytes and part.media_type:
                encoded = base64.b64encode(part.image_bytes).decode("ascii")
                images.append(ImageSource(data_url=f"data:{part.media_type};base64,{encoded}"))
    if not text_parts and not images:
        return None
    return RetrievalQuery(texts=["\n".join(text_parts)] if text_parts else [], images=images)


def additive_judgment(candidates: list[CandidateMemory]) -> JudgmentResult:
    """Turn extraction-time links into the existing bidirectional write protocol."""
    linked_ids = sorted(
        {link.memory_id for candidate in candidates for link in candidate.links},
        key=str,
    )
    aliases = {memory_id: f"existing-{index}" for index, memory_id in enumerate(linked_ids)}
    relations = []
    for index, candidate in enumerate(candidates):
        for link in candidate.links:
            relations.append(RelationDecision(
                left=f"candidate-{index}",
                right=aliases[link.memory_id],
                reason=link.reason,
            ))
    return JudgmentResult(
        payload=JudgmentPayload(
            decisions=[
                MemoryDecision(candidate=f"candidate-{index}", action="ADD")
                for index in range(len(candidates))
            ],
            relations=relations,
        ),
        mode="additive",
        requests=[],
        responses=[],
        existing_aliases={alias: memory_id for memory_id, alias in aliases.items()},
    )
