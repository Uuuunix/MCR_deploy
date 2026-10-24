"""LLM-only ADD/SKIP and related_to judgment."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from uuid import UUID

from mcr.ingestion.errors import JudgmentError
from mcr.ingestion.models import JudgmentPayload, MemoryDecision, RelationDecision
from mcr.llm import LLMClient, LLMMessage, ResponseFormat, TextPart
from mcr.memory import MemoryItem
from mcr.ingestion.models import PendingMemory
from mcr.retrieval import RetrievedMemory

_FENCE = re.compile(r"\A\s*```(?:json)?\s*(.*?)\s*```\s*\Z", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = """You are MCR's immutable memory admission judge.

You may ONLY decide whether each candidate is ADD or SKIP and which supplied
memories are related. Never rewrite candidate x, meta, D, or r. Never emit
UPDATE, DELETE, MERGE, confidence, or an ID absent from the input.

First identify a semantically equivalent memory among the supplied aliases.
If none exists, you MUST choose ADD with duplicate_of: null. Choose SKIP only
after identifying that memory; duplicate_of MUST be its supplied alias. SKIP
requires semantic equivalence with no meaningful new fact, time, degree,
person, event, or context. Never use SKIP to reject a candidate because it is
public, unimportant, unrelated to retrieved memories, or outside the user's
personal life. When uncertain whether two memories are duplicates, choose ADD.
Related but non-duplicate information is ADD. Contradictions and changed state
are both retained as ADD and related.

Relations use one implicit type, related_to. Give a concise reason; there is no
hard length limit. Emit a relation only when the two memories share a meaningful
subject, event, entity, preference, plan, place, time, or causal context. If two
memories are different or unrelated, emit no relation for that pair; never use a
relation merely to explain that they are different. Return one JSON object with
decisions and relations arrays. Valid decisions:
{"candidate":"candidate-0","action":"ADD","duplicate_of":null}
{"candidate":"candidate-0","action":"SKIP","duplicate_of":"existing-2"}
Invalid decision (never output this):
{"candidate":"candidate-0","action":"SKIP","duplicate_of":null}
For ADD, duplicate_of is null. For SKIP, duplicate_of names a supplied
semantically equivalent candidate or existing memory. The relations array may
be empty; otherwise each relation has left, right, and reason fields.
"""


@dataclass
class JudgmentResult:
    payload: JudgmentPayload
    mode: str
    requests: list[dict]
    responses: list[str]
    existing_aliases: dict[str, UUID]


def _memory_projection(memory: MemoryItem | PendingMemory) -> dict:
    """Return only semantic evidence needed for ADD/SKIP/link judgment.

    Source relations, existing memory links, generated IDs, and extraction
    bookkeeping do not help semantic comparison. Including them causes the
    judgment prompt to grow recursively as the graph gains links.
    """

    return {
        "x": memory.x.model_dump(mode="json"),
        "meta": {
            "session_id": memory.meta.session_id,
            "turn_id": memory.meta.turn_id,
            "observed_at": memory.meta.extra.get("observed_at"),
        },
        "descriptions": [
            item.model_dump(mode="json", exclude={"id", "embedding"})
            for item in memory.descriptions
        ],
    }


def _parse(text: str) -> JudgmentPayload:
    candidate = text.strip()
    fence = _FENCE.fullmatch(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        return JudgmentPayload.model_validate(json.loads(candidate))
    except Exception as exc:
        raise JudgmentError("judgment response is not valid structured JSON") from exc


class MemoryJudge:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def judge(
        self,
        candidates: list[MemoryItem | PendingMemory],
        retrieved: dict[str, list[RetrievedMemory]],
        *,
        allow_fallback: bool = True,
    ) -> JudgmentResult:
        aliases = {f"candidate-{index}" for index in range(len(candidates))}
        existing = {}
        for results in retrieved.values():
            for result in results:
                existing.setdefault(result.memory.id, result.memory)
        existing_aliases = {
            memory_id: f"existing-{index}"
            for index, memory_id in enumerate(sorted(existing, key=str))
        }
        existing_catalog = {
            existing_aliases[memory_id]: _memory_projection(memory)
            for memory_id, memory in existing.items()
        }
        allowed = aliases | set(existing_aliases.values())
        duplicate_targets = {
            alias: (aliases - {alias})
            | {
                existing_aliases[result.memory.id]
                for result in retrieved.get(alias, [])
            }
            for alias in aliases
        }

        batch_input = {
            "candidates": [
                {"id": f"candidate-{index}", "memory": _memory_projection(memory)}
                for index, memory in enumerate(candidates)
            ],
            "existing_memories": [
                {"id": alias, "memory": existing_catalog[alias]}
                for alias in sorted(existing_catalog)
            ],
            "retrieved": {
                alias: [
                    {
                        "id": existing_aliases[result.memory.id],
                        "score": result.score,
                    }
                    for result in retrieved.get(alias, [])
                ]
                for alias in aliases
            },
        }
        requests = []
        responses = []
        try:
            payload, raw, request = await self._call(batch_input)
            self._validate(payload, aliases, allowed, duplicate_targets)
            requests.append(request)
            responses.append(raw)
            return JudgmentResult(
                payload,
                "batch",
                requests,
                responses,
                {alias: memory_id for memory_id, alias in existing_aliases.items()},
            )
        except Exception:
            if not allow_fallback:
                raise
            decisions = []
            relations = []
            for index, candidate in enumerate(candidates):
                alias = f"candidate-{index}"
                single_input = {
                    "candidate": {"id": alias, "memory": _memory_projection(candidate)},
                    "retrieved": batch_input["retrieved"][alias],
                    "existing_memories": [
                        {
                            "id": item["id"],
                            "memory": existing_catalog[item["id"]],
                        }
                        for item in batch_input["retrieved"][alias]
                    ],
                    "peer_candidates": [
                        {"id": f"candidate-{peer}", "x": candidates[peer].x.model_dump(mode="json")}
                        for peer in range(len(candidates))
                        if peer != index
                    ],
                }
                last_error = None
                for _ in range(2):
                    try:
                        payload, raw, request = await self._call(
                            single_input, current_candidate=alias
                        )
                        own = [
                            decision for decision in payload.decisions
                            if decision.candidate == alias
                        ]
                        if len(own) != 1:
                            raise JudgmentError(
                                f"judgment must return exactly one decision for {alias}"
                            )
                        # Peers are comparison context, not additional decisions.
                        # Each peer receives its own fallback call below.
                        related = [
                            relation for relation in payload.relations
                            if alias in (relation.left, relation.right)
                        ]
                        unique_relations = {
                            tuple(sorted((relation.left, relation.right))): relation
                            for relation in related
                        }
                        current = JudgmentPayload(
                            decisions=own, relations=list(unique_relations.values())
                        )
                        self._validate(
                            current,
                            {alias},
                            allowed,
                            duplicate_targets,
                        )
                        break
                    except Exception as exc:
                        last_error = exc
                else:
                    raise JudgmentError(
                        f"judgment failed for {alias}: {last_error}"
                    ) from last_error
                decisions.extend(own)
                relations.extend(current.relations)
                requests.append(request)
                responses.append(raw)
            # The same peer-to-peer relation can be returned in both peers'
            # calls. Keep one undirected relation before the final validation.
            relations = list({
                tuple(sorted((relation.left, relation.right))): relation
                for relation in relations
            }.values())
            merged = JudgmentPayload(decisions=decisions, relations=relations)
            self._validate(merged, aliases, allowed, duplicate_targets)
            return JudgmentResult(
                merged,
                "per_candidate_fallback",
                requests,
                responses,
                {alias: memory_id for memory_id, alias in existing_aliases.items()},
            )

    async def _call(
        self, value: dict, *, current_candidate: str | None = None
    ) -> tuple[JudgmentPayload, str, dict]:
        system_prompt = SYSTEM_PROMPT
        if current_candidate is not None:
            system_prompt += (
                "\nThis is a single-candidate fallback. Return exactly one decision "
                f"for {current_candidate}. peer_candidates are comparison context only; "
                "do not decide them. Relations may include the current candidate "
                "and a supplied peer or existing memory.\n"
            )
        messages = [
            LLMMessage(role="system", content=[TextPart(text=system_prompt)]),
            LLMMessage(
                role="user",
                content=[TextPart(text=json.dumps(value, ensure_ascii=False))],
            ),
        ]
        response = await self._llm.complete(messages, ResponseFormat())
        return _parse(response.text), response.text, response.request or {
            "messages": [message.model_dump(mode="json") for message in messages]
        }

    @staticmethod
    def _validate(
        payload: JudgmentPayload,
        required_candidates: set[str],
        allowed: set[str],
        duplicate_targets: dict[str, set[str]],
        require_all: bool = True,
    ) -> None:
        decision_ids = [decision.candidate for decision in payload.decisions]
        if len(decision_ids) != len(set(decision_ids)):
            raise JudgmentError("candidate decisions must be unique")
        if require_all and set(decision_ids) != required_candidates:
            raise JudgmentError("judgment must cover every candidate exactly once")
        for decision in payload.decisions:
            if decision.candidate not in required_candidates:
                raise JudgmentError("judgment references an invalid candidate")
            if decision.duplicate_of is not None:
                if decision.duplicate_of not in duplicate_targets[decision.candidate]:
                    raise JudgmentError("duplicate_of references an invalid memory")
        seen_relations = set()
        for relation in payload.relations:
            if relation.left not in allowed or relation.right not in allowed:
                raise JudgmentError("relation references an invalid memory")
            if relation.left == relation.right:
                raise JudgmentError("a memory cannot relate to itself")
            key = tuple(sorted((relation.left, relation.right)))
            if key in seen_relations:
                raise JudgmentError("duplicate relation pair")
            seen_relations.add(key)
