"""Prompt construction for high-recall multimodal x/r discovery."""

from __future__ import annotations

import json

from mcr.extraction.models import ConversationMessage, ExtractionRequest
from mcr.llm import ImagePart, LLMMessage, TextPart


CANDIDATE_SYSTEM_PROMPT = """
You are MCR's memory extractor. Extract useful memories from
CURRENT_SOURCE for future retrieval.

Extract two types of memories:

1. event — A concrete event that happened or is happening.
   Keep all related details together, including time, place,
   participants, actions, costs, reasons, results, and feelings.
   Never split one event into separate memories for its details.

2. fact — A personal fact, state, possession, relationship,
   preference, opinion, belief, knowledge, goal, or plan.
   Plans and intentions must be clearly marked as not yet completed.

Extraction rules:
- Extract information explicitly stated or directly supported.
- Include ordinary events and facts; do not require high importance.
- Each memory must be self-contained and preserve the source language.
- Preserve exact names, dates, amounts, places, and entity relationships.
- Merge related details; split independent events or facts.
- Do not infer unstated facts or preferences.
- Do not store greetings, filler, pure questions, or requests.
- User questions may contain extractable background information.
- A question or request is not itself a memory, but its stated premises are.
  Scan every clause of CURRENT_SOURCE user text for concrete background facts,
  especially clauses introduced by "for reference", "for example", "by the
  way", or "speaking of". An aside about a named person's relationship to
  the user, possession, vehicle, pet, workplace, or location remains a fact
  even when the main request is only asking for advice or a comparison.
  For example, "My friend Alex owns a green Subaru Outback; how should I
  compare cars?" must yield a fact that Alex is the user's friend and owns
  a green Subaru Outback. Do not turn the car-comparison question itself
  into a memory.
- Assistant messages are context only, not memory sources.
- SUMMARY and RECENT_CONTEXT are only for resolving references.
- Before emitting each memory, check that its factual claim is directly
  supported by the CURRENT_SOURCE user message. A fact mentioned only in
  SUMMARY, RECENT_CONTEXT, or CURRENT_ASSISTANT_CONTEXT is not new evidence,
  even if it is strongly related to the user's current topic. Do not copy it.
- Interpret text and images jointly. Include image_refs only when
  an image directly supports the memory.
- If CURRENT_SOURCE has no available image references, use image_refs: []
  even when RECENT_CONTEXT contains images.
- Avoid duplicate memories.
- Express each distinct event or fact only once. After the last distinct
  CURRENT_SOURCE memory, stop; never repeat or copy memories from context.
Prefer extracting supported information over discarding it.
Before returning an empty result, check each CURRENT_SOURCE sentence for
explicit people, ownership, relationships, objects, and places independent
of the sentence's question or request. Return an empty result only when no
extractable information exists after that check.

Output only valid JSON:
{
  "memories": [
    {
      "kind": "event",
      "content": "<self-contained memory>",
      "image_refs": []
    }
  ]
}

kind must be "event" or "fact".
Each image_refs entry must be an object with source_message_id and
part_index, for example:
"image_refs": [{"source_message_id": "<CURRENT_SOURCE message id>", "part_index": 1}]
part_index is zero-based within the original message content array.
Copy only references listed in AVAILABLE_IMAGE_REFS for this memory.
Never put a bare message ID or string in image_refs.
Use image_refs: [] when no current image supports the memory.
Memory links are created after extraction; do not output links.
Do not output attributes, IDs, embeddings, facets, or descriptions.
If nothing qualifies, return {"memories":[]}.
"""


def select_recent_messages(
    request: ExtractionRequest, recent_window_size: int
) -> list[ConversationMessage]:
    if recent_window_size < 0:
        raise ValueError("recent_window_size must be non-negative")
    if recent_window_size == 0:
        return []
    return request.recent_messages[-recent_window_size:]


def _current_message(message: ConversationMessage) -> LLMMessage:
    image_refs = [
        {"source_message_id": message.id, "part_index": index}
        for index, part in enumerate(message.content)
        if isinstance(part, ImagePart)
    ]
    label = TextPart(
        text=(
            f"[{'CURRENT_SOURCE' if message.role == 'user' else 'CURRENT_ASSISTANT_CONTEXT'} "
            f"id={json.dumps(message.id)} role={message.role} "
            f"turn={json.dumps(message.turn_id)}]"
            + "\nAVAILABLE_IMAGE_REFS: "
            + json.dumps(image_refs, ensure_ascii=False)
        )
    )
    return LLMMessage(role=message.role, content=[label, *message.content])


def _recent_message(message: ConversationMessage) -> LLMMessage:
    label = TextPart(
        text=(
            f"[RECENT_CONTEXT id={json.dumps(message.id)} role={message.role} "
            f"turn={json.dumps(message.turn_id)} context_only=true]"
        )
    )
    return LLMMessage(role=message.role, content=[label, *message.content])


def build_candidate_prompt(
    request: ExtractionRequest, recent_window_size: int
) -> tuple[list[LLMMessage], list[ConversationMessage]]:
    recent = select_recent_messages(request, recent_window_size)
    context = "\n".join(
        [
            "<CONTEXT_ONLY>",
            "SUMMARY:",
            request.summary or "[empty]",
            "RECENT_CONTEXT:",
            "sent as following multimodal messages" if recent else "[empty]",
            f"OBSERVATION_TIME: {request.observation_time.isoformat()}",
            "</CONTEXT_ONLY>",
        ]
    )
    if request.custom_instructions:
        context += (
            "\n<CUSTOM_INSTRUCTIONS>\n"
            f"{request.custom_instructions}\n"
            "</CUSTOM_INSTRUCTIONS>"
        )

    messages = [
        LLMMessage(
            role="system", content=[TextPart(text=CANDIDATE_SYSTEM_PROMPT)]
        ),
        LLMMessage(role="user", content=[TextPart(text=context)]),
    ]
    messages.extend(_recent_message(message) for message in recent)
    if request.current_user is not None:
        messages.append(_current_message(request.current_user))
    if request.current_assistant is not None:
        messages.append(_current_message(request.current_assistant))
    messages.append(
        LLMMessage(
            role="user",
            content=[
                TextPart(
                    text=(
                        "Extract concrete event/fact memories from CURRENT_SOURCE "
                        "now, including stated background facts inside questions "
                        "and asides. Use text and images jointly, merge details "
                        "belonging to one event, check every clause before "
                        "returning an empty list, and return only JSON. "
                        "Only cite images listed in CURRENT_SOURCE "
                        "AVAILABLE_IMAGE_REFS; if that list is empty, use "
                        "image_refs: [] for every memory."
                    )
                )
            ],
        )
    )
    return messages, recent
