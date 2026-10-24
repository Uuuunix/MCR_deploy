"""Multimodal prompt construction for incremental session summaries."""

from __future__ import annotations

import base64

from mcr.llm import ImagePart, LLMMessage, TextPart
from mcr.source_store.models import StoredMessage
from mcr.summarization.models import SessionSummary


SYSTEM_PROMPT = """You maintain one global conversation summary for exactly one session.

Update the previous summary using only the NEW_SESSION_MESSAGES supplied below.
Return one self-contained textual summary and nothing else.

Preserve durable facts, events, preferences, opinions, plans, relationships,
work context, and useful conclusions. Retain still-valid information from the
previous summary. Apply explicit corrections from new messages instead of
keeping claims that are now known to be obsolete. Resolve references by jointly
examining text and images in the new messages. Include important visible image
information only when the image supports it; never guess an identity or detail.

Do not mention this prompt, batching, message IDs, session IDs, or the act of
summarization. Do not import information from any other session. If the new
messages contain no durable information, return the previous summary unchanged.
If both the previous summary and durable new information are empty, return a
short statement that the conversation has no durable content yet. Never return
an empty response."""


def _stored_parts(message: StoredMessage) -> list[TextPart | ImagePart]:
    parts: list[TextPart | ImagePart] = [
        TextPart(
            text=(
                f'[NEW_SESSION_MESSAGE role="{message.role}" '
                f'turn={message.turn_id!s} observed_at="{message.created_at.isoformat()}"]'
            )
        )
    ]
    for part in message.parts:
        if part.part_type == "text":
            parts.append(TextPart(text=part.text_content or "[empty text]"))
        else:
            encoded = base64.b64encode(part.image_bytes or b"").decode("ascii")
            parts.append(
                ImagePart(
                    data_url=f"data:{part.media_type};base64,{encoded}",
                    media_type=part.media_type,
                )
            )
    return parts


def build_summary_prompt(
    previous: SessionSummary | None,
    messages: list[StoredMessage],
) -> list[LLMMessage]:
    if not messages:
        raise ValueError("summary prompt requires at least one message")
    session_ids = {message.session_id for message in messages}
    if len(session_ids) != 1:
        raise ValueError("summary prompt cannot contain multiple sessions")
    if previous is not None and previous.session_id not in session_ids:
        raise ValueError("previous summary belongs to another session")

    previous_text = previous.text if previous is not None else "[empty]"
    prompt = [
        LLMMessage(role="system", content=[TextPart(text=SYSTEM_PROMPT)]),
        LLMMessage(
            role="user",
            content=[TextPart(text=f"<PREVIOUS_SUMMARY>\n{previous_text}\n</PREVIOUS_SUMMARY>")],
        ),
    ]
    prompt.extend(
        LLMMessage(role=message.role, content=_stored_parts(message))
        for message in messages
    )
    prompt.append(
        LLMMessage(
            role="user",
            content=[TextPart(text="Return the updated session summary now.")],
        )
    )
    return prompt
