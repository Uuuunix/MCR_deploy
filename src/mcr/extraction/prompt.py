"""Evidence-bounded multimodal prompt construction."""

from __future__ import annotations

import json

from mcr.extraction.models import ConversationMessage, ExtractionRequest
from mcr.llm import ImagePart, LLMMessage, TextPart

SYSTEM_PROMPT = """You are MCR's evidence-bound memory extractor.

Extract useful candidate memories ONLY from messages labeled CURRENT_SOURCE.
SUMMARY and RECENT_CONTEXT are context-only: use them to resolve names,
pronouns, references, and relative time, but NEVER extract a memory solely from
them and NEVER cite their message IDs.

Accuracy and completeness are both critical. A missed extraction loses context
that cannot be recovered later. Extract every distinct piece of memorable
information from the CURRENT_SOURCE messages; downstream judgment and
deduplication decide what is ultimately stored. When in doubt between omitting
a meaningful supported memory and extracting it, extract it.

Extract from both current user and current assistant messages. Useful memory
includes events, facts, feelings, opinions, preferences, plans, intentions,
constraints, current decision states, and other self-contained knowledge.
User messages may reveal things done, not done, not yet done, being considered,
delayed, stopped, or conditional on something else. A pending or negative state
such as "has not bought it yet", "is still deciding", or "is waiting for a
discount" is memorable when it would help interpret a future conversation.

Assistant content is eligible only when it adds a substantive recommendation,
plan, solution, researched fact, or factual conclusion the user may later
reference. Attribute it correctly: write "The assistant recommended ..." or
"The user was advised ..." rather than turning unaccepted advice into a user
preference. Do not duplicate an assistant echo of user content. Ignore greetings,
filler, vague acknowledgments, unsupported inference, and commentary about the
model.

Scan every CURRENT_SOURCE message and every topic within it. Do not let the
main request or a dominant topic hide incidental personal facts in an aside.
Extract each separate topic as a separate x. Implicit preferences supported by
the user's stated constraints, comparisons, or reasons are eligible. Casual
topics are not filler when they reveal specific facts, experiences, plans,
opinions, preferences, or current states. Return an empty memories list only
when the CURRENT_SOURCE messages are purely phatic or contain no specific,
useful information at all; uncertainty or incompleteness alone is not a reason
to return empty.

Do not treat a concrete fact as unimportant merely because it appears in an
aside or inside a longer request about another topic. Extract explicit durable
details such as purchases, owned objects, exact prices, dates, places, payment
methods, and completed actions as separate candidates. Preserve exact amounts,
currencies, quantities, and units in x.content whenever CURRENT_SOURCE states
them.

Treat every text and image part in a CURRENT_SOURCE message as evidence and
reason over them jointly. Resolve cross-modal references such as "this",
"this place", "this school", "in the image", and equivalent expressions in
other languages from the accompanying image. A short text is not vague when
an image grounds its referent. Read clearly visible names, signs, objects, and
events from images, but do not guess an identity that the image does not
support. If the exact identity is uncertain, retain a useful generic reference
to the pictured entity instead of inventing a name or dropping the memory.
Treat the image itself as direct evidence rather than merely repeating the
surrounding text. When text reports buying or owning "this" object and gives a
price, combine that text with a clearly recognizable pictured object in one
self-contained x. If an assistant's object label conflicts with what is
directly visible, prefer the supported visual description and do not repeat the
unsupported label.

Perform a visual entity coverage check for EVERY image in CURRENT_SOURCE before
finishing the response. Identify each primary visible object, place, document,
or event using the most specific common name that the pixels defensibly support.
Do not stop at a generic phrase such as "item", "thing", "device", "tool",
"contraption", or "kitchen gadget" when the object category is visually
recognizable. If accompanying text uses a vague reference such as "this" or
"thingamajig", resolve it to the visually identified category in x.content.
For an identifiable pictured object, include an entity_type facet whose text
contains both its specific common category and its useful broader functional
class (for example, a cordless drill is also a power tool). This broader class
must be grounded in the object's normal function, not invented from the user's
question. Also include a purpose or attribute facet when the image clearly
supports what the object is used for. Assistant guesses are textual claims, not
ground truth: inspect the image independently, and never downgrade a clear
visual identity to a vague noun merely because surrounding text is uncertain.

Create separate candidates for independent information, but include enough
context for each candidate to stand alone. Preserve the source language.
Resolve relative dates using OBSERVATION_TIME when the evidence supports it.
Every candidate must cite one or both source IDs from CURRENT_SOURCE messages.

For every candidate, classify it into any applicable subset of the six memory
domains: biography, experiences, preferences, social_circle, work, and
psychometrics. Each domain is optional, but every candidate must contain at
least one domain. Emit each applicable domain at most once. Every emitted D
must contain a domain-oriented core facet plus any evidence-supported atomic
facets. Do not include id or embedding; MCR generates them after extraction.

x.content is the complete memory. Every d_j text is a domain-specific retrieval
facet derived from x: select only the information relevant to that domain and
add useful domain context. Never copy x.content verbatim into a d_j text. Each
d_j has exactly one primary facet_type, but its text must repeat enough of the
entity or event anchor to be independently understandable and retrievable.
Do not emit a preference merely because the user bought or used something.

Every d_j uses {"facet_type":"...","text":"...","attributes":{...}}.
Use only these facet types and their exact attribute keys; omit unknown fields
instead of using null, and never add keys:
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

Numeric limits are strict: intensity and strength are 0..1; polarity is -1..1.
sentiment is positive, neutral, negative, or mixed. time.precision is exact,
day, week, month, year, relative, or unknown. An attributes object is always
required, even when it is empty. A D may contain multiple facets of the same
type when they describe different values.

Every source_refs entry must identify an exact part of a CURRENT_SOURCE message.
Part indexes follow the original content array: the label inserted by MCR is not
part 0. When a memory depends on both text and an image, source_refs MUST cite
every supporting part, including the image part. For example, text part 0
"我在图中的学校上过学" plus image part 1 with a clearly visible university name
should produce x.content "用户曾就读于该大学" with the visible name resolved,
a non-verbatim biography core d_j such as "教育经历：该大学校友", and source_refs
for both part 0 and part 1.

Coverage example for pending decisions, implicit preferences, and assistant
recommendations:
CURRENT_SOURCE user: "Comparing too many adjacent laptop models sends me down a
research rabbit hole. I still haven't replaced my laptop and I'm waiting for a
meaningful price drop before I commit."
CURRENT_SOURCE assistant: "For your workload, prioritize at least 16 GB of RAM
and set a purchase ceiling of 1,200 USD."
This contains multiple memories and MUST NOT return an empty list. It should
produce the equivalent of:
{"memories":[{"kind":"plan","content":"The user has not yet replaced their laptop and is waiting for a meaningful price drop before committing.","attributes":{},"domains":[{"domain":"preferences","items":[{"facet_type":"core","text":"The user's laptop replacement is conditional on a meaningful price drop","attributes":{"subject":"the user","predicate":"is delaying","object":"laptop replacement"}},{"facet_type":"status","text":"The laptop replacement remains pending until the price drops","attributes":{"subject":"laptop replacement","status":"waiting for a meaningful price drop"}}]}],"source_refs":[{"source_message_id":"user-id","part_index":0}]},{"kind":"preference","content":"The user prefers narrowly scoped laptop choices because comparing many adjacent models leads to an unproductive research rabbit hole.","attributes":{},"domains":[{"domain":"preferences","items":[{"facet_type":"core","text":"The user favors narrow laptop comparisons over many adjacent models","attributes":{"subject":"the user","predicate":"prefers","object":"narrowly scoped laptop choices"}},{"facet_type":"preference","text":"The user prefers fewer laptop options when making a purchase decision","attributes":{"target":"narrow laptop comparisons","polarity":1.0,"strength":0.8}}]}],"source_refs":[{"source_message_id":"user-id","part_index":0}]},{"kind":"recommendation","content":"The assistant recommended a laptop with at least 16 GB of RAM and a purchase ceiling of 1,200 USD.","attributes":{},"domains":[{"domain":"experiences","items":[{"facet_type":"core","text":"The user received a laptop memory and price recommendation","attributes":{"subject":"the assistant","predicate":"recommended","object":"laptop configuration"}},{"facet_type":"attribute","text":"The recommended laptop has at least 16 GB of RAM","attributes":{"entity":"laptop","name":"minimum RAM","value":16,"unit":"GB"}},{"facet_type":"cost","text":"The recommended laptop purchase ceiling is 1,200 USD","attributes":{"item":"laptop","amount":1200,"currency":"USD"}}]}],"source_refs":[{"source_message_id":"assistant-id","part_index":0}]}]}

Return exactly one JSON object with this shape and no additional keys:
{"memories":[{"kind":"event","content":"The user paid 15 USD for an espresso today.","attributes":{},"domains":[{"domain":"experiences","items":[{"facet_type":"core","text":"The user purchased a cup of coffee","attributes":{"subject":"the user","predicate":"purchased","object":"coffee"}},{"facet_type":"entity_type","text":"The purchased coffee was an espresso","attributes":{"entity":"coffee","type":"espresso"}},{"facet_type":"cost","text":"The user spent 15 USD on the espresso","attributes":{"item":"espresso","amount":15,"currency":"USD"}}]}],"source_refs":[{"source_message_id":"...","part_index":0}]}]}
If nothing memorable is present, return {"memories":[]}.
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
    label = TextPart(
        text=(
            f"[CURRENT_SOURCE id={json.dumps(message.id)} role={message.role} "
            f"turn={json.dumps(message.turn_id)}]"
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


def build_extraction_prompt(
    request: ExtractionRequest,
    recent_window_size: int,
    visual_grounding: str | None = None,
) -> tuple[list[LLMMessage], list[ConversationMessage]]:
    recent = select_recent_messages(request, recent_window_size)
    context = "\n".join(
        [
            "<CONTEXT_ONLY>",
            "[SUMMARY_CONTEXT]",
            "SUMMARY:",
            request.summary or "[empty]",
            "RECENT_CONTEXT:",
            "sent as following multimodal messages" if recent else "[empty]",
            f"OBSERVATION_TIME: {request.observation_time.isoformat()}",
            "</CONTEXT_ONLY>",
        ]
    )
    if request.custom_instructions:
        context += f"\n<CUSTOM_INSTRUCTIONS>\n{request.custom_instructions}\n</CUSTOM_INSTRUCTIONS>"
    if visual_grounding:
        context += (
            "\n<VISUAL_GROUNDING_FROM_CURRENT_IMAGES>\n"
            "The following compact observation was produced by an image-only "
            "grounding pass over CURRENT_SOURCE images. It is an observation of "
            "those CURRENT_SOURCE image parts, not historical context. Verify it "
            "against the images, use it to resolve vague textual references, and "
            "cite the original image parts rather than this context block. For "
            "each grounded entity, check whether CURRENT_SOURCE text states a "
            "memorable action or fact about it. A statement that the user bought, "
            "owns, found, used, visited, created, or photographed the grounded "
            "entity MUST produce a candidate; a simultaneous question and an "
            "incorrect assistant guess do not erase that supported fact. Do not "
            "return an empty memories list when such a supported fact exists.\n"
            f"{visual_grounding}\n"
            "</VISUAL_GROUNDING_FROM_CURRENT_IMAGES>"
        )

    messages = [
        LLMMessage(role="system", content=[TextPart(text=SYSTEM_PROMPT)]),
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
                        "Extract only from CURRENT_SOURCE messages above and return "
                        "the JSON object now. Complete the visual-grounding coverage "
                        "check before deciding that memories is empty."
                    )
                )
            ],
        )
    )
    return messages, recent


# The extraction API was split into candidate discovery and post-admission
# description generation. Keep this module path as a compatibility alias while
# ensuring callers receive the candidate-only prompt rather than the retired
# all-in-one domain prompt above.
from mcr.extraction.candidate_discovery import (  # noqa: E402
    CANDIDATE_SYSTEM_PROMPT as SYSTEM_PROMPT,
    build_candidate_prompt as build_extraction_prompt,
    select_recent_messages,
)


# Compatibility exports: candidate discovery now owns the public extraction
# prompt. The former all-in-one prompt remains above only for old imports while
# downstream code migrates; these final bindings guarantee current behavior.
from mcr.extraction.candidate_discovery import (  # noqa: E402
    CANDIDATE_SYSTEM_PROMPT,
    build_candidate_prompt,
)

SYSTEM_PROMPT = CANDIDATE_SYSTEM_PROMPT
build_extraction_prompt = build_candidate_prompt

