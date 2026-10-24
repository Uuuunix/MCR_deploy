"""The normalized memory content represented by x."""

from __future__ import annotations

from pydantic import Field, JsonValue

from mcr.memory.common import DomainModel, NonEmptyStr


class MemoryContent(DomainModel):
    """One extracted fact, event, or other normalized memory unit."""

    kind: NonEmptyStr
    content: NonEmptyStr
    attributes: dict[str, JsonValue] = Field(default_factory=dict)

