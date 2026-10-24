"""Relations between persisted memories."""

from __future__ import annotations

from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from mcr.memory.common import DomainModel, NonEmptyStr


def memory_relation_id(left: UUID, right: UUID) -> UUID:
    """Return one stable ID for an unordered pair of memories."""

    first, second = sorted((str(left), str(right)))
    return uuid5(NAMESPACE_URL, f"mcr:related:{first}:{second}")


class MemoryLink(DomainModel):
    relation_id: UUID
    target_memory_id: UUID
    relation_type: Literal["related_to"] = "related_to"
    reason: NonEmptyStr
